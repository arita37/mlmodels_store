
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb3a0353ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb3a0353ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb40b90a048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb40b90a048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb425638e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb425638e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3b918a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3b918a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3b918a8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 453396.39it/s] 92%|█████████▏| 9076736/9912422 [00:00<00:01, 646296.08it/s]9920512it [00:00, 45567200.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 675980.25it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 951868.01it/s]1654784it [00:00, 12160949.15it/s]                         
0it [00:00, ?it/s]8192it [00:00, 301200.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3b64ac518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3b66e77b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb3b918a510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb3b918a510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb3b918a510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:31,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:30,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:30,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:29,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:28,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:28,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:27,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:27,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:26,  1.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<01:00,  2.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:00,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:00,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:00,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:59,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:59,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:58,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:58,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:58,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:57,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:57,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:56,  2.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:39,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:39,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:39,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:38,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:38,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:38,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:37,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:37,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:37,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:37,  3.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:25,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:25,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:25,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:25,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:24,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:24,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:24,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:24,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:15,  6.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:10,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 13.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 29.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 37.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 45.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.11 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.11 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.53 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.44s/ url]
0 examples [00:00, ? examples/s]2020-06-18 18:11:26.605504: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-18 18:11:26.620869: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-18 18:11:26.621051: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5623f9fec4a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-18 18:11:26.621067: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
43 examples [00:00, 427.47 examples/s]144 examples [00:00, 516.43 examples/s]245 examples [00:00, 604.45 examples/s]348 examples [00:00, 689.20 examples/s]442 examples [00:00, 748.00 examples/s]549 examples [00:00, 820.58 examples/s]659 examples [00:00, 887.00 examples/s]759 examples [00:00, 917.36 examples/s]860 examples [00:00, 941.08 examples/s]970 examples [00:01, 983.33 examples/s]1081 examples [00:01, 1015.41 examples/s]1185 examples [00:01, 1003.64 examples/s]1293 examples [00:01, 1022.95 examples/s]1397 examples [00:01, 1022.09 examples/s]1500 examples [00:01, 1013.51 examples/s]1603 examples [00:01, 1018.34 examples/s]1707 examples [00:01, 1023.98 examples/s]1814 examples [00:01, 1034.26 examples/s]1924 examples [00:01, 1052.45 examples/s]2032 examples [00:02, 1059.67 examples/s]2139 examples [00:02, 1024.38 examples/s]2245 examples [00:02, 1032.16 examples/s]2350 examples [00:02, 1035.15 examples/s]2455 examples [00:02, 1021.61 examples/s]2560 examples [00:02, 1028.95 examples/s]2665 examples [00:02, 1035.10 examples/s]2775 examples [00:02, 1053.50 examples/s]2883 examples [00:02, 1059.35 examples/s]2993 examples [00:02, 1069.30 examples/s]3101 examples [00:03, 1042.60 examples/s]3206 examples [00:03, 1033.24 examples/s]3310 examples [00:03, 1029.23 examples/s]3415 examples [00:03, 1033.38 examples/s]3519 examples [00:03, 1034.21 examples/s]3625 examples [00:03, 1040.06 examples/s]3730 examples [00:03, 1040.99 examples/s]3838 examples [00:03, 1049.45 examples/s]3943 examples [00:03, 1036.74 examples/s]4051 examples [00:03, 1048.53 examples/s]4156 examples [00:04, 1027.36 examples/s]4265 examples [00:04, 1044.91 examples/s]4375 examples [00:04, 1060.46 examples/s]4484 examples [00:04, 1067.77 examples/s]4591 examples [00:04, 1053.64 examples/s]4697 examples [00:04, 1037.60 examples/s]4801 examples [00:04, 1028.30 examples/s]4906 examples [00:04, 1032.57 examples/s]5010 examples [00:04, 1016.65 examples/s]5117 examples [00:04, 1029.66 examples/s]5228 examples [00:05, 1050.01 examples/s]5334 examples [00:05, 1038.51 examples/s]5442 examples [00:05, 1050.11 examples/s]5549 examples [00:05, 1055.76 examples/s]5655 examples [00:05, 1046.63 examples/s]5763 examples [00:05, 1056.10 examples/s]5869 examples [00:05, 1055.22 examples/s]5975 examples [00:05, 1054.80 examples/s]6082 examples [00:05, 1059.07 examples/s]6188 examples [00:06, 1053.58 examples/s]6294 examples [00:06, 1049.90 examples/s]6400 examples [00:06, 1023.01 examples/s]6503 examples [00:06, 1011.34 examples/s]6607 examples [00:06, 1017.25 examples/s]6714 examples [00:06, 1030.51 examples/s]6820 examples [00:06, 1038.58 examples/s]6926 examples [00:06, 1043.32 examples/s]7031 examples [00:06, 1019.08 examples/s]7136 examples [00:06, 1026.16 examples/s]7242 examples [00:07, 1033.60 examples/s]7346 examples [00:07, 1006.76 examples/s]7453 examples [00:07, 1024.67 examples/s]7556 examples [00:07, 1006.43 examples/s]7665 examples [00:07, 1029.34 examples/s]7773 examples [00:07, 1042.35 examples/s]7878 examples [00:07, 1036.10 examples/s]7983 examples [00:07, 1038.14 examples/s]8087 examples [00:07, 1010.13 examples/s]8191 examples [00:07, 1015.82 examples/s]8293 examples [00:08, 1015.24 examples/s]8398 examples [00:08, 1025.01 examples/s]8509 examples [00:08, 1048.99 examples/s]8617 examples [00:08, 1058.06 examples/s]8723 examples [00:08, 1052.05 examples/s]8833 examples [00:08, 1065.58 examples/s]8943 examples [00:08, 1073.06 examples/s]9053 examples [00:08, 1078.34 examples/s]9161 examples [00:08, 1074.80 examples/s]9272 examples [00:08, 1082.35 examples/s]9381 examples [00:09, 1079.10 examples/s]9491 examples [00:09, 1083.42 examples/s]9600 examples [00:09, 1053.56 examples/s]9708 examples [00:09, 1061.02 examples/s]9817 examples [00:09, 1068.87 examples/s]9925 examples [00:09, 1037.52 examples/s]10030 examples [00:09, 988.65 examples/s]10139 examples [00:09, 1016.70 examples/s]10244 examples [00:09, 1024.01 examples/s]10349 examples [00:10, 1029.81 examples/s]10459 examples [00:10, 1047.63 examples/s]10569 examples [00:10, 1061.52 examples/s]10680 examples [00:10, 1075.57 examples/s]10792 examples [00:10, 1086.79 examples/s]10903 examples [00:10, 1092.61 examples/s]11014 examples [00:10, 1095.40 examples/s]11124 examples [00:10, 1056.50 examples/s]11231 examples [00:10, 1055.91 examples/s]11337 examples [00:10, 1056.08 examples/s]11443 examples [00:11, 1046.33 examples/s]11548 examples [00:11, 1042.49 examples/s]11653 examples [00:11, 1014.65 examples/s]11763 examples [00:11, 1037.12 examples/s]11868 examples [00:11, 1035.48 examples/s]11972 examples [00:11, 1018.82 examples/s]12075 examples [00:11, 1018.63 examples/s]12182 examples [00:11, 1031.85 examples/s]12287 examples [00:11, 1035.06 examples/s]12392 examples [00:11, 1037.28 examples/s]12498 examples [00:12, 1042.10 examples/s]12603 examples [00:12, 1038.66 examples/s]12712 examples [00:12, 1051.14 examples/s]12818 examples [00:12, 1020.14 examples/s]12922 examples [00:12, 1023.97 examples/s]13031 examples [00:12, 1035.22 examples/s]13137 examples [00:12, 1040.28 examples/s]13242 examples [00:12, 1034.03 examples/s]13351 examples [00:12, 1049.14 examples/s]13462 examples [00:12, 1064.36 examples/s]13572 examples [00:13, 1070.88 examples/s]13680 examples [00:13, 1062.30 examples/s]13789 examples [00:13, 1067.84 examples/s]13896 examples [00:13, 1067.91 examples/s]14003 examples [00:13, 1061.65 examples/s]14115 examples [00:13, 1077.14 examples/s]14223 examples [00:13, 1069.50 examples/s]14331 examples [00:13, 1067.18 examples/s]14438 examples [00:13, 1056.12 examples/s]14548 examples [00:14, 1067.10 examples/s]14656 examples [00:14, 1069.53 examples/s]14764 examples [00:14, 1070.22 examples/s]14874 examples [00:14, 1076.49 examples/s]14983 examples [00:14, 1077.74 examples/s]15093 examples [00:14, 1082.60 examples/s]15202 examples [00:14, 1082.70 examples/s]15311 examples [00:14, 1078.13 examples/s]15419 examples [00:14, 1061.91 examples/s]15528 examples [00:14, 1069.82 examples/s]15638 examples [00:15, 1076.73 examples/s]15746 examples [00:15, 1052.74 examples/s]15852 examples [00:15, 1015.36 examples/s]15955 examples [00:15, 1016.00 examples/s]16057 examples [00:15, 997.76 examples/s] 16158 examples [00:15, 992.33 examples/s]16260 examples [00:15, 999.99 examples/s]16369 examples [00:15, 1023.63 examples/s]16476 examples [00:15, 1035.93 examples/s]16586 examples [00:15, 1052.13 examples/s]16692 examples [00:16, 1050.85 examples/s]16798 examples [00:16, 1043.17 examples/s]16903 examples [00:16, 1021.96 examples/s]17014 examples [00:16, 1044.79 examples/s]17119 examples [00:16, 1045.78 examples/s]17224 examples [00:16, 1043.37 examples/s]17333 examples [00:16, 1056.47 examples/s]17439 examples [00:16, 1041.79 examples/s]17546 examples [00:16, 1048.97 examples/s]17652 examples [00:16, 1046.26 examples/s]17757 examples [00:17, 1046.25 examples/s]17865 examples [00:17, 1054.72 examples/s]17974 examples [00:17, 1062.70 examples/s]18085 examples [00:17, 1073.77 examples/s]18195 examples [00:17, 1080.05 examples/s]18304 examples [00:17, 1079.09 examples/s]18415 examples [00:17, 1085.15 examples/s]18525 examples [00:17, 1088.23 examples/s]18635 examples [00:17, 1089.12 examples/s]18744 examples [00:17, 1088.24 examples/s]18854 examples [00:18, 1089.06 examples/s]18963 examples [00:18, 1087.29 examples/s]19072 examples [00:18, 1066.33 examples/s]19182 examples [00:18, 1073.44 examples/s]19290 examples [00:18, 1032.28 examples/s]19397 examples [00:18, 1040.81 examples/s]19507 examples [00:18, 1055.82 examples/s]19617 examples [00:18, 1066.03 examples/s]19727 examples [00:18, 1073.68 examples/s]19836 examples [00:19, 1075.86 examples/s]19944 examples [00:19, 1065.03 examples/s]20051 examples [00:19, 1001.22 examples/s]20152 examples [00:19, 990.74 examples/s] 20263 examples [00:19, 1022.72 examples/s]20370 examples [00:19, 1017.20 examples/s]20477 examples [00:19, 1032.45 examples/s]20585 examples [00:19, 1044.63 examples/s]20690 examples [00:19, 1040.20 examples/s]20795 examples [00:19, 1042.47 examples/s]20906 examples [00:20, 1059.45 examples/s]21014 examples [00:20, 1065.50 examples/s]21122 examples [00:20, 1068.70 examples/s]21231 examples [00:20, 1072.69 examples/s]21341 examples [00:20, 1079.49 examples/s]21450 examples [00:20, 1065.67 examples/s]21557 examples [00:20, 1058.95 examples/s]21667 examples [00:20, 1069.27 examples/s]21774 examples [00:20, 1069.16 examples/s]21881 examples [00:20, 1067.31 examples/s]21988 examples [00:21, 1067.40 examples/s]22095 examples [00:21, 1043.57 examples/s]22200 examples [00:21, 1035.35 examples/s]22304 examples [00:21, 1016.21 examples/s]22406 examples [00:21, 1015.87 examples/s]22508 examples [00:21, 1009.30 examples/s]22610 examples [00:21, 982.87 examples/s] 22717 examples [00:21, 1005.78 examples/s]22823 examples [00:21, 1021.12 examples/s]22933 examples [00:21, 1042.77 examples/s]23040 examples [00:22, 1049.59 examples/s]23149 examples [00:22, 1058.99 examples/s]23256 examples [00:22, 1021.80 examples/s]23359 examples [00:22, 984.19 examples/s] 23462 examples [00:22, 997.22 examples/s]23565 examples [00:22, 1006.23 examples/s]23673 examples [00:22, 1025.81 examples/s]23783 examples [00:22, 1045.09 examples/s]23889 examples [00:22, 1048.38 examples/s]23995 examples [00:23, 1041.85 examples/s]24102 examples [00:23, 1047.65 examples/s]24209 examples [00:23, 1053.61 examples/s]24315 examples [00:23, 1054.00 examples/s]24422 examples [00:23, 1057.26 examples/s]24528 examples [00:23, 1048.99 examples/s]24638 examples [00:23, 1061.31 examples/s]24746 examples [00:23, 1063.84 examples/s]24857 examples [00:23, 1075.03 examples/s]24965 examples [00:23, 1076.18 examples/s]25073 examples [00:24, 1041.45 examples/s]25181 examples [00:24, 1052.58 examples/s]25287 examples [00:24, 1051.50 examples/s]25396 examples [00:24, 1061.90 examples/s]25506 examples [00:24, 1072.30 examples/s]25614 examples [00:24, 1043.57 examples/s]25721 examples [00:24, 1050.90 examples/s]25829 examples [00:24, 1058.62 examples/s]25938 examples [00:24, 1066.61 examples/s]26046 examples [00:24, 1067.97 examples/s]26156 examples [00:25, 1075.29 examples/s]26265 examples [00:25, 1076.59 examples/s]26373 examples [00:25, 1071.93 examples/s]26481 examples [00:25, 1069.19 examples/s]26588 examples [00:25, 1066.26 examples/s]26696 examples [00:25, 1068.35 examples/s]26803 examples [00:25, 1051.77 examples/s]26909 examples [00:25, 1011.85 examples/s]27014 examples [00:25, 1022.57 examples/s]27119 examples [00:25, 1030.25 examples/s]27224 examples [00:26, 1035.30 examples/s]27331 examples [00:26, 1044.58 examples/s]27440 examples [00:26, 1055.55 examples/s]27548 examples [00:26, 1062.21 examples/s]27655 examples [00:26, 1008.68 examples/s]27757 examples [00:26, 988.92 examples/s] 27857 examples [00:26, 980.20 examples/s]27956 examples [00:26, 982.23 examples/s]28055 examples [00:26, 969.91 examples/s]28153 examples [00:27, 971.62 examples/s]28262 examples [00:27, 1003.07 examples/s]28363 examples [00:27, 1003.62 examples/s]28467 examples [00:27, 1013.78 examples/s]28574 examples [00:27, 1029.41 examples/s]28680 examples [00:27, 1036.59 examples/s]28784 examples [00:27, 1005.91 examples/s]28888 examples [00:27, 1013.70 examples/s]28990 examples [00:27, 1011.43 examples/s]29098 examples [00:27, 1028.54 examples/s]29202 examples [00:28, 1031.65 examples/s]29308 examples [00:28, 1037.81 examples/s]29412 examples [00:28, 1031.72 examples/s]29520 examples [00:28, 1044.31 examples/s]29625 examples [00:28, 1038.49 examples/s]29729 examples [00:28, 1020.82 examples/s]29836 examples [00:28, 1034.87 examples/s]29945 examples [00:28, 1047.91 examples/s]30050 examples [00:28, 992.14 examples/s] 30150 examples [00:28, 993.67 examples/s]30254 examples [00:29, 1005.55 examples/s]30355 examples [00:29, 998.44 examples/s] 30456 examples [00:29, 998.03 examples/s]30559 examples [00:29, 1000.98 examples/s]30662 examples [00:29, 1008.22 examples/s]30763 examples [00:29, 1001.27 examples/s]30871 examples [00:29, 1021.19 examples/s]30974 examples [00:29, 1018.11 examples/s]31081 examples [00:29, 1031.55 examples/s]31185 examples [00:29, 1032.44 examples/s]31291 examples [00:30, 1040.01 examples/s]31396 examples [00:30, 1022.22 examples/s]31500 examples [00:30, 1027.04 examples/s]31603 examples [00:30, 1027.27 examples/s]31709 examples [00:30, 1034.80 examples/s]31814 examples [00:30, 1036.78 examples/s]31918 examples [00:30, 992.95 examples/s] 32019 examples [00:30, 997.81 examples/s]32120 examples [00:30, 990.56 examples/s]32223 examples [00:31, 1001.50 examples/s]32325 examples [00:31, 1006.24 examples/s]32434 examples [00:31, 1029.87 examples/s]32538 examples [00:31, 1029.22 examples/s]32643 examples [00:31, 1034.41 examples/s]32747 examples [00:31, 1026.86 examples/s]32851 examples [00:31, 1029.34 examples/s]32958 examples [00:31, 1036.10 examples/s]33062 examples [00:31, 1028.24 examples/s]33165 examples [00:31, 1025.87 examples/s]33273 examples [00:32, 1039.62 examples/s]33381 examples [00:32, 1049.58 examples/s]33488 examples [00:32, 1053.46 examples/s]33594 examples [00:32, 1014.30 examples/s]33703 examples [00:32, 1034.64 examples/s]33810 examples [00:32, 1044.75 examples/s]33915 examples [00:32, 1000.02 examples/s]34020 examples [00:32, 1013.07 examples/s]34124 examples [00:32, 1019.72 examples/s]34232 examples [00:32, 1034.66 examples/s]34336 examples [00:33, 1023.36 examples/s]34445 examples [00:33, 1040.28 examples/s]34553 examples [00:33, 1051.59 examples/s]34661 examples [00:33, 1057.65 examples/s]34771 examples [00:33, 1068.27 examples/s]34878 examples [00:33, 1060.37 examples/s]34985 examples [00:33, 1058.20 examples/s]35091 examples [00:33, 1056.86 examples/s]35199 examples [00:33, 1061.02 examples/s]35306 examples [00:33, 1044.18 examples/s]35411 examples [00:34, 1035.84 examples/s]35516 examples [00:34, 1037.44 examples/s]35623 examples [00:34, 1046.62 examples/s]35731 examples [00:34, 1054.74 examples/s]35839 examples [00:34, 1059.91 examples/s]35947 examples [00:34, 1065.37 examples/s]36054 examples [00:34, 1055.12 examples/s]36163 examples [00:34, 1064.36 examples/s]36270 examples [00:34, 1061.26 examples/s]36377 examples [00:34, 1063.31 examples/s]36484 examples [00:35, 1060.73 examples/s]36591 examples [00:35, 1063.49 examples/s]36701 examples [00:35, 1071.81 examples/s]36809 examples [00:35, 1061.41 examples/s]36916 examples [00:35, 1060.10 examples/s]37023 examples [00:35, 1061.21 examples/s]37130 examples [00:35, 1050.79 examples/s]37236 examples [00:35, 1031.29 examples/s]37340 examples [00:35, 1010.88 examples/s]37448 examples [00:36, 1030.28 examples/s]37553 examples [00:36, 1035.97 examples/s]37661 examples [00:36, 1047.51 examples/s]37771 examples [00:36, 1061.51 examples/s]37880 examples [00:36, 1067.95 examples/s]37987 examples [00:36, 1055.77 examples/s]38093 examples [00:36, 1036.42 examples/s]38197 examples [00:36, 1003.78 examples/s]38305 examples [00:36, 1025.04 examples/s]38411 examples [00:36, 1033.21 examples/s]38515 examples [00:37, 1022.14 examples/s]38618 examples [00:37, 970.03 examples/s] 38723 examples [00:37, 992.20 examples/s]38832 examples [00:37, 1019.49 examples/s]38939 examples [00:37, 1033.62 examples/s]39047 examples [00:37, 1045.64 examples/s]39152 examples [00:37, 1040.94 examples/s]39258 examples [00:37, 1045.55 examples/s]39365 examples [00:37, 1052.12 examples/s]39471 examples [00:37, 1039.00 examples/s]39576 examples [00:38, 1040.89 examples/s]39681 examples [00:38, 1039.63 examples/s]39789 examples [00:38, 1050.53 examples/s]39895 examples [00:38, 1041.26 examples/s]40001 examples [00:38, 993.39 examples/s] 40110 examples [00:38, 1018.28 examples/s]40218 examples [00:38, 1034.70 examples/s]40322 examples [00:38, 1010.87 examples/s]40424 examples [00:38, 975.47 examples/s] 40532 examples [00:39, 1003.74 examples/s]40639 examples [00:39, 1022.17 examples/s]40748 examples [00:39, 1038.99 examples/s]40857 examples [00:39, 1052.35 examples/s]40963 examples [00:39, 1051.78 examples/s]41072 examples [00:39, 1062.23 examples/s]41180 examples [00:39, 1067.14 examples/s]41288 examples [00:39, 1069.22 examples/s]41396 examples [00:39, 1063.98 examples/s]41503 examples [00:39, 1047.53 examples/s]41611 examples [00:40, 1055.91 examples/s]41720 examples [00:40, 1064.63 examples/s]41827 examples [00:40, 1064.20 examples/s]41934 examples [00:40, 1053.89 examples/s]42040 examples [00:40, 1045.03 examples/s]42149 examples [00:40, 1055.35 examples/s]42255 examples [00:40, 1055.29 examples/s]42362 examples [00:40, 1059.22 examples/s]42468 examples [00:40, 1032.86 examples/s]42577 examples [00:40, 1046.75 examples/s]42682 examples [00:41, 1029.85 examples/s]42786 examples [00:41, 1031.74 examples/s]42892 examples [00:41, 1039.60 examples/s]42997 examples [00:41, 1018.81 examples/s]43104 examples [00:41, 1033.21 examples/s]43212 examples [00:41, 1044.32 examples/s]43317 examples [00:41, 1038.86 examples/s]43422 examples [00:41, 1038.79 examples/s]43526 examples [00:41, 1014.76 examples/s]43628 examples [00:41, 962.55 examples/s] 43726 examples [00:42, 964.00 examples/s]43833 examples [00:42, 993.18 examples/s]43935 examples [00:42, 1000.68 examples/s]44039 examples [00:42, 1009.93 examples/s]44143 examples [00:42, 1016.61 examples/s]44251 examples [00:42, 1034.28 examples/s]44358 examples [00:42, 1042.87 examples/s]44463 examples [00:42, 1040.65 examples/s]44569 examples [00:42, 1043.90 examples/s]44676 examples [00:42, 1050.73 examples/s]44782 examples [00:43, 1005.64 examples/s]44888 examples [00:43, 1021.28 examples/s]44996 examples [00:43, 1036.88 examples/s]45104 examples [00:43, 1048.89 examples/s]45210 examples [00:43, 1043.07 examples/s]45318 examples [00:43, 1052.52 examples/s]45424 examples [00:43, 1049.03 examples/s]45531 examples [00:43, 1055.09 examples/s]45640 examples [00:43, 1063.86 examples/s]45748 examples [00:44, 1067.92 examples/s]45855 examples [00:44, 1062.67 examples/s]45962 examples [00:44, 1063.28 examples/s]46071 examples [00:44, 1070.49 examples/s]46179 examples [00:44, 1065.32 examples/s]46288 examples [00:44, 1072.15 examples/s]46396 examples [00:44, 1060.65 examples/s]46503 examples [00:44, 1063.25 examples/s]46610 examples [00:44, 1051.06 examples/s]46716 examples [00:44, 1026.60 examples/s]46819 examples [00:45, 1002.62 examples/s]46920 examples [00:45, 996.41 examples/s] 47020 examples [00:45, 980.30 examples/s]47125 examples [00:45, 998.14 examples/s]47226 examples [00:45, 999.11 examples/s]47335 examples [00:45, 1022.64 examples/s]47443 examples [00:45, 1036.87 examples/s]47550 examples [00:45, 1046.52 examples/s]47655 examples [00:45, 1030.68 examples/s]47763 examples [00:45, 1044.25 examples/s]47871 examples [00:46, 1054.16 examples/s]47979 examples [00:46, 1058.85 examples/s]48088 examples [00:46, 1065.95 examples/s]48196 examples [00:46, 1068.00 examples/s]48305 examples [00:46, 1072.69 examples/s]48413 examples [00:46, 1065.69 examples/s]48521 examples [00:46, 1066.95 examples/s]48628 examples [00:46, 1067.67 examples/s]48735 examples [00:46, 1048.64 examples/s]48842 examples [00:46, 1053.42 examples/s]48949 examples [00:47, 1057.73 examples/s]49055 examples [00:47, 1041.29 examples/s]49160 examples [00:47, 1037.80 examples/s]49266 examples [00:47, 1042.88 examples/s]49371 examples [00:47, 1044.82 examples/s]49479 examples [00:47, 1052.75 examples/s]49585 examples [00:47, 1051.66 examples/s]49691 examples [00:47, 1052.14 examples/s]49797 examples [00:47, 1032.40 examples/s]49901 examples [00:47, 1026.96 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6780/50000 [00:00<00:00, 67799.74 examples/s] 40%|███▉      | 19900/50000 [00:00<00:00, 79294.77 examples/s] 66%|██████▋   | 33202/50000 [00:00<00:00, 90225.83 examples/s] 93%|█████████▎| 46545/50000 [00:00<00:00, 99932.30 examples/s]                                                               0 examples [00:00, ? examples/s]90 examples [00:00, 898.23 examples/s]196 examples [00:00, 940.64 examples/s]307 examples [00:00, 985.62 examples/s]414 examples [00:00, 1007.57 examples/s]526 examples [00:00, 1036.95 examples/s]638 examples [00:00, 1059.34 examples/s]735 examples [00:00, 1029.90 examples/s]847 examples [00:00, 1052.75 examples/s]958 examples [00:00, 1066.95 examples/s]1067 examples [00:01, 1072.91 examples/s]1178 examples [00:01, 1082.71 examples/s]1285 examples [00:01, 1075.94 examples/s]1394 examples [00:01, 1078.43 examples/s]1502 examples [00:01, 1068.58 examples/s]1609 examples [00:01, 1052.56 examples/s]1714 examples [00:01, 1049.78 examples/s]1820 examples [00:01, 1050.45 examples/s]1926 examples [00:01, 1052.69 examples/s]2033 examples [00:01, 1055.48 examples/s]2144 examples [00:02, 1068.52 examples/s]2251 examples [00:02, 1066.35 examples/s]2358 examples [00:02, 1054.91 examples/s]2464 examples [00:02, 1039.94 examples/s]2569 examples [00:02, 1030.88 examples/s]2673 examples [00:02, 1002.79 examples/s]2782 examples [00:02, 1027.11 examples/s]2886 examples [00:02, 1023.78 examples/s]2996 examples [00:02, 1045.13 examples/s]3108 examples [00:02, 1064.81 examples/s]3216 examples [00:03, 1066.89 examples/s]3323 examples [00:03, 1054.15 examples/s]3430 examples [00:03, 1055.98 examples/s]3536 examples [00:03, 1041.33 examples/s]3648 examples [00:03, 1062.99 examples/s]3759 examples [00:03, 1076.23 examples/s]3868 examples [00:03, 1080.07 examples/s]3977 examples [00:03, 1061.51 examples/s]4087 examples [00:03, 1071.77 examples/s]4198 examples [00:03, 1080.32 examples/s]4307 examples [00:04, 1082.99 examples/s]4416 examples [00:04, 1084.17 examples/s]4525 examples [00:04, 1065.06 examples/s]4632 examples [00:04, 1052.16 examples/s]4743 examples [00:04, 1066.38 examples/s]4851 examples [00:04, 1070.26 examples/s]4961 examples [00:04, 1076.86 examples/s]5069 examples [00:04, 1035.80 examples/s]5180 examples [00:04, 1055.37 examples/s]5286 examples [00:04, 1052.67 examples/s]5398 examples [00:05, 1071.18 examples/s]5507 examples [00:05, 1076.36 examples/s]5615 examples [00:05, 1058.22 examples/s]5724 examples [00:05, 1066.96 examples/s]5835 examples [00:05, 1077.80 examples/s]5943 examples [00:05, 1057.38 examples/s]6049 examples [00:05, 1055.90 examples/s]6155 examples [00:05, 1040.35 examples/s]6260 examples [00:05, 1042.09 examples/s]6365 examples [00:06, 1027.64 examples/s]6473 examples [00:06, 1040.44 examples/s]6583 examples [00:06, 1055.85 examples/s]6689 examples [00:06, 1038.03 examples/s]6799 examples [00:06, 1054.50 examples/s]6910 examples [00:06, 1069.51 examples/s]7018 examples [00:06, 1070.57 examples/s]7126 examples [00:06, 1049.88 examples/s]7232 examples [00:06, 1019.39 examples/s]7341 examples [00:06, 1037.93 examples/s]7449 examples [00:07, 1048.44 examples/s]7560 examples [00:07, 1063.28 examples/s]7670 examples [00:07, 1073.82 examples/s]7778 examples [00:07, 1055.88 examples/s]7888 examples [00:07, 1067.65 examples/s]7997 examples [00:07, 1072.98 examples/s]8106 examples [00:07, 1077.13 examples/s]8216 examples [00:07, 1082.88 examples/s]8325 examples [00:07, 1071.23 examples/s]8433 examples [00:07, 1030.96 examples/s]8537 examples [00:08, 1023.20 examples/s]8648 examples [00:08, 1045.70 examples/s]8753 examples [00:08, 1035.80 examples/s]8857 examples [00:08, 1011.70 examples/s]8960 examples [00:08, 1015.31 examples/s]9064 examples [00:08, 1016.52 examples/s]9166 examples [00:08, 995.75 examples/s] 9268 examples [00:08, 993.35 examples/s]9370 examples [00:08, 1000.12 examples/s]9474 examples [00:09, 1010.60 examples/s]9582 examples [00:09, 1028.27 examples/s]9692 examples [00:09, 1045.59 examples/s]9799 examples [00:09, 1051.33 examples/s]9907 examples [00:09, 1059.31 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNZ4567/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNZ4567/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3b918a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3b918a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3b918a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb40e00d240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb33e603b00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb3b918a8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb3b918a8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb3b918a8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3b66e7390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3b66e7518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb33c087840> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb33c087840> 

  function with postional parmater data_info <function split_train_valid at 0x7fb33c087840> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb33c087950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb33c087950> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb33c087950> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=531739478a29db603c8053a99b79016b527ea62cfb9f482be0c09dd98ad1713c
  Stored in directory: /tmp/pip-ephem-wheel-cache-ar251f10/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:20:02, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:28:09, 16.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:11:00, 23.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:08:13, 33.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.10M/862M [00:01<4:59:11, 47.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.72M/862M [00:01<3:28:40, 68.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.9M/862M [00:01<2:25:27, 97.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:41:17, 139kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 19.2M/862M [00:01<1:10:45, 199kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:01<49:20, 283kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.7M/862M [00:01<34:30, 403kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<24:06, 574kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:02<16:54, 814kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<11:52, 1.15MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.2M/862M [00:02<08:26, 1.61MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:02<05:58, 2.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<04:59, 2.70MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.8M/862M [00:03<03:40, 3.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:26, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<08:01, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<06:30, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:46, 2.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:27, 1.79MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<06:57, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:15, 2.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:17, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<06:11, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<04:41, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<03:27, 3.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<17:42, 747kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<14:08, 934kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<10:16, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:11<07:21, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<17:45, 740kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<14:09, 928kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<10:19, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<09:46, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<08:33, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:15<04:36, 2.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<25:21, 513kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:17<19:29, 667kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<14:02, 924kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<12:20, 1.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<10:22, 1.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<07:37, 1.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<05:32, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<11:23, 1.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<09:38, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:21<07:08, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<07:30, 1.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<06:54, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<05:14, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<06:09, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<04:35, 2.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:40, 2.23MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:39, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:19, 2.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<03:12, 3.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<13:14, 952kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<12:33, 1.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<09:36, 1.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:55, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<08:44, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:47, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<05:51, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:42, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<07:58, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<07:10, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:08, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<12:17, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<10:14, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:59, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:46, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:58, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:11, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:25, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<03:56, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<24:04, 509kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<18:17, 669kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<13:03, 936kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<09:18, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<17:27, 698kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<13:35, 896kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<09:50, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<09:28, 1.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<09:21, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<07:09, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<07:09, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:45<06:25, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<04:49, 2.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:54, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:44, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<04:23, 2.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:23, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:20, 2.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:06, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:11, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<06:42, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:21, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:59, 2.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:48, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:35, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<04:14, 2.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<03:06, 3.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<23:57, 488kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<18:08, 644kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<13:01, 896kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<11:33, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<11:17, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:39, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:12, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:47, 1.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<07:42, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<05:42, 2.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:08, 2.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<16:36, 691kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<12:58, 885kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<09:23, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:59, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:03<09:17, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<07:11, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:11, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:40, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:52, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:07, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<03:43, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<17:52, 631kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<13:58, 807kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<10:08, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:15, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:00, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:57, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<06:20, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:54, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<04:27, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:13, 3.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<5:31:13, 33.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<3:53:11, 47.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:43:16, 67.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<1:55:59, 94.8kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<1:22:39, 133kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<58:03, 189kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<40:41, 269kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<36:50, 297kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<27:12, 401kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<19:22, 563kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<15:38, 694kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<12:22, 877kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:00, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<08:23, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:16, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:53, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:33, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:14, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:02, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:54, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:45, 2.83MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:42, 2.25MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:40, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:35, 2.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<02:41, 3.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<07:07, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:23, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:48, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:50, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<09:51, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<08:15, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:07, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:44, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:17, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:11, 3.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:18, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<06:29, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:50, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<03:30, 2.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<36:09, 283kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<26:29, 386kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<18:45, 544kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<13:12, 770kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<1:04:28, 158kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<46:28, 219kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<32:46, 309kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<22:59, 440kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<34:08, 296kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<25:12, 400kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<17:53, 563kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<12:37, 795kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<19:31, 514kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<14:59, 669kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<10:48, 926kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<09:28, 1.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:55, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<05:51, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:33, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:09, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:02, 3.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<10:02, 978kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:18, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<06:04, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:24, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<09:12, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:43, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<05:40, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:04, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<24:18, 399kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<18:17, 529kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<13:03, 739kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<09:13, 1.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<35:02, 274kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<25:39, 374kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<18:11, 527kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<14:43, 648kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<12:51, 742kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<09:32, 1.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<07:00, 1.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:40, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:46, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:17, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:00, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:26, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<02:32, 3.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<10:11, 917kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<09:13, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:54, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:45, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:08, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:16, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:09, 2.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:30, 2.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:24, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<02:30, 3.64MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<11:31, 793kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<09:15, 985kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:43, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:49, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<11:26, 791kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<09:11, 985kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:40, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:46, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<12:00, 749kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<09:38, 932kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:00, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<06:37, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:40, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:48, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:40, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:32, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:18, 2.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<06:48, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:15, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:49, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:33, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:28, 2.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:06, 2.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:05, 2.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:50, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<02:53, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:10, 3.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<05:45, 1.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<05:08, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:52, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:21, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<04:08, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:07, 2.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:19, 3.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<06:43, 1.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<05:49, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:18, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:06, 2.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<12:43, 652kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<09:52, 840kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<07:07, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<06:43, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:45, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:31, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:36, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:03, 2.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:41, 3.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:20, 3.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:11, 3.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:03, 3.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<26:13, 311kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<20:42, 394kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<15:01, 542kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<11:00, 740kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<08:10, 994kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:09, 1.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:42, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:45, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<11:48, 685kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<16:27, 492kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<13:26, 601kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<09:58, 810kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<07:23, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<05:34, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:15, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:24, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<04:38, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:39, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:57, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:27, 3.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<02:02, 3.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<08:44, 910kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<10:05, 788kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<08:08, 977kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:04, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:41, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:38, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<02:58, 2.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<02:30, 3.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<02:10, 3.63MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<29:31, 267kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<22:43, 347kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<16:28, 478kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<11:55, 659kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<08:41, 903kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:28, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:54, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<08:27, 924kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<09:51, 793kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<07:53, 989kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:06, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:35, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:43, 2.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:56, 2.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<02:30, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<02:05, 3.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<11:20, 683kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<09:58, 777kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<07:31, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:35, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:23, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<02:49, 2.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<02:18, 3.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:50<13:09, 583kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<13:03, 588kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<10:10, 754kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:26, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:44, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:18, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<02:45, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:25, 3.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<24:57, 305kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<19:27, 391kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<14:10, 536kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<10:16, 738kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:32, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:37, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:17, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<5:21:32, 23.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<3:48:47, 33.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<2:40:58, 46.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<1:52:53, 66.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<1:19:13, 94.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<55:41, 135kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<39:18, 190kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<31:53, 234kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<25:43, 290kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<18:53, 395kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<13:36, 548kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<09:47, 759kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:13, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:20, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<04:06, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<2:36:46, 47.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:58<1:53:08, 65.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<1:19:57, 92.5kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<56:12, 131kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<39:33, 186kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<27:51, 264kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<19:51, 369kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<24:25, 300kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<20:08, 364kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<14:49, 494kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<10:38, 687kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<07:45, 941kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:39, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:15, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<08:49, 824kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<08:56, 813kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<06:51, 1.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:05, 1.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:49, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:55, 2.47MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:16, 3.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<20:44, 347kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<16:47, 429kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<13:08, 547kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<09:34, 750kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<06:58, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:07, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:48, 1.87MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:03, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<07:08, 999kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<05:34, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:06, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:06, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:20, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<12:05, 584kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<15:33, 454kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<12:33, 562kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<09:11, 766kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:34, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:49, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<06:30, 1.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<06:05, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<05:11, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:54, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:10, 3.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<11:16, 614kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<11:19, 611kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<08:44, 791kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:18, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:33, 1.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<05:44, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<06:53, 993kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<05:33, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:05, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<15:20, 442kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<13:22, 507kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<10:02, 675kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:10, 941kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:08, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<07:23, 908kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<07:33, 888kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<05:54, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<04:17, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:07, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<06:08, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<06:40, 996kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:13, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:58, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:51, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<07:36, 863kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<09:29, 692kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<07:37, 862kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<05:31, 1.19MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:59, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<08:32, 762kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<07:53, 824kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<05:53, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<04:22, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:11, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:52, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<04:50, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<08:47, 732kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<07:25, 866kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<05:27, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:01, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:02, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:24, 2.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<05:42, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:28<06:25, 992kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<05:04, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:45, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:49, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<04:20, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<05:17, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<04:13, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:08, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:23, 2.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:32<04:17, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<05:05, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<03:59, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<02:59, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:17, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<01:49, 3.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:36, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<05:15, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:08, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:04, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:19, 2.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:33, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<05:13, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<04:03, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:05, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:30, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<01:53, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<05:30, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<06:35, 913kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<05:24, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<04:05, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:06, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:24, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<01:54, 3.12MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<01:34, 3.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<15:05, 395kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<13:02, 456kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<09:38, 616kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<06:59, 849kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<05:06, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:50, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:52, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<06:15, 941kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<06:51, 858kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<05:22, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<03:59, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:19, 2.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<01:50, 3.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<06:45, 860kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<07:10, 810kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<05:35, 1.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<04:07, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:05, 1.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:21, 2.45MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:45<08:19, 690kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<08:01, 715kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<06:04, 944kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:20, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:32, 2.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<01:58, 2.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:47<08:13, 690kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<07:56, 714kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<06:06, 927kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<04:33, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<03:23, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:34, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<01:59, 2.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<05:41, 984kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<06:06, 917kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<04:48, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:34, 1.56MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:41, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:04, 2.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<10:31, 526kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<09:29, 583kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<07:10, 771kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<05:11, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:51, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:51, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:12, 2.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<10:11, 537kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<09:13, 593kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<06:58, 784kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:03, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:44, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:46, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<10:17, 525kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<09:17, 581kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<07:00, 769kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<05:05, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:44, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:48, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<10:22, 514kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<09:18, 572kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<07:01, 758kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<05:04, 1.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:42, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:48, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<20:33, 256kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<16:25, 320kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<11:58, 439kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<08:33, 612kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<06:09, 848kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:28, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<13:23, 388kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<11:25, 454kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<08:27, 613kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<06:06, 846kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:25, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:13, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<12:32, 409kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<10:35, 484kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<07:47, 656kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<05:55, 863kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:15, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:16, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:30, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<05:41, 889kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<06:45, 747kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<05:23, 937kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:58, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<03:01, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:19, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<01:53, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<01:35, 3.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<04:39, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<06:01, 828kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<04:50, 1.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:36, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:44, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:06, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:43, 2.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<04:26, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<05:22, 916kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:20, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:14, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:32, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<01:56, 2.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<01:38, 2.95MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<03:11, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<04:26, 1.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:33, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:47, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:09, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<01:40, 2.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<01:25, 3.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:15, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<04:25, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:37, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:44, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:09, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<01:44, 2.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<01:26, 3.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<01:12, 3.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<05:17, 889kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<05:50, 806kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<04:33, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:25, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:33, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:58, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:35, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<04:38, 998kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<03:46, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:10, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:42, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:18, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<04:23, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:34, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:42, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:02, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:41, 2.68MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:21, 3.34MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<05:39, 795kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<05:59, 751kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<04:37, 974kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:32, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:39, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:03, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:36, 2.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:19, 3.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<05:25, 818kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<05:49, 762kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<04:34, 968kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<03:23, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:32, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<01:57, 2.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<03:29, 1.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<04:16, 1.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:28, 1.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:36, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:59, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:33, 2.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<03:26, 1.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<04:21, 986kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<03:30, 1.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:38, 1.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:36, 2.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<01:14, 3.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<12:56, 327kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<10:48, 391kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<07:59, 528kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<05:45, 730kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:10, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:02, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<05:16, 788kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<05:17, 786kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<04:06, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<03:00, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:12, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:44, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<03:19, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<03:46, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<02:57, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<02:14, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:40, 2.42MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:19, 3.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<04:02, 994kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<04:09, 968kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<03:15, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:35, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:02, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:41, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:25, 2.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:14, 3.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:04, 3.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<00:58, 4.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<06:24, 616kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:37<05:08, 767kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<03:49, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:48, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:03, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:32, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<05:30, 705kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<06:00, 647kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<04:44, 817kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<03:26, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:32, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:51, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<03:33, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<03:26, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:38, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:56, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<02:22, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<03:30, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<02:53, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:07, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:33, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<02:35, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<03:19, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<02:40, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:57, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:27, 2.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:06, 3.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<08:16, 436kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<07:10, 503kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<05:22, 670kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:49, 934kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:42, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<08:28, 418kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<07:16, 486kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<05:20, 660kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<03:50, 913kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:45, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<03:01, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<03:26, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<02:44, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:00, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:27, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:54, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<03:01, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<02:19, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:44, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:14, 2.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<04:50, 687kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<04:22, 761kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<03:17, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:20, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:35, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<02:47, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<02:13, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:39, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:11, 2.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:55, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<02:51, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:12, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<02:28, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<01:54, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:27, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:02, 2.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<03:22, 903kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:56, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:03<02:25, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:45, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:16, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<02:54, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<02:39, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:59, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:26, 2.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:59, 1.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<02:29, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:59, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:28, 1.96MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:03, 2.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<04:20, 655kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<03:46, 754kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<02:48, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:58, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<03:02, 911kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<02:36, 1.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:55, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:21, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<03:43, 726kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<03:14, 836kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<02:23, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:42, 1.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:52, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:43, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:18, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:21, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<01:31, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:12, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<00:52, 2.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:25, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:32, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<01:19, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<00:59, 2.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:39, 3.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:04, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<02:32, 959kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<02:02, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:31, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:09, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<00:53, 2.68MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:43, 3.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<09:21, 253kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<07:31, 314kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<05:29, 429kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<03:55, 597kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:48, 828kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:02, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<03:21, 683kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<03:16, 702kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<02:29, 915kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:49, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:20, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:00, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<05:41, 391kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<04:50, 460kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<03:32, 625kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<02:34, 855kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:51, 1.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:22, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:01, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<08:44, 247kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<06:54, 312kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<05:04, 424kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<03:40, 584kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<02:38, 808kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:53, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:23, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<02:17, 908kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<02:18, 904kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:47, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:17, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:59, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:44, 2.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<02:01, 996kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<02:05, 962kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:36, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:13, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:54, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:42, 2.76MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<01:14, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<01:34, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:15, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:55, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:43, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:33, 3.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<01:22, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<01:35, 1.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:16, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:56, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:44, 2.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:33, 3.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:44, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:49, 989kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:23, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:04, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:47, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:37, 2.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:05, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:23, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:05, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:38, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:30, 3.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:02, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:12, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:09, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:53, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:40, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:31, 3.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:24, 3.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:40, 954kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:42, 940kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<01:19, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:43, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<01:06, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<01:19, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<01:03, 1.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:46, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:35, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<01:06, 1.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<02:15, 648kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<01:55, 759kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<01:25, 1.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:01, 1.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:44, 1.90MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<01:19, 1.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<01:23, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<01:04, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:47, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:35, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:27, 2.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<03:02, 437kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<02:32, 521kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<01:51, 710kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<01:21, 964kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:58, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:42, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<01:05, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:54<01:09, 1.09MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:53, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:39, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:29, 2.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:22, 3.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<01:15, 943kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<01:32, 770kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<01:15, 942kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:57<00:54, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:39, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:29, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:57, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:59, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:46, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:33, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:24, 2.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:49, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<01:08, 915kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:56, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<00:41, 1.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:43, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:45, 1.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:35, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:25, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:19, 2.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:38, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:54, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:44, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:32, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:23, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:17, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<01:28, 575kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<01:24, 599kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<01:04, 780kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:45, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:38, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:47, 968kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:38, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:27, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:19, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<01:03, 666kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<01:02, 682kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:47, 885kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:32, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:23, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:31, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:36, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:28, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:20, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:14, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:28, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:31, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:24, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:11, 2.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:43, 695kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:39, 760kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:28, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:20, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:13, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:30, 840kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:28, 890kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:21, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:20<01:04, 334kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:51, 421kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:36, 580kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:24, 810kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:18, 934kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:17, 989kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:12, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:09, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:07, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:08, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:04, 2.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:04, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:05, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 2.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:02, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:02, 2.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 1.87MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:07:54,  7.34it/s]  0%|          | 869/400000 [00:00<10:34:23, 10.49it/s]  0%|          | 1662/400000 [00:00<7:23:26, 14.97it/s]  1%|          | 2496/400000 [00:00<5:09:59, 21.37it/s]  1%|          | 3263/400000 [00:00<3:36:50, 30.49it/s]  1%|          | 4045/400000 [00:00<2:31:44, 43.49it/s]  1%|          | 4916/400000 [00:00<1:46:12, 62.00it/s]  1%|▏         | 5782/400000 [00:00<1:14:24, 88.30it/s]  2%|▏         | 6643/400000 [00:00<52:12, 125.58it/s]   2%|▏         | 7489/400000 [00:01<36:41, 178.27it/s]  2%|▏         | 8332/400000 [00:01<25:51, 252.38it/s]  2%|▏         | 9204/400000 [00:01<18:17, 356.13it/s]  3%|▎         | 10067/400000 [00:01<13:00, 499.85it/s]  3%|▎         | 10909/400000 [00:01<09:19, 695.73it/s]  3%|▎         | 11774/400000 [00:01<06:44, 960.76it/s]  3%|▎         | 12617/400000 [00:01<04:56, 1308.34it/s]  3%|▎         | 13490/400000 [00:01<03:40, 1756.17it/s]  4%|▎         | 14341/400000 [00:01<02:48, 2283.49it/s]  4%|▍         | 15211/400000 [00:01<02:11, 2932.26it/s]  4%|▍         | 16084/400000 [00:02<01:44, 3661.37it/s]  4%|▍         | 16939/400000 [00:02<01:26, 4419.04it/s]  4%|▍         | 17790/400000 [00:02<01:14, 5142.04it/s]  5%|▍         | 18635/400000 [00:02<01:05, 5808.37it/s]  5%|▍         | 19477/400000 [00:02<01:00, 6283.18it/s]  5%|▌         | 20342/400000 [00:02<00:55, 6843.69it/s]  5%|▌         | 21212/400000 [00:02<00:51, 7309.97it/s]  6%|▌         | 22078/400000 [00:02<00:49, 7666.76it/s]  6%|▌         | 22950/400000 [00:02<00:47, 7952.82it/s]  6%|▌         | 23816/400000 [00:02<00:46, 8152.34it/s]  6%|▌         | 24676/400000 [00:03<00:45, 8267.26it/s]  6%|▋         | 25547/400000 [00:03<00:44, 8394.25it/s]  7%|▋         | 26424/400000 [00:03<00:43, 8500.73it/s]  7%|▋         | 27310/400000 [00:03<00:43, 8603.04it/s]  7%|▋         | 28182/400000 [00:03<00:43, 8618.32it/s]  7%|▋         | 29052/400000 [00:03<00:42, 8635.51it/s]  7%|▋         | 29922/400000 [00:03<00:43, 8564.23it/s]  8%|▊         | 30783/400000 [00:03<00:43, 8553.90it/s]  8%|▊         | 31642/400000 [00:03<00:43, 8545.19it/s]  8%|▊         | 32499/400000 [00:03<00:43, 8493.15it/s]  8%|▊         | 33377/400000 [00:04<00:42, 8574.68it/s]  9%|▊         | 34236/400000 [00:04<00:42, 8556.56it/s]  9%|▉         | 35098/400000 [00:04<00:42, 8573.73it/s]  9%|▉         | 35967/400000 [00:04<00:42, 8606.06it/s]  9%|▉         | 36835/400000 [00:04<00:42, 8626.60it/s]  9%|▉         | 37707/400000 [00:04<00:41, 8654.09it/s] 10%|▉         | 38573/400000 [00:04<00:41, 8615.95it/s] 10%|▉         | 39451/400000 [00:04<00:41, 8663.53it/s] 10%|█         | 40323/400000 [00:04<00:41, 8678.12it/s] 10%|█         | 41195/400000 [00:04<00:41, 8688.92it/s] 11%|█         | 42065/400000 [00:05<00:41, 8586.83it/s] 11%|█         | 42929/400000 [00:05<00:41, 8602.27it/s] 11%|█         | 43805/400000 [00:05<00:41, 8646.90it/s] 11%|█         | 44670/400000 [00:05<00:41, 8516.07it/s] 11%|█▏        | 45541/400000 [00:05<00:41, 8572.54it/s] 12%|█▏        | 46401/400000 [00:05<00:41, 8580.13it/s] 12%|█▏        | 47260/400000 [00:05<00:41, 8531.67it/s] 12%|█▏        | 48114/400000 [00:05<00:41, 8517.34it/s] 12%|█▏        | 48990/400000 [00:05<00:40, 8588.23it/s] 12%|█▏        | 49850/400000 [00:05<00:41, 8418.74it/s] 13%|█▎        | 50703/400000 [00:06<00:41, 8451.74it/s] 13%|█▎        | 51552/400000 [00:06<00:41, 8460.76it/s] 13%|█▎        | 52403/400000 [00:06<00:41, 8475.23it/s] 13%|█▎        | 53285/400000 [00:06<00:40, 8574.62it/s] 14%|█▎        | 54143/400000 [00:06<00:41, 8246.89it/s] 14%|█▍        | 55002/400000 [00:06<00:41, 8346.06it/s] 14%|█▍        | 55860/400000 [00:06<00:40, 8412.69it/s] 14%|█▍        | 56734/400000 [00:06<00:40, 8506.01it/s] 14%|█▍        | 57611/400000 [00:06<00:39, 8581.21it/s] 15%|█▍        | 58482/400000 [00:07<00:39, 8616.91it/s] 15%|█▍        | 59359/400000 [00:07<00:39, 8660.30it/s] 15%|█▌        | 60226/400000 [00:07<00:39, 8635.08it/s] 15%|█▌        | 61106/400000 [00:07<00:39, 8683.66it/s] 15%|█▌        | 61986/400000 [00:07<00:38, 8718.08it/s] 16%|█▌        | 62859/400000 [00:07<00:38, 8690.80it/s] 16%|█▌        | 63735/400000 [00:07<00:38, 8710.41it/s] 16%|█▌        | 64607/400000 [00:07<00:38, 8675.75it/s] 16%|█▋        | 65480/400000 [00:07<00:38, 8689.53it/s] 17%|█▋        | 66350/400000 [00:07<00:39, 8539.37it/s] 17%|█▋        | 67205/400000 [00:08<00:39, 8470.09it/s] 17%|█▋        | 68069/400000 [00:08<00:39, 8483.90it/s] 17%|█▋        | 68924/400000 [00:08<00:38, 8502.42it/s] 17%|█▋        | 69793/400000 [00:08<00:38, 8555.01it/s] 18%|█▊        | 70675/400000 [00:08<00:38, 8631.27it/s] 18%|█▊        | 71539/400000 [00:08<00:38, 8614.46it/s] 18%|█▊        | 72422/400000 [00:08<00:37, 8675.27it/s] 18%|█▊        | 73290/400000 [00:08<00:37, 8610.11it/s] 19%|█▊        | 74180/400000 [00:08<00:37, 8693.92it/s] 19%|█▉        | 75053/400000 [00:08<00:37, 8702.89it/s] 19%|█▉        | 75924/400000 [00:09<00:37, 8546.80it/s] 19%|█▉        | 76800/400000 [00:09<00:37, 8607.11it/s] 19%|█▉        | 77670/400000 [00:09<00:37, 8633.50it/s] 20%|█▉        | 78545/400000 [00:09<00:37, 8666.21it/s] 20%|█▉        | 79416/400000 [00:09<00:36, 8678.75it/s] 20%|██        | 80285/400000 [00:09<00:36, 8680.00it/s] 20%|██        | 81155/400000 [00:09<00:36, 8684.33it/s] 21%|██        | 82024/400000 [00:09<00:36, 8645.79it/s] 21%|██        | 82904/400000 [00:09<00:36, 8691.41it/s] 21%|██        | 83775/400000 [00:09<00:36, 8694.51it/s] 21%|██        | 84645/400000 [00:10<00:36, 8675.01it/s] 21%|██▏       | 85518/400000 [00:10<00:36, 8690.10it/s] 22%|██▏       | 86388/400000 [00:10<00:36, 8623.87it/s] 22%|██▏       | 87266/400000 [00:10<00:36, 8670.02it/s] 22%|██▏       | 88134/400000 [00:10<00:36, 8546.61it/s] 22%|██▏       | 88990/400000 [00:10<00:37, 8349.76it/s] 22%|██▏       | 89867/400000 [00:10<00:36, 8470.22it/s] 23%|██▎       | 90738/400000 [00:10<00:36, 8540.09it/s] 23%|██▎       | 91603/400000 [00:10<00:35, 8570.35it/s] 23%|██▎       | 92484/400000 [00:10<00:35, 8639.74it/s] 23%|██▎       | 93350/400000 [00:11<00:35, 8643.15it/s] 24%|██▎       | 94231/400000 [00:11<00:35, 8690.22it/s] 24%|██▍       | 95101/400000 [00:11<00:35, 8601.65it/s] 24%|██▍       | 95964/400000 [00:11<00:35, 8608.18it/s] 24%|██▍       | 96826/400000 [00:11<00:35, 8519.65it/s] 24%|██▍       | 97692/400000 [00:11<00:35, 8559.95it/s] 25%|██▍       | 98573/400000 [00:11<00:34, 8632.07it/s] 25%|██▍       | 99437/400000 [00:11<00:35, 8422.08it/s] 25%|██▌       | 100281/400000 [00:11<00:36, 8315.92it/s] 25%|██▌       | 101120/400000 [00:11<00:35, 8309.04it/s] 25%|██▌       | 101956/400000 [00:12<00:35, 8322.66it/s] 26%|██▌       | 102789/400000 [00:12<00:35, 8308.66it/s] 26%|██▌       | 103624/400000 [00:12<00:35, 8320.87it/s] 26%|██▌       | 104457/400000 [00:12<00:35, 8320.17it/s] 26%|██▋       | 105334/400000 [00:12<00:34, 8449.43it/s] 27%|██▋       | 106180/400000 [00:12<00:34, 8431.75it/s] 27%|██▋       | 107039/400000 [00:12<00:34, 8477.86it/s] 27%|██▋       | 107888/400000 [00:12<00:35, 8282.39it/s] 27%|██▋       | 108718/400000 [00:12<00:35, 8220.45it/s] 27%|██▋       | 109574/400000 [00:12<00:34, 8318.07it/s] 28%|██▊       | 110447/400000 [00:13<00:34, 8436.45it/s] 28%|██▊       | 111292/400000 [00:13<00:34, 8435.15it/s] 28%|██▊       | 112137/400000 [00:13<00:34, 8374.59it/s] 28%|██▊       | 113009/400000 [00:13<00:33, 8473.63it/s] 28%|██▊       | 113894/400000 [00:13<00:33, 8581.40it/s] 29%|██▊       | 114753/400000 [00:13<00:33, 8572.24it/s] 29%|██▉       | 115611/400000 [00:13<00:33, 8495.20it/s] 29%|██▉       | 116477/400000 [00:13<00:33, 8542.30it/s] 29%|██▉       | 117332/400000 [00:13<00:33, 8508.29it/s] 30%|██▉       | 118184/400000 [00:13<00:33, 8488.45it/s] 30%|██▉       | 119034/400000 [00:14<00:33, 8449.55it/s] 30%|██▉       | 119889/400000 [00:14<00:33, 8476.80it/s] 30%|███       | 120753/400000 [00:14<00:32, 8523.93it/s] 30%|███       | 121606/400000 [00:14<00:32, 8514.50it/s] 31%|███       | 122458/400000 [00:14<00:32, 8473.15it/s] 31%|███       | 123326/400000 [00:14<00:32, 8532.23it/s] 31%|███       | 124203/400000 [00:14<00:32, 8602.14it/s] 31%|███▏      | 125075/400000 [00:14<00:31, 8637.15it/s] 31%|███▏      | 125942/400000 [00:14<00:31, 8646.27it/s] 32%|███▏      | 126823/400000 [00:15<00:31, 8694.32it/s] 32%|███▏      | 127693/400000 [00:15<00:31, 8657.84it/s] 32%|███▏      | 128567/400000 [00:15<00:31, 8681.09it/s] 32%|███▏      | 129436/400000 [00:15<00:31, 8664.61it/s] 33%|███▎      | 130303/400000 [00:15<00:31, 8665.58it/s] 33%|███▎      | 131179/400000 [00:15<00:30, 8693.24it/s] 33%|███▎      | 132049/400000 [00:15<00:30, 8682.25it/s] 33%|███▎      | 132920/400000 [00:15<00:30, 8690.40it/s] 33%|███▎      | 133797/400000 [00:15<00:30, 8711.86it/s] 34%|███▎      | 134669/400000 [00:15<00:31, 8426.61it/s] 34%|███▍      | 135522/400000 [00:16<00:31, 8456.39it/s] 34%|███▍      | 136370/400000 [00:16<00:31, 8435.21it/s] 34%|███▍      | 137221/400000 [00:16<00:31, 8456.65it/s] 35%|███▍      | 138068/400000 [00:16<00:30, 8458.84it/s] 35%|███▍      | 138915/400000 [00:16<00:30, 8444.92it/s] 35%|███▍      | 139768/400000 [00:16<00:30, 8470.03it/s] 35%|███▌      | 140634/400000 [00:16<00:30, 8523.56it/s] 35%|███▌      | 141487/400000 [00:16<00:30, 8404.88it/s] 36%|███▌      | 142340/400000 [00:16<00:30, 8439.90it/s] 36%|███▌      | 143206/400000 [00:16<00:30, 8503.93it/s] 36%|███▌      | 144085/400000 [00:17<00:29, 8586.08it/s] 36%|███▌      | 144956/400000 [00:17<00:29, 8620.44it/s] 36%|███▋      | 145822/400000 [00:17<00:29, 8631.04it/s] 37%|███▋      | 146699/400000 [00:17<00:29, 8669.08it/s] 37%|███▋      | 147567/400000 [00:17<00:29, 8523.58it/s] 37%|███▋      | 148445/400000 [00:17<00:29, 8597.34it/s] 37%|███▋      | 149314/400000 [00:17<00:29, 8622.55it/s] 38%|███▊      | 150190/400000 [00:17<00:28, 8661.34it/s] 38%|███▊      | 151066/400000 [00:17<00:28, 8689.31it/s] 38%|███▊      | 151936/400000 [00:17<00:28, 8669.82it/s] 38%|███▊      | 152812/400000 [00:18<00:28, 8694.11it/s] 38%|███▊      | 153683/400000 [00:18<00:28, 8698.84it/s] 39%|███▊      | 154553/400000 [00:18<00:28, 8618.00it/s] 39%|███▉      | 155421/400000 [00:18<00:28, 8635.08it/s] 39%|███▉      | 156285/400000 [00:18<00:28, 8578.13it/s] 39%|███▉      | 157156/400000 [00:18<00:28, 8615.59it/s] 40%|███▉      | 158026/400000 [00:18<00:28, 8640.70it/s] 40%|███▉      | 158902/400000 [00:18<00:27, 8673.77it/s] 40%|███▉      | 159775/400000 [00:18<00:27, 8688.59it/s] 40%|████      | 160652/400000 [00:18<00:27, 8712.23it/s] 40%|████      | 161524/400000 [00:19<00:27, 8706.89it/s] 41%|████      | 162395/400000 [00:19<00:27, 8558.80it/s] 41%|████      | 163259/400000 [00:19<00:27, 8582.67it/s] 41%|████      | 164129/400000 [00:19<00:27, 8616.97it/s] 41%|████      | 164992/400000 [00:19<00:27, 8568.55it/s] 41%|████▏     | 165858/400000 [00:19<00:27, 8592.95it/s] 42%|████▏     | 166718/400000 [00:19<00:27, 8439.33it/s] 42%|████▏     | 167569/400000 [00:19<00:27, 8459.35it/s] 42%|████▏     | 168441/400000 [00:19<00:27, 8535.82it/s] 42%|████▏     | 169314/400000 [00:19<00:26, 8590.48it/s] 43%|████▎     | 170181/400000 [00:20<00:26, 8614.13it/s] 43%|████▎     | 171043/400000 [00:20<00:27, 8465.88it/s] 43%|████▎     | 171891/400000 [00:20<00:26, 8464.32it/s] 43%|████▎     | 172764/400000 [00:20<00:26, 8540.48it/s] 43%|████▎     | 173640/400000 [00:20<00:26, 8603.90it/s] 44%|████▎     | 174501/400000 [00:20<00:26, 8560.90it/s] 44%|████▍     | 175367/400000 [00:20<00:26, 8589.23it/s] 44%|████▍     | 176227/400000 [00:20<00:26, 8561.01it/s] 44%|████▍     | 177084/400000 [00:20<00:26, 8531.25it/s] 44%|████▍     | 177938/400000 [00:20<00:26, 8374.28it/s] 45%|████▍     | 178804/400000 [00:21<00:26, 8456.81it/s] 45%|████▍     | 179668/400000 [00:21<00:25, 8510.63it/s] 45%|████▌     | 180520/400000 [00:21<00:25, 8503.38it/s] 45%|████▌     | 181371/400000 [00:21<00:26, 8244.86it/s] 46%|████▌     | 182236/400000 [00:21<00:26, 8359.15it/s] 46%|████▌     | 183074/400000 [00:21<00:26, 8323.92it/s] 46%|████▌     | 183943/400000 [00:21<00:25, 8429.59it/s] 46%|████▌     | 184821/400000 [00:21<00:25, 8530.99it/s] 46%|████▋     | 185693/400000 [00:21<00:24, 8584.84it/s] 47%|████▋     | 186563/400000 [00:21<00:24, 8618.18it/s] 47%|████▋     | 187431/400000 [00:22<00:24, 8634.61it/s] 47%|████▋     | 188309/400000 [00:22<00:24, 8676.21it/s] 47%|████▋     | 189178/400000 [00:22<00:24, 8679.44it/s] 48%|████▊     | 190047/400000 [00:22<00:24, 8587.35it/s] 48%|████▊     | 190908/400000 [00:22<00:24, 8592.13it/s] 48%|████▊     | 191768/400000 [00:22<00:24, 8511.22it/s] 48%|████▊     | 192620/400000 [00:22<00:24, 8494.53it/s] 48%|████▊     | 193483/400000 [00:22<00:24, 8531.90it/s] 49%|████▊     | 194347/400000 [00:22<00:24, 8563.83it/s] 49%|████▉     | 195204/400000 [00:22<00:23, 8534.57it/s] 49%|████▉     | 196068/400000 [00:23<00:23, 8564.98it/s] 49%|████▉     | 196940/400000 [00:23<00:23, 8610.83it/s] 49%|████▉     | 197816/400000 [00:23<00:23, 8653.42it/s] 50%|████▉     | 198695/400000 [00:23<00:23, 8692.82it/s] 50%|████▉     | 199569/400000 [00:23<00:23, 8705.12it/s] 50%|█████     | 200440/400000 [00:23<00:22, 8702.04it/s] 50%|█████     | 201311/400000 [00:23<00:22, 8681.00it/s] 51%|█████     | 202193/400000 [00:23<00:22, 8720.02it/s] 51%|█████     | 203066/400000 [00:23<00:23, 8556.79it/s] 51%|█████     | 203923/400000 [00:23<00:23, 8516.57it/s] 51%|█████     | 204776/400000 [00:24<00:23, 8399.77it/s] 51%|█████▏    | 205649/400000 [00:24<00:22, 8494.29it/s] 52%|█████▏    | 206515/400000 [00:24<00:22, 8542.16it/s] 52%|█████▏    | 207389/400000 [00:24<00:22, 8598.31it/s] 52%|█████▏    | 208254/400000 [00:24<00:22, 8611.45it/s] 52%|█████▏    | 209124/400000 [00:24<00:22, 8637.47it/s] 52%|█████▏    | 209999/400000 [00:24<00:21, 8669.13it/s] 53%|█████▎    | 210879/400000 [00:24<00:21, 8705.70it/s] 53%|█████▎    | 211750/400000 [00:24<00:21, 8690.11it/s] 53%|█████▎    | 212620/400000 [00:25<00:21, 8677.05it/s] 53%|█████▎    | 213491/400000 [00:25<00:21, 8685.41it/s] 54%|█████▎    | 214370/400000 [00:25<00:21, 8715.15it/s] 54%|█████▍    | 215252/400000 [00:25<00:21, 8744.13it/s] 54%|█████▍    | 216128/400000 [00:25<00:21, 8747.63it/s] 54%|█████▍    | 217003/400000 [00:25<00:21, 8711.96it/s] 54%|█████▍    | 217878/400000 [00:25<00:20, 8720.51it/s] 55%|█████▍    | 218756/400000 [00:25<00:20, 8737.85it/s] 55%|█████▍    | 219635/400000 [00:25<00:20, 8749.84it/s] 55%|█████▌    | 220511/400000 [00:25<00:20, 8751.32it/s] 55%|█████▌    | 221387/400000 [00:26<00:20, 8659.05it/s] 56%|█████▌    | 222259/400000 [00:26<00:20, 8676.26it/s] 56%|█████▌    | 223139/400000 [00:26<00:20, 8712.09it/s] 56%|█████▌    | 224011/400000 [00:26<00:20, 8706.30it/s] 56%|█████▌    | 224888/400000 [00:26<00:20, 8724.27it/s] 56%|█████▋    | 225761/400000 [00:26<00:20, 8703.07it/s] 57%|█████▋    | 226632/400000 [00:26<00:19, 8695.05it/s] 57%|█████▋    | 227518/400000 [00:26<00:19, 8743.12it/s] 57%|█████▋    | 228394/400000 [00:26<00:19, 8746.92it/s] 57%|█████▋    | 229275/400000 [00:26<00:19, 8763.66it/s] 58%|█████▊    | 230152/400000 [00:27<00:19, 8650.77it/s] 58%|█████▊    | 231018/400000 [00:27<00:19, 8583.98it/s] 58%|█████▊    | 231892/400000 [00:27<00:19, 8628.38it/s] 58%|█████▊    | 232770/400000 [00:27<00:19, 8672.03it/s] 58%|█████▊    | 233644/400000 [00:27<00:19, 8691.57it/s] 59%|█████▊    | 234515/400000 [00:27<00:19, 8694.31it/s] 59%|█████▉    | 235385/400000 [00:27<00:18, 8693.74it/s] 59%|█████▉    | 236264/400000 [00:27<00:18, 8719.40it/s] 59%|█████▉    | 237142/400000 [00:27<00:18, 8737.46it/s] 60%|█████▉    | 238016/400000 [00:27<00:18, 8629.43it/s] 60%|█████▉    | 238880/400000 [00:28<00:18, 8531.09it/s] 60%|█████▉    | 239734/400000 [00:28<00:18, 8485.36it/s] 60%|██████    | 240583/400000 [00:28<00:18, 8414.34it/s] 60%|██████    | 241458/400000 [00:28<00:18, 8512.14it/s] 61%|██████    | 242327/400000 [00:28<00:18, 8563.86it/s] 61%|██████    | 243192/400000 [00:28<00:18, 8587.24it/s] 61%|██████    | 244074/400000 [00:28<00:18, 8655.41it/s] 61%|██████    | 244945/400000 [00:28<00:17, 8671.47it/s] 61%|██████▏   | 245825/400000 [00:28<00:17, 8708.13it/s] 62%|██████▏   | 246697/400000 [00:28<00:17, 8523.24it/s] 62%|██████▏   | 247551/400000 [00:29<00:17, 8487.27it/s] 62%|██████▏   | 248401/400000 [00:29<00:18, 8391.56it/s] 62%|██████▏   | 249241/400000 [00:29<00:17, 8393.18it/s] 63%|██████▎   | 250119/400000 [00:29<00:17, 8502.89it/s] 63%|██████▎   | 250971/400000 [00:29<00:17, 8506.57it/s] 63%|██████▎   | 251829/400000 [00:29<00:17, 8526.58it/s] 63%|██████▎   | 252699/400000 [00:29<00:17, 8576.12it/s] 63%|██████▎   | 253557/400000 [00:29<00:17, 8544.35it/s] 64%|██████▎   | 254432/400000 [00:29<00:16, 8604.86it/s] 64%|██████▍   | 255293/400000 [00:29<00:16, 8577.52it/s] 64%|██████▍   | 256151/400000 [00:30<00:16, 8514.45it/s] 64%|██████▍   | 257030/400000 [00:30<00:16, 8592.99it/s] 64%|██████▍   | 257890/400000 [00:30<00:16, 8587.81it/s] 65%|██████▍   | 258776/400000 [00:30<00:16, 8666.60it/s] 65%|██████▍   | 259644/400000 [00:30<00:16, 8660.12it/s] 65%|██████▌   | 260511/400000 [00:30<00:16, 8634.01it/s] 65%|██████▌   | 261383/400000 [00:30<00:16, 8657.96it/s] 66%|██████▌   | 262259/400000 [00:30<00:15, 8688.06it/s] 66%|██████▌   | 263136/400000 [00:30<00:15, 8709.67it/s] 66%|██████▌   | 264008/400000 [00:30<00:15, 8688.90it/s] 66%|██████▌   | 264877/400000 [00:31<00:15, 8462.57it/s] 66%|██████▋   | 265751/400000 [00:31<00:15, 8543.33it/s] 67%|██████▋   | 266607/400000 [00:31<00:15, 8411.59it/s] 67%|██████▋   | 267487/400000 [00:31<00:15, 8523.76it/s] 67%|██████▋   | 268368/400000 [00:31<00:15, 8607.45it/s] 67%|██████▋   | 269230/400000 [00:31<00:15, 8521.23it/s] 68%|██████▊   | 270106/400000 [00:31<00:15, 8589.15it/s] 68%|██████▊   | 270985/400000 [00:31<00:14, 8646.94it/s] 68%|██████▊   | 271864/400000 [00:31<00:14, 8686.83it/s] 68%|██████▊   | 272734/400000 [00:31<00:14, 8626.89it/s] 68%|██████▊   | 273598/400000 [00:32<00:14, 8555.02it/s] 69%|██████▊   | 274454/400000 [00:32<00:14, 8517.21it/s] 69%|██████▉   | 275330/400000 [00:32<00:14, 8587.22it/s] 69%|██████▉   | 276198/400000 [00:32<00:14, 8614.27it/s] 69%|██████▉   | 277060/400000 [00:32<00:14, 8515.35it/s] 69%|██████▉   | 277912/400000 [00:32<00:14, 8483.34it/s] 70%|██████▉   | 278783/400000 [00:32<00:14, 8549.04it/s] 70%|██████▉   | 279639/400000 [00:32<00:14, 8518.16it/s] 70%|███████   | 280516/400000 [00:32<00:13, 8591.41it/s] 70%|███████   | 281396/400000 [00:32<00:13, 8651.04it/s] 71%|███████   | 282262/400000 [00:33<00:13, 8643.21it/s] 71%|███████   | 283140/400000 [00:33<00:13, 8682.19it/s] 71%|███████   | 284009/400000 [00:33<00:13, 8655.68it/s] 71%|███████   | 284886/400000 [00:33<00:13, 8688.48it/s] 71%|███████▏  | 285755/400000 [00:33<00:13, 8680.17it/s] 72%|███████▏  | 286624/400000 [00:33<00:13, 8652.41it/s] 72%|███████▏  | 287495/400000 [00:33<00:12, 8668.82it/s] 72%|███████▏  | 288375/400000 [00:33<00:12, 8707.48it/s] 72%|███████▏  | 289246/400000 [00:33<00:12, 8594.05it/s] 73%|███████▎  | 290106/400000 [00:33<00:12, 8583.98it/s] 73%|███████▎  | 290965/400000 [00:34<00:12, 8498.98it/s] 73%|███████▎  | 291834/400000 [00:34<00:12, 8553.63it/s] 73%|███████▎  | 292713/400000 [00:34<00:12, 8620.46it/s] 73%|███████▎  | 293579/400000 [00:34<00:12, 8631.68it/s] 74%|███████▎  | 294453/400000 [00:34<00:12, 8661.89it/s] 74%|███████▍  | 295320/400000 [00:34<00:12, 8650.28it/s] 74%|███████▍  | 296191/400000 [00:34<00:11, 8665.82it/s] 74%|███████▍  | 297070/400000 [00:34<00:11, 8699.88it/s] 74%|███████▍  | 297941/400000 [00:34<00:11, 8699.40it/s] 75%|███████▍  | 298814/400000 [00:35<00:11, 8707.44it/s] 75%|███████▍  | 299685/400000 [00:35<00:11, 8697.08it/s] 75%|███████▌  | 300555/400000 [00:35<00:11, 8538.30it/s] 75%|███████▌  | 301433/400000 [00:35<00:11, 8607.21it/s] 76%|███████▌  | 302306/400000 [00:35<00:11, 8641.16it/s] 76%|███████▌  | 303171/400000 [00:35<00:11, 8482.02it/s] 76%|███████▌  | 304033/400000 [00:35<00:11, 8522.27it/s] 76%|███████▌  | 304901/400000 [00:35<00:11, 8567.15it/s] 76%|███████▋  | 305768/400000 [00:35<00:10, 8597.25it/s] 77%|███████▋  | 306629/400000 [00:35<00:11, 8444.93it/s] 77%|███████▋  | 307482/400000 [00:36<00:10, 8469.57it/s] 77%|███████▋  | 308334/400000 [00:36<00:10, 8483.03it/s] 77%|███████▋  | 309205/400000 [00:36<00:10, 8549.00it/s] 78%|███████▊  | 310081/400000 [00:36<00:10, 8609.17it/s] 78%|███████▊  | 310943/400000 [00:36<00:10, 8606.73it/s] 78%|███████▊  | 311804/400000 [00:36<00:10, 8483.45it/s] 78%|███████▊  | 312659/400000 [00:36<00:10, 8502.30it/s] 78%|███████▊  | 313510/400000 [00:36<00:10, 8437.70it/s] 79%|███████▊  | 314367/400000 [00:36<00:10, 8475.62it/s] 79%|███████▉  | 315234/400000 [00:36<00:09, 8531.15it/s] 79%|███████▉  | 316088/400000 [00:37<00:09, 8403.96it/s] 79%|███████▉  | 316957/400000 [00:37<00:09, 8485.48it/s] 79%|███████▉  | 317825/400000 [00:37<00:09, 8541.81it/s] 80%|███████▉  | 318701/400000 [00:37<00:09, 8604.60it/s] 80%|███████▉  | 319583/400000 [00:37<00:09, 8666.26it/s] 80%|████████  | 320462/400000 [00:37<00:09, 8702.09it/s] 80%|████████  | 321343/400000 [00:37<00:09, 8733.34it/s] 81%|████████  | 322217/400000 [00:37<00:08, 8701.60it/s] 81%|████████  | 323094/400000 [00:37<00:08, 8721.71it/s] 81%|████████  | 323972/400000 [00:37<00:08, 8737.62it/s] 81%|████████  | 324846/400000 [00:38<00:08, 8684.69it/s] 81%|████████▏ | 325726/400000 [00:38<00:08, 8717.28it/s] 82%|████████▏ | 326599/400000 [00:38<00:08, 8720.36it/s] 82%|████████▏ | 327472/400000 [00:38<00:08, 8550.80it/s] 82%|████████▏ | 328344/400000 [00:38<00:08, 8600.28it/s] 82%|████████▏ | 329219/400000 [00:38<00:08, 8642.07it/s] 83%|████████▎ | 330095/400000 [00:38<00:08, 8676.91it/s] 83%|████████▎ | 330965/400000 [00:38<00:07, 8681.06it/s] 83%|████████▎ | 331836/400000 [00:38<00:07, 8688.45it/s] 83%|████████▎ | 332720/400000 [00:38<00:07, 8732.97it/s] 83%|████████▎ | 333601/400000 [00:39<00:07, 8754.99it/s] 84%|████████▎ | 334481/400000 [00:39<00:07, 8767.02it/s] 84%|████████▍ | 335358/400000 [00:39<00:07, 8726.17it/s] 84%|████████▍ | 336231/400000 [00:39<00:07, 8692.08it/s] 84%|████████▍ | 337105/400000 [00:39<00:07, 8705.42it/s] 84%|████████▍ | 337980/400000 [00:39<00:07, 8716.68it/s] 85%|████████▍ | 338852/400000 [00:39<00:07, 8670.85it/s] 85%|████████▍ | 339720/400000 [00:39<00:06, 8660.34it/s] 85%|████████▌ | 340597/400000 [00:39<00:06, 8692.15it/s] 85%|████████▌ | 341467/400000 [00:39<00:06, 8652.95it/s] 86%|████████▌ | 342341/400000 [00:40<00:06, 8673.66it/s] 86%|████████▌ | 343218/400000 [00:40<00:06, 8700.41it/s] 86%|████████▌ | 344089/400000 [00:40<00:06, 8679.45it/s] 86%|████████▌ | 344958/400000 [00:40<00:06, 8659.99it/s] 86%|████████▋ | 345825/400000 [00:40<00:06, 8458.78it/s] 87%|████████▋ | 346694/400000 [00:40<00:06, 8525.95it/s] 87%|████████▋ | 347568/400000 [00:40<00:06, 8586.56it/s] 87%|████████▋ | 348436/400000 [00:40<00:05, 8613.76it/s] 87%|████████▋ | 349307/400000 [00:40<00:05, 8641.68it/s] 88%|████████▊ | 350172/400000 [00:40<00:05, 8539.93it/s] 88%|████████▊ | 351027/400000 [00:41<00:05, 8404.59it/s] 88%|████████▊ | 351902/400000 [00:41<00:05, 8504.84it/s] 88%|████████▊ | 352767/400000 [00:41<00:05, 8545.54it/s] 88%|████████▊ | 353623/400000 [00:41<00:05, 8471.65it/s] 89%|████████▊ | 354491/400000 [00:41<00:05, 8531.19it/s] 89%|████████▉ | 355345/400000 [00:41<00:05, 8502.09it/s] 89%|████████▉ | 356223/400000 [00:41<00:05, 8582.13it/s] 89%|████████▉ | 357088/400000 [00:41<00:04, 8601.79it/s] 89%|████████▉ | 357949/400000 [00:41<00:04, 8599.28it/s] 90%|████████▉ | 358810/400000 [00:41<00:04, 8538.97it/s] 90%|████████▉ | 359665/400000 [00:42<00:04, 8520.33it/s] 90%|█████████ | 360518/400000 [00:42<00:04, 8485.55it/s] 90%|█████████ | 361367/400000 [00:42<00:04, 8463.26it/s] 91%|█████████ | 362214/400000 [00:42<00:04, 8304.60it/s] 91%|█████████ | 363067/400000 [00:42<00:04, 8370.64it/s] 91%|█████████ | 363929/400000 [00:42<00:04, 8443.79it/s] 91%|█████████ | 364802/400000 [00:42<00:04, 8527.43it/s] 91%|█████████▏| 365681/400000 [00:42<00:03, 8603.44it/s] 92%|█████████▏| 366542/400000 [00:42<00:03, 8448.44it/s] 92%|█████████▏| 367388/400000 [00:42<00:03, 8389.01it/s] 92%|█████████▏| 368235/400000 [00:43<00:03, 8410.17it/s] 92%|█████████▏| 369077/400000 [00:43<00:03, 8388.29it/s] 92%|█████████▏| 369932/400000 [00:43<00:03, 8433.90it/s] 93%|█████████▎| 370776/400000 [00:43<00:03, 8394.45it/s] 93%|█████████▎| 371629/400000 [00:43<00:03, 8433.06it/s] 93%|█████████▎| 372473/400000 [00:43<00:03, 8402.73it/s] 93%|█████████▎| 373314/400000 [00:43<00:03, 8371.99it/s] 94%|█████████▎| 374163/400000 [00:43<00:03, 8405.55it/s] 94%|█████████▍| 375014/400000 [00:43<00:02, 8435.73it/s] 94%|█████████▍| 375890/400000 [00:44<00:02, 8529.10it/s] 94%|█████████▍| 376744/400000 [00:44<00:02, 8401.74it/s] 94%|█████████▍| 377623/400000 [00:44<00:02, 8512.07it/s] 95%|█████████▍| 378486/400000 [00:44<00:02, 8544.48it/s] 95%|█████████▍| 379349/400000 [00:44<00:02, 8569.07it/s] 95%|█████████▌| 380207/400000 [00:44<00:02, 8373.83it/s] 95%|█████████▌| 381067/400000 [00:44<00:02, 8438.95it/s] 95%|█████████▌| 381946/400000 [00:44<00:02, 8539.51it/s] 96%|█████████▌| 382810/400000 [00:44<00:02, 8567.15it/s] 96%|█████████▌| 383678/400000 [00:44<00:01, 8599.62it/s] 96%|█████████▌| 384560/400000 [00:45<00:01, 8663.80it/s] 96%|█████████▋| 385427/400000 [00:45<00:01, 8664.20it/s] 97%|█████████▋| 386318/400000 [00:45<00:01, 8735.21it/s] 97%|█████████▋| 387192/400000 [00:45<00:01, 8559.65it/s] 97%|█████████▋| 388063/400000 [00:45<00:01, 8603.06it/s] 97%|█████████▋| 388930/400000 [00:45<00:01, 8621.28it/s] 97%|█████████▋| 389796/400000 [00:45<00:01, 8632.59it/s] 98%|█████████▊| 390660/400000 [00:45<00:01, 8596.73it/s] 98%|█████████▊| 391532/400000 [00:45<00:00, 8632.48it/s] 98%|█████████▊| 392396/400000 [00:45<00:00, 8583.73it/s] 98%|█████████▊| 393280/400000 [00:46<00:00, 8658.39it/s] 99%|█████████▊| 394165/400000 [00:46<00:00, 8712.98it/s] 99%|█████████▉| 395043/400000 [00:46<00:00, 8732.80it/s] 99%|█████████▉| 395917/400000 [00:46<00:00, 8698.17it/s] 99%|█████████▉| 396788/400000 [00:46<00:00, 8668.68it/s] 99%|█████████▉| 397661/400000 [00:46<00:00, 8684.43it/s]100%|█████████▉| 398530/400000 [00:46<00:00, 8527.23it/s]100%|█████████▉| 399396/400000 [00:46<00:00, 8565.38it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8546.14it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb4253bb5c0>, <torchtext.data.dataset.TabularDataset object at 0x7fb4253bb470>, <torchtext.vocab.Vocab object at 0x7fb4253bb550>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 24.91 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 48.57 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 29.17 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 29.17 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.07 file/s]2020-06-18 18:20:36.373366: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-18 18:20:36.377690: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-18 18:20:36.377868: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba5737dd30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-18 18:20:36.377883: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 162981.40it/s] 87%|████████▋ | 8617984/9912422 [00:00<00:05, 232641.17it/s]9920512it [00:00, 47135209.50it/s]                           
0it [00:00, ?it/s]32768it [00:00, 619189.22it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 992222.18it/s]1654784it [00:00, 11969680.56it/s]                          
0it [00:00, ?it/s]8192it [00:00, 242278.81it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
