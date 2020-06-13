
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/9e11198fe5a58761eefd3822c50ee8f86a42b427', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '9e11198fe5a58761eefd3822c50ee8f86a42b427', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/9e11198fe5a58761eefd3822c50ee8f86a42b427

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/9e11198fe5a58761eefd3822c50ee8f86a42b427

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/9e11198fe5a58761eefd3822c50ee8f86a42b427

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc52811d6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc52811d6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc5936e50d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc5936e50d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc5ad436ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc5ad436ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc540f62950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc540f62950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc540f62950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3244032/9912422 [00:00<00:00, 32276134.57it/s]9920512it [00:00, 37778763.24it/s]                             
0it [00:00, ?it/s]32768it [00:00, 638339.46it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466357.43it/s]1654784it [00:00, 11848691.31it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203047.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc53e382d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc53e3789e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc540f62598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc540f62598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc540f62598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:10,  2.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:10,  2.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:50,  3.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:50,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:35,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:35,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:00<00:26,  5.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:26,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:18,  7.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:18,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:17,  7.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:13, 10.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:13, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:13, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:12, 10.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:09, 14.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:09, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:09, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:09, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 14.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 18.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 24.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 48.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 56.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:00, 64.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 64.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 71.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 71.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 76.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 76.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 81.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 81.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 84.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 84.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 86.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 86.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 88.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 88.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.71 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 88.71 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.53 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.44s/ url]
0 examples [00:00, ? examples/s]2020-06-13 13:39:50.054018: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-13 13:39:50.080210: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-06-13 13:39:50.080794: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5627ae2f62e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 13:39:50.080817: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
3 examples [00:00, 29.96 examples/s]108 examples [00:00, 42.29 examples/s]226 examples [00:00, 59.50 examples/s]339 examples [00:00, 83.11 examples/s]452 examples [00:00, 115.08 examples/s]560 examples [00:00, 157.16 examples/s]674 examples [00:00, 211.88 examples/s]789 examples [00:00, 280.39 examples/s]908 examples [00:00, 363.58 examples/s]1018 examples [00:01, 454.91 examples/s]1127 examples [00:01, 549.97 examples/s]1242 examples [00:01, 651.64 examples/s]1355 examples [00:01, 745.25 examples/s]1472 examples [00:01, 836.29 examples/s]1587 examples [00:01, 909.40 examples/s]1701 examples [00:01, 963.65 examples/s]1817 examples [00:01, 1013.75 examples/s]1931 examples [00:01, 1039.05 examples/s]2044 examples [00:01, 1056.65 examples/s]2157 examples [00:02, 1075.46 examples/s]2270 examples [00:02, 1089.65 examples/s]2383 examples [00:02, 1088.88 examples/s]2499 examples [00:02, 1108.59 examples/s]2619 examples [00:02, 1132.35 examples/s]2734 examples [00:02, 1125.67 examples/s]2848 examples [00:02, 1121.48 examples/s]2970 examples [00:02, 1130.08 examples/s]3092 examples [00:02, 1154.26 examples/s]3209 examples [00:02, 1156.02 examples/s]3332 examples [00:03, 1176.99 examples/s]3451 examples [00:03, 1150.02 examples/s]3567 examples [00:03, 1137.44 examples/s]3690 examples [00:03, 1161.44 examples/s]3807 examples [00:03, 1118.68 examples/s]3920 examples [00:03, 1105.32 examples/s]4031 examples [00:03, 1094.42 examples/s]4145 examples [00:03, 1106.70 examples/s]4258 examples [00:03, 1111.44 examples/s]4371 examples [00:03, 1115.58 examples/s]4483 examples [00:04, 1105.57 examples/s]4594 examples [00:04, 1099.38 examples/s]4711 examples [00:04, 1118.45 examples/s]4823 examples [00:04, 1109.56 examples/s]4939 examples [00:04, 1122.51 examples/s]5052 examples [00:04, 1124.34 examples/s]5165 examples [00:04, 1115.28 examples/s]5278 examples [00:04, 1117.53 examples/s]5394 examples [00:04, 1128.57 examples/s]5510 examples [00:04, 1137.32 examples/s]5624 examples [00:05, 1135.08 examples/s]5738 examples [00:05, 1123.57 examples/s]5851 examples [00:05, 1076.60 examples/s]5963 examples [00:05, 1086.96 examples/s]6074 examples [00:05, 1091.88 examples/s]6190 examples [00:05, 1109.20 examples/s]6303 examples [00:05, 1112.90 examples/s]6422 examples [00:05, 1134.56 examples/s]6539 examples [00:05, 1144.12 examples/s]6654 examples [00:06, 1123.06 examples/s]6776 examples [00:06, 1149.49 examples/s]6898 examples [00:06, 1167.80 examples/s]7016 examples [00:06, 1158.76 examples/s]7137 examples [00:06, 1173.55 examples/s]7260 examples [00:06, 1189.81 examples/s]7392 examples [00:06, 1224.05 examples/s]7515 examples [00:06, 1191.48 examples/s]7650 examples [00:06, 1232.53 examples/s]7774 examples [00:06, 1218.12 examples/s]7897 examples [00:07, 1202.71 examples/s]8018 examples [00:07, 1194.63 examples/s]8138 examples [00:07, 1180.12 examples/s]8258 examples [00:07, 1183.60 examples/s]8383 examples [00:07, 1199.91 examples/s]8515 examples [00:07, 1231.44 examples/s]8640 examples [00:07, 1233.93 examples/s]8770 examples [00:07, 1251.45 examples/s]8900 examples [00:07, 1264.42 examples/s]9032 examples [00:07, 1277.84 examples/s]9160 examples [00:08, 1273.40 examples/s]9289 examples [00:08, 1277.65 examples/s]9417 examples [00:08, 1264.03 examples/s]9544 examples [00:08, 1232.83 examples/s]9670 examples [00:08, 1239.66 examples/s]9799 examples [00:08, 1252.18 examples/s]9925 examples [00:08, 1223.86 examples/s]10048 examples [00:08, 1174.93 examples/s]10178 examples [00:08, 1209.01 examples/s]10307 examples [00:08, 1231.60 examples/s]10432 examples [00:09, 1236.37 examples/s]10561 examples [00:09, 1249.91 examples/s]10688 examples [00:09, 1255.34 examples/s]10821 examples [00:09, 1275.04 examples/s]10955 examples [00:09, 1291.67 examples/s]11088 examples [00:09, 1299.86 examples/s]11219 examples [00:09, 1281.85 examples/s]11348 examples [00:09, 1281.11 examples/s]11483 examples [00:09, 1299.56 examples/s]11615 examples [00:10, 1304.68 examples/s]11749 examples [00:10, 1314.81 examples/s]11881 examples [00:10, 1302.83 examples/s]12012 examples [00:10, 1273.08 examples/s]12140 examples [00:10, 1257.12 examples/s]12267 examples [00:10, 1259.63 examples/s]12395 examples [00:10, 1264.61 examples/s]12522 examples [00:10, 1262.28 examples/s]12661 examples [00:10, 1297.77 examples/s]12792 examples [00:10, 1297.78 examples/s]12922 examples [00:11, 1295.24 examples/s]13058 examples [00:11, 1311.62 examples/s]13190 examples [00:11, 1304.20 examples/s]13321 examples [00:11, 1253.78 examples/s]13452 examples [00:11, 1267.76 examples/s]13586 examples [00:11, 1287.56 examples/s]13719 examples [00:11, 1296.81 examples/s]13849 examples [00:11, 1264.87 examples/s]13976 examples [00:11, 1241.41 examples/s]14101 examples [00:11, 1238.07 examples/s]14233 examples [00:12, 1259.88 examples/s]14364 examples [00:12, 1272.90 examples/s]14492 examples [00:12, 1266.28 examples/s]14619 examples [00:12, 1229.46 examples/s]14743 examples [00:12, 1220.41 examples/s]14866 examples [00:12, 1217.26 examples/s]14995 examples [00:12, 1236.80 examples/s]15119 examples [00:12, 1217.95 examples/s]15245 examples [00:12, 1229.86 examples/s]15372 examples [00:12, 1241.48 examples/s]15501 examples [00:13, 1254.46 examples/s]15627 examples [00:13, 1232.64 examples/s]15751 examples [00:13, 1185.66 examples/s]15871 examples [00:13, 1188.17 examples/s]16002 examples [00:13, 1221.15 examples/s]16125 examples [00:13, 1223.18 examples/s]16255 examples [00:13, 1243.16 examples/s]16385 examples [00:13, 1258.55 examples/s]16515 examples [00:13, 1269.03 examples/s]16643 examples [00:14, 1267.07 examples/s]16774 examples [00:14, 1277.13 examples/s]16905 examples [00:14, 1285.68 examples/s]17034 examples [00:14, 1284.74 examples/s]17163 examples [00:14, 1256.79 examples/s]17294 examples [00:14, 1271.59 examples/s]17430 examples [00:14, 1294.75 examples/s]17567 examples [00:14, 1314.60 examples/s]17699 examples [00:14, 1313.57 examples/s]17836 examples [00:14, 1327.76 examples/s]17969 examples [00:15, 1315.61 examples/s]18106 examples [00:15, 1329.97 examples/s]18240 examples [00:15, 1325.61 examples/s]18373 examples [00:15, 1273.61 examples/s]18501 examples [00:15, 1250.16 examples/s]18634 examples [00:15, 1272.87 examples/s]18762 examples [00:15, 1242.15 examples/s]18887 examples [00:15, 1240.71 examples/s]19013 examples [00:15, 1246.38 examples/s]19146 examples [00:15, 1269.07 examples/s]19274 examples [00:16, 1248.60 examples/s]19407 examples [00:16, 1269.68 examples/s]19535 examples [00:16, 1261.68 examples/s]19662 examples [00:16, 1263.55 examples/s]19799 examples [00:16, 1291.90 examples/s]19932 examples [00:16, 1301.16 examples/s]20063 examples [00:16, 1235.13 examples/s]20189 examples [00:16, 1238.23 examples/s]20314 examples [00:16, 1214.27 examples/s]20443 examples [00:16, 1235.59 examples/s]20574 examples [00:17, 1256.29 examples/s]20701 examples [00:17, 1233.35 examples/s]20832 examples [00:17, 1254.26 examples/s]20958 examples [00:17, 1235.19 examples/s]21082 examples [00:17, 1215.99 examples/s]21217 examples [00:17, 1252.84 examples/s]21345 examples [00:17, 1257.56 examples/s]21472 examples [00:17, 1244.34 examples/s]21598 examples [00:17, 1245.90 examples/s]21723 examples [00:18, 1223.22 examples/s]21847 examples [00:18, 1227.91 examples/s]21980 examples [00:18, 1256.07 examples/s]22116 examples [00:18, 1283.14 examples/s]22245 examples [00:18, 1272.95 examples/s]22373 examples [00:18, 1263.50 examples/s]22500 examples [00:18, 1236.76 examples/s]22632 examples [00:18, 1258.83 examples/s]22759 examples [00:18, 1251.89 examples/s]22885 examples [00:18, 1240.17 examples/s]23010 examples [00:19, 1224.87 examples/s]23135 examples [00:19, 1230.33 examples/s]23259 examples [00:19, 1222.57 examples/s]23382 examples [00:19, 1219.09 examples/s]23504 examples [00:19, 1208.42 examples/s]23625 examples [00:19, 1193.03 examples/s]23745 examples [00:19, 1193.44 examples/s]23865 examples [00:19, 1175.11 examples/s]23983 examples [00:19, 1163.64 examples/s]24100 examples [00:19, 1162.56 examples/s]24219 examples [00:20, 1167.58 examples/s]24336 examples [00:20, 1163.39 examples/s]24458 examples [00:20, 1178.16 examples/s]24580 examples [00:20, 1187.33 examples/s]24699 examples [00:20, 1181.74 examples/s]24818 examples [00:20, 1165.38 examples/s]24935 examples [00:20, 1140.79 examples/s]25050 examples [00:20, 1131.80 examples/s]25168 examples [00:20, 1144.00 examples/s]25290 examples [00:21, 1164.19 examples/s]25409 examples [00:21, 1171.67 examples/s]25535 examples [00:21, 1194.99 examples/s]25660 examples [00:21, 1208.78 examples/s]25782 examples [00:21, 1184.03 examples/s]25909 examples [00:21, 1207.72 examples/s]26031 examples [00:21, 1202.15 examples/s]26158 examples [00:21, 1220.02 examples/s]26281 examples [00:21, 1205.84 examples/s]26402 examples [00:21, 1204.33 examples/s]26524 examples [00:22, 1208.79 examples/s]26650 examples [00:22, 1221.71 examples/s]26775 examples [00:22, 1229.87 examples/s]26899 examples [00:22, 1206.03 examples/s]27020 examples [00:22, 1178.91 examples/s]27139 examples [00:22, 1167.30 examples/s]27268 examples [00:22, 1200.58 examples/s]27395 examples [00:22, 1219.44 examples/s]27518 examples [00:22, 1222.11 examples/s]27641 examples [00:22, 1218.67 examples/s]27765 examples [00:23, 1224.95 examples/s]27892 examples [00:23, 1236.02 examples/s]28016 examples [00:23, 1223.62 examples/s]28145 examples [00:23, 1242.20 examples/s]28270 examples [00:23, 1218.26 examples/s]28393 examples [00:23, 1211.36 examples/s]28515 examples [00:23, 1176.85 examples/s]28639 examples [00:23, 1192.18 examples/s]28762 examples [00:23, 1203.22 examples/s]28883 examples [00:23, 1168.85 examples/s]29001 examples [00:24, 1146.46 examples/s]29120 examples [00:24, 1158.28 examples/s]29243 examples [00:24, 1178.75 examples/s]29372 examples [00:24, 1208.60 examples/s]29494 examples [00:24, 1204.07 examples/s]29617 examples [00:24, 1211.57 examples/s]29741 examples [00:24, 1218.51 examples/s]29864 examples [00:24, 1219.60 examples/s]29990 examples [00:24, 1229.68 examples/s]30114 examples [00:25, 1159.14 examples/s]30241 examples [00:25, 1189.46 examples/s]30365 examples [00:25, 1202.84 examples/s]30491 examples [00:25, 1219.05 examples/s]30617 examples [00:25, 1229.09 examples/s]30741 examples [00:25, 1224.91 examples/s]30864 examples [00:25, 1214.83 examples/s]30986 examples [00:25, 1189.55 examples/s]31106 examples [00:25, 1176.45 examples/s]31224 examples [00:25, 1159.18 examples/s]31341 examples [00:26, 1153.32 examples/s]31457 examples [00:26, 1116.70 examples/s]31570 examples [00:26, 1065.77 examples/s]31695 examples [00:26, 1113.13 examples/s]31823 examples [00:26, 1157.06 examples/s]31950 examples [00:26, 1186.89 examples/s]32083 examples [00:26, 1225.50 examples/s]32207 examples [00:26, 1210.89 examples/s]32332 examples [00:26, 1221.06 examples/s]32455 examples [00:26, 1223.10 examples/s]32578 examples [00:27, 1153.02 examples/s]32703 examples [00:27, 1178.10 examples/s]32830 examples [00:27, 1201.93 examples/s]32951 examples [00:27, 1203.68 examples/s]33072 examples [00:27, 1186.16 examples/s]33192 examples [00:27, 1165.90 examples/s]33314 examples [00:27, 1179.69 examples/s]33444 examples [00:27, 1212.18 examples/s]33566 examples [00:27, 1213.89 examples/s]33688 examples [00:28, 1193.37 examples/s]33808 examples [00:28, 1176.16 examples/s]33934 examples [00:28, 1198.40 examples/s]34063 examples [00:28, 1224.40 examples/s]34196 examples [00:28, 1252.27 examples/s]34322 examples [00:28, 1242.86 examples/s]34447 examples [00:28, 1238.39 examples/s]34572 examples [00:28, 1234.44 examples/s]34696 examples [00:28, 1234.23 examples/s]34820 examples [00:28, 1231.00 examples/s]34945 examples [00:29, 1235.79 examples/s]35069 examples [00:29, 1221.84 examples/s]35192 examples [00:29, 1200.41 examples/s]35319 examples [00:29, 1217.89 examples/s]35441 examples [00:29, 1198.87 examples/s]35562 examples [00:29, 1195.44 examples/s]35685 examples [00:29, 1203.92 examples/s]35806 examples [00:29, 1190.28 examples/s]35927 examples [00:29, 1194.88 examples/s]36053 examples [00:29, 1212.90 examples/s]36180 examples [00:30, 1228.62 examples/s]36303 examples [00:30, 1198.19 examples/s]36426 examples [00:30, 1204.91 examples/s]36552 examples [00:30, 1218.94 examples/s]36675 examples [00:30, 1210.26 examples/s]36798 examples [00:30, 1215.58 examples/s]36920 examples [00:30, 1210.49 examples/s]37045 examples [00:30, 1219.85 examples/s]37168 examples [00:30, 1217.95 examples/s]37290 examples [00:30, 1217.53 examples/s]37412 examples [00:31, 1189.37 examples/s]37532 examples [00:31, 1184.70 examples/s]37651 examples [00:31, 1183.86 examples/s]37772 examples [00:31, 1190.09 examples/s]37895 examples [00:31, 1199.33 examples/s]38015 examples [00:31, 1174.87 examples/s]38139 examples [00:31, 1191.37 examples/s]38273 examples [00:31, 1231.72 examples/s]38400 examples [00:31, 1241.03 examples/s]38531 examples [00:31, 1258.49 examples/s]38658 examples [00:32, 1242.98 examples/s]38785 examples [00:32, 1250.68 examples/s]38911 examples [00:32, 1216.74 examples/s]39034 examples [00:32, 1193.98 examples/s]39154 examples [00:32, 1169.81 examples/s]39272 examples [00:32, 1157.44 examples/s]39391 examples [00:32, 1165.47 examples/s]39527 examples [00:32, 1215.09 examples/s]39662 examples [00:32, 1250.53 examples/s]39795 examples [00:33, 1273.15 examples/s]39923 examples [00:33, 1267.16 examples/s]40051 examples [00:33, 1183.47 examples/s]40186 examples [00:33, 1226.36 examples/s]40311 examples [00:33, 1185.23 examples/s]40440 examples [00:33, 1213.42 examples/s]40572 examples [00:33, 1242.26 examples/s]40702 examples [00:33, 1257.83 examples/s]40836 examples [00:33, 1280.25 examples/s]40972 examples [00:33, 1302.46 examples/s]41105 examples [00:34, 1310.42 examples/s]41237 examples [00:34, 1255.70 examples/s]41364 examples [00:34, 1231.03 examples/s]41491 examples [00:34, 1242.24 examples/s]41616 examples [00:34, 1221.80 examples/s]41740 examples [00:34, 1226.52 examples/s]41867 examples [00:34, 1235.03 examples/s]41999 examples [00:34, 1258.52 examples/s]42132 examples [00:34, 1278.09 examples/s]42264 examples [00:35, 1289.58 examples/s]42394 examples [00:35, 1245.93 examples/s]42523 examples [00:35, 1256.12 examples/s]42659 examples [00:35, 1283.56 examples/s]42793 examples [00:35, 1297.90 examples/s]42924 examples [00:35, 1300.25 examples/s]43055 examples [00:35, 1292.96 examples/s]43185 examples [00:35, 1285.96 examples/s]43316 examples [00:35, 1291.36 examples/s]43446 examples [00:35, 1259.01 examples/s]43573 examples [00:36, 1250.87 examples/s]43700 examples [00:36, 1256.18 examples/s]43826 examples [00:36, 1223.49 examples/s]43955 examples [00:36, 1237.98 examples/s]44081 examples [00:36, 1243.73 examples/s]44210 examples [00:36, 1256.05 examples/s]44341 examples [00:36, 1270.90 examples/s]44469 examples [00:36, 1268.49 examples/s]44598 examples [00:36, 1273.22 examples/s]44731 examples [00:36, 1287.20 examples/s]44861 examples [00:37, 1289.64 examples/s]44994 examples [00:37, 1301.43 examples/s]45125 examples [00:37, 1301.03 examples/s]45261 examples [00:37, 1316.70 examples/s]45393 examples [00:37, 1258.75 examples/s]45520 examples [00:37, 1201.24 examples/s]45642 examples [00:37, 1200.38 examples/s]45765 examples [00:37, 1206.63 examples/s]45900 examples [00:37, 1246.06 examples/s]46027 examples [00:37, 1252.99 examples/s]46153 examples [00:38, 1241.19 examples/s]46278 examples [00:38, 1241.05 examples/s]46411 examples [00:38, 1264.12 examples/s]46538 examples [00:38, 1262.04 examples/s]46667 examples [00:38, 1269.05 examples/s]46795 examples [00:38, 1269.19 examples/s]46923 examples [00:38, 1198.08 examples/s]47044 examples [00:38, 1183.42 examples/s]47164 examples [00:38, 1169.34 examples/s]47282 examples [00:39, 1165.53 examples/s]47402 examples [00:39, 1173.82 examples/s]47520 examples [00:39, 1161.51 examples/s]47637 examples [00:39, 1112.87 examples/s]47757 examples [00:39, 1136.60 examples/s]47883 examples [00:39, 1169.57 examples/s]48010 examples [00:39, 1197.30 examples/s]48136 examples [00:39, 1212.81 examples/s]48258 examples [00:39, 1200.29 examples/s]48387 examples [00:39, 1223.89 examples/s]48510 examples [00:40, 1213.81 examples/s]48632 examples [00:40, 1206.13 examples/s]48759 examples [00:40, 1222.96 examples/s]48882 examples [00:40, 1224.61 examples/s]49011 examples [00:40, 1243.30 examples/s]49142 examples [00:40, 1260.95 examples/s]49270 examples [00:40, 1264.59 examples/s]49397 examples [00:40, 1247.67 examples/s]49526 examples [00:40, 1258.32 examples/s]49652 examples [00:40, 1251.03 examples/s]49778 examples [00:41, 1216.05 examples/s]49904 examples [00:41, 1228.48 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9553/50000 [00:00<00:00, 95525.99 examples/s] 50%|█████     | 25100/50000 [00:00<00:00, 108020.50 examples/s] 80%|████████  | 40180/50000 [00:00<00:00, 118067.85 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 863.29 examples/s]205 examples [00:00, 937.78 examples/s]333 examples [00:00, 1017.96 examples/s]455 examples [00:00, 1070.67 examples/s]580 examples [00:00, 1118.52 examples/s]686 examples [00:00, 1099.27 examples/s]797 examples [00:00, 1100.33 examples/s]908 examples [00:00, 1102.61 examples/s]1035 examples [00:00, 1146.58 examples/s]1171 examples [00:01, 1201.31 examples/s]1291 examples [00:01, 1179.66 examples/s]1409 examples [00:01, 945.21 examples/s] 1541 examples [00:01, 1032.64 examples/s]1663 examples [00:01, 1081.99 examples/s]1790 examples [00:01, 1131.51 examples/s]1914 examples [00:01, 1160.05 examples/s]2037 examples [00:01, 1178.71 examples/s]2158 examples [00:01, 1156.90 examples/s]2276 examples [00:02, 1148.57 examples/s]2400 examples [00:02, 1172.54 examples/s]2524 examples [00:02, 1189.92 examples/s]2649 examples [00:02, 1205.54 examples/s]2774 examples [00:02, 1215.51 examples/s]2898 examples [00:02, 1219.64 examples/s]3021 examples [00:02, 1213.36 examples/s]3143 examples [00:02, 1186.93 examples/s]3262 examples [00:02, 1154.04 examples/s]3378 examples [00:02, 1155.27 examples/s]3500 examples [00:03, 1172.22 examples/s]3625 examples [00:03, 1192.19 examples/s]3758 examples [00:03, 1230.01 examples/s]3886 examples [00:03, 1243.72 examples/s]4021 examples [00:03, 1273.17 examples/s]4149 examples [00:03, 1247.15 examples/s]4276 examples [00:03, 1246.72 examples/s]4401 examples [00:03, 1217.30 examples/s]4524 examples [00:03, 1207.88 examples/s]4659 examples [00:03, 1245.24 examples/s]4790 examples [00:04, 1261.29 examples/s]4917 examples [00:04, 1262.73 examples/s]5044 examples [00:04, 1196.32 examples/s]5165 examples [00:04, 1187.03 examples/s]5286 examples [00:04, 1191.93 examples/s]5420 examples [00:04, 1232.25 examples/s]5554 examples [00:04, 1261.71 examples/s]5689 examples [00:04, 1284.21 examples/s]5823 examples [00:04, 1297.75 examples/s]5954 examples [00:04, 1299.00 examples/s]6089 examples [00:05, 1313.51 examples/s]6221 examples [00:05, 1300.46 examples/s]6352 examples [00:05, 1288.51 examples/s]6486 examples [00:05, 1300.76 examples/s]6617 examples [00:05, 1298.21 examples/s]6749 examples [00:05, 1302.60 examples/s]6883 examples [00:05, 1311.01 examples/s]7015 examples [00:05, 1306.34 examples/s]7148 examples [00:05, 1310.91 examples/s]7283 examples [00:05, 1321.89 examples/s]7416 examples [00:06, 1292.92 examples/s]7547 examples [00:06, 1296.49 examples/s]7677 examples [00:06, 1289.17 examples/s]7811 examples [00:06, 1301.45 examples/s]7946 examples [00:06, 1313.78 examples/s]8080 examples [00:06, 1321.20 examples/s]8213 examples [00:06, 1321.27 examples/s]8346 examples [00:06, 1283.73 examples/s]8475 examples [00:06, 1272.83 examples/s]8604 examples [00:07, 1277.69 examples/s]8732 examples [00:07, 1273.55 examples/s]8860 examples [00:07, 1256.15 examples/s]8986 examples [00:07, 1252.41 examples/s]9112 examples [00:07, 1246.33 examples/s]9244 examples [00:07, 1267.22 examples/s]9371 examples [00:07, 1261.45 examples/s]9498 examples [00:07, 1231.30 examples/s]9622 examples [00:07, 1198.34 examples/s]9745 examples [00:07, 1206.96 examples/s]9874 examples [00:08, 1230.46 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQXVV74/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQXVV74/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc540f62950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc540f62950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc540f62950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc58c94d3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4ca3eb898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc540f62950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc540f62950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc540f62950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc53e597d30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc53e5973c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc4c80992f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc4c80992f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fc4c80992f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc4c8099400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc4c8099400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc4c8099400> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:30:17, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:35:10, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:15:25, 23.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:11:18, 33.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.12M/862M [00:01<5:01:19, 47.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.37M/862M [00:01<3:30:17, 67.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.4M/862M [00:01<2:26:25, 96.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:01<1:42:04, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<1:11:10, 197kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.7M/862M [00:01<49:39, 281kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<34:39, 401kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.5M/862M [00:01<24:14, 570kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<16:58, 809kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:02<11:54, 1.15MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:02<08:23, 1.62MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<05:56, 2.28MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<05:39, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<05:51, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<05:55, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<04:32, 2.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:43, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<05:38, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<04:05, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<13:13, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<10:23, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<07:33, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:09<05:30, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<29:15, 452kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<21:49, 605kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<15:35, 846kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<14:00, 939kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:12<12:28, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<09:23, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<08:40, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<07:25, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:14<05:28, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:51, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<06:07, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:16<04:36, 2.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:17, 2.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:42, 2.26MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<04:19, 2.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:04, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<05:32, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<04:11, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:59, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:50, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<05:20, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:22<03:55, 3.26MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<08:09, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:47, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<05:01, 2.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<03:40, 3.45MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<53:40, 236kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<40:08, 316kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<28:37, 442kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<20:08, 626kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<21:14, 593kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:08, 780kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<11:33, 1.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:00, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:58, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<06:35, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:15, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<9:51:12, 21.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<6:54:04, 30.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<4:48:49, 42.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<3:41:29, 56.0kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:37:31, 78.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:50:45, 112kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:19:13, 156kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<56:42, 217kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<39:56, 308kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<30:43, 399kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<22:44, 539kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<16:11, 755kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<14:11, 859kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<11:10, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<08:03, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:39, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:31, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:28, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:41, 2.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:16, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:44, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:40, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:46, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:10, 1.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:03, 2.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:29:03, 134kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:03:31, 188kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<44:40, 266kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<33:57, 349kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<26:16, 451kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<18:54, 626kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<13:22, 882kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<14:03, 838kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:03, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:01, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<08:20, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:19, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:20, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:34, 2.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:31, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:52, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:47, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:45, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:06, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<04:00, 2.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:16:45, 150kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<54:52, 210kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<38:37, 297kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<29:38, 386kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<21:53, 522kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<15:35, 732kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<13:32, 839kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<11:47, 964kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:44, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<06:14, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<11:46, 959kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:24, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:49, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:24, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:19, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:42, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:55, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:25, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:59, 2.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:35, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<19:45, 562kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<14:57, 742kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<10:40, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<10:03, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:17, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:58, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<09:34, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:49, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:44, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:34, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:48, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:27, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:44, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:35, 3.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:03, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:29, 3.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:57, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:38, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:24, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:12, 3.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:22, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:28, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:25, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:56, 2.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<45:16, 233kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<33:49, 312kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<24:07, 436kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<18:31, 566kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<14:01, 747kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:03, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:27, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:43, 1.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:34, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:42, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:39, 817kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:55, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<07:09, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:25, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:17, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:36, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:34, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:56, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:42, 2.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:58, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:27, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:15, 2.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:04, 3.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<17:28, 576kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<13:15, 758kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<09:29, 1.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<08:58, 1.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<08:18, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:15, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:29, 2.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<10:33, 941kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:24, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:07, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<06:34, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:37, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:04, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:39, 2.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<09:52, 992kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:54, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:46, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:17, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:23, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:32, 2.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:02, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:37, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:50, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:37, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:52, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:31, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:16, 2.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<08:45, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:05, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:10, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:50, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:02, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:45, 2.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:50, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:17, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:06, 2.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:00, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:46, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:59, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:43, 2.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:46, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:13, 2.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:24, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:56, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:55, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:51, 3.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<1:16:21, 119kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<54:21, 167kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<38:10, 238kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<28:43, 315kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<21:56, 412kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<15:43, 574kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<11:05, 811kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<12:50, 698kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<09:55, 904kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<07:08, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<07:03, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:45, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:10, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:02, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:26, 1.99MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:19, 2.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:21, 2.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:49, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:49, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:04, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:44, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:50, 3.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:59, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:32, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:34, 2.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:34, 3.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<12:32, 683kB/s] .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:39, 885kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<06:56, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:48, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:29, 1.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:55, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<03:30, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<13:16, 635kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<10:03, 837kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<07:14, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:58, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:35, 1.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:59, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:33, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<14:24, 575kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<10:55, 758kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<07:49, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<07:23, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:50, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:11, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:56, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:17, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:10, 2.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:06, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:31, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:30, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:32, 3.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<06:07, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:07, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:44, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:28, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:56, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<02:55, 2.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:55, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:20, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:25, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:39, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:22, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:32, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:33, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:14, 2.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:30, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:27, 3.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:28, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:13, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:26, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:28, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:57, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:06, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:15, 3.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:25, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:07, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:44, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<02:42, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<31:43, 233kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<22:56, 322kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<16:11, 455kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<13:00, 563kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<10:37, 690kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<07:47, 938kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:35, 1.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:22, 1.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:54, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:23, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:41, 1.94MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:45, 2.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:37, 1.97MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:59, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:05, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:18, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:37, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:27, 2.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:20, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:04, 2.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:19, 2.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:14, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:45, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:56, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:08, 3.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<17:58, 381kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<13:17, 515kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<09:26, 721kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<08:08, 832kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<07:06, 953kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:18, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:46, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:02, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:58, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:37, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:12, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:22, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<01:44, 3.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<50:08, 131kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<35:44, 184kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<25:05, 261kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<19:00, 342kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<13:57, 466kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<09:53, 654kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<08:24, 765kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<06:32, 983kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:43, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:47, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:07, 2.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<1:16:28, 82.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<54:47, 115kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<38:33, 163kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<26:54, 232kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<22:19, 279kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<16:15, 383kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<11:29, 539kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<09:26, 652kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<07:13, 851kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<05:11, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:03, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:09, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:03, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:33, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:39, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:02, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:56, 1.20MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:44, 1.25MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:34, 1.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:35, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:41, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:05, 1.90MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:25, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:44, 3.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<12:12, 476kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<09:43, 598kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:05, 818kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:51, 982kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:41, 1.22MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:24, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:41, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<03:44, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:04, 2.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<43:31, 129kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<31:00, 181kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<21:44, 257kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<16:25, 337kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<12:37, 439kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<09:04, 609kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<06:21, 861kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<09:02, 605kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:54, 792kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<04:55, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:41, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:45, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:44, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<01:58, 2.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<11:28, 465kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<09:06, 585kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<06:36, 806kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<04:40, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<05:31, 952kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:18, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<03:07, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<02:14, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<22:29, 231kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<16:15, 320kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<11:25, 452kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<09:09, 560kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:27, 688kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:26, 940kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<03:50, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<07:33, 669kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:48, 870kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:10, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:04, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:17, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:28, 2.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:46, 2.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<21:04, 234kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<15:14, 323kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<10:43, 456kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<08:36, 564kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<07:00, 693kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:05, 949kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<03:37, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:29, 1.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:38, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:39, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:57, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:32, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:52, 2.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:24, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:36, 2.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:12, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:28, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:57, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:24, 3.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:41, 961kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:45, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:43, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:55, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:00, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:19, 1.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:39, 2.63MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<32:21, 135kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<23:04, 189kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<16:09, 269kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<12:13, 352kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<09:25, 456kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<06:45, 634kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<04:45, 895kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<05:16, 803kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<04:07, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:57, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:02, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:33, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:52, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:16, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:56, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:26, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:03, 3.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<16:59, 237kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<12:17, 328kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<08:38, 463kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<06:56, 572kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<05:15, 753kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:45, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:32, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:16, 1.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:01, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:29, 2.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:54, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:05, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:39, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:44, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:12, 3.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:40, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:32, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:10, 3.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:38, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:08, 3.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:50, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:26, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:01, 3.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:14, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:37, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:54, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:06, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:49, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:21, 2.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:42, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:31, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:07, 2.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:33, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:44, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:21, 2.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:58, 3.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<07:03, 445kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<05:15, 596kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<03:43, 836kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:17, 932kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:36, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:01, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:42, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:15, 2.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:33, 1.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:41, 1.74MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:18, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:56, 3.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:29, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:02, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:29, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:41, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:44, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:21, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:22, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:14, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:55, 2.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:15, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:08, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:51, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:12, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:06, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:50, 3.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:48, 3.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:02, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:46, 3.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:06, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:15, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<00:59, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:03, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<00:58, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:44, 3.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:10, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:55, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:39, 3.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:40, 815kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:05, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:31, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:29, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:09, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:07, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:59, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:44, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:57, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:04, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:50, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:53, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:46, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:34, 3.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:25, 4.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:46, 487kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:01, 607kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:11, 833kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:31, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:15, 784kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:45, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:15, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:13, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:39, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:20, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:06, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:48, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:55, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:57, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:43, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:31, 2.89MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:57, 1.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:35, 2.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:44, 1.92MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:29, 2.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:35, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:26, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:36, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:33, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:24, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:29, 2.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:22, 3.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:29, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:18, 3.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<58:43, 17.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<40:57, 24.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<27:57, 35.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<19:04, 49.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<13:29, 69.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<09:21, 99.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<06:19, 139kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<04:29, 194kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<03:04, 275kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<02:14, 360kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:37, 493kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:07, 695kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:45, 978kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<03:33, 208kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<02:32, 289kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<01:43, 409kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:18, 513kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:02, 637kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:45, 870kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:29, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<34:59, 17.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<24:17, 24.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<16:17, 35.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<10:47, 49.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<07:31, 70.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<05:00, 100kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<03:22, 138kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<02:26, 190kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:40, 268kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<01:05, 381kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:55, 429kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:40, 576kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:27, 806kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:21, 906kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:16, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:10, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:10, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:14, 232kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:09, 311kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:05, 435kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 614kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 107/400000 [00:00<06:13, 1069.56it/s]  0%|          | 1010/400000 [00:00<04:34, 1454.09it/s]  0%|          | 1879/400000 [00:00<03:25, 1938.20it/s]  1%|          | 2598/400000 [00:00<02:40, 2481.96it/s]  1%|          | 3440/400000 [00:00<02:05, 3147.81it/s]  1%|          | 4355/400000 [00:00<01:40, 3918.71it/s]  1%|▏         | 5262/400000 [00:00<01:23, 4722.70it/s]  2%|▏         | 6208/400000 [00:00<01:10, 5556.71it/s]  2%|▏         | 7088/400000 [00:00<01:02, 6247.19it/s]  2%|▏         | 7951/400000 [00:01<00:57, 6810.85it/s]  2%|▏         | 8881/400000 [00:01<00:52, 7403.73it/s]  2%|▏         | 9864/400000 [00:01<00:48, 7993.90it/s]  3%|▎         | 10830/400000 [00:01<00:46, 8429.66it/s]  3%|▎         | 11754/400000 [00:01<00:46, 8438.18it/s]  3%|▎         | 12655/400000 [00:01<00:45, 8567.92it/s]  3%|▎         | 13552/400000 [00:01<00:44, 8656.03it/s]  4%|▎         | 14447/400000 [00:01<00:44, 8741.44it/s]  4%|▍         | 15344/400000 [00:01<00:43, 8807.03it/s]  4%|▍         | 16250/400000 [00:01<00:43, 8881.11it/s]  4%|▍         | 17149/400000 [00:02<00:43, 8870.07it/s]  5%|▍         | 18072/400000 [00:02<00:42, 8973.88it/s]  5%|▍         | 18975/400000 [00:02<00:42, 8928.14it/s]  5%|▍         | 19872/400000 [00:02<00:42, 8864.32it/s]  5%|▌         | 20764/400000 [00:02<00:42, 8878.92it/s]  5%|▌         | 21657/400000 [00:02<00:42, 8893.13it/s]  6%|▌         | 22592/400000 [00:02<00:41, 9024.26it/s]  6%|▌         | 23496/400000 [00:02<00:41, 8984.67it/s]  6%|▌         | 24418/400000 [00:02<00:41, 9051.10it/s]  6%|▋         | 25324/400000 [00:02<00:41, 9011.76it/s]  7%|▋         | 26226/400000 [00:03<00:41, 8935.09it/s]  7%|▋         | 27149/400000 [00:03<00:41, 9021.55it/s]  7%|▋         | 28052/400000 [00:03<00:41, 8953.79it/s]  7%|▋         | 28948/400000 [00:03<00:41, 8866.51it/s]  7%|▋         | 29836/400000 [00:03<00:42, 8802.15it/s]  8%|▊         | 30787/400000 [00:03<00:41, 9001.67it/s]  8%|▊         | 31761/400000 [00:03<00:39, 9209.38it/s]  8%|▊         | 32685/400000 [00:03<00:39, 9195.92it/s]  8%|▊         | 33607/400000 [00:03<00:40, 8969.83it/s]  9%|▊         | 34507/400000 [00:03<00:41, 8773.08it/s]  9%|▉         | 35427/400000 [00:04<00:40, 8896.73it/s]  9%|▉         | 36319/400000 [00:04<00:41, 8820.71it/s]  9%|▉         | 37233/400000 [00:04<00:40, 8911.79it/s] 10%|▉         | 38126/400000 [00:04<00:41, 8800.55it/s] 10%|▉         | 39008/400000 [00:04<00:41, 8666.00it/s] 10%|▉         | 39876/400000 [00:04<00:41, 8652.29it/s] 10%|█         | 40836/400000 [00:04<00:40, 8915.85it/s] 10%|█         | 41731/400000 [00:04<00:40, 8815.45it/s] 11%|█         | 42615/400000 [00:04<00:40, 8766.05it/s] 11%|█         | 43494/400000 [00:04<00:40, 8732.61it/s] 11%|█         | 44375/400000 [00:05<00:40, 8753.54it/s] 11%|█▏        | 45256/400000 [00:05<00:40, 8769.75it/s] 12%|█▏        | 46151/400000 [00:05<00:40, 8821.25it/s] 12%|█▏        | 47034/400000 [00:05<00:40, 8692.33it/s] 12%|█▏        | 47904/400000 [00:05<00:40, 8658.87it/s] 12%|█▏        | 48783/400000 [00:05<00:40, 8696.36it/s] 12%|█▏        | 49668/400000 [00:05<00:40, 8740.38it/s] 13%|█▎        | 50543/400000 [00:05<00:40, 8626.11it/s] 13%|█▎        | 51422/400000 [00:05<00:40, 8673.55it/s] 13%|█▎        | 52290/400000 [00:05<00:40, 8654.34it/s] 13%|█▎        | 53167/400000 [00:06<00:39, 8685.92it/s] 14%|█▎        | 54036/400000 [00:06<00:39, 8663.44it/s] 14%|█▎        | 54922/400000 [00:06<00:39, 8719.86it/s] 14%|█▍        | 55795/400000 [00:06<00:39, 8674.04it/s] 14%|█▍        | 56663/400000 [00:06<00:39, 8657.31it/s] 14%|█▍        | 57545/400000 [00:06<00:39, 8703.35it/s] 15%|█▍        | 58428/400000 [00:06<00:39, 8739.92it/s] 15%|█▍        | 59328/400000 [00:06<00:38, 8813.62it/s] 15%|█▌        | 60210/400000 [00:06<00:38, 8756.35it/s] 15%|█▌        | 61092/400000 [00:06<00:38, 8773.47it/s] 15%|█▌        | 61970/400000 [00:07<00:38, 8729.87it/s] 16%|█▌        | 62844/400000 [00:07<00:38, 8732.44it/s] 16%|█▌        | 63730/400000 [00:07<00:38, 8767.86it/s] 16%|█▌        | 64640/400000 [00:07<00:37, 8864.04it/s] 16%|█▋        | 65528/400000 [00:07<00:37, 8868.80it/s] 17%|█▋        | 66456/400000 [00:07<00:37, 8986.25it/s] 17%|█▋        | 67360/400000 [00:07<00:36, 9001.12it/s] 17%|█▋        | 68261/400000 [00:07<00:37, 8880.52it/s] 17%|█▋        | 69150/400000 [00:07<00:37, 8872.31it/s] 18%|█▊        | 70038/400000 [00:08<00:37, 8798.60it/s] 18%|█▊        | 70925/400000 [00:08<00:37, 8817.85it/s] 18%|█▊        | 71862/400000 [00:08<00:36, 8976.53it/s] 18%|█▊        | 72761/400000 [00:08<00:36, 8883.56it/s] 18%|█▊        | 73721/400000 [00:08<00:35, 9086.30it/s] 19%|█▊        | 74632/400000 [00:08<00:36, 8982.56it/s] 19%|█▉        | 75537/400000 [00:08<00:36, 9002.02it/s] 19%|█▉        | 76439/400000 [00:08<00:36, 8928.08it/s] 19%|█▉        | 77333/400000 [00:08<00:36, 8775.32it/s] 20%|█▉        | 78217/400000 [00:08<00:36, 8792.24it/s] 20%|█▉        | 79098/400000 [00:09<00:36, 8680.92it/s] 20%|██        | 80001/400000 [00:09<00:36, 8781.59it/s] 20%|██        | 80881/400000 [00:09<00:36, 8670.16it/s] 20%|██        | 81749/400000 [00:09<00:37, 8559.71it/s] 21%|██        | 82626/400000 [00:09<00:36, 8621.05it/s] 21%|██        | 83489/400000 [00:09<00:36, 8574.96it/s] 21%|██        | 84361/400000 [00:09<00:36, 8617.33it/s] 21%|██▏       | 85255/400000 [00:09<00:36, 8708.66it/s] 22%|██▏       | 86158/400000 [00:09<00:35, 8801.07it/s] 22%|██▏       | 87048/400000 [00:09<00:35, 8829.55it/s] 22%|██▏       | 87942/400000 [00:10<00:35, 8860.04it/s] 22%|██▏       | 88829/400000 [00:10<00:35, 8860.77it/s] 22%|██▏       | 89721/400000 [00:10<00:34, 8876.32it/s] 23%|██▎       | 90609/400000 [00:10<00:34, 8844.19it/s] 23%|██▎       | 91516/400000 [00:10<00:34, 8910.18it/s] 23%|██▎       | 92408/400000 [00:10<00:34, 8836.34it/s] 23%|██▎       | 93326/400000 [00:10<00:34, 8934.57it/s] 24%|██▎       | 94220/400000 [00:10<00:34, 8917.95it/s] 24%|██▍       | 95148/400000 [00:10<00:33, 9020.75it/s] 24%|██▍       | 96051/400000 [00:10<00:33, 9009.22it/s] 24%|██▍       | 96953/400000 [00:11<00:34, 8867.28it/s] 24%|██▍       | 97841/400000 [00:11<00:34, 8867.64it/s] 25%|██▍       | 98741/400000 [00:11<00:33, 8905.96it/s] 25%|██▍       | 99674/400000 [00:11<00:33, 9026.01it/s] 25%|██▌       | 100587/400000 [00:11<00:33, 9056.78it/s] 25%|██▌       | 101494/400000 [00:11<00:33, 8951.66it/s] 26%|██▌       | 102390/400000 [00:11<00:33, 8829.25it/s] 26%|██▌       | 103274/400000 [00:11<00:34, 8724.02it/s] 26%|██▌       | 104151/400000 [00:11<00:33, 8737.34it/s] 26%|██▋       | 105036/400000 [00:11<00:33, 8769.21it/s] 26%|██▋       | 105914/400000 [00:12<00:33, 8748.03it/s] 27%|██▋       | 106807/400000 [00:12<00:33, 8800.79it/s] 27%|██▋       | 107688/400000 [00:12<00:33, 8731.56it/s] 27%|██▋       | 108634/400000 [00:12<00:32, 8937.79it/s] 27%|██▋       | 109530/400000 [00:12<00:32, 8943.58it/s] 28%|██▊       | 110446/400000 [00:12<00:32, 9006.92it/s] 28%|██▊       | 111400/400000 [00:12<00:31, 9160.39it/s] 28%|██▊       | 112318/400000 [00:12<00:31, 9088.10it/s] 28%|██▊       | 113256/400000 [00:12<00:31, 9171.70it/s] 29%|██▊       | 114175/400000 [00:12<00:31, 9090.70it/s] 29%|██▉       | 115110/400000 [00:13<00:31, 9165.66it/s] 29%|██▉       | 116028/400000 [00:13<00:31, 8995.83it/s] 29%|██▉       | 116929/400000 [00:13<00:31, 8930.01it/s] 29%|██▉       | 117844/400000 [00:13<00:31, 8993.84it/s] 30%|██▉       | 118809/400000 [00:13<00:30, 9180.25it/s] 30%|██▉       | 119729/400000 [00:13<00:30, 9107.97it/s] 30%|███       | 120641/400000 [00:13<00:31, 8921.76it/s] 30%|███       | 121535/400000 [00:13<00:31, 8906.56it/s] 31%|███       | 122427/400000 [00:13<00:31, 8839.60it/s] 31%|███       | 123323/400000 [00:13<00:31, 8874.61it/s] 31%|███       | 124224/400000 [00:14<00:30, 8912.90it/s] 31%|███▏      | 125119/400000 [00:14<00:30, 8921.18it/s] 32%|███▏      | 126030/400000 [00:14<00:30, 8975.31it/s] 32%|███▏      | 126963/400000 [00:14<00:30, 9078.83it/s] 32%|███▏      | 127872/400000 [00:14<00:30, 9046.88it/s] 32%|███▏      | 128778/400000 [00:14<00:30, 8957.84it/s] 32%|███▏      | 129681/400000 [00:14<00:30, 8977.80it/s] 33%|███▎      | 130580/400000 [00:14<00:30, 8938.57it/s] 33%|███▎      | 131475/400000 [00:14<00:30, 8896.01it/s] 33%|███▎      | 132365/400000 [00:15<00:30, 8862.80it/s] 33%|███▎      | 133252/400000 [00:15<00:30, 8760.45it/s] 34%|███▎      | 134141/400000 [00:15<00:30, 8798.36it/s] 34%|███▍      | 135022/400000 [00:15<00:30, 8767.12it/s] 34%|███▍      | 135927/400000 [00:15<00:29, 8848.21it/s] 34%|███▍      | 136831/400000 [00:15<00:29, 8904.33it/s] 34%|███▍      | 137744/400000 [00:15<00:29, 8969.15it/s] 35%|███▍      | 138688/400000 [00:15<00:28, 9102.89it/s] 35%|███▍      | 139600/400000 [00:15<00:28, 9019.91it/s] 35%|███▌      | 140600/400000 [00:15<00:27, 9291.19it/s] 35%|███▌      | 141532/400000 [00:16<00:28, 9195.66it/s] 36%|███▌      | 142454/400000 [00:16<00:28, 9029.60it/s] 36%|███▌      | 143391/400000 [00:16<00:28, 9128.46it/s] 36%|███▌      | 144343/400000 [00:16<00:27, 9219.94it/s] 36%|███▋      | 145426/400000 [00:16<00:26, 9649.98it/s] 37%|███▋      | 146508/400000 [00:16<00:25, 9971.62it/s] 37%|███▋      | 147514/400000 [00:16<00:25, 9997.35it/s] 37%|███▋      | 148632/400000 [00:16<00:24, 10323.01it/s] 37%|███▋      | 149759/400000 [00:16<00:23, 10589.86it/s] 38%|███▊      | 150864/400000 [00:16<00:23, 10721.96it/s] 38%|███▊      | 151973/400000 [00:17<00:22, 10829.63it/s] 38%|███▊      | 153113/400000 [00:17<00:22, 10991.68it/s] 39%|███▊      | 154229/400000 [00:17<00:22, 11040.36it/s] 39%|███▉      | 155336/400000 [00:17<00:22, 10977.51it/s] 39%|███▉      | 156443/400000 [00:17<00:22, 11004.56it/s] 39%|███▉      | 157559/400000 [00:17<00:21, 11048.07it/s] 40%|███▉      | 158665/400000 [00:17<00:22, 10957.70it/s] 40%|███▉      | 159762/400000 [00:17<00:22, 10868.28it/s] 40%|████      | 160856/400000 [00:17<00:21, 10888.68it/s] 40%|████      | 161980/400000 [00:17<00:21, 10989.00it/s] 41%|████      | 163080/400000 [00:18<00:21, 10855.09it/s] 41%|████      | 164167/400000 [00:18<00:21, 10780.43it/s] 41%|████▏     | 165246/400000 [00:18<00:21, 10726.78it/s] 42%|████▏     | 166320/400000 [00:18<00:22, 10598.94it/s] 42%|████▏     | 167381/400000 [00:18<00:21, 10584.97it/s] 42%|████▏     | 168489/400000 [00:18<00:21, 10725.99it/s] 42%|████▏     | 169650/400000 [00:18<00:20, 10975.47it/s] 43%|████▎     | 170750/400000 [00:18<00:21, 10871.65it/s] 43%|████▎     | 171839/400000 [00:18<00:21, 10534.66it/s] 43%|████▎     | 172930/400000 [00:18<00:21, 10643.03it/s] 44%|████▎     | 174010/400000 [00:19<00:21, 10686.74it/s] 44%|████▍     | 175081/400000 [00:19<00:21, 10463.10it/s] 44%|████▍     | 176130/400000 [00:19<00:22, 9924.90it/s]  44%|████▍     | 177183/400000 [00:19<00:22, 10096.99it/s] 45%|████▍     | 178244/400000 [00:19<00:21, 10245.56it/s] 45%|████▍     | 179289/400000 [00:19<00:21, 10304.82it/s] 45%|████▌     | 180323/400000 [00:19<00:21, 10067.09it/s] 45%|████▌     | 181334/400000 [00:19<00:21, 9969.50it/s]  46%|████▌     | 182347/400000 [00:19<00:21, 10015.89it/s] 46%|████▌     | 183351/400000 [00:19<00:21, 9910.11it/s]  46%|████▌     | 184412/400000 [00:20<00:21, 10108.83it/s] 46%|████▋     | 185469/400000 [00:20<00:20, 10242.30it/s] 47%|████▋     | 186518/400000 [00:20<00:20, 10315.22it/s] 47%|████▋     | 187638/400000 [00:20<00:20, 10563.44it/s] 47%|████▋     | 188697/400000 [00:20<00:20, 10503.66it/s] 47%|████▋     | 189750/400000 [00:20<00:20, 10264.87it/s] 48%|████▊     | 190803/400000 [00:20<00:20, 10340.65it/s] 48%|████▊     | 191842/400000 [00:20<00:20, 10354.13it/s] 48%|████▊     | 192966/400000 [00:20<00:19, 10603.73it/s] 49%|████▊     | 194029/400000 [00:21<00:19, 10487.18it/s] 49%|████▉     | 195151/400000 [00:21<00:19, 10695.06it/s] 49%|████▉     | 196235/400000 [00:21<00:18, 10737.57it/s] 49%|████▉     | 197311/400000 [00:21<00:19, 10628.12it/s] 50%|████▉     | 198402/400000 [00:21<00:18, 10710.80it/s] 50%|████▉     | 199475/400000 [00:21<00:18, 10667.49it/s] 50%|█████     | 200543/400000 [00:21<00:18, 10666.58it/s] 50%|█████     | 201611/400000 [00:21<00:18, 10481.50it/s] 51%|█████     | 202749/400000 [00:21<00:18, 10735.09it/s] 51%|█████     | 203843/400000 [00:21<00:18, 10794.75it/s] 51%|█████     | 204925/400000 [00:22<00:18, 10554.88it/s] 51%|█████▏    | 205983/400000 [00:22<00:18, 10483.41it/s] 52%|█████▏    | 207034/400000 [00:22<00:18, 10287.75it/s] 52%|█████▏    | 208101/400000 [00:22<00:18, 10396.50it/s] 52%|█████▏    | 209143/400000 [00:22<00:18, 10080.38it/s] 53%|█████▎    | 210171/400000 [00:22<00:18, 10138.46it/s] 53%|█████▎    | 211189/400000 [00:22<00:18, 10148.77it/s] 53%|█████▎    | 212206/400000 [00:22<00:19, 9826.14it/s]  53%|█████▎    | 213193/400000 [00:22<00:19, 9690.17it/s] 54%|█████▎    | 214179/400000 [00:22<00:19, 9738.84it/s] 54%|█████▍    | 215179/400000 [00:23<00:18, 9813.01it/s] 54%|█████▍    | 216231/400000 [00:23<00:18, 10013.96it/s] 54%|█████▍    | 217235/400000 [00:23<00:18, 9951.00it/s]  55%|█████▍    | 218232/400000 [00:23<00:18, 9929.18it/s] 55%|█████▍    | 219227/400000 [00:23<00:18, 9750.03it/s] 55%|█████▌    | 220247/400000 [00:23<00:18, 9877.93it/s] 55%|█████▌    | 221237/400000 [00:23<00:18, 9861.12it/s] 56%|█████▌    | 222252/400000 [00:23<00:17, 9945.27it/s] 56%|█████▌    | 223325/400000 [00:23<00:17, 10167.61it/s] 56%|█████▌    | 224413/400000 [00:23<00:16, 10371.03it/s] 56%|█████▋    | 225538/400000 [00:24<00:16, 10619.52it/s] 57%|█████▋    | 226666/400000 [00:24<00:16, 10808.06it/s] 57%|█████▋    | 227755/400000 [00:24<00:15, 10830.48it/s] 57%|█████▋    | 228872/400000 [00:24<00:15, 10928.82it/s] 58%|█████▊    | 230016/400000 [00:24<00:15, 11075.40it/s] 58%|█████▊    | 231155/400000 [00:24<00:15, 11165.58it/s] 58%|█████▊    | 232273/400000 [00:24<00:15, 11002.42it/s] 58%|█████▊    | 233388/400000 [00:24<00:15, 11044.72it/s] 59%|█████▊    | 234518/400000 [00:24<00:14, 11118.47it/s] 59%|█████▉    | 235631/400000 [00:24<00:14, 11026.40it/s] 59%|█████▉    | 236735/400000 [00:25<00:15, 10863.46it/s] 59%|█████▉    | 237828/400000 [00:25<00:14, 10883.02it/s] 60%|█████▉    | 238937/400000 [00:25<00:14, 10942.67it/s] 60%|██████    | 240055/400000 [00:25<00:14, 11010.39it/s] 60%|██████    | 241177/400000 [00:25<00:14, 11071.86it/s] 61%|██████    | 242285/400000 [00:25<00:14, 10852.49it/s] 61%|██████    | 243372/400000 [00:25<00:14, 10583.77it/s] 61%|██████    | 244433/400000 [00:25<00:14, 10499.19it/s] 61%|██████▏   | 245543/400000 [00:25<00:14, 10670.65it/s] 62%|██████▏   | 246680/400000 [00:26<00:14, 10870.54it/s] 62%|██████▏   | 247770/400000 [00:26<00:14, 10620.64it/s] 62%|██████▏   | 248895/400000 [00:26<00:13, 10800.31it/s] 62%|██████▏   | 249994/400000 [00:26<00:13, 10855.81it/s] 63%|██████▎   | 251082/400000 [00:26<00:14, 10571.85it/s] 63%|██████▎   | 252143/400000 [00:26<00:14, 10142.25it/s] 63%|██████▎   | 253164/400000 [00:26<00:14, 9867.76it/s]  64%|██████▎   | 254234/400000 [00:26<00:14, 10102.66it/s] 64%|██████▍   | 255280/400000 [00:26<00:14, 10206.05it/s] 64%|██████▍   | 256416/400000 [00:26<00:13, 10525.08it/s] 64%|██████▍   | 257528/400000 [00:27<00:13, 10696.23it/s] 65%|██████▍   | 258604/400000 [00:27<00:13, 10713.07it/s] 65%|██████▍   | 259697/400000 [00:27<00:13, 10776.39it/s] 65%|██████▌   | 260788/400000 [00:27<00:12, 10815.16it/s] 65%|██████▌   | 261872/400000 [00:27<00:12, 10782.65it/s] 66%|██████▌   | 262996/400000 [00:27<00:12, 10912.96it/s] 66%|██████▌   | 264089/400000 [00:27<00:12, 10894.75it/s] 66%|██████▋   | 265216/400000 [00:27<00:12, 11002.83it/s] 67%|██████▋   | 266318/400000 [00:27<00:12, 10870.31it/s] 67%|██████▋   | 267486/400000 [00:27<00:11, 11099.33it/s] 67%|██████▋   | 268605/400000 [00:28<00:11, 11124.33it/s] 67%|██████▋   | 269788/400000 [00:28<00:11, 11326.10it/s] 68%|██████▊   | 270941/400000 [00:28<00:11, 11382.73it/s] 68%|██████▊   | 272081/400000 [00:28<00:11, 11170.07it/s] 68%|██████▊   | 273200/400000 [00:28<00:11, 10600.12it/s] 69%|██████▊   | 274268/400000 [00:28<00:11, 10523.31it/s] 69%|██████▉   | 275406/400000 [00:28<00:11, 10764.49it/s] 69%|██████▉   | 276505/400000 [00:28<00:11, 10829.95it/s] 69%|██████▉   | 277592/400000 [00:28<00:11, 10817.11it/s] 70%|██████▉   | 278706/400000 [00:28<00:11, 10909.68it/s] 70%|██████▉   | 279867/400000 [00:29<00:10, 11108.90it/s] 70%|███████   | 280989/400000 [00:29<00:10, 11141.08it/s] 71%|███████   | 282124/400000 [00:29<00:10, 11199.74it/s] 71%|███████   | 283246/400000 [00:29<00:10, 11130.85it/s] 71%|███████   | 284391/400000 [00:29<00:10, 11224.63it/s] 71%|███████▏  | 285515/400000 [00:29<00:10, 11192.12it/s] 72%|███████▏  | 286635/400000 [00:29<00:10, 11128.30it/s] 72%|███████▏  | 287749/400000 [00:29<00:10, 11075.38it/s] 72%|███████▏  | 288857/400000 [00:29<00:10, 11052.90it/s] 72%|███████▏  | 289981/400000 [00:29<00:09, 11107.46it/s] 73%|███████▎  | 291093/400000 [00:30<00:09, 11031.90it/s] 73%|███████▎  | 292197/400000 [00:30<00:09, 10895.93it/s] 73%|███████▎  | 293288/400000 [00:30<00:09, 10812.66it/s] 74%|███████▎  | 294370/400000 [00:30<00:09, 10745.12it/s] 74%|███████▍  | 295512/400000 [00:30<00:09, 10935.18it/s] 74%|███████▍  | 296635/400000 [00:30<00:09, 11019.95it/s] 74%|███████▍  | 297738/400000 [00:30<00:09, 11005.78it/s] 75%|███████▍  | 298853/400000 [00:30<00:09, 11048.51it/s] 75%|███████▍  | 299959/400000 [00:30<00:09, 10974.10it/s] 75%|███████▌  | 301096/400000 [00:31<00:08, 11089.82it/s] 76%|███████▌  | 302228/400000 [00:31<00:08, 11157.75it/s] 76%|███████▌  | 303389/400000 [00:31<00:08, 11287.16it/s] 76%|███████▌  | 304519/400000 [00:31<00:08, 10948.54it/s] 76%|███████▋  | 305617/400000 [00:31<00:08, 10706.20it/s] 77%|███████▋  | 306691/400000 [00:31<00:08, 10587.60it/s] 77%|███████▋  | 307753/400000 [00:31<00:08, 10596.55it/s] 77%|███████▋  | 308867/400000 [00:31<00:08, 10752.31it/s] 77%|███████▋  | 309944/400000 [00:31<00:08, 10641.48it/s] 78%|███████▊  | 311010/400000 [00:31<00:08, 10616.35it/s] 78%|███████▊  | 312110/400000 [00:32<00:08, 10728.29it/s] 78%|███████▊  | 313245/400000 [00:32<00:07, 10906.29it/s] 79%|███████▊  | 314418/400000 [00:32<00:07, 11138.23it/s] 79%|███████▉  | 315572/400000 [00:32<00:07, 11254.83it/s] 79%|███████▉  | 316700/400000 [00:32<00:07, 11194.48it/s] 79%|███████▉  | 317821/400000 [00:32<00:07, 11153.62it/s] 80%|███████▉  | 318938/400000 [00:32<00:07, 11153.54it/s] 80%|████████  | 320080/400000 [00:32<00:07, 11231.05it/s] 80%|████████  | 321204/400000 [00:32<00:07, 11164.66it/s] 81%|████████  | 322321/400000 [00:32<00:07, 10982.76it/s] 81%|████████  | 323447/400000 [00:33<00:06, 11063.14it/s] 81%|████████  | 324565/400000 [00:33<00:06, 11096.31it/s] 81%|████████▏ | 325676/400000 [00:33<00:06, 10891.63it/s] 82%|████████▏ | 326767/400000 [00:33<00:06, 10760.33it/s] 82%|████████▏ | 327856/400000 [00:33<00:06, 10797.95it/s] 82%|████████▏ | 328937/400000 [00:33<00:06, 10636.02it/s] 83%|████████▎ | 330063/400000 [00:33<00:06, 10815.56it/s] 83%|████████▎ | 331178/400000 [00:33<00:06, 10911.16it/s] 83%|████████▎ | 332331/400000 [00:33<00:06, 11089.28it/s] 83%|████████▎ | 333442/400000 [00:33<00:06, 10699.64it/s] 84%|████████▎ | 334549/400000 [00:34<00:06, 10808.06it/s] 84%|████████▍ | 335633/400000 [00:34<00:05, 10811.31it/s] 84%|████████▍ | 336763/400000 [00:34<00:05, 10952.79it/s] 84%|████████▍ | 337930/400000 [00:34<00:05, 11156.10it/s] 85%|████████▍ | 339048/400000 [00:34<00:05, 11045.58it/s] 85%|████████▌ | 340155/400000 [00:34<00:05, 10772.33it/s] 85%|████████▌ | 341236/400000 [00:34<00:05, 10762.85it/s] 86%|████████▌ | 342368/400000 [00:34<00:05, 10922.16it/s] 86%|████████▌ | 343512/400000 [00:34<00:05, 11069.75it/s] 86%|████████▌ | 344627/400000 [00:34<00:04, 11092.36it/s] 86%|████████▋ | 345780/400000 [00:35<00:04, 11218.99it/s] 87%|████████▋ | 346904/400000 [00:35<00:04, 10933.87it/s] 87%|████████▋ | 348000/400000 [00:35<00:04, 10909.41it/s] 87%|████████▋ | 349149/400000 [00:35<00:04, 11075.89it/s] 88%|████████▊ | 350259/400000 [00:35<00:04, 11066.54it/s] 88%|████████▊ | 351386/400000 [00:35<00:04, 11123.18it/s] 88%|████████▊ | 352500/400000 [00:35<00:04, 10921.71it/s] 88%|████████▊ | 353594/400000 [00:35<00:04, 10730.47it/s] 89%|████████▊ | 354756/400000 [00:35<00:04, 10979.37it/s] 89%|████████▉ | 355857/400000 [00:36<00:04, 10853.21it/s] 89%|████████▉ | 357001/400000 [00:36<00:03, 11021.54it/s] 90%|████████▉ | 358162/400000 [00:36<00:03, 11189.24it/s] 90%|████████▉ | 359284/400000 [00:36<00:03, 11075.06it/s] 90%|█████████ | 360394/400000 [00:36<00:03, 10897.91it/s] 90%|█████████ | 361486/400000 [00:36<00:03, 10763.76it/s] 91%|█████████ | 362663/400000 [00:36<00:03, 11046.47it/s] 91%|█████████ | 363780/400000 [00:36<00:03, 11082.87it/s] 91%|█████████ | 364891/400000 [00:36<00:03, 10968.55it/s] 92%|█████████▏| 366010/400000 [00:36<00:03, 11032.00it/s] 92%|█████████▏| 367115/400000 [00:37<00:03, 10882.98it/s] 92%|█████████▏| 368249/400000 [00:37<00:02, 11015.78it/s] 92%|█████████▏| 369374/400000 [00:37<00:02, 11084.07it/s] 93%|█████████▎| 370501/400000 [00:37<00:02, 11138.73it/s] 93%|█████████▎| 371640/400000 [00:37<00:02, 11210.56it/s] 93%|█████████▎| 372762/400000 [00:37<00:02, 11086.55it/s] 93%|█████████▎| 373872/400000 [00:37<00:02, 10991.43it/s] 94%|█████████▎| 374985/400000 [00:37<00:02, 11031.41it/s] 94%|█████████▍| 376089/400000 [00:37<00:02, 10999.35it/s] 94%|█████████▍| 377204/400000 [00:37<00:02, 11043.38it/s] 95%|█████████▍| 378309/400000 [00:38<00:01, 11032.39it/s] 95%|█████████▍| 379447/400000 [00:38<00:01, 11133.01it/s] 95%|█████████▌| 380589/400000 [00:38<00:01, 11214.59it/s] 95%|█████████▌| 381711/400000 [00:38<00:01, 11167.36it/s] 96%|█████████▌| 382829/400000 [00:38<00:01, 11029.30it/s] 96%|█████████▌| 383933/400000 [00:38<00:01, 10703.31it/s] 96%|█████████▋| 385006/400000 [00:38<00:01, 10555.97it/s] 97%|█████████▋| 386140/400000 [00:38<00:01, 10779.47it/s] 97%|█████████▋| 387246/400000 [00:38<00:01, 10860.60it/s] 97%|█████████▋| 388364/400000 [00:38<00:01, 10952.52it/s] 97%|█████████▋| 389461/400000 [00:39<00:00, 10795.27it/s] 98%|█████████▊| 390548/400000 [00:39<00:00, 10815.81it/s] 98%|█████████▊| 391632/400000 [00:39<00:00, 10820.85it/s] 98%|█████████▊| 392715/400000 [00:39<00:00, 10811.61it/s] 98%|█████████▊| 393809/400000 [00:39<00:00, 10849.72it/s] 99%|█████████▊| 394924/400000 [00:39<00:00, 10935.85it/s] 99%|█████████▉| 396040/400000 [00:39<00:00, 11001.39it/s] 99%|█████████▉| 397170/400000 [00:39<00:00, 11086.51it/s]100%|█████████▉| 398287/400000 [00:39<00:00, 11110.54it/s]100%|█████████▉| 399399/400000 [00:39<00:00, 10913.00it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9989.39it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc4bf163860>, <torchtext.data.dataset.TabularDataset object at 0x7fc4bf1639b0>, <torchtext.vocab.Vocab object at 0x7fc4bf1638d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.29 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.42 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 15.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.95 file/s]2020-06-13 13:48:30.835376: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-13 13:48:30.839447: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095105000 Hz
2020-06-13 13:48:30.839599: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557b05ff1cd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 13:48:30.839612: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3620864/9912422 [00:00<00:00, 35359730.75it/s]9920512it [00:00, 38980596.81it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1110994.87it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157473.70it/s]1654784it [00:00, 11131533.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207151.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
