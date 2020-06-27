
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5e74118ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5e74118ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5edf3d81e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5edf3d81e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5efd720e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5efd720e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e8c700488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e8c700488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e8c700488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 463366.92it/s] 65%|██████▍   | 6438912/9912422 [00:00<00:05, 659901.24it/s]9920512it [00:00, 41756710.30it/s]                           
0it [00:00, ?it/s]32768it [00:00, 645335.11it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1019113.89it/s]1654784it [00:00, 12635246.31it/s]                           
0it [00:00, ?it/s]8192it [00:00, 203411.96it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e89bbae10>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e89bc0630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5e8c7000d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5e8c7000d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5e8c7000d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:09,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:22,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:22,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 11.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 20.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 30.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 39.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 48.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 50.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 51.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 51.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 54.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 54.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 54.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 54.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 54.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 54.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 55.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 55.50 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 55.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 55.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.17s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 55.18 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.68s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 55.18 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.68s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.68s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.52 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.68s/ url]
0 examples [00:00, ? examples/s]2020-06-27 18:09:13.758705: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-27 18:09:13.782346: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-27 18:09:13.783038: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b6e8c7a0c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-27 18:09:13.783059: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
10 examples [00:00, 99.62 examples/s]102 examples [00:00, 136.00 examples/s]193 examples [00:00, 182.51 examples/s]286 examples [00:00, 240.50 examples/s]380 examples [00:00, 309.37 examples/s]474 examples [00:00, 387.21 examples/s]566 examples [00:00, 467.96 examples/s]656 examples [00:00, 546.48 examples/s]743 examples [00:00, 614.29 examples/s]838 examples [00:01, 686.98 examples/s]929 examples [00:01, 741.24 examples/s]1026 examples [00:01, 795.98 examples/s]1117 examples [00:01, 817.83 examples/s]1208 examples [00:01, 841.58 examples/s]1303 examples [00:01, 870.53 examples/s]1399 examples [00:01, 893.20 examples/s]1495 examples [00:01, 911.05 examples/s]1591 examples [00:01, 924.64 examples/s]1686 examples [00:01, 925.19 examples/s]1782 examples [00:02, 933.51 examples/s]1877 examples [00:02, 934.03 examples/s]1973 examples [00:02, 940.75 examples/s]2068 examples [00:02, 937.03 examples/s]2163 examples [00:02, 938.12 examples/s]2258 examples [00:02, 908.27 examples/s]2356 examples [00:02, 927.44 examples/s]2453 examples [00:02, 937.98 examples/s]2551 examples [00:02, 950.12 examples/s]2647 examples [00:02, 921.29 examples/s]2740 examples [00:03, 879.20 examples/s]2832 examples [00:03, 889.28 examples/s]2928 examples [00:03, 909.09 examples/s]3020 examples [00:03, 874.06 examples/s]3113 examples [00:03, 887.57 examples/s]3204 examples [00:03, 892.58 examples/s]3298 examples [00:03, 905.33 examples/s]3389 examples [00:03, 900.26 examples/s]3480 examples [00:03, 883.92 examples/s]3574 examples [00:03, 897.34 examples/s]3665 examples [00:04, 898.31 examples/s]3755 examples [00:04, 870.50 examples/s]3849 examples [00:04, 888.11 examples/s]3944 examples [00:04, 903.76 examples/s]4035 examples [00:04, 901.07 examples/s]4131 examples [00:04, 916.91 examples/s]4228 examples [00:04, 930.48 examples/s]4323 examples [00:04, 934.81 examples/s]4417 examples [00:04, 931.10 examples/s]4513 examples [00:05, 937.06 examples/s]4607 examples [00:05, 932.72 examples/s]4701 examples [00:05, 932.50 examples/s]4795 examples [00:05, 929.26 examples/s]4888 examples [00:05, 922.84 examples/s]4981 examples [00:05, 922.68 examples/s]5074 examples [00:05, 870.55 examples/s]5171 examples [00:05, 895.74 examples/s]5262 examples [00:05, 898.83 examples/s]5353 examples [00:05, 883.40 examples/s]5442 examples [00:06, 850.57 examples/s]5539 examples [00:06, 882.43 examples/s]5636 examples [00:06, 905.23 examples/s]5729 examples [00:06, 910.78 examples/s]5826 examples [00:06, 925.48 examples/s]5919 examples [00:06, 896.16 examples/s]6014 examples [00:06, 909.20 examples/s]6110 examples [00:06, 922.15 examples/s]6203 examples [00:06, 919.18 examples/s]6296 examples [00:06, 912.85 examples/s]6389 examples [00:07, 915.32 examples/s]6481 examples [00:07, 915.98 examples/s]6573 examples [00:07, 908.23 examples/s]6664 examples [00:07, 888.99 examples/s]6755 examples [00:07, 894.18 examples/s]6852 examples [00:07, 913.15 examples/s]6950 examples [00:07, 930.25 examples/s]7046 examples [00:07, 936.63 examples/s]7144 examples [00:07, 948.56 examples/s]7239 examples [00:07, 946.00 examples/s]7334 examples [00:08, 930.97 examples/s]7432 examples [00:08, 942.66 examples/s]7527 examples [00:08, 939.05 examples/s]7621 examples [00:08, 927.80 examples/s]7714 examples [00:08, 916.39 examples/s]7807 examples [00:08, 919.11 examples/s]7902 examples [00:08, 927.97 examples/s]7997 examples [00:08, 932.32 examples/s]8091 examples [00:08, 900.98 examples/s]8182 examples [00:09, 898.12 examples/s]8277 examples [00:09, 911.25 examples/s]8373 examples [00:09, 923.73 examples/s]8466 examples [00:09, 921.43 examples/s]8559 examples [00:09, 922.03 examples/s]8654 examples [00:09, 929.88 examples/s]8748 examples [00:09, 931.24 examples/s]8844 examples [00:09, 937.71 examples/s]8940 examples [00:09, 941.42 examples/s]9035 examples [00:09, 919.71 examples/s]9131 examples [00:10, 928.53 examples/s]9227 examples [00:10, 937.57 examples/s]9324 examples [00:10, 947.04 examples/s]9419 examples [00:10, 929.29 examples/s]9513 examples [00:10, 921.15 examples/s]9608 examples [00:10, 928.49 examples/s]9701 examples [00:10, 926.11 examples/s]9797 examples [00:10, 935.23 examples/s]9891 examples [00:10, 933.45 examples/s]9985 examples [00:10, 935.37 examples/s]10079 examples [00:11, 900.73 examples/s]10173 examples [00:11, 911.88 examples/s]10268 examples [00:11, 920.84 examples/s]10361 examples [00:11, 916.65 examples/s]10455 examples [00:11, 923.31 examples/s]10549 examples [00:11, 925.40 examples/s]10642 examples [00:11, 914.50 examples/s]10737 examples [00:11, 924.32 examples/s]10830 examples [00:11, 917.95 examples/s]10922 examples [00:11, 911.43 examples/s]11015 examples [00:12, 915.32 examples/s]11107 examples [00:12, 911.92 examples/s]11200 examples [00:12, 915.32 examples/s]11292 examples [00:12, 906.60 examples/s]11386 examples [00:12, 915.87 examples/s]11482 examples [00:12, 928.55 examples/s]11575 examples [00:12, 919.07 examples/s]11673 examples [00:12, 935.81 examples/s]11769 examples [00:12, 941.12 examples/s]11870 examples [00:13, 958.57 examples/s]11968 examples [00:13, 962.75 examples/s]12065 examples [00:13, 956.15 examples/s]12161 examples [00:13, 956.80 examples/s]12257 examples [00:13, 920.15 examples/s]12353 examples [00:13, 931.27 examples/s]12447 examples [00:13, 917.99 examples/s]12540 examples [00:13, 919.97 examples/s]12633 examples [00:13, 908.92 examples/s]12725 examples [00:13, 890.27 examples/s]12818 examples [00:14, 900.52 examples/s]12910 examples [00:14, 903.26 examples/s]13001 examples [00:14, 904.07 examples/s]13094 examples [00:14, 911.59 examples/s]13186 examples [00:14, 911.70 examples/s]13279 examples [00:14, 914.44 examples/s]13371 examples [00:14, 910.55 examples/s]13463 examples [00:14, 897.92 examples/s]13559 examples [00:14, 915.64 examples/s]13651 examples [00:14, 913.56 examples/s]13743 examples [00:15, 911.96 examples/s]13839 examples [00:15, 924.06 examples/s]13932 examples [00:15, 923.80 examples/s]14025 examples [00:15, 919.70 examples/s]14118 examples [00:15, 892.46 examples/s]14209 examples [00:15, 895.75 examples/s]14303 examples [00:15, 907.52 examples/s]14394 examples [00:15, 905.44 examples/s]14491 examples [00:15, 923.59 examples/s]14587 examples [00:15, 931.96 examples/s]14683 examples [00:16, 937.77 examples/s]14777 examples [00:16, 926.74 examples/s]14870 examples [00:16, 923.17 examples/s]14963 examples [00:16, 916.95 examples/s]15055 examples [00:16, 910.77 examples/s]15148 examples [00:16, 915.05 examples/s]15244 examples [00:16, 926.38 examples/s]15339 examples [00:16, 931.19 examples/s]15434 examples [00:16, 934.10 examples/s]15528 examples [00:16, 916.39 examples/s]15623 examples [00:17, 924.97 examples/s]15716 examples [00:17, 889.48 examples/s]15812 examples [00:17, 908.10 examples/s]15904 examples [00:17, 903.55 examples/s]16000 examples [00:17, 917.96 examples/s]16097 examples [00:17, 931.47 examples/s]16194 examples [00:17, 941.61 examples/s]16289 examples [00:17, 933.68 examples/s]16383 examples [00:17, 930.37 examples/s]16477 examples [00:18, 925.06 examples/s]16574 examples [00:18, 937.12 examples/s]16669 examples [00:18, 938.88 examples/s]16763 examples [00:18, 935.37 examples/s]16857 examples [00:18, 927.18 examples/s]16950 examples [00:18, 924.43 examples/s]17043 examples [00:18, 897.93 examples/s]17133 examples [00:18, 896.90 examples/s]17223 examples [00:18, 878.26 examples/s]17314 examples [00:18, 885.07 examples/s]17407 examples [00:19, 895.92 examples/s]17502 examples [00:19, 908.87 examples/s]17594 examples [00:19, 905.49 examples/s]17687 examples [00:19, 911.98 examples/s]17779 examples [00:19, 902.22 examples/s]17871 examples [00:19, 906.90 examples/s]17962 examples [00:19, 898.61 examples/s]18053 examples [00:19, 900.95 examples/s]18144 examples [00:19, 903.45 examples/s]18240 examples [00:19, 917.26 examples/s]18334 examples [00:20, 923.57 examples/s]18434 examples [00:20, 944.00 examples/s]18530 examples [00:20, 947.30 examples/s]18625 examples [00:20, 947.43 examples/s]18720 examples [00:20, 941.94 examples/s]18815 examples [00:20, 935.55 examples/s]18912 examples [00:20, 945.09 examples/s]19007 examples [00:20, 939.56 examples/s]19102 examples [00:20, 913.52 examples/s]19194 examples [00:20, 882.81 examples/s]19289 examples [00:21, 901.38 examples/s]19387 examples [00:21, 921.25 examples/s]19480 examples [00:21, 918.73 examples/s]19573 examples [00:21, 916.33 examples/s]19665 examples [00:21, 912.69 examples/s]19759 examples [00:21, 920.62 examples/s]19855 examples [00:21, 930.29 examples/s]19949 examples [00:21, 930.99 examples/s]20043 examples [00:21, 859.92 examples/s]20133 examples [00:22, 871.36 examples/s]20227 examples [00:22, 889.02 examples/s]20321 examples [00:22, 901.14 examples/s]20412 examples [00:22, 901.07 examples/s]20503 examples [00:22, 883.73 examples/s]20597 examples [00:22, 897.49 examples/s]20695 examples [00:22, 918.27 examples/s]20789 examples [00:22, 924.21 examples/s]20883 examples [00:22, 927.23 examples/s]20980 examples [00:22, 937.94 examples/s]21075 examples [00:23, 941.25 examples/s]21174 examples [00:23, 952.92 examples/s]21270 examples [00:23, 951.66 examples/s]21366 examples [00:23, 948.76 examples/s]21461 examples [00:23, 946.02 examples/s]21556 examples [00:23, 944.06 examples/s]21651 examples [00:23, 945.78 examples/s]21746 examples [00:23, 924.86 examples/s]21839 examples [00:23, 890.51 examples/s]21929 examples [00:23, 873.93 examples/s]22023 examples [00:24, 891.66 examples/s]22113 examples [00:24, 890.89 examples/s]22203 examples [00:24, 872.57 examples/s]22299 examples [00:24, 894.84 examples/s]22389 examples [00:24, 874.59 examples/s]22477 examples [00:24, 875.11 examples/s]22565 examples [00:24, 863.83 examples/s]22658 examples [00:24, 881.97 examples/s]22747 examples [00:24, 879.19 examples/s]22836 examples [00:25, 868.75 examples/s]22928 examples [00:25, 883.25 examples/s]23024 examples [00:25, 903.67 examples/s]23116 examples [00:25, 908.13 examples/s]23213 examples [00:25, 923.28 examples/s]23306 examples [00:25, 904.39 examples/s]23397 examples [00:25, 901.42 examples/s]23489 examples [00:25, 904.66 examples/s]23580 examples [00:25, 871.42 examples/s]23668 examples [00:25, 873.46 examples/s]23761 examples [00:26, 888.41 examples/s]23852 examples [00:26, 892.89 examples/s]23944 examples [00:26, 899.01 examples/s]24035 examples [00:26, 898.76 examples/s]24125 examples [00:26, 893.26 examples/s]24216 examples [00:26, 896.48 examples/s]24311 examples [00:26, 909.54 examples/s]24404 examples [00:26, 914.26 examples/s]24496 examples [00:26, 915.02 examples/s]24588 examples [00:26, 878.07 examples/s]24680 examples [00:27, 888.28 examples/s]24770 examples [00:27, 883.81 examples/s]24862 examples [00:27, 893.17 examples/s]24956 examples [00:27, 904.20 examples/s]25047 examples [00:27, 883.41 examples/s]25136 examples [00:27, 873.04 examples/s]25228 examples [00:27, 885.37 examples/s]25317 examples [00:27, 866.02 examples/s]25407 examples [00:27, 875.56 examples/s]25495 examples [00:27, 862.53 examples/s]25587 examples [00:28, 877.53 examples/s]25675 examples [00:28, 868.84 examples/s]25763 examples [00:28, 867.24 examples/s]25855 examples [00:28, 881.35 examples/s]25946 examples [00:28, 888.89 examples/s]26038 examples [00:28, 897.79 examples/s]26132 examples [00:28, 909.55 examples/s]26224 examples [00:28, 901.11 examples/s]26317 examples [00:28, 909.16 examples/s]26413 examples [00:29, 922.00 examples/s]26506 examples [00:29, 924.33 examples/s]26599 examples [00:29, 895.11 examples/s]26694 examples [00:29, 910.68 examples/s]26790 examples [00:29, 923.47 examples/s]26885 examples [00:29, 929.09 examples/s]26979 examples [00:29, 932.03 examples/s]27074 examples [00:29, 935.33 examples/s]27169 examples [00:29, 937.24 examples/s]27263 examples [00:29, 935.38 examples/s]27357 examples [00:30, 927.19 examples/s]27450 examples [00:30, 911.32 examples/s]27547 examples [00:30, 926.54 examples/s]27641 examples [00:30, 927.69 examples/s]27734 examples [00:30, 893.83 examples/s]27824 examples [00:30, 874.95 examples/s]27919 examples [00:30, 894.75 examples/s]28014 examples [00:30, 908.06 examples/s]28109 examples [00:30, 917.70 examples/s]28203 examples [00:30, 923.73 examples/s]28296 examples [00:31, 893.81 examples/s]28390 examples [00:31, 905.55 examples/s]28481 examples [00:31, 906.33 examples/s]28574 examples [00:31, 912.76 examples/s]28666 examples [00:31, 897.64 examples/s]28756 examples [00:31, 892.64 examples/s]28846 examples [00:31, 880.33 examples/s]28940 examples [00:31, 895.05 examples/s]29035 examples [00:31, 908.87 examples/s]29127 examples [00:31, 889.72 examples/s]29217 examples [00:32, 887.30 examples/s]29308 examples [00:32, 893.27 examples/s]29401 examples [00:32, 901.90 examples/s]29496 examples [00:32, 915.46 examples/s]29593 examples [00:32, 931.07 examples/s]29687 examples [00:32, 911.11 examples/s]29779 examples [00:32, 872.62 examples/s]29874 examples [00:32, 893.31 examples/s]29964 examples [00:32, 887.73 examples/s]30054 examples [00:33, 857.89 examples/s]30143 examples [00:33, 866.14 examples/s]30232 examples [00:33, 872.56 examples/s]30320 examples [00:33, 863.32 examples/s]30409 examples [00:33, 871.15 examples/s]30499 examples [00:33, 877.18 examples/s]30588 examples [00:33, 880.86 examples/s]30677 examples [00:33, 860.42 examples/s]30771 examples [00:33, 882.50 examples/s]30862 examples [00:33, 885.52 examples/s]30953 examples [00:34, 891.40 examples/s]31043 examples [00:34, 879.78 examples/s]31132 examples [00:34, 882.13 examples/s]31223 examples [00:34, 887.57 examples/s]31312 examples [00:34, 887.55 examples/s]31401 examples [00:34, 880.82 examples/s]31494 examples [00:34, 893.28 examples/s]31584 examples [00:34, 893.95 examples/s]31674 examples [00:34, 890.15 examples/s]31765 examples [00:34, 895.76 examples/s]31857 examples [00:35, 900.99 examples/s]31951 examples [00:35, 910.68 examples/s]32043 examples [00:35, 895.92 examples/s]32133 examples [00:35, 894.51 examples/s]32223 examples [00:35, 893.47 examples/s]32315 examples [00:35, 901.03 examples/s]32407 examples [00:35, 904.96 examples/s]32498 examples [00:35, 896.22 examples/s]32591 examples [00:35, 905.38 examples/s]32682 examples [00:35, 902.36 examples/s]32782 examples [00:36, 927.94 examples/s]32880 examples [00:36, 939.39 examples/s]32975 examples [00:36, 941.88 examples/s]33070 examples [00:36, 923.74 examples/s]33163 examples [00:36, 904.91 examples/s]33258 examples [00:36, 917.68 examples/s]33355 examples [00:36, 931.39 examples/s]33449 examples [00:36, 932.01 examples/s]33543 examples [00:36, 895.26 examples/s]33633 examples [00:37, 879.84 examples/s]33722 examples [00:37, 880.07 examples/s]33818 examples [00:37, 901.04 examples/s]33912 examples [00:37, 909.47 examples/s]34004 examples [00:37, 898.84 examples/s]34097 examples [00:37, 905.38 examples/s]34193 examples [00:37, 918.24 examples/s]34285 examples [00:37, 902.36 examples/s]34380 examples [00:37, 915.27 examples/s]34475 examples [00:37, 923.22 examples/s]34569 examples [00:38, 926.65 examples/s]34662 examples [00:38, 917.21 examples/s]34759 examples [00:38, 932.15 examples/s]34853 examples [00:38, 930.21 examples/s]34947 examples [00:38, 932.73 examples/s]35041 examples [00:38, 931.74 examples/s]35135 examples [00:38, 933.51 examples/s]35233 examples [00:38, 944.14 examples/s]35328 examples [00:38, 920.45 examples/s]35421 examples [00:38, 902.01 examples/s]35512 examples [00:39, 897.04 examples/s]35604 examples [00:39, 902.56 examples/s]35695 examples [00:39, 885.51 examples/s]35784 examples [00:39, 877.43 examples/s]35872 examples [00:39, 870.95 examples/s]35960 examples [00:39, 853.11 examples/s]36049 examples [00:39, 862.80 examples/s]36140 examples [00:39, 873.66 examples/s]36231 examples [00:39, 883.49 examples/s]36323 examples [00:39, 891.17 examples/s]36414 examples [00:40, 895.74 examples/s]36506 examples [00:40, 900.78 examples/s]36601 examples [00:40, 914.98 examples/s]36694 examples [00:40, 919.01 examples/s]36786 examples [00:40, 904.16 examples/s]36882 examples [00:40, 918.98 examples/s]36979 examples [00:40, 932.86 examples/s]37073 examples [00:40, 926.71 examples/s]37166 examples [00:40, 924.53 examples/s]37263 examples [00:41, 937.08 examples/s]37357 examples [00:41, 915.03 examples/s]37449 examples [00:41, 905.21 examples/s]37544 examples [00:41, 916.89 examples/s]37640 examples [00:41, 928.58 examples/s]37737 examples [00:41, 937.76 examples/s]37834 examples [00:41, 946.08 examples/s]37930 examples [00:41, 949.50 examples/s]38029 examples [00:41, 958.88 examples/s]38125 examples [00:41, 947.24 examples/s]38220 examples [00:42, 923.49 examples/s]38313 examples [00:42, 921.12 examples/s]38412 examples [00:42, 939.49 examples/s]38507 examples [00:42, 940.23 examples/s]38602 examples [00:42, 940.53 examples/s]38697 examples [00:42, 894.46 examples/s]38788 examples [00:42, 897.54 examples/s]38886 examples [00:42, 919.35 examples/s]38979 examples [00:42, 907.83 examples/s]39072 examples [00:42, 913.28 examples/s]39165 examples [00:43, 917.39 examples/s]39257 examples [00:43, 911.97 examples/s]39349 examples [00:43, 908.94 examples/s]39440 examples [00:43, 883.35 examples/s]39529 examples [00:43, 867.25 examples/s]39616 examples [00:43, 866.13 examples/s]39705 examples [00:43, 872.21 examples/s]39795 examples [00:43, 879.93 examples/s]39884 examples [00:43, 870.27 examples/s]39974 examples [00:43, 877.88 examples/s]40062 examples [00:44, 835.54 examples/s]40154 examples [00:44, 844.67 examples/s]40247 examples [00:44, 868.25 examples/s]40338 examples [00:44, 879.16 examples/s]40429 examples [00:44, 887.55 examples/s]40520 examples [00:44, 892.32 examples/s]40610 examples [00:44, 890.92 examples/s]40705 examples [00:44, 905.84 examples/s]40796 examples [00:44, 907.05 examples/s]40887 examples [00:45, 905.80 examples/s]40981 examples [00:45, 914.43 examples/s]41075 examples [00:45, 921.15 examples/s]41169 examples [00:45, 925.73 examples/s]41262 examples [00:45, 898.98 examples/s]41357 examples [00:45, 912.92 examples/s]41449 examples [00:45, 863.12 examples/s]41539 examples [00:45, 871.78 examples/s]41633 examples [00:45, 891.04 examples/s]41723 examples [00:45, 891.49 examples/s]41816 examples [00:46, 901.50 examples/s]41908 examples [00:46, 906.02 examples/s]42001 examples [00:46, 911.49 examples/s]42093 examples [00:46, 901.45 examples/s]42184 examples [00:46, 899.12 examples/s]42274 examples [00:46, 897.22 examples/s]42367 examples [00:46, 904.67 examples/s]42462 examples [00:46, 917.67 examples/s]42558 examples [00:46, 928.51 examples/s]42651 examples [00:46, 914.74 examples/s]42749 examples [00:47, 932.73 examples/s]42846 examples [00:47, 941.61 examples/s]42944 examples [00:47, 951.42 examples/s]43040 examples [00:47, 951.70 examples/s]43136 examples [00:47, 931.39 examples/s]43230 examples [00:47, 929.73 examples/s]43324 examples [00:47, 928.39 examples/s]43417 examples [00:47, 907.07 examples/s]43508 examples [00:47, 900.90 examples/s]43599 examples [00:48, 874.79 examples/s]43692 examples [00:48, 888.28 examples/s]43785 examples [00:48, 897.70 examples/s]43881 examples [00:48, 913.33 examples/s]43973 examples [00:48, 913.31 examples/s]44065 examples [00:48, 890.29 examples/s]44157 examples [00:48, 896.53 examples/s]44251 examples [00:48, 908.79 examples/s]44344 examples [00:48, 912.46 examples/s]44436 examples [00:48, 910.35 examples/s]44528 examples [00:49, 886.45 examples/s]44621 examples [00:49, 897.26 examples/s]44711 examples [00:49, 887.48 examples/s]44800 examples [00:49, 888.17 examples/s]44893 examples [00:49, 899.46 examples/s]44984 examples [00:49, 896.20 examples/s]45074 examples [00:49, 865.33 examples/s]45166 examples [00:49, 878.84 examples/s]45260 examples [00:49, 896.14 examples/s]45350 examples [00:49, 894.96 examples/s]45443 examples [00:50, 904.11 examples/s]45534 examples [00:50, 896.58 examples/s]45626 examples [00:50, 902.37 examples/s]45723 examples [00:50, 920.37 examples/s]45819 examples [00:50, 930.91 examples/s]45913 examples [00:50, 930.99 examples/s]46007 examples [00:50, 930.96 examples/s]46101 examples [00:50, 927.85 examples/s]46195 examples [00:50, 929.67 examples/s]46289 examples [00:50, 900.69 examples/s]46381 examples [00:51, 904.97 examples/s]46474 examples [00:51, 911.65 examples/s]46572 examples [00:51, 929.96 examples/s]46666 examples [00:51, 922.83 examples/s]46761 examples [00:51, 928.91 examples/s]46855 examples [00:51, 929.69 examples/s]46949 examples [00:51, 931.47 examples/s]47043 examples [00:51, 932.25 examples/s]47137 examples [00:51, 905.73 examples/s]47228 examples [00:52, 887.97 examples/s]47318 examples [00:52, 886.54 examples/s]47411 examples [00:52, 897.02 examples/s]47506 examples [00:52, 912.12 examples/s]47598 examples [00:52, 911.09 examples/s]47690 examples [00:52, 892.34 examples/s]47782 examples [00:52, 900.22 examples/s]47875 examples [00:52, 907.93 examples/s]47970 examples [00:52, 919.58 examples/s]48063 examples [00:52, 922.56 examples/s]48156 examples [00:53, 898.33 examples/s]48247 examples [00:53, 891.79 examples/s]48341 examples [00:53, 905.02 examples/s]48435 examples [00:53, 914.36 examples/s]48527 examples [00:53, 911.84 examples/s]48619 examples [00:53, 881.41 examples/s]48711 examples [00:53, 892.19 examples/s]48805 examples [00:53, 904.26 examples/s]48896 examples [00:53, 905.41 examples/s]48990 examples [00:53, 914.98 examples/s]49082 examples [00:54, 902.59 examples/s]49178 examples [00:54, 917.84 examples/s]49273 examples [00:54, 926.97 examples/s]49375 examples [00:54, 950.66 examples/s]49471 examples [00:54, 940.47 examples/s]49566 examples [00:54, 938.28 examples/s]49660 examples [00:54, 928.68 examples/s]49753 examples [00:54, 923.94 examples/s]49846 examples [00:54, 898.67 examples/s]49940 examples [00:54, 909.89 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6531/50000 [00:00<00:00, 65306.64 examples/s] 37%|███▋      | 18691/50000 [00:00<00:00, 75838.50 examples/s] 63%|██████▎   | 31453/50000 [00:00<00:00, 86348.90 examples/s] 88%|████████▊ | 44097/50000 [00:00<00:00, 95425.45 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 774.58 examples/s]175 examples [00:00, 823.52 examples/s]272 examples [00:00, 861.45 examples/s]371 examples [00:00, 894.78 examples/s]462 examples [00:00, 898.42 examples/s]557 examples [00:00, 911.84 examples/s]652 examples [00:00, 922.11 examples/s]747 examples [00:00, 928.58 examples/s]839 examples [00:00, 925.90 examples/s]938 examples [00:01, 943.10 examples/s]1032 examples [00:01, 941.34 examples/s]1127 examples [00:01, 940.78 examples/s]1221 examples [00:01, 939.54 examples/s]1315 examples [00:01, 924.47 examples/s]1408 examples [00:01, 922.42 examples/s]1503 examples [00:01, 930.03 examples/s]1605 examples [00:01, 953.01 examples/s]1705 examples [00:01, 964.56 examples/s]1802 examples [00:01, 946.34 examples/s]1897 examples [00:02, 947.13 examples/s]1992 examples [00:02, 943.68 examples/s]2087 examples [00:02, 943.02 examples/s]2185 examples [00:02, 951.13 examples/s]2281 examples [00:02, 937.30 examples/s]2377 examples [00:02, 943.68 examples/s]2472 examples [00:02, 928.46 examples/s]2565 examples [00:02, 892.15 examples/s]2657 examples [00:02, 899.16 examples/s]2748 examples [00:02, 888.38 examples/s]2841 examples [00:03, 898.16 examples/s]2931 examples [00:03, 878.50 examples/s]3024 examples [00:03, 892.66 examples/s]3117 examples [00:03, 900.64 examples/s]3211 examples [00:03, 911.39 examples/s]3308 examples [00:03, 926.79 examples/s]3401 examples [00:03, 927.21 examples/s]3494 examples [00:03, 922.08 examples/s]3591 examples [00:03, 935.25 examples/s]3687 examples [00:03, 942.39 examples/s]3783 examples [00:04, 945.33 examples/s]3878 examples [00:04, 937.01 examples/s]3972 examples [00:04, 932.99 examples/s]4066 examples [00:04, 934.90 examples/s]4160 examples [00:04, 934.95 examples/s]4254 examples [00:04, 908.61 examples/s]4346 examples [00:04, 891.61 examples/s]4436 examples [00:04, 885.49 examples/s]4532 examples [00:04, 906.48 examples/s]4632 examples [00:04, 930.09 examples/s]4727 examples [00:05, 935.63 examples/s]4823 examples [00:05, 940.14 examples/s]4919 examples [00:05, 944.28 examples/s]5014 examples [00:05, 939.61 examples/s]5109 examples [00:05, 938.61 examples/s]5206 examples [00:05, 947.07 examples/s]5301 examples [00:05, 926.01 examples/s]5394 examples [00:05, 921.27 examples/s]5489 examples [00:05, 929.39 examples/s]5583 examples [00:06, 916.83 examples/s]5675 examples [00:06, 910.49 examples/s]5767 examples [00:06, 889.29 examples/s]5861 examples [00:06, 902.61 examples/s]5954 examples [00:06, 908.11 examples/s]6053 examples [00:06, 929.20 examples/s]6147 examples [00:06, 930.59 examples/s]6244 examples [00:06, 939.64 examples/s]6346 examples [00:06, 961.27 examples/s]6444 examples [00:06, 965.40 examples/s]6541 examples [00:07, 963.01 examples/s]6638 examples [00:07, 925.57 examples/s]6735 examples [00:07, 935.95 examples/s]6834 examples [00:07, 951.05 examples/s]6930 examples [00:07, 944.85 examples/s]7025 examples [00:07, 913.45 examples/s]7119 examples [00:07, 920.02 examples/s]7213 examples [00:07, 925.18 examples/s]7306 examples [00:07, 908.84 examples/s]7399 examples [00:07, 912.33 examples/s]7498 examples [00:08, 931.97 examples/s]7592 examples [00:08, 927.38 examples/s]7691 examples [00:08, 943.11 examples/s]7786 examples [00:08, 944.30 examples/s]7881 examples [00:08, 939.07 examples/s]7977 examples [00:08, 945.08 examples/s]8072 examples [00:08, 941.65 examples/s]8167 examples [00:08, 919.54 examples/s]8260 examples [00:08, 917.92 examples/s]8353 examples [00:08, 919.42 examples/s]8446 examples [00:09, 917.09 examples/s]8540 examples [00:09, 923.69 examples/s]8633 examples [00:09, 921.01 examples/s]8731 examples [00:09, 935.79 examples/s]8827 examples [00:09, 941.12 examples/s]8922 examples [00:09, 914.74 examples/s]9018 examples [00:09, 926.66 examples/s]9115 examples [00:09, 937.09 examples/s]9209 examples [00:09, 902.36 examples/s]9310 examples [00:10, 930.41 examples/s]9405 examples [00:10, 935.94 examples/s]9499 examples [00:10, 932.17 examples/s]9596 examples [00:10, 941.63 examples/s]9696 examples [00:10, 957.90 examples/s]9793 examples [00:10, 958.82 examples/s]9890 examples [00:10, 948.79 examples/s]9987 examples [00:10, 953.31 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSBM9EM/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSBM9EM/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e8c700488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e8c700488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e8c700488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e101a4fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e8992a2e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e8c700488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e8c700488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5e8c700488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e89bc0400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e89bc0128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5e100671e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5e100671e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f5e100671e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5e100672f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5e100672f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5e100672f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=aacf7e3becf82b29949d29668261692ac59f3d731d14a63cd5f8be3c5f1b7aed
  Stored in directory: /tmp/pip-ephem-wheel-cache-jl_njwmv/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:13:26, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:29:30, 14.5kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:36:00, 20.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<8:07:41, 29.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.53M/862M [00:01<5:40:32, 42.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.45M/862M [00:01<3:56:48, 60.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:45:11, 85.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:55:03, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.2M/862M [00:01<1:20:14, 174kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<55:53, 249kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:01<39:05, 354kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.4M/862M [00:02<27:15, 505kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.5M/862M [00:02<19:08, 716kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:02<13:23, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:02<09:27, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<07:14, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:57, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<07:11, 1.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:36, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<06:13, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:33, 2.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:09, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:45, 2.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:09<04:10, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<15:05, 876kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<11:55, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<08:37, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<09:06, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<09:01, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:59, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:15<06:14, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<04:41, 2.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:18, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:02, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<03:59, 3.25MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<10:46, 1.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<08:52, 1.46MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<06:31, 1.98MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<07:34, 1.70MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<06:22, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:20<04:46, 2.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<06:22, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<07:03, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<05:28, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:23<03:59, 3.20MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<08:46, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<8:04:19, 26.3kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<5:38:57, 37.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<3:58:42, 53.1kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<2:49:36, 74.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:59:08, 106kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<1:23:17, 152kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:06:33, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<47:52, 263kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<33:42, 373kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<26:29, 473kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<19:48, 633kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<14:09, 884kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:50, 972kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:30, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:39, 1.44MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:03, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:15, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:19, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<04:33, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<12:12, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:50, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<07:41, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<49:57, 246kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<36:11, 339kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<25:35, 478kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<20:45, 588kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<15:45, 773kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<11:16, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<10:46, 1.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:58, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:35, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:25, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:42:10, 17.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<8:12:30, 24.5kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<5:44:17, 34.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<4:03:05, 49.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<2:51:18, 70.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<1:59:57, 99.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:26:31, 138kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:02:57, 189kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<44:37, 267kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<33:01, 359kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<24:18, 488kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<17:13, 686kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<14:48, 796kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<12:45, 924kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:30, 1.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:31, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:10, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:16, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:24, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:50, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:16, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:50, 3.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:20, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:44, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:39, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:39, 1.73MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:58, 1.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:53, 2.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:18, 1.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:03, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<06:34, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:20, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<7:49:23, 24.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<5:28:52, 34.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<3:49:44, 49.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:44:11, 68.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<1:57:31, 96.3kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:22:44, 137kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<57:51, 195kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<46:32, 242kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<33:44, 333kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<23:49, 471kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<19:10, 583kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<15:40, 713kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:26, 976kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<08:06, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<13:25, 828kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<10:30, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:34, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:54, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:33, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:49, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:59, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:24, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:01, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:16, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:47, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:37, 3.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:04, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:44, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:28, 2.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:14, 3.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<12:43, 846kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:00, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:15, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<07:35, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:27, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:40, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:06, 2.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:36, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:50, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:52, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:14, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:50, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:05, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:37, 2.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:26, 3.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:52, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:30, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:22, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:44, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:29, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:17, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:07, 3.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:32, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:20, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:41, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:38, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:01, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:43, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:56, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:28, 2.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:23, 2.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:43, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:22, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:10, 2.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:03, 3.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:28, 1.55MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:33, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:08, 2.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:36, 2.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<6:24:09, 25.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<4:28:41, 37.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<3:09:02, 52.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<2:14:34, 73.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<1:34:33, 104kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<1:06:05, 149kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<50:15, 195kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<36:08, 271kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<25:28, 384kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<20:13, 482kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<14:46, 659kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<10:33, 919kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<09:35, 1.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<07:41, 1.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<05:36, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<06:10, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<06:15, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:47, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<03:26, 2.77MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:15, 930kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<08:10, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:54, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:20, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:21, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:50, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:29, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:11, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:42, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:52, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:34, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:46, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:30, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<04:38, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:07, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:59, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:53, 3.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<08:35, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:56, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<05:39, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:48, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:31, 2.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:14, 2.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<8:47:45, 17.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<6:10:05, 24.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<4:18:33, 34.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<3:02:22, 49.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<2:09:26, 69.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<1:30:51, 98.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<1:03:26, 141kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<49:50, 179kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<35:46, 249kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<25:09, 353kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<19:37, 451kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<15:31, 570kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<11:17, 782kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<09:17, 946kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:24, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:23, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:47, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:56, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:38, 2.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:33, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:56, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:45, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<19:27, 441kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<5:35:40, 25.6kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<3:54:44, 36.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<2:44:57, 51.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<1:57:29, 72.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<1:22:34, 103kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<57:42, 147kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<43:40, 193kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<30:35, 275kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<21:25, 391kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<8:19:33, 16.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<5:50:07, 23.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<4:04:32, 34.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<2:52:24, 48.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<2:02:19, 67.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<1:25:54, 96.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<1:01:05, 135kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<44:25, 186kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<31:28, 261kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<21:59, 372kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<1:07:51, 120kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<48:17, 169kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<33:54, 240kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<25:32, 317kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<19:30, 415kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<13:59, 578kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<09:50, 818kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<14:18, 562kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<10:49, 742kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<07:45, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:16, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:42, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:01, 1.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:36, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:40, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:27, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:00, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:38, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:01, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:00, 2.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:56, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:19, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:21, 2.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:25, 3.17MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:39, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<05:26, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:57, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:33, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:39, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:34, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:35, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:12, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:07, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:45, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:23, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:38, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:35, 2.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:36, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:39, 1.59MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:07, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:22, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:46, 2.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:30, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:12, 2.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:25, 3.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:08, 3.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<4:50:12, 24.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<3:23:19, 35.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<2:21:38, 50.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<1:42:26, 69.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<1:13:21, 97.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<51:35, 138kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<36:03, 197kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<27:50, 255kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<20:12, 350kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<14:16, 494kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<11:35, 606kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:31, 736kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:00, 999kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:00, 1.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:53, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:35, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:06, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:15, 1.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:16, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:12<02:21, 2.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:27, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:31, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:17, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:04, 1.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:08, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:17, 2.93MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:45, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:00, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:57, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:36, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:11, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:23, 2.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:13, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:54, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:11, 2.97MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:03, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:26, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:43, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:20, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:37, 2.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<01:54, 3.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:48, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:00, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:56, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:43, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:52, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:04, 2.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:05, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:11, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:03, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:34, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:44, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:55, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:00, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:19, 1.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:35, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<01:53, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:52, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:19, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:26, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:05, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:21, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:07, 2.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<3:41:31, 26.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<2:34:45, 37.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<1:48:29, 53.4kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<1:17:18, 74.9kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<54:20, 106kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<37:48, 152kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<31:03, 184kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<22:18, 256kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<15:41, 363kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<12:13, 463kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<09:41, 583kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<07:03, 799kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:48, 963kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:33, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:18, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:34, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:04, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:16, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:52, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:06, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:26, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:34, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:53, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:39, 3.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<39:11, 136kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<27:55, 190kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<19:34, 270kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<14:50, 353kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<11:24, 460kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<08:13, 635kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<05:45, 899kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<5:04:04, 17.0kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<3:33:07, 24.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<2:28:34, 34.6kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<1:44:28, 48.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<1:13:32, 69.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<51:19, 98.8kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<36:52, 137kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<26:49, 188kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<18:58, 265kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<13:57, 356kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<10:15, 484kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<07:14, 681kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<06:11, 791kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<05:19, 920kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:55, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:01, 963kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<04:00, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:54, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:08, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:41, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:59, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:29, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:12, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<01:39, 2.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:14, 2.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:58, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:29, 3.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:56, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:26, 3.13MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:19, 3.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<3:00:04, 25.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<2:06:03, 35.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<1:27:30, 50.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<1:03:10, 70.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<45:13, 97.9kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<31:49, 139kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<22:05, 198kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<20:43, 210kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<14:56, 291kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<10:29, 413kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<08:17, 517kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<06:40, 643kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:52, 878kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<04:03, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:15, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:21, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:36, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:14, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:38, 2.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:07, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:18, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:54, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:44, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:18, 3.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:49, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:04, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:37, 2.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:10, 3.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:04, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:32, 1.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:50, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:10, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<02:17, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:45, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:15, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:54, 633kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:31, 827kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:13, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:05, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:54, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:06, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:12, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:41, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:12, 2.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<04:39, 761kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:36, 979kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:34, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:36, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:31, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:54, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:21, 2.51MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:05, 830kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<03:12, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:18, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:22, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:19, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:46, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:15, 2.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:57, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:24, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:45, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:01, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:34, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:13, 2.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<1:58:27, 26.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<1:22:31, 37.8kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<57:24, 53.4kB/s]  .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<40:56, 74.9kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<28:42, 106kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<19:54, 152kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<14:59, 200kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<10:47, 277kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<07:34, 392kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<05:52, 498kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<04:41, 622kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<03:24, 853kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<02:22, 1.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:59, 318kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<06:34, 434kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<04:37, 610kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:51, 724kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<03:15, 856kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:24, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:05, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:02, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<01:05, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<10:59, 241kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<07:54, 335kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<05:32, 474kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<03:51, 670kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<07:35, 341kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<05:51, 441kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<04:12, 610kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<02:55, 862kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<20:03, 125kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<14:15, 176kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<09:56, 250kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<07:26, 329kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<05:26, 449kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:49, 631kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:12, 743kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:43, 874kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:00, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:45, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:27, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:03, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:21, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:44, 2.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:39, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:00, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:11, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:15, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:57, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:41, 2.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:37, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:20, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:58, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:08, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:11, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:55, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:56, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:49, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:36, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:57, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:45, 2.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:36, 2.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<1:05:40, 27.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<45:33, 38.5kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<31:17, 54.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<22:21, 76.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<15:40, 108kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<10:44, 154kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<08:09, 201kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<05:53, 277kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<04:06, 391kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:06, 503kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:18, 674kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:38, 937kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:26, 1.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:44, 854kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<01:11, 1.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:31, 935kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:12, 1.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:51, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:54, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:53, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:41, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:28, 2.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<1:15:08, 17.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<52:28, 24.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<36:01, 34.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<24:46, 49.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<17:32, 69.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<12:12, 98.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<08:21, 138kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<05:56, 193kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<04:05, 274kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<03:01, 358kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [05:58<02:12, 487kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:32, 683kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:16, 792kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:59, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:43, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:30, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:47, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<01:03, 894kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:51, 1.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:37, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:26, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:37, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:51, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:43, 1.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:31, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:22, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:46, 1.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:38, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:27, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:20, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:29, 1.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:44, 996kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:36, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:26, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:19, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:27, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:40, 987kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:17, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:36, 992kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:29, 1.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:21, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:15, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:21, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:32, 984kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:26, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:18, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:28, 987kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:23, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:16, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:11, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:16, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:22, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:18, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:18, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:15, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:07, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:10, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:15, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:12, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:11, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<03:55, 33.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<02:38, 46.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<01:33, 66.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:47, 95.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:28, 127kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:20, 168kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:13, 233kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:05, 330kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 467kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<42:03:41,  2.64it/s]  0%|          | 805/400000 [00:00<29:23:16,  3.77it/s]  0%|          | 1582/400000 [00:00<20:32:08,  5.39it/s]  1%|          | 2334/400000 [00:00<14:21:08,  7.70it/s]  1%|          | 3154/400000 [00:00<10:01:47, 10.99it/s]  1%|          | 3878/400000 [00:00<7:00:45, 15.69it/s]   1%|          | 4688/400000 [00:00<4:54:10, 22.40it/s]  1%|▏         | 5446/400000 [00:01<3:25:47, 31.95it/s]  2%|▏         | 6179/400000 [00:01<2:24:03, 45.56it/s]  2%|▏         | 6957/400000 [00:01<1:40:53, 64.93it/s]  2%|▏         | 7760/400000 [00:01<1:10:43, 92.43it/s]  2%|▏         | 8516/400000 [00:01<49:40, 131.36it/s]   2%|▏         | 9313/400000 [00:01<34:56, 186.33it/s]  3%|▎         | 10130/400000 [00:01<24:38, 263.61it/s]  3%|▎         | 10912/400000 [00:01<17:28, 370.97it/s]  3%|▎         | 11703/400000 [00:01<12:27, 519.51it/s]  3%|▎         | 12480/400000 [00:01<08:57, 721.31it/s]  3%|▎         | 13255/400000 [00:02<06:31, 988.87it/s]  4%|▎         | 14018/400000 [00:02<04:49, 1331.08it/s]  4%|▎         | 14760/400000 [00:02<03:38, 1761.25it/s]  4%|▍         | 15544/400000 [00:02<02:47, 2295.10it/s]  4%|▍         | 16309/400000 [00:02<02:12, 2905.00it/s]  4%|▍         | 17112/400000 [00:02<01:46, 3592.62it/s]  4%|▍         | 17896/400000 [00:02<01:29, 4289.43it/s]  5%|▍         | 18707/400000 [00:02<01:16, 4994.52it/s]  5%|▍         | 19492/400000 [00:02<01:08, 5548.68it/s]  5%|▌         | 20319/400000 [00:03<01:01, 6155.60it/s]  5%|▌         | 21109/400000 [00:03<00:58, 6486.23it/s]  5%|▌         | 21905/400000 [00:03<00:55, 6866.45it/s]  6%|▌         | 22687/400000 [00:03<00:53, 7112.44it/s]  6%|▌         | 23467/400000 [00:03<00:51, 7289.94it/s]  6%|▌         | 24268/400000 [00:03<00:50, 7491.66it/s]  6%|▋         | 25053/400000 [00:03<00:50, 7491.45it/s]  6%|▋         | 25828/400000 [00:03<00:49, 7506.23it/s]  7%|▋         | 26628/400000 [00:03<00:48, 7646.22it/s]  7%|▋         | 27406/400000 [00:03<00:48, 7613.10it/s]  7%|▋         | 28177/400000 [00:04<00:48, 7636.90it/s]  7%|▋         | 28948/400000 [00:04<00:48, 7652.87it/s]  7%|▋         | 29718/400000 [00:04<00:48, 7594.53it/s]  8%|▊         | 30536/400000 [00:04<00:47, 7760.12it/s]  8%|▊         | 31316/400000 [00:04<00:47, 7755.60it/s]  8%|▊         | 32094/400000 [00:04<00:47, 7757.13it/s]  8%|▊         | 32898/400000 [00:04<00:46, 7838.93it/s]  8%|▊         | 33703/400000 [00:04<00:46, 7897.97it/s]  9%|▊         | 34508/400000 [00:04<00:46, 7942.07it/s]  9%|▉         | 35303/400000 [00:04<00:46, 7897.36it/s]  9%|▉         | 36114/400000 [00:05<00:45, 7958.20it/s]  9%|▉         | 36911/400000 [00:05<00:46, 7878.99it/s]  9%|▉         | 37700/400000 [00:05<00:48, 7497.98it/s] 10%|▉         | 38496/400000 [00:05<00:47, 7630.46it/s] 10%|▉         | 39287/400000 [00:05<00:46, 7709.05it/s] 10%|█         | 40097/400000 [00:05<00:46, 7819.88it/s] 10%|█         | 40897/400000 [00:05<00:45, 7872.36it/s] 10%|█         | 41711/400000 [00:05<00:45, 7949.64it/s] 11%|█         | 42531/400000 [00:05<00:44, 8021.15it/s] 11%|█         | 43383/400000 [00:05<00:43, 8163.41it/s] 11%|█         | 44245/400000 [00:06<00:42, 8294.64it/s] 11%|█▏        | 45076/400000 [00:06<00:43, 8089.24it/s] 11%|█▏        | 45888/400000 [00:06<00:44, 7906.15it/s] 12%|█▏        | 46682/400000 [00:06<00:45, 7810.52it/s] 12%|█▏        | 47466/400000 [00:06<00:45, 7770.29it/s] 12%|█▏        | 48280/400000 [00:06<00:44, 7875.06it/s] 12%|█▏        | 49107/400000 [00:06<00:43, 7987.51it/s] 12%|█▏        | 49911/400000 [00:06<00:43, 8000.45it/s] 13%|█▎        | 50712/400000 [00:06<00:45, 7622.74it/s] 13%|█▎        | 51479/400000 [00:07<00:46, 7512.69it/s] 13%|█▎        | 52289/400000 [00:07<00:45, 7679.42it/s] 13%|█▎        | 53061/400000 [00:07<00:45, 7657.68it/s] 13%|█▎        | 53830/400000 [00:07<00:46, 7473.75it/s] 14%|█▎        | 54623/400000 [00:07<00:45, 7602.21it/s] 14%|█▍        | 55413/400000 [00:07<00:44, 7688.33it/s] 14%|█▍        | 56214/400000 [00:07<00:44, 7779.60it/s] 14%|█▍        | 56994/400000 [00:07<00:44, 7693.97it/s] 14%|█▍        | 57766/400000 [00:07<00:44, 7696.32it/s] 15%|█▍        | 58537/400000 [00:07<00:44, 7665.16it/s] 15%|█▍        | 59307/400000 [00:08<00:44, 7674.20it/s] 15%|█▌        | 60102/400000 [00:08<00:43, 7754.57it/s] 15%|█▌        | 60909/400000 [00:08<00:43, 7846.15it/s] 15%|█▌        | 61717/400000 [00:08<00:42, 7912.39it/s] 16%|█▌        | 62509/400000 [00:08<00:43, 7824.85it/s] 16%|█▌        | 63293/400000 [00:08<00:43, 7699.15it/s] 16%|█▌        | 64064/400000 [00:08<00:43, 7668.68it/s] 16%|█▌        | 64832/400000 [00:08<00:43, 7624.07it/s] 16%|█▋        | 65632/400000 [00:08<00:43, 7732.30it/s] 17%|█▋        | 66431/400000 [00:08<00:42, 7806.72it/s] 17%|█▋        | 67216/400000 [00:09<00:42, 7818.14it/s] 17%|█▋        | 68020/400000 [00:09<00:42, 7880.61it/s] 17%|█▋        | 68809/400000 [00:09<00:43, 7667.41it/s] 17%|█▋        | 69607/400000 [00:09<00:42, 7756.74it/s] 18%|█▊        | 70423/400000 [00:09<00:41, 7871.14it/s] 18%|█▊        | 71212/400000 [00:09<00:42, 7737.69it/s] 18%|█▊        | 71999/400000 [00:09<00:42, 7774.75it/s] 18%|█▊        | 72827/400000 [00:09<00:41, 7917.22it/s] 18%|█▊        | 73653/400000 [00:09<00:40, 8015.43it/s] 19%|█▊        | 74456/400000 [00:09<00:40, 7954.53it/s] 19%|█▉        | 75253/400000 [00:10<00:40, 7953.87it/s] 19%|█▉        | 76050/400000 [00:10<00:40, 7944.97it/s] 19%|█▉        | 76845/400000 [00:10<00:41, 7799.95it/s] 19%|█▉        | 77643/400000 [00:10<00:41, 7852.26it/s] 20%|█▉        | 78460/400000 [00:10<00:40, 7942.45it/s] 20%|█▉        | 79275/400000 [00:10<00:40, 8003.35it/s] 20%|██        | 80076/400000 [00:10<00:40, 7950.34it/s] 20%|██        | 80872/400000 [00:10<00:40, 7951.66it/s] 20%|██        | 81668/400000 [00:10<00:40, 7892.08it/s] 21%|██        | 82458/400000 [00:10<00:40, 7882.29it/s] 21%|██        | 83247/400000 [00:11<00:40, 7864.90it/s] 21%|██        | 84034/400000 [00:11<00:40, 7802.88it/s] 21%|██        | 84815/400000 [00:11<00:41, 7645.80it/s] 21%|██▏       | 85581/400000 [00:11<00:41, 7644.91it/s] 22%|██▏       | 86347/400000 [00:11<00:41, 7547.45it/s] 22%|██▏       | 87121/400000 [00:11<00:41, 7602.39it/s] 22%|██▏       | 87933/400000 [00:11<00:40, 7748.28it/s] 22%|██▏       | 88709/400000 [00:11<00:42, 7381.45it/s] 22%|██▏       | 89452/400000 [00:11<00:42, 7302.54it/s] 23%|██▎       | 90186/400000 [00:12<00:42, 7240.80it/s] 23%|██▎       | 90920/400000 [00:12<00:42, 7269.85it/s] 23%|██▎       | 91690/400000 [00:12<00:41, 7392.21it/s] 23%|██▎       | 92490/400000 [00:12<00:40, 7562.52it/s] 23%|██▎       | 93265/400000 [00:12<00:40, 7616.88it/s] 24%|██▎       | 94076/400000 [00:12<00:39, 7756.03it/s] 24%|██▎       | 94890/400000 [00:12<00:38, 7867.03it/s] 24%|██▍       | 95680/400000 [00:12<00:38, 7876.64it/s] 24%|██▍       | 96469/400000 [00:12<00:38, 7852.48it/s] 24%|██▍       | 97256/400000 [00:12<00:38, 7812.00it/s] 25%|██▍       | 98077/400000 [00:13<00:38, 7926.42it/s] 25%|██▍       | 98871/400000 [00:13<00:38, 7816.24it/s] 25%|██▍       | 99690/400000 [00:13<00:37, 7922.76it/s] 25%|██▌       | 100484/400000 [00:13<00:37, 7918.29it/s] 25%|██▌       | 101277/400000 [00:13<00:37, 7897.12it/s] 26%|██▌       | 102068/400000 [00:13<00:38, 7673.30it/s] 26%|██▌       | 102869/400000 [00:13<00:38, 7770.92it/s] 26%|██▌       | 103648/400000 [00:13<00:38, 7749.72it/s] 26%|██▌       | 104446/400000 [00:13<00:37, 7814.53it/s] 26%|██▋       | 105229/400000 [00:13<00:37, 7789.40it/s] 27%|██▋       | 106012/400000 [00:14<00:37, 7799.46it/s] 27%|██▋       | 106808/400000 [00:14<00:37, 7846.30it/s] 27%|██▋       | 107619/400000 [00:14<00:36, 7922.39it/s] 27%|██▋       | 108412/400000 [00:14<00:36, 7890.18it/s] 27%|██▋       | 109202/400000 [00:14<00:37, 7743.33it/s] 27%|██▋       | 109978/400000 [00:14<00:37, 7736.94it/s] 28%|██▊       | 110753/400000 [00:14<00:37, 7735.42it/s] 28%|██▊       | 111589/400000 [00:14<00:36, 7911.86it/s] 28%|██▊       | 112382/400000 [00:14<00:36, 7862.46it/s] 28%|██▊       | 113170/400000 [00:14<00:36, 7786.73it/s] 29%|██▊       | 114001/400000 [00:15<00:36, 7935.62it/s] 29%|██▊       | 114796/400000 [00:15<00:36, 7810.19it/s] 29%|██▉       | 115584/400000 [00:15<00:36, 7830.62it/s] 29%|██▉       | 116380/400000 [00:15<00:36, 7867.37it/s] 29%|██▉       | 117175/400000 [00:15<00:35, 7890.63it/s] 29%|██▉       | 117966/400000 [00:15<00:35, 7894.29it/s] 30%|██▉       | 118757/400000 [00:15<00:35, 7898.38it/s] 30%|██▉       | 119574/400000 [00:15<00:35, 7975.82it/s] 30%|███       | 120372/400000 [00:15<00:35, 7882.59it/s] 30%|███       | 121161/400000 [00:15<00:35, 7876.13it/s] 30%|███       | 121973/400000 [00:16<00:34, 7947.51it/s] 31%|███       | 122769/400000 [00:16<00:35, 7739.65it/s] 31%|███       | 123545/400000 [00:16<00:35, 7714.78it/s] 31%|███       | 124318/400000 [00:16<00:35, 7703.60it/s] 31%|███▏      | 125090/400000 [00:16<00:36, 7621.19it/s] 31%|███▏      | 125867/400000 [00:16<00:35, 7660.80it/s] 32%|███▏      | 126655/400000 [00:16<00:35, 7722.60it/s] 32%|███▏      | 127428/400000 [00:16<00:35, 7660.98it/s] 32%|███▏      | 128211/400000 [00:16<00:35, 7710.43it/s] 32%|███▏      | 129005/400000 [00:16<00:34, 7777.29it/s] 32%|███▏      | 129847/400000 [00:17<00:33, 7958.64it/s] 33%|███▎      | 130675/400000 [00:17<00:33, 8049.89it/s] 33%|███▎      | 131482/400000 [00:17<00:34, 7880.64it/s] 33%|███▎      | 132272/400000 [00:17<00:34, 7797.45it/s] 33%|███▎      | 133054/400000 [00:17<00:34, 7775.41it/s] 33%|███▎      | 133833/400000 [00:17<00:34, 7744.82it/s] 34%|███▎      | 134627/400000 [00:17<00:34, 7800.54it/s] 34%|███▍      | 135439/400000 [00:17<00:33, 7891.60it/s] 34%|███▍      | 136229/400000 [00:17<00:33, 7856.62it/s] 34%|███▍      | 137056/400000 [00:17<00:32, 7974.34it/s] 34%|███▍      | 137855/400000 [00:18<00:32, 7972.00it/s] 35%|███▍      | 138653/400000 [00:18<00:32, 7939.54it/s] 35%|███▍      | 139476/400000 [00:18<00:32, 8022.87it/s] 35%|███▌      | 140279/400000 [00:18<00:33, 7870.28it/s] 35%|███▌      | 141068/400000 [00:18<00:33, 7843.97it/s] 35%|███▌      | 141854/400000 [00:18<00:32, 7839.58it/s] 36%|███▌      | 142686/400000 [00:18<00:32, 7975.87it/s] 36%|███▌      | 143485/400000 [00:18<00:32, 7920.49it/s] 36%|███▌      | 144278/400000 [00:18<00:32, 7807.51it/s] 36%|███▋      | 145060/400000 [00:19<00:32, 7750.51it/s] 36%|███▋      | 145849/400000 [00:19<00:32, 7790.55it/s] 37%|███▋      | 146636/400000 [00:19<00:32, 7812.11it/s] 37%|███▋      | 147445/400000 [00:19<00:32, 7891.93it/s] 37%|███▋      | 148235/400000 [00:19<00:32, 7790.45it/s] 37%|███▋      | 149015/400000 [00:19<00:32, 7681.83it/s] 37%|███▋      | 149805/400000 [00:19<00:32, 7743.92it/s] 38%|███▊      | 150635/400000 [00:19<00:31, 7901.86it/s] 38%|███▊      | 151470/400000 [00:19<00:30, 8026.01it/s] 38%|███▊      | 152276/400000 [00:19<00:30, 8033.86it/s] 38%|███▊      | 153081/400000 [00:20<00:30, 7993.69it/s] 38%|███▊      | 153882/400000 [00:20<00:30, 7979.13it/s] 39%|███▊      | 154681/400000 [00:20<00:30, 7954.61it/s] 39%|███▉      | 155477/400000 [00:20<00:30, 7943.09it/s] 39%|███▉      | 156272/400000 [00:20<00:31, 7667.71it/s] 39%|███▉      | 157054/400000 [00:20<00:31, 7710.55it/s] 39%|███▉      | 157827/400000 [00:20<00:31, 7626.11it/s] 40%|███▉      | 158624/400000 [00:20<00:31, 7725.87it/s] 40%|███▉      | 159443/400000 [00:20<00:30, 7858.35it/s] 40%|████      | 160259/400000 [00:20<00:30, 7943.89it/s] 40%|████      | 161069/400000 [00:21<00:29, 7987.84it/s] 40%|████      | 161897/400000 [00:21<00:29, 8071.68it/s] 41%|████      | 162706/400000 [00:21<00:29, 8014.40it/s] 41%|████      | 163509/400000 [00:21<00:29, 8004.03it/s] 41%|████      | 164310/400000 [00:21<00:29, 7947.58it/s] 41%|████▏     | 165106/400000 [00:21<00:29, 7892.26it/s] 41%|████▏     | 165896/400000 [00:21<00:29, 7808.96it/s] 42%|████▏     | 166681/400000 [00:21<00:29, 7819.02it/s] 42%|████▏     | 167483/400000 [00:21<00:29, 7877.36it/s] 42%|████▏     | 168272/400000 [00:21<00:29, 7764.63it/s] 42%|████▏     | 169097/400000 [00:22<00:29, 7902.20it/s] 42%|████▏     | 169895/400000 [00:22<00:29, 7922.73it/s] 43%|████▎     | 170688/400000 [00:22<00:29, 7898.35it/s] 43%|████▎     | 171496/400000 [00:22<00:28, 7947.92it/s] 43%|████▎     | 172292/400000 [00:22<00:28, 7874.01it/s] 43%|████▎     | 173080/400000 [00:22<00:28, 7864.55it/s] 43%|████▎     | 173867/400000 [00:22<00:28, 7816.53it/s] 44%|████▎     | 174653/400000 [00:22<00:28, 7829.14it/s] 44%|████▍     | 175453/400000 [00:22<00:28, 7877.14it/s] 44%|████▍     | 176248/400000 [00:22<00:28, 7897.32it/s] 44%|████▍     | 177038/400000 [00:23<00:28, 7760.01it/s] 44%|████▍     | 177815/400000 [00:23<00:28, 7761.89it/s] 45%|████▍     | 178592/400000 [00:23<00:28, 7705.82it/s] 45%|████▍     | 179363/400000 [00:23<00:28, 7621.23it/s] 45%|████▌     | 180126/400000 [00:23<00:29, 7529.71it/s] 45%|████▌     | 180920/400000 [00:23<00:28, 7648.07it/s] 45%|████▌     | 181707/400000 [00:23<00:28, 7713.32it/s] 46%|████▌     | 182504/400000 [00:23<00:27, 7787.46it/s] 46%|████▌     | 183291/400000 [00:23<00:27, 7809.78it/s] 46%|████▌     | 184112/400000 [00:23<00:27, 7924.57it/s] 46%|████▌     | 184942/400000 [00:24<00:26, 8033.42it/s] 46%|████▋     | 185760/400000 [00:24<00:26, 8074.09it/s] 47%|████▋     | 186569/400000 [00:24<00:26, 8054.31it/s] 47%|████▋     | 187375/400000 [00:24<00:26, 8009.12it/s] 47%|████▋     | 188177/400000 [00:24<00:26, 7888.02it/s] 47%|████▋     | 188978/400000 [00:24<00:26, 7923.73it/s] 47%|████▋     | 189789/400000 [00:24<00:26, 7978.18it/s] 48%|████▊     | 190625/400000 [00:24<00:25, 8087.36it/s] 48%|████▊     | 191448/400000 [00:24<00:25, 8128.40it/s] 48%|████▊     | 192262/400000 [00:24<00:26, 7852.54it/s] 48%|████▊     | 193052/400000 [00:25<00:26, 7865.98it/s] 48%|████▊     | 193856/400000 [00:25<00:26, 7914.93it/s] 49%|████▊     | 194649/400000 [00:25<00:25, 7917.63it/s] 49%|████▉     | 195442/400000 [00:25<00:26, 7806.04it/s] 49%|████▉     | 196224/400000 [00:25<00:26, 7663.47it/s] 49%|████▉     | 197000/400000 [00:25<00:26, 7688.64it/s] 49%|████▉     | 197772/400000 [00:25<00:26, 7696.19it/s] 50%|████▉     | 198582/400000 [00:25<00:25, 7811.74it/s] 50%|████▉     | 199421/400000 [00:25<00:25, 7973.78it/s] 50%|█████     | 200220/400000 [00:26<00:25, 7777.98it/s] 50%|█████     | 201002/400000 [00:26<00:25, 7788.51it/s] 50%|█████     | 201783/400000 [00:26<00:25, 7779.01it/s] 51%|█████     | 202593/400000 [00:26<00:25, 7869.89it/s] 51%|█████     | 203405/400000 [00:26<00:24, 7942.60it/s] 51%|█████     | 204201/400000 [00:26<00:24, 7855.90it/s] 51%|█████     | 204988/400000 [00:26<00:24, 7831.66it/s] 51%|█████▏    | 205785/400000 [00:26<00:24, 7870.33it/s] 52%|█████▏    | 206592/400000 [00:26<00:24, 7928.82it/s] 52%|█████▏    | 207388/400000 [00:26<00:24, 7938.08it/s] 52%|█████▏    | 208193/400000 [00:27<00:24, 7968.02it/s] 52%|█████▏    | 208991/400000 [00:27<00:24, 7796.39it/s] 52%|█████▏    | 209788/400000 [00:27<00:24, 7846.57it/s] 53%|█████▎    | 210594/400000 [00:27<00:23, 7907.44it/s] 53%|█████▎    | 211443/400000 [00:27<00:23, 8071.45it/s] 53%|█████▎    | 212252/400000 [00:27<00:23, 8023.77it/s] 53%|█████▎    | 213056/400000 [00:27<00:23, 7810.88it/s] 53%|█████▎    | 213840/400000 [00:27<00:23, 7771.99it/s] 54%|█████▎    | 214627/400000 [00:27<00:23, 7798.48it/s] 54%|█████▍    | 215408/400000 [00:27<00:23, 7702.26it/s] 54%|█████▍    | 216180/400000 [00:28<00:23, 7690.71it/s] 54%|█████▍    | 216955/400000 [00:28<00:23, 7708.40it/s] 54%|█████▍    | 217727/400000 [00:28<00:23, 7610.69it/s] 55%|█████▍    | 218489/400000 [00:28<00:24, 7511.46it/s] 55%|█████▍    | 219243/400000 [00:28<00:24, 7518.09it/s] 55%|█████▍    | 219996/400000 [00:28<00:24, 7488.83it/s] 55%|█████▌    | 220827/400000 [00:28<00:23, 7717.51it/s] 55%|█████▌    | 221601/400000 [00:28<00:23, 7613.67it/s] 56%|█████▌    | 222417/400000 [00:28<00:22, 7769.24it/s] 56%|█████▌    | 223215/400000 [00:28<00:22, 7829.16it/s] 56%|█████▌    | 224006/400000 [00:29<00:22, 7851.46it/s] 56%|█████▌    | 224841/400000 [00:29<00:21, 7992.60it/s] 56%|█████▋    | 225652/400000 [00:29<00:21, 8026.73it/s] 57%|█████▋    | 226456/400000 [00:29<00:21, 7913.73it/s] 57%|█████▋    | 227256/400000 [00:29<00:21, 7938.27it/s] 57%|█████▋    | 228051/400000 [00:29<00:21, 7826.59it/s] 57%|█████▋    | 228889/400000 [00:29<00:21, 7982.50it/s] 57%|█████▋    | 229715/400000 [00:29<00:21, 8062.65it/s] 58%|█████▊    | 230523/400000 [00:29<00:21, 7871.47it/s] 58%|█████▊    | 231312/400000 [00:29<00:21, 7733.42it/s] 58%|█████▊    | 232088/400000 [00:30<00:22, 7524.89it/s] 58%|█████▊    | 232843/400000 [00:30<00:23, 7151.19it/s] 58%|█████▊    | 233606/400000 [00:30<00:22, 7284.95it/s] 59%|█████▊    | 234340/400000 [00:30<00:22, 7283.78it/s] 59%|█████▉    | 235083/400000 [00:30<00:22, 7326.16it/s] 59%|█████▉    | 235818/400000 [00:30<00:22, 7301.45it/s] 59%|█████▉    | 236593/400000 [00:30<00:21, 7429.60it/s] 59%|█████▉    | 237386/400000 [00:30<00:21, 7572.84it/s] 60%|█████▉    | 238167/400000 [00:30<00:21, 7640.16it/s] 60%|█████▉    | 238965/400000 [00:31<00:20, 7738.83it/s] 60%|█████▉    | 239766/400000 [00:31<00:20, 7816.36it/s] 60%|██████    | 240549/400000 [00:31<00:20, 7749.79it/s] 60%|██████    | 241346/400000 [00:31<00:20, 7812.97it/s] 61%|██████    | 242141/400000 [00:31<00:20, 7847.43it/s] 61%|██████    | 242927/400000 [00:31<00:20, 7673.69it/s] 61%|██████    | 243696/400000 [00:31<00:20, 7645.70it/s] 61%|██████    | 244462/400000 [00:31<00:20, 7507.52it/s] 61%|██████▏   | 245244/400000 [00:31<00:20, 7597.18it/s] 62%|██████▏   | 246005/400000 [00:31<00:20, 7568.11it/s] 62%|██████▏   | 246775/400000 [00:32<00:20, 7605.80it/s] 62%|██████▏   | 247537/400000 [00:32<00:20, 7575.00it/s] 62%|██████▏   | 248314/400000 [00:32<00:19, 7631.23it/s] 62%|██████▏   | 249078/400000 [00:32<00:19, 7587.62it/s] 62%|██████▏   | 249844/400000 [00:32<00:19, 7608.03it/s] 63%|██████▎   | 250606/400000 [00:32<00:19, 7574.44it/s] 63%|██████▎   | 251382/400000 [00:32<00:19, 7627.26it/s] 63%|██████▎   | 252166/400000 [00:32<00:19, 7689.12it/s] 63%|██████▎   | 252936/400000 [00:32<00:19, 7495.70it/s] 63%|██████▎   | 253726/400000 [00:32<00:19, 7609.88it/s] 64%|██████▎   | 254490/400000 [00:33<00:19, 7618.22it/s] 64%|██████▍   | 255258/400000 [00:33<00:18, 7635.00it/s] 64%|██████▍   | 256023/400000 [00:33<00:18, 7579.19it/s] 64%|██████▍   | 256835/400000 [00:33<00:18, 7732.81it/s] 64%|██████▍   | 257654/400000 [00:33<00:18, 7863.20it/s] 65%|██████▍   | 258452/400000 [00:33<00:17, 7895.31it/s] 65%|██████▍   | 259243/400000 [00:33<00:18, 7802.00it/s] 65%|██████▌   | 260025/400000 [00:33<00:18, 7599.32it/s] 65%|██████▌   | 260795/400000 [00:33<00:18, 7626.99it/s] 65%|██████▌   | 261569/400000 [00:33<00:18, 7658.81it/s] 66%|██████▌   | 262406/400000 [00:34<00:17, 7858.70it/s] 66%|██████▌   | 263194/400000 [00:34<00:17, 7821.08it/s] 66%|██████▌   | 263982/400000 [00:34<00:17, 7836.24it/s] 66%|██████▌   | 264767/400000 [00:34<00:17, 7753.13it/s] 66%|██████▋   | 265546/400000 [00:34<00:17, 7763.61it/s] 67%|██████▋   | 266323/400000 [00:34<00:17, 7707.10it/s] 67%|██████▋   | 267095/400000 [00:34<00:17, 7617.99it/s] 67%|██████▋   | 267861/400000 [00:34<00:17, 7629.55it/s] 67%|██████▋   | 268625/400000 [00:34<00:17, 7558.05it/s] 67%|██████▋   | 269389/400000 [00:34<00:17, 7580.65it/s] 68%|██████▊   | 270148/400000 [00:35<00:17, 7578.29it/s] 68%|██████▊   | 270907/400000 [00:35<00:17, 7543.77it/s] 68%|██████▊   | 271687/400000 [00:35<00:16, 7616.67it/s] 68%|██████▊   | 272490/400000 [00:35<00:16, 7734.96it/s] 68%|██████▊   | 273265/400000 [00:35<00:16, 7687.90it/s] 69%|██████▊   | 274039/400000 [00:35<00:16, 7703.42it/s] 69%|██████▊   | 274810/400000 [00:35<00:16, 7581.47it/s] 69%|██████▉   | 275590/400000 [00:35<00:16, 7643.25it/s] 69%|██████▉   | 276391/400000 [00:35<00:15, 7746.86it/s] 69%|██████▉   | 277180/400000 [00:35<00:15, 7787.46it/s] 69%|██████▉   | 277998/400000 [00:36<00:15, 7899.93it/s] 70%|██████▉   | 278789/400000 [00:36<00:15, 7898.40it/s] 70%|██████▉   | 279590/400000 [00:36<00:15, 7929.10it/s] 70%|███████   | 280413/400000 [00:36<00:14, 8014.93it/s] 70%|███████   | 281216/400000 [00:36<00:15, 7701.26it/s] 70%|███████   | 281990/400000 [00:36<00:15, 7646.29it/s] 71%|███████   | 282765/400000 [00:36<00:15, 7676.79it/s] 71%|███████   | 283584/400000 [00:36<00:14, 7823.19it/s] 71%|███████   | 284396/400000 [00:36<00:14, 7906.41it/s] 71%|███████▏  | 285200/400000 [00:37<00:14, 7944.30it/s] 71%|███████▏  | 285996/400000 [00:37<00:14, 7936.96it/s] 72%|███████▏  | 286791/400000 [00:37<00:14, 7856.12it/s] 72%|███████▏  | 287578/400000 [00:37<00:14, 7717.52it/s] 72%|███████▏  | 288359/400000 [00:37<00:14, 7744.12it/s] 72%|███████▏  | 289145/400000 [00:37<00:14, 7778.31it/s] 72%|███████▏  | 289924/400000 [00:37<00:14, 7765.76it/s] 73%|███████▎  | 290704/400000 [00:37<00:14, 7775.28it/s] 73%|███████▎  | 291508/400000 [00:37<00:13, 7852.73it/s] 73%|███████▎  | 292308/400000 [00:37<00:13, 7893.57it/s] 73%|███████▎  | 293126/400000 [00:38<00:13, 7975.65it/s] 73%|███████▎  | 293924/400000 [00:38<00:13, 7778.24it/s] 74%|███████▎  | 294704/400000 [00:38<00:14, 7495.10it/s] 74%|███████▍  | 295503/400000 [00:38<00:13, 7635.93it/s] 74%|███████▍  | 296290/400000 [00:38<00:13, 7704.36it/s] 74%|███████▍  | 297098/400000 [00:38<00:13, 7812.76it/s] 74%|███████▍  | 297929/400000 [00:38<00:12, 7954.71it/s] 75%|███████▍  | 298727/400000 [00:38<00:13, 7770.44it/s] 75%|███████▍  | 299524/400000 [00:38<00:12, 7827.95it/s] 75%|███████▌  | 300316/400000 [00:38<00:12, 7854.61it/s] 75%|███████▌  | 301143/400000 [00:39<00:12, 7972.55it/s] 75%|███████▌  | 301942/400000 [00:39<00:12, 7881.34it/s] 76%|███████▌  | 302732/400000 [00:39<00:12, 7776.46it/s] 76%|███████▌  | 303541/400000 [00:39<00:12, 7865.99it/s] 76%|███████▌  | 304341/400000 [00:39<00:12, 7905.02it/s] 76%|███████▋  | 305133/400000 [00:39<00:12, 7719.39it/s] 76%|███████▋  | 305945/400000 [00:39<00:12, 7832.39it/s] 77%|███████▋  | 306730/400000 [00:39<00:12, 7634.75it/s] 77%|███████▋  | 307500/400000 [00:39<00:12, 7652.34it/s] 77%|███████▋  | 308271/400000 [00:39<00:11, 7668.38it/s] 77%|███████▋  | 309055/400000 [00:40<00:11, 7717.26it/s] 77%|███████▋  | 309851/400000 [00:40<00:11, 7786.14it/s] 78%|███████▊  | 310640/400000 [00:40<00:11, 7816.67it/s] 78%|███████▊  | 311423/400000 [00:40<00:11, 7654.70it/s] 78%|███████▊  | 312211/400000 [00:40<00:11, 7717.59it/s] 78%|███████▊  | 312984/400000 [00:40<00:11, 7694.02it/s] 78%|███████▊  | 313755/400000 [00:40<00:11, 7667.10it/s] 79%|███████▊  | 314542/400000 [00:40<00:11, 7725.97it/s] 79%|███████▉  | 315336/400000 [00:40<00:10, 7786.52it/s] 79%|███████▉  | 316116/400000 [00:40<00:10, 7680.69it/s] 79%|███████▉  | 316903/400000 [00:41<00:10, 7734.36it/s] 79%|███████▉  | 317678/400000 [00:41<00:10, 7738.09it/s] 80%|███████▉  | 318453/400000 [00:41<00:10, 7671.18it/s] 80%|███████▉  | 319236/400000 [00:41<00:10, 7717.99it/s] 80%|████████  | 320009/400000 [00:41<00:10, 7528.79it/s] 80%|████████  | 320765/400000 [00:41<00:10, 7535.60it/s] 80%|████████  | 321580/400000 [00:41<00:10, 7708.43it/s] 81%|████████  | 322353/400000 [00:41<00:10, 7645.59it/s] 81%|████████  | 323146/400000 [00:41<00:09, 7726.41it/s] 81%|████████  | 323920/400000 [00:42<00:10, 7558.05it/s] 81%|████████  | 324689/400000 [00:42<00:09, 7581.36it/s] 81%|████████▏ | 325449/400000 [00:42<00:09, 7567.13it/s] 82%|████████▏ | 326225/400000 [00:42<00:09, 7621.11it/s] 82%|████████▏ | 327048/400000 [00:42<00:09, 7793.06it/s] 82%|████████▏ | 327829/400000 [00:42<00:09, 7748.15it/s] 82%|████████▏ | 328605/400000 [00:42<00:09, 7685.73it/s] 82%|████████▏ | 329408/400000 [00:42<00:09, 7782.09it/s] 83%|████████▎ | 330197/400000 [00:42<00:08, 7814.09it/s] 83%|████████▎ | 330985/400000 [00:42<00:08, 7833.66it/s] 83%|████████▎ | 331769/400000 [00:43<00:08, 7677.85it/s] 83%|████████▎ | 332538/400000 [00:43<00:08, 7563.72it/s] 83%|████████▎ | 333296/400000 [00:43<00:08, 7478.59it/s] 84%|████████▎ | 334105/400000 [00:43<00:08, 7649.49it/s] 84%|████████▎ | 334893/400000 [00:43<00:08, 7716.61it/s] 84%|████████▍ | 335720/400000 [00:43<00:08, 7873.82it/s] 84%|████████▍ | 336510/400000 [00:43<00:08, 7775.70it/s] 84%|████████▍ | 337290/400000 [00:43<00:08, 7731.46it/s] 85%|████████▍ | 338089/400000 [00:43<00:07, 7804.65it/s] 85%|████████▍ | 338871/400000 [00:43<00:07, 7694.66it/s] 85%|████████▍ | 339642/400000 [00:44<00:07, 7680.62it/s] 85%|████████▌ | 340419/400000 [00:44<00:07, 7706.12it/s] 85%|████████▌ | 341191/400000 [00:44<00:07, 7670.96it/s] 85%|████████▌ | 341959/400000 [00:44<00:07, 7563.73it/s] 86%|████████▌ | 342747/400000 [00:44<00:07, 7652.94it/s] 86%|████████▌ | 343541/400000 [00:44<00:07, 7736.33it/s] 86%|████████▌ | 344346/400000 [00:44<00:07, 7827.44it/s] 86%|████████▋ | 345130/400000 [00:44<00:07, 7782.06it/s] 86%|████████▋ | 345909/400000 [00:44<00:07, 7604.88it/s] 87%|████████▋ | 346681/400000 [00:44<00:06, 7636.26it/s] 87%|████████▋ | 347463/400000 [00:45<00:06, 7689.83it/s] 87%|████████▋ | 348246/400000 [00:45<00:06, 7725.20it/s] 87%|████████▋ | 349020/400000 [00:45<00:06, 7700.18it/s] 87%|████████▋ | 349791/400000 [00:45<00:06, 7595.11it/s] 88%|████████▊ | 350552/400000 [00:45<00:06, 7576.59it/s] 88%|████████▊ | 351349/400000 [00:45<00:06, 7689.06it/s] 88%|████████▊ | 352151/400000 [00:45<00:06, 7783.54it/s] 88%|████████▊ | 352931/400000 [00:45<00:06, 7788.36it/s] 88%|████████▊ | 353711/400000 [00:45<00:05, 7765.07it/s] 89%|████████▊ | 354488/400000 [00:45<00:06, 7573.62it/s] 89%|████████▉ | 355275/400000 [00:46<00:05, 7657.83it/s] 89%|████████▉ | 356078/400000 [00:46<00:05, 7764.94it/s] 89%|████████▉ | 356883/400000 [00:46<00:05, 7846.26it/s] 89%|████████▉ | 357669/400000 [00:46<00:05, 7765.98it/s] 90%|████████▉ | 358447/400000 [00:46<00:05, 7682.80it/s] 90%|████████▉ | 359236/400000 [00:46<00:05, 7743.15it/s] 90%|█████████ | 360063/400000 [00:46<00:05, 7892.92it/s] 90%|█████████ | 360905/400000 [00:46<00:04, 8041.91it/s] 90%|█████████ | 361711/400000 [00:46<00:04, 7980.52it/s] 91%|█████████ | 362511/400000 [00:47<00:04, 7947.63it/s] 91%|█████████ | 363343/400000 [00:47<00:04, 8055.05it/s] 91%|█████████ | 364150/400000 [00:47<00:04, 8045.90it/s] 91%|█████████ | 364956/400000 [00:47<00:04, 7987.42it/s] 91%|█████████▏| 365756/400000 [00:47<00:04, 7728.12it/s] 92%|█████████▏| 366537/400000 [00:47<00:04, 7751.20it/s] 92%|█████████▏| 367320/400000 [00:47<00:04, 7772.04it/s] 92%|█████████▏| 368099/400000 [00:47<00:04, 7708.48it/s] 92%|█████████▏| 368871/400000 [00:47<00:04, 7673.35it/s] 92%|█████████▏| 369671/400000 [00:47<00:03, 7765.99it/s] 93%|█████████▎| 370449/400000 [00:48<00:03, 7710.65it/s] 93%|█████████▎| 371221/400000 [00:48<00:03, 7646.35it/s] 93%|█████████▎| 372025/400000 [00:48<00:03, 7760.19it/s] 93%|█████████▎| 372802/400000 [00:48<00:03, 7731.77it/s] 93%|█████████▎| 373576/400000 [00:48<00:03, 7673.29it/s] 94%|█████████▎| 374382/400000 [00:48<00:03, 7784.91it/s] 94%|█████████▍| 375162/400000 [00:48<00:03, 7756.40it/s] 94%|█████████▍| 375957/400000 [00:48<00:03, 7812.08it/s] 94%|█████████▍| 376739/400000 [00:48<00:02, 7804.12it/s] 94%|█████████▍| 377534/400000 [00:48<00:02, 7845.73it/s] 95%|█████████▍| 378319/400000 [00:49<00:02, 7819.22it/s] 95%|█████████▍| 379146/400000 [00:49<00:02, 7949.00it/s] 95%|█████████▍| 379942/400000 [00:49<00:02, 7823.06it/s] 95%|█████████▌| 380726/400000 [00:49<00:02, 7805.27it/s] 95%|█████████▌| 381531/400000 [00:49<00:02, 7877.06it/s] 96%|█████████▌| 382320/400000 [00:49<00:02, 7792.43it/s] 96%|█████████▌| 383137/400000 [00:49<00:02, 7899.84it/s] 96%|█████████▌| 383928/400000 [00:49<00:02, 7823.26it/s] 96%|█████████▌| 384728/400000 [00:49<00:01, 7872.94it/s] 96%|█████████▋| 385547/400000 [00:49<00:01, 7965.28it/s] 97%|█████████▋| 386362/400000 [00:50<00:01, 8018.12it/s] 97%|█████████▋| 387189/400000 [00:50<00:01, 8090.21it/s] 97%|█████████▋| 387999/400000 [00:50<00:01, 7910.79it/s] 97%|█████████▋| 388792/400000 [00:50<00:01, 7793.25it/s] 97%|█████████▋| 389573/400000 [00:50<00:01, 7579.00it/s] 98%|█████████▊| 390351/400000 [00:50<00:01, 7635.54it/s] 98%|█████████▊| 391169/400000 [00:50<00:01, 7790.52it/s] 98%|█████████▊| 391960/400000 [00:50<00:01, 7825.10it/s] 98%|█████████▊| 392749/400000 [00:50<00:00, 7841.86it/s] 98%|█████████▊| 393535/400000 [00:50<00:00, 7815.64it/s] 99%|█████████▊| 394340/400000 [00:51<00:00, 7882.81it/s] 99%|█████████▉| 395137/400000 [00:51<00:00, 7907.60it/s] 99%|█████████▉| 395929/400000 [00:51<00:00, 7864.80it/s] 99%|█████████▉| 396716/400000 [00:51<00:00, 7747.67it/s] 99%|█████████▉| 397496/400000 [00:51<00:00, 7762.11it/s]100%|█████████▉| 398326/400000 [00:51<00:00, 7915.58it/s]100%|█████████▉| 399135/400000 [00:51<00:00, 7966.61it/s]100%|█████████▉| 399965/400000 [00:51<00:00, 8063.23it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7723.92it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5e100823c8>, <torchtext.data.dataset.TabularDataset object at 0x7f5e10082518>, <torchtext.vocab.Vocab object at 0x7f5e10082438>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.04 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.04 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.04 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.59 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.95 file/s]2020-06-27 18:18:33.275438: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-27 18:18:33.279741: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-27 18:18:33.279935: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561d825720b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-27 18:18:33.279955: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3268608/9912422 [00:00<00:00, 32260083.93it/s]9920512it [00:00, 36251207.22it/s]                             
0it [00:00, ?it/s]32768it [00:00, 710330.22it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1039290.35it/s]1654784it [00:00, 12874401.60it/s]                           
0it [00:00, ?it/s]8192it [00:00, 273953.04it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
