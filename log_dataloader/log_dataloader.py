
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff7d52790d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff7d52790d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff8405c61e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff8405c61e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff85e90ce18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff85e90ce18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff7ed8f4620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff7ed8f4620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff7ed8f4620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3842048/9912422 [00:00<00:00, 37904139.06it/s]9920512it [00:00, 35149781.22it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1233233.62it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1036714.18it/s]1654784it [00:00, 12630440.71it/s]                           
0it [00:00, ?it/s]8192it [00:00, 235974.25it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff7eace6940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff7d4928be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff7ed8f4268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff7ed8f4268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff7ed8f4268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:12,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:12,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:12,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:11,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:11,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:10,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:10,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:48,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:48,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:45,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:31,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:30,  4.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  6.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 11.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 34.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 51.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 86.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 86.22 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 86.22 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.34 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ url]
0 examples [00:00, ? examples/s]2020-07-01 18:10:51.115150: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 18:10:51.130050: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-01 18:10:51.130243: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5598add74aa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 18:10:51.130261: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
28 examples [00:00, 279.40 examples/s]131 examples [00:00, 357.49 examples/s]237 examples [00:00, 446.01 examples/s]343 examples [00:00, 539.05 examples/s]449 examples [00:00, 631.88 examples/s]556 examples [00:00, 719.90 examples/s]663 examples [00:00, 797.45 examples/s]770 examples [00:00, 861.55 examples/s]876 examples [00:00, 891.98 examples/s]981 examples [00:01, 932.65 examples/s]1088 examples [00:01, 968.96 examples/s]1194 examples [00:01, 993.00 examples/s]1300 examples [00:01, 1010.01 examples/s]1406 examples [00:01, 1022.80 examples/s]1511 examples [00:01, 1027.44 examples/s]1616 examples [00:01, 1032.16 examples/s]1722 examples [00:01, 1038.63 examples/s]1827 examples [00:01, 1019.04 examples/s]1930 examples [00:01, 1020.80 examples/s]2033 examples [00:02, 1022.55 examples/s]2140 examples [00:02, 1036.01 examples/s]2247 examples [00:02, 1045.24 examples/s]2354 examples [00:02, 1051.44 examples/s]2462 examples [00:02, 1057.45 examples/s]2568 examples [00:02, 1052.55 examples/s]2674 examples [00:02, 1008.00 examples/s]2782 examples [00:02, 1026.59 examples/s]2887 examples [00:02, 1031.20 examples/s]2992 examples [00:02, 1036.11 examples/s]3098 examples [00:03, 1041.05 examples/s]3203 examples [00:03, 1015.41 examples/s]3311 examples [00:03, 1032.77 examples/s]3415 examples [00:03, 1034.30 examples/s]3522 examples [00:03, 1044.33 examples/s]3631 examples [00:03, 1055.12 examples/s]3738 examples [00:03, 1058.95 examples/s]3847 examples [00:03, 1067.61 examples/s]3956 examples [00:03, 1073.36 examples/s]4065 examples [00:03, 1075.84 examples/s]4174 examples [00:04, 1079.19 examples/s]4283 examples [00:04, 1080.61 examples/s]4392 examples [00:04, 1076.03 examples/s]4500 examples [00:04, 1071.50 examples/s]4608 examples [00:04, 1068.95 examples/s]4715 examples [00:04, 1066.00 examples/s]4823 examples [00:04, 1068.04 examples/s]4932 examples [00:04, 1072.36 examples/s]5040 examples [00:04, 1070.50 examples/s]5148 examples [00:04, 1070.76 examples/s]5256 examples [00:05, 1068.01 examples/s]5363 examples [00:05, 1056.64 examples/s]5472 examples [00:05, 1066.04 examples/s]5580 examples [00:05, 1069.89 examples/s]5688 examples [00:05, 1066.25 examples/s]5795 examples [00:05, 1066.86 examples/s]5904 examples [00:05, 1072.73 examples/s]6012 examples [00:05, 1070.37 examples/s]6120 examples [00:05, 1061.81 examples/s]6227 examples [00:05, 1061.07 examples/s]6336 examples [00:06, 1068.92 examples/s]6443 examples [00:06, 1035.86 examples/s]6551 examples [00:06, 1046.13 examples/s]6657 examples [00:06, 1048.34 examples/s]6765 examples [00:06, 1055.86 examples/s]6871 examples [00:06, 1041.59 examples/s]6981 examples [00:06, 1057.70 examples/s]7087 examples [00:06, 1049.14 examples/s]7193 examples [00:06, 1027.75 examples/s]7301 examples [00:07, 1042.43 examples/s]7409 examples [00:07, 1050.98 examples/s]7516 examples [00:07, 1056.23 examples/s]7622 examples [00:07, 1056.90 examples/s]7729 examples [00:07, 1058.08 examples/s]7837 examples [00:07, 1062.95 examples/s]7945 examples [00:07, 1067.54 examples/s]8053 examples [00:07, 1069.17 examples/s]8161 examples [00:07, 1071.30 examples/s]8270 examples [00:07, 1076.07 examples/s]8378 examples [00:08, 1062.85 examples/s]8487 examples [00:08, 1068.74 examples/s]8596 examples [00:08, 1072.69 examples/s]8705 examples [00:08, 1076.51 examples/s]8814 examples [00:08, 1078.73 examples/s]8923 examples [00:08, 1080.47 examples/s]9032 examples [00:08, 1061.56 examples/s]9139 examples [00:08, 1034.46 examples/s]9248 examples [00:08, 1048.03 examples/s]9353 examples [00:08, 993.96 examples/s] 9461 examples [00:09, 1017.23 examples/s]9564 examples [00:09, 1019.02 examples/s]9667 examples [00:09, 1002.99 examples/s]9774 examples [00:09, 1020.19 examples/s]9879 examples [00:09, 1027.72 examples/s]9988 examples [00:09, 1044.50 examples/s]10093 examples [00:09, 974.22 examples/s]10201 examples [00:09, 1003.65 examples/s]10309 examples [00:09, 1024.45 examples/s]10415 examples [00:09, 1034.06 examples/s]10519 examples [00:10, 1030.56 examples/s]10627 examples [00:10, 1042.35 examples/s]10732 examples [00:10, 1043.71 examples/s]10839 examples [00:10, 1051.06 examples/s]10945 examples [00:10, 1048.23 examples/s]11051 examples [00:10, 1051.33 examples/s]11158 examples [00:10, 1054.57 examples/s]11266 examples [00:10, 1060.61 examples/s]11373 examples [00:10, 1054.98 examples/s]11480 examples [00:11, 1056.62 examples/s]11589 examples [00:11, 1066.37 examples/s]11696 examples [00:11, 1062.79 examples/s]11806 examples [00:11, 1072.33 examples/s]11914 examples [00:11, 1069.05 examples/s]12023 examples [00:11, 1075.18 examples/s]12132 examples [00:11, 1077.09 examples/s]12240 examples [00:11, 1054.79 examples/s]12346 examples [00:11, 1049.00 examples/s]12454 examples [00:11, 1056.92 examples/s]12560 examples [00:12, 1047.76 examples/s]12669 examples [00:12, 1057.55 examples/s]12775 examples [00:12, 1056.10 examples/s]12881 examples [00:12, 1033.64 examples/s]12989 examples [00:12, 1046.80 examples/s]13095 examples [00:12, 1050.67 examples/s]13203 examples [00:12, 1057.20 examples/s]13309 examples [00:12, 1049.87 examples/s]13417 examples [00:12, 1057.40 examples/s]13525 examples [00:12, 1061.83 examples/s]13633 examples [00:13, 1065.41 examples/s]13740 examples [00:13, 1066.20 examples/s]13847 examples [00:13, 1051.60 examples/s]13955 examples [00:13, 1057.46 examples/s]14063 examples [00:13, 1062.69 examples/s]14171 examples [00:13, 1066.94 examples/s]14278 examples [00:13, 1066.72 examples/s]14385 examples [00:13, 1058.89 examples/s]14494 examples [00:13, 1066.92 examples/s]14602 examples [00:13, 1068.95 examples/s]14709 examples [00:14, 1061.81 examples/s]14817 examples [00:14, 1065.36 examples/s]14925 examples [00:14, 1067.63 examples/s]15032 examples [00:14, 1063.28 examples/s]15141 examples [00:14, 1069.78 examples/s]15251 examples [00:14, 1076.89 examples/s]15359 examples [00:14, 1069.08 examples/s]15468 examples [00:14, 1072.45 examples/s]15578 examples [00:14, 1079.41 examples/s]15686 examples [00:14, 1079.33 examples/s]15796 examples [00:15, 1083.14 examples/s]15905 examples [00:15, 1068.91 examples/s]16012 examples [00:15, 1049.05 examples/s]16118 examples [00:15, 1022.38 examples/s]16226 examples [00:15, 1038.46 examples/s]16335 examples [00:15, 1050.80 examples/s]16443 examples [00:15, 1058.82 examples/s]16551 examples [00:15, 1062.91 examples/s]16658 examples [00:15, 1037.72 examples/s]16764 examples [00:15, 1043.46 examples/s]16872 examples [00:16, 1054.11 examples/s]16980 examples [00:16, 1061.06 examples/s]17088 examples [00:16, 1066.55 examples/s]17195 examples [00:16, 1038.84 examples/s]17303 examples [00:16, 1049.19 examples/s]17411 examples [00:16, 1057.92 examples/s]17517 examples [00:16, 1058.32 examples/s]17623 examples [00:16, 1048.08 examples/s]17731 examples [00:16, 1054.60 examples/s]17837 examples [00:17, 1051.31 examples/s]17943 examples [00:17, 1049.01 examples/s]18048 examples [00:17, 1039.87 examples/s]18153 examples [00:17, 1037.45 examples/s]18260 examples [00:17, 1045.48 examples/s]18368 examples [00:17, 1053.21 examples/s]18477 examples [00:17, 1062.86 examples/s]18586 examples [00:17, 1069.24 examples/s]18693 examples [00:17, 1062.45 examples/s]18800 examples [00:17, 1063.08 examples/s]18907 examples [00:18, 1062.87 examples/s]19014 examples [00:18, 1064.29 examples/s]19121 examples [00:18, 1061.66 examples/s]19228 examples [00:18, 1061.29 examples/s]19335 examples [00:18, 1019.91 examples/s]19443 examples [00:18, 1036.36 examples/s]19547 examples [00:18, 994.91 examples/s] 19654 examples [00:18, 1014.70 examples/s]19762 examples [00:18, 1031.53 examples/s]19871 examples [00:18, 1047.04 examples/s]19979 examples [00:19, 1055.56 examples/s]20085 examples [00:19, 1008.68 examples/s]20192 examples [00:19, 1026.06 examples/s]20299 examples [00:19, 1036.35 examples/s]20406 examples [00:19, 1045.14 examples/s]20514 examples [00:19, 1053.68 examples/s]20622 examples [00:19, 1059.60 examples/s]20729 examples [00:19, 1060.31 examples/s]20837 examples [00:19, 1065.60 examples/s]20945 examples [00:19, 1069.70 examples/s]21053 examples [00:20, 1069.18 examples/s]21160 examples [00:20, 1066.77 examples/s]21267 examples [00:20, 1060.36 examples/s]21374 examples [00:20, 1055.39 examples/s]21480 examples [00:20, 1041.20 examples/s]21588 examples [00:20, 1051.22 examples/s]21697 examples [00:20, 1062.13 examples/s]21804 examples [00:20, 1059.26 examples/s]21913 examples [00:20, 1066.22 examples/s]22020 examples [00:20, 1065.93 examples/s]22129 examples [00:21, 1072.20 examples/s]22239 examples [00:21, 1077.73 examples/s]22348 examples [00:21, 1079.02 examples/s]22456 examples [00:21, 1065.46 examples/s]22563 examples [00:21, 1040.95 examples/s]22671 examples [00:21, 1052.15 examples/s]22778 examples [00:21, 1057.24 examples/s]22884 examples [00:21, 1055.38 examples/s]22990 examples [00:21, 1033.76 examples/s]23094 examples [00:22, 1014.53 examples/s]23203 examples [00:22, 1034.76 examples/s]23312 examples [00:22, 1049.20 examples/s]23419 examples [00:22, 1053.30 examples/s]23525 examples [00:22, 1047.30 examples/s]23631 examples [00:22, 1049.63 examples/s]23740 examples [00:22, 1058.86 examples/s]23849 examples [00:22, 1066.86 examples/s]23956 examples [00:22, 1056.86 examples/s]24062 examples [00:22, 1037.90 examples/s]24166 examples [00:23, 1030.02 examples/s]24276 examples [00:23, 1048.18 examples/s]24385 examples [00:23, 1057.65 examples/s]24493 examples [00:23, 1064.12 examples/s]24600 examples [00:23, 1055.21 examples/s]24709 examples [00:23, 1064.17 examples/s]24817 examples [00:23, 1066.95 examples/s]24926 examples [00:23, 1071.19 examples/s]25034 examples [00:23, 1072.08 examples/s]25142 examples [00:23, 1057.45 examples/s]25250 examples [00:24, 1062.32 examples/s]25357 examples [00:24, 1062.85 examples/s]25464 examples [00:24, 1057.74 examples/s]25570 examples [00:24, 1041.64 examples/s]25675 examples [00:24, 1039.81 examples/s]25780 examples [00:24, 1015.17 examples/s]25884 examples [00:24, 1021.90 examples/s]25992 examples [00:24, 1036.10 examples/s]26101 examples [00:24, 1050.56 examples/s]26207 examples [00:24, 1053.12 examples/s]26313 examples [00:25, 1043.94 examples/s]26418 examples [00:25, 1024.35 examples/s]26524 examples [00:25, 1032.34 examples/s]26628 examples [00:25, 1029.40 examples/s]26732 examples [00:25, 1029.58 examples/s]26838 examples [00:25, 1037.99 examples/s]26946 examples [00:25, 1049.13 examples/s]27051 examples [00:25, 1046.43 examples/s]27158 examples [00:25, 1050.73 examples/s]27264 examples [00:25, 1042.19 examples/s]27370 examples [00:26, 1044.82 examples/s]27476 examples [00:26, 1047.36 examples/s]27582 examples [00:26, 1049.43 examples/s]27688 examples [00:26, 1052.46 examples/s]27794 examples [00:26, 1035.88 examples/s]27898 examples [00:26, 1030.24 examples/s]28002 examples [00:26, 1028.11 examples/s]28108 examples [00:26, 1037.01 examples/s]28212 examples [00:26, 1026.03 examples/s]28315 examples [00:27, 1009.23 examples/s]28420 examples [00:27, 1020.86 examples/s]28525 examples [00:27, 1026.74 examples/s]28628 examples [00:27, 1015.87 examples/s]28736 examples [00:27, 1032.85 examples/s]28842 examples [00:27, 1039.68 examples/s]28947 examples [00:27, 1009.08 examples/s]29055 examples [00:27, 1028.72 examples/s]29164 examples [00:27, 1044.30 examples/s]29274 examples [00:27, 1058.84 examples/s]29381 examples [00:28, 1055.19 examples/s]29488 examples [00:28, 1058.90 examples/s]29595 examples [00:28, 1048.71 examples/s]29700 examples [00:28, 1048.78 examples/s]29808 examples [00:28, 1056.17 examples/s]29915 examples [00:28, 1059.24 examples/s]30021 examples [00:28, 1001.80 examples/s]30131 examples [00:28, 1028.11 examples/s]30240 examples [00:28, 1044.24 examples/s]30348 examples [00:28, 1054.28 examples/s]30454 examples [00:29, 1052.72 examples/s]30561 examples [00:29, 1055.00 examples/s]30668 examples [00:29, 1059.22 examples/s]30776 examples [00:29, 1064.60 examples/s]30884 examples [00:29, 1067.64 examples/s]30991 examples [00:29, 1067.68 examples/s]31098 examples [00:29, 1028.96 examples/s]31202 examples [00:29, 1028.69 examples/s]31306 examples [00:29, 981.02 examples/s] 31407 examples [00:29, 988.86 examples/s]31507 examples [00:30, 989.53 examples/s]31607 examples [00:30, 968.05 examples/s]31712 examples [00:30, 990.45 examples/s]31812 examples [00:30, 963.54 examples/s]31916 examples [00:30, 983.35 examples/s]32019 examples [00:30, 996.82 examples/s]32120 examples [00:30, 999.33 examples/s]32221 examples [00:30, 997.81 examples/s]32328 examples [00:30, 1018.14 examples/s]32431 examples [00:31, 1010.22 examples/s]32538 examples [00:31, 1026.07 examples/s]32641 examples [00:31, 1025.33 examples/s]32746 examples [00:31, 1030.57 examples/s]32850 examples [00:31, 1032.65 examples/s]32955 examples [00:31, 1037.77 examples/s]33063 examples [00:31, 1049.33 examples/s]33171 examples [00:31, 1056.13 examples/s]33280 examples [00:31, 1064.21 examples/s]33387 examples [00:31, 1047.15 examples/s]33492 examples [00:32, 1032.50 examples/s]33599 examples [00:32, 1041.70 examples/s]33706 examples [00:32, 1047.56 examples/s]33811 examples [00:32, 1040.99 examples/s]33918 examples [00:32, 1048.37 examples/s]34023 examples [00:32, 1048.31 examples/s]34130 examples [00:32, 1053.58 examples/s]34237 examples [00:32, 1056.92 examples/s]34345 examples [00:32, 1062.78 examples/s]34453 examples [00:32, 1031.99 examples/s]34558 examples [00:33, 1036.75 examples/s]34664 examples [00:33, 1041.81 examples/s]34769 examples [00:33, 1033.57 examples/s]34873 examples [00:33, 1024.60 examples/s]34979 examples [00:33, 1034.00 examples/s]35084 examples [00:33, 1037.98 examples/s]35188 examples [00:33, 999.84 examples/s] 35293 examples [00:33, 1011.84 examples/s]35395 examples [00:33, 989.46 examples/s] 35502 examples [00:33, 1010.48 examples/s]35610 examples [00:34, 1027.89 examples/s]35717 examples [00:34, 1039.38 examples/s]35822 examples [00:34, 1034.04 examples/s]35929 examples [00:34, 1043.76 examples/s]36034 examples [00:34, 1034.53 examples/s]36140 examples [00:34, 1039.42 examples/s]36245 examples [00:34, 1038.00 examples/s]36352 examples [00:34, 1044.67 examples/s]36460 examples [00:34, 1052.61 examples/s]36566 examples [00:34, 1052.78 examples/s]36674 examples [00:35, 1059.91 examples/s]36781 examples [00:35, 1055.48 examples/s]36887 examples [00:35, 1049.87 examples/s]36993 examples [00:35, 1044.64 examples/s]37099 examples [00:35, 1046.50 examples/s]37204 examples [00:35, 1043.29 examples/s]37310 examples [00:35, 1046.12 examples/s]37415 examples [00:35, 1044.02 examples/s]37520 examples [00:35, 1030.85 examples/s]37624 examples [00:36, 1022.30 examples/s]37729 examples [00:36, 1030.43 examples/s]37838 examples [00:36, 1044.98 examples/s]37944 examples [00:36, 1048.14 examples/s]38050 examples [00:36, 1050.26 examples/s]38159 examples [00:36, 1058.96 examples/s]38267 examples [00:36, 1062.54 examples/s]38374 examples [00:36, 1054.03 examples/s]38480 examples [00:36, 1050.40 examples/s]38586 examples [00:36, 1008.51 examples/s]38690 examples [00:37, 1017.69 examples/s]38798 examples [00:37, 1035.36 examples/s]38905 examples [00:37, 1043.61 examples/s]39010 examples [00:37, 1024.09 examples/s]39117 examples [00:37, 1036.66 examples/s]39224 examples [00:37, 1043.64 examples/s]39329 examples [00:37, 1043.75 examples/s]39436 examples [00:37, 1051.18 examples/s]39544 examples [00:37, 1058.23 examples/s]39653 examples [00:37, 1065.66 examples/s]39761 examples [00:38, 1069.41 examples/s]39868 examples [00:38, 1060.26 examples/s]39977 examples [00:38, 1067.03 examples/s]40084 examples [00:38, 993.52 examples/s] 40188 examples [00:38, 1004.37 examples/s]40297 examples [00:38, 1026.25 examples/s]40404 examples [00:38, 1036.37 examples/s]40513 examples [00:38, 1049.43 examples/s]40620 examples [00:38, 1054.60 examples/s]40726 examples [00:38, 1052.75 examples/s]40832 examples [00:39, 1046.43 examples/s]40937 examples [00:39, 1043.22 examples/s]41044 examples [00:39, 1050.45 examples/s]41150 examples [00:39, 1047.73 examples/s]41255 examples [00:39, 1023.63 examples/s]41360 examples [00:39, 1030.71 examples/s]41467 examples [00:39, 1039.87 examples/s]41575 examples [00:39, 1050.85 examples/s]41682 examples [00:39, 1055.25 examples/s]41788 examples [00:40, 1020.90 examples/s]41895 examples [00:40, 1034.74 examples/s]42001 examples [00:40, 1040.66 examples/s]42108 examples [00:40, 1048.91 examples/s]42215 examples [00:40, 1053.41 examples/s]42322 examples [00:40, 1057.58 examples/s]42428 examples [00:40, 1037.93 examples/s]42535 examples [00:40, 1045.62 examples/s]42643 examples [00:40, 1054.98 examples/s]42749 examples [00:40, 1012.20 examples/s]42853 examples [00:41, 1018.74 examples/s]42959 examples [00:41, 1030.13 examples/s]43065 examples [00:41, 1037.35 examples/s]43169 examples [00:41, 1013.04 examples/s]43273 examples [00:41, 1019.54 examples/s]43383 examples [00:41, 1039.93 examples/s]43491 examples [00:41, 1050.57 examples/s]43599 examples [00:41, 1058.78 examples/s]43708 examples [00:41, 1067.04 examples/s]43815 examples [00:41, 1058.34 examples/s]43921 examples [00:42, 1039.54 examples/s]44026 examples [00:42, 1040.42 examples/s]44132 examples [00:42, 1044.40 examples/s]44237 examples [00:42, 1033.36 examples/s]44343 examples [00:42, 1041.02 examples/s]44449 examples [00:42, 1045.53 examples/s]44555 examples [00:42, 1049.35 examples/s]44662 examples [00:42, 1054.91 examples/s]44770 examples [00:42, 1062.05 examples/s]44877 examples [00:42, 1054.83 examples/s]44983 examples [00:43, 1024.88 examples/s]45089 examples [00:43, 1032.69 examples/s]45196 examples [00:43, 1040.99 examples/s]45302 examples [00:43, 1044.36 examples/s]45409 examples [00:43, 1046.03 examples/s]45517 examples [00:43, 1055.68 examples/s]45625 examples [00:43, 1060.89 examples/s]45734 examples [00:43, 1067.24 examples/s]45843 examples [00:43, 1072.08 examples/s]45951 examples [00:43, 1072.77 examples/s]46059 examples [00:44, 1071.55 examples/s]46168 examples [00:44, 1074.21 examples/s]46276 examples [00:44, 1069.97 examples/s]46384 examples [00:44, 1069.95 examples/s]46492 examples [00:44, 1067.43 examples/s]46599 examples [00:44, 1067.44 examples/s]46706 examples [00:44, 1053.37 examples/s]46812 examples [00:44, 1039.67 examples/s]46917 examples [00:44, 1035.29 examples/s]47022 examples [00:45, 1039.51 examples/s]47127 examples [00:45, 1040.22 examples/s]47233 examples [00:45, 1044.91 examples/s]47338 examples [00:45, 1042.13 examples/s]47445 examples [00:45, 1048.12 examples/s]47552 examples [00:45, 1051.78 examples/s]47658 examples [00:45, 1043.28 examples/s]47764 examples [00:45, 1047.85 examples/s]47872 examples [00:45, 1054.67 examples/s]47978 examples [00:45, 1044.31 examples/s]48085 examples [00:46, 1050.43 examples/s]48191 examples [00:46, 1021.26 examples/s]48294 examples [00:46, 1011.81 examples/s]48396 examples [00:46, 1007.93 examples/s]48499 examples [00:46, 1013.23 examples/s]48606 examples [00:46, 1028.37 examples/s]48712 examples [00:46, 1036.14 examples/s]48818 examples [00:46, 1042.11 examples/s]48926 examples [00:46, 1050.99 examples/s]49032 examples [00:46, 1039.96 examples/s]49138 examples [00:47, 1045.64 examples/s]49245 examples [00:47, 1052.32 examples/s]49351 examples [00:47, 1053.80 examples/s]49457 examples [00:47, 1052.56 examples/s]49563 examples [00:47, 1052.50 examples/s]49669 examples [00:47, 1047.88 examples/s]49777 examples [00:47, 1055.81 examples/s]49883 examples [00:47, 1033.35 examples/s]49988 examples [00:47, 1037.17 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6370/50000 [00:00<00:00, 63699.30 examples/s] 38%|███▊      | 18800/50000 [00:00<00:00, 74612.01 examples/s] 63%|██████▎   | 31733/50000 [00:00<00:00, 85458.67 examples/s] 89%|████████▉ | 44615/50000 [00:00<00:00, 95056.63 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 755.99 examples/s]185 examples [00:00, 830.87 examples/s]293 examples [00:00, 891.81 examples/s]401 examples [00:00, 939.60 examples/s]509 examples [00:00, 975.86 examples/s]619 examples [00:00, 1009.17 examples/s]727 examples [00:00, 1027.40 examples/s]825 examples [00:00, 1010.69 examples/s]933 examples [00:00, 1030.28 examples/s]1043 examples [00:01, 1049.23 examples/s]1147 examples [00:01, 1025.75 examples/s]1250 examples [00:01, 1026.62 examples/s]1358 examples [00:01, 1039.51 examples/s]1465 examples [00:01, 1048.15 examples/s]1576 examples [00:01, 1064.51 examples/s]1684 examples [00:01, 1066.42 examples/s]1791 examples [00:01, 1055.64 examples/s]1898 examples [00:01, 1058.45 examples/s]2005 examples [00:01, 1059.26 examples/s]2111 examples [00:02, 1057.83 examples/s]2220 examples [00:02, 1064.16 examples/s]2330 examples [00:02, 1072.48 examples/s]2438 examples [00:02, 1072.38 examples/s]2546 examples [00:02, 1065.24 examples/s]2653 examples [00:02, 1063.39 examples/s]2762 examples [00:02, 1070.02 examples/s]2870 examples [00:02, 1041.05 examples/s]2978 examples [00:02, 1051.44 examples/s]3085 examples [00:02, 1055.32 examples/s]3192 examples [00:03, 1058.39 examples/s]3298 examples [00:03, 1053.48 examples/s]3404 examples [00:03, 1046.82 examples/s]3512 examples [00:03, 1055.90 examples/s]3620 examples [00:03, 1061.28 examples/s]3728 examples [00:03, 1064.94 examples/s]3835 examples [00:03, 1055.31 examples/s]3944 examples [00:03, 1064.30 examples/s]4051 examples [00:03, 1044.93 examples/s]4156 examples [00:03, 1014.96 examples/s]4263 examples [00:04, 1028.52 examples/s]4370 examples [00:04, 1038.39 examples/s]4476 examples [00:04, 1044.45 examples/s]4584 examples [00:04, 1054.41 examples/s]4692 examples [00:04, 1060.79 examples/s]4801 examples [00:04, 1067.91 examples/s]4909 examples [00:04, 1069.02 examples/s]5018 examples [00:04, 1073.50 examples/s]5127 examples [00:04, 1076.34 examples/s]5235 examples [00:04, 1073.49 examples/s]5344 examples [00:05, 1075.51 examples/s]5452 examples [00:05, 1068.29 examples/s]5562 examples [00:05, 1074.99 examples/s]5670 examples [00:05, 1075.19 examples/s]5778 examples [00:05, 1076.08 examples/s]5889 examples [00:05, 1083.45 examples/s]5998 examples [00:05, 1075.86 examples/s]6108 examples [00:05, 1081.50 examples/s]6217 examples [00:05, 1081.51 examples/s]6326 examples [00:05, 1065.70 examples/s]6434 examples [00:06, 1067.97 examples/s]6541 examples [00:06, 1068.23 examples/s]6648 examples [00:06, 1067.53 examples/s]6758 examples [00:06, 1074.53 examples/s]6869 examples [00:06, 1082.21 examples/s]6979 examples [00:06, 1085.30 examples/s]7088 examples [00:06, 1079.01 examples/s]7197 examples [00:06, 1079.43 examples/s]7305 examples [00:06, 1048.46 examples/s]7411 examples [00:07, 1044.60 examples/s]7517 examples [00:07, 1045.49 examples/s]7625 examples [00:07, 1055.39 examples/s]7735 examples [00:07, 1066.23 examples/s]7843 examples [00:07, 1069.72 examples/s]7951 examples [00:07, 1069.05 examples/s]8060 examples [00:07, 1074.14 examples/s]8168 examples [00:07, 1071.33 examples/s]8276 examples [00:07, 1063.00 examples/s]8384 examples [00:07, 1065.42 examples/s]8491 examples [00:08, 1039.93 examples/s]8596 examples [00:08, 1020.18 examples/s]8699 examples [00:08, 1014.19 examples/s]8807 examples [00:08, 1031.01 examples/s]8911 examples [00:08, 982.44 examples/s] 9019 examples [00:08, 1008.08 examples/s]9122 examples [00:08, 1014.46 examples/s]9227 examples [00:08, 1023.74 examples/s]9336 examples [00:08, 1041.03 examples/s]9443 examples [00:08, 1048.86 examples/s]9551 examples [00:09, 1056.83 examples/s]9660 examples [00:09, 1064.01 examples/s]9767 examples [00:09, 1060.51 examples/s]9875 examples [00:09, 1064.47 examples/s]9982 examples [00:09, 1065.29 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUZTDCR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUZTDCR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff7ed8f4620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff7ed8f4620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff7ed8f4620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff776d2ef60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff776d69588>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff7ed8f4620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff7ed8f4620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff7ed8f4620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff7d49285c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff7d4928320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff76535c2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff76535c2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7ff76535c2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff76535c400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff76535c400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff76535c400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=c49503922d7806812da36cd3c58fa334b764b64ad7f935a09a6e2177d0d3f7b6
  Stored in directory: /tmp/pip-ephem-wheel-cache-v_rodrug/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:17:21, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:02:19, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<9:11:04, 26.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<6:26:18, 37.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.02M/862M [00:01<4:29:57, 53.0kB/s].vector_cache/glove.6B.zip:   1%|          | 6.62M/862M [00:01<3:08:18, 75.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.9M/862M [00:01<2:11:04, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:01<1:31:31, 154kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:03:43, 220kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:01<44:34, 313kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.0M/862M [00:01<31:05, 447kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<21:47, 634kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.5M/862M [00:01<15:15, 901kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<10:45, 1.27MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<07:33, 1.80MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<05:23, 2.51MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:02<04:04, 3.32MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<04:45, 2.83MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<07:03, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:52, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:04<04:19, 3.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<08:47, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<07:35, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<05:37, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<04:17, 3.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<10:57, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<09:04, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<06:40, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<07:45, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<06:46, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<05:04, 2.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<06:38, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<04:31, 2.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<06:12, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<05:26, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<04:10, 3.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:16<03:05, 4.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<1:39:32, 130kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<1:12:19, 179kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<51:08, 253kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:18<35:51, 360kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<32:29, 397kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<24:03, 535kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<17:09, 750kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<14:58, 856kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<11:47, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<08:34, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<08:58, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<08:53, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:48, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:52, 2.60MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<38:59, 325kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<28:36, 443kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<20:18, 623kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<17:06, 737kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<13:31, 933kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:46, 1.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<06:59, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<29:58, 419kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<23:55, 524kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<17:30, 716kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<12:22, 1.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<16:28, 758kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<13:01, 958kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:28, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<09:10, 1.35MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:28, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:24, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:21, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<11:31, 1.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:35, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:02, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:26, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:14, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:24, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:43, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:03, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:36, 2.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:41, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:08, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:44, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:11, 2.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:49, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:01, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:03, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<05:05, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:31, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:59, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:20, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:55, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:53, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:27, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:10, 2.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:20, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:39, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:22, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:54, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<09:48, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:18, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:07, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:40, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:43, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:01, 1.93MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:27, 2.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:04, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:40, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:19, 2.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:36, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:15, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:48, 3.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:42, 1.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:28, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:34, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:13, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:44, 1.98MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:19, 2.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:20, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:07, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:55, 2.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:02, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:45, 2.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:54, 2.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:11, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:55, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:35, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:46, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:54, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:30, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:19, 3.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:15, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:38, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:40, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:11, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:03, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<05:34, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:03, 2.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:07, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:31, 1.44MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:35, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:02, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<56:44, 191kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<42:15, 256kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<30:07, 358kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<21:10, 508kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<18:48, 571kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<14:29, 740kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<10:27, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:28, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:15, 1.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<07:08, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<05:07, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<10:10, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<08:22, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<06:10, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:30, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:10, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:39, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<04:05, 2.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:23, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:52, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<05:46, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<04:08, 2.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:46:29, 97.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<1:15:35, 137kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<53:07, 195kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:31<37:10, 278kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<1:13:40, 140kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<52:46, 196kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<37:10, 277kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<28:01, 366kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<22:02, 465kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<15:56, 642kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<11:15, 906kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<16:46, 607kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<12:59, 783kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<09:20, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:34, 1.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<07:14, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<05:19, 1.90MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:47, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:32, 1.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:11, 1.93MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<03:46, 2.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:59, 1.11MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:24, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:25, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:55, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:35, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:07, 1.93MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:44, 2.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:39, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:08, 1.91MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:53, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:43, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:41, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:35, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:20, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<08:27, 1.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:06, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<05:13, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:37, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:05, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:49, 2.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:35, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:26, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:12, 2.97MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:09, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:54, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:06, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:28, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:59, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:44, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:30, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:27, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:23, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:11, 2.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:00, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:42, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:58, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:02, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:44, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:24, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:35, 1.20MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:26, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:44, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:10, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:52, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:34, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:20, 2.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:19, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:50, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<03:36, 2.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:20, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:07, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:09, 2.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:00, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:59, 1.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:02, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<02:56, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<07:30, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:09, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:36, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:20, 2.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:34, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:20, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:41, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:04, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:23, 1.97MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:20, 2.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:10, 2.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:01, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:57, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:59, 2.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:58, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<03:49, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:54, 2.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:46, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<03:40, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<02:49, 2.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:41, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:37, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<02:44, 3.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:01, 4.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<14:50, 560kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<11:22, 730kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<08:11, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<07:23, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:11, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:31, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:15, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<20:25, 400kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<16:17, 501kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<11:49, 690kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<08:21, 972kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<08:56, 906kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:14, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<05:17, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:20, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:43, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:31, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:05, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:56, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<02:51, 2.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:35, 1.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:51, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:31, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:14, 2.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:52, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:04, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:44, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:11, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:53, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<02:57, 2.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:37, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:25, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:35, 2.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:39, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:21, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<03:07, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:58, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:07, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:46, 1.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:10, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:45, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:42, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:43, 2.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:38, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:44, 2.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:30, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:17, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:18, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:08, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:20, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:25, 2.96MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:16, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:16, 1.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:53, 1.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:10, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:41, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:39, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:38, 2.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:02, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:23, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:42, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:18, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:27, 2.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:12, 1.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:47, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:51, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:23, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:12, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:25, 2.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:05, 2.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:40, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:56, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:07, 3.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:56, 966kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:36, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:04, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:16, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:38, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:36, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:35, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:01, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:10, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:06, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:29, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:02, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:11, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:17, 2.81MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:06, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:21, 1.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:14, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:33, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:14, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:27, 2.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:00, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:30, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:45, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<01:59, 3.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:09, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:21, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:13, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:30, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:59, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:07, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:14, 2.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:47, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:05, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:02, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:20, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:05, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:19, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:50, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:43, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:59, 2.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:26, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:09, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:22, 2.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:50, 2.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:03, 2.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:32, 3.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:24, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:40, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:49, 2.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:02, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:49, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:07, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:01, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:11, 2.56MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<08:53, 630kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:58, 802kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:01, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<03:34, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<10:14, 539kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<07:52, 702kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:39, 973kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:02, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:16, 1.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:09, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:16, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:58, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:14, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:30, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:53, 2.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:23, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:19, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:45, 2.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:16, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:13, 2.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:41, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:13, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:48, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:16, 2.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:39, 3.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:22, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:36, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:39, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:55, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:09, 1.57MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:29, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:47, 2.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<06:29, 755kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<05:07, 955kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<03:43, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:34, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:05, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<02:16, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:57, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:19, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:40, 2.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:28, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:00, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<02:14, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:29, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:54, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:18, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:40, 2.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:04, 1.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:23, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:30, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:39, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:24, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:47, 2.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<01:17, 3.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<34:01, 130kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<24:47, 178kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<17:31, 251kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<12:14, 357kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<10:11, 427kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<07:39, 567kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<05:28, 790kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:39, 917kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:44, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:43, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:48, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:28, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:50, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:07, 1.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:59, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:30, 2.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:18, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:20, 3.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:09, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:59, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:30, 2.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:51, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:46, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:20, 2.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:43, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:09, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:42, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:38<01:14, 3.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:35, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:16, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:41, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:56, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:15, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:48, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<01:18, 2.80MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:13, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:41, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:59, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:06, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:16, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:47, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<01:17, 2.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<05:08, 683kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<04:01, 872kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:53, 1.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:42, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:44, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:07, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:30, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:09, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:54, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:00, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:12, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:43, 1.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:14, 2.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:49, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:22, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:44, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:51, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:00, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:07, 2.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:21, 711kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:23, 913kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:26, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:20, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:00, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:28, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:37, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:51, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:28, 1.99MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:04, 2.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:34, 1.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:07, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:32, 1.85MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:40, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:52, 1.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:27, 1.93MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:03, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:35, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:24, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:03, 2.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:46, 3.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:08, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:49, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:57, 2.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<23:03, 113kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<16:45, 156kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<11:49, 220kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<08:14, 312kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<06:26, 395kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<04:48, 528kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<03:24, 739kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:50, 869kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:35, 951kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:56, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:24, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:32, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:20, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<00:59, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:11, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:04, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:48, 2.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:35, 3.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:59, 568kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:17, 687kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:25, 929kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:41, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<05:06, 430kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:52, 566kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:44, 792kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:56, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:37, 813kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:10, 976kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:34, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:07, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:01, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:45, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:15, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:54, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:53, 1.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:38, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:14, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:52, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:51, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:31, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:47, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:51, 994kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:46, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:22, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:58, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:07, 1.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:50, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:36, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:22, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:10, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:51, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:55, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:00, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:46, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:34, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:50, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:45, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:33, 2.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:42, 2.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:51, 1.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:29, 2.99MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:59, 1.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:50, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:28, 2.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:39, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:26, 2.89MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:19, 3.87MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:59, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:00, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:45, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:32, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:40, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:27, 2.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:37, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:30, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:21, 2.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:53, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:44, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:32, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:33, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:37, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:28, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:21, 2.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:26, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:18, 2.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:22, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:19, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:12, 3.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:33, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:28, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:22, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:39, 803kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:31, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:21, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:19, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:16, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:12, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:05, 3.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:03, 3.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:14, 728kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:11, 922kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:07, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:03, 692kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 904kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.26MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:58:28,  5.86it/s]  0%|          | 824/400000 [00:00<13:15:32,  8.36it/s]  0%|          | 1664/400000 [00:00<9:15:56, 11.94it/s]  1%|          | 2505/400000 [00:00<6:28:34, 17.05it/s]  1%|          | 3324/400000 [00:00<4:31:41, 24.33it/s]  1%|          | 4131/400000 [00:00<3:10:02, 34.72it/s]  1%|          | 4975/400000 [00:00<2:12:58, 49.51it/s]  1%|▏         | 5816/400000 [00:00<1:33:07, 70.55it/s]  2%|▏         | 6658/400000 [00:00<1:05:16, 100.43it/s]  2%|▏         | 7509/400000 [00:01<45:49, 142.74it/s]    2%|▏         | 8359/400000 [00:01<32:14, 202.46it/s]  2%|▏         | 9189/400000 [00:01<22:45, 286.24it/s]  3%|▎         | 10012/400000 [00:01<16:08, 402.84it/s]  3%|▎         | 10857/400000 [00:01<11:30, 563.97it/s]  3%|▎         | 11715/400000 [00:01<08:15, 783.57it/s]  3%|▎         | 12552/400000 [00:01<06:01, 1071.19it/s]  3%|▎         | 13371/400000 [00:01<04:26, 1448.96it/s]  4%|▎         | 14210/400000 [00:01<03:20, 1927.19it/s]  4%|▍         | 15030/400000 [00:01<02:34, 2485.97it/s]  4%|▍         | 15847/400000 [00:02<02:02, 3141.36it/s]  4%|▍         | 16689/400000 [00:02<01:39, 3868.64it/s]  4%|▍         | 17541/400000 [00:02<01:22, 4625.71it/s]  5%|▍         | 18396/400000 [00:02<01:11, 5363.34it/s]  5%|▍         | 19240/400000 [00:02<01:03, 6020.73it/s]  5%|▌         | 20091/400000 [00:02<00:57, 6598.73it/s]  5%|▌         | 20933/400000 [00:02<00:53, 7037.08it/s]  5%|▌         | 21787/400000 [00:02<00:50, 7428.04it/s]  6%|▌         | 22643/400000 [00:02<00:48, 7733.35it/s]  6%|▌         | 23490/400000 [00:02<00:47, 7846.31it/s]  6%|▌         | 24346/400000 [00:03<00:46, 8045.36it/s]  6%|▋         | 25189/400000 [00:03<00:45, 8148.98it/s]  7%|▋         | 26031/400000 [00:03<00:46, 8125.01it/s]  7%|▋         | 26877/400000 [00:03<00:45, 8222.30it/s]  7%|▋         | 27713/400000 [00:03<00:45, 8225.95it/s]  7%|▋         | 28546/400000 [00:03<00:44, 8255.67it/s]  7%|▋         | 29393/400000 [00:03<00:44, 8317.65it/s]  8%|▊         | 30230/400000 [00:03<00:44, 8322.08it/s]  8%|▊         | 31066/400000 [00:03<00:44, 8330.03it/s]  8%|▊         | 31904/400000 [00:04<00:44, 8342.60it/s]  8%|▊         | 32740/400000 [00:04<00:44, 8296.06it/s]  8%|▊         | 33585/400000 [00:04<00:43, 8338.78it/s]  9%|▊         | 34439/400000 [00:04<00:43, 8396.51it/s]  9%|▉         | 35280/400000 [00:04<00:43, 8374.26it/s]  9%|▉         | 36118/400000 [00:04<00:43, 8363.01it/s]  9%|▉         | 36955/400000 [00:04<00:43, 8296.62it/s]  9%|▉         | 37786/400000 [00:04<00:44, 8217.62it/s] 10%|▉         | 38609/400000 [00:04<00:44, 8134.69it/s] 10%|▉         | 39464/400000 [00:04<00:43, 8253.67it/s] 10%|█         | 40306/400000 [00:05<00:43, 8302.19it/s] 10%|█         | 41138/400000 [00:05<00:43, 8307.10it/s] 10%|█         | 41986/400000 [00:05<00:42, 8358.13it/s] 11%|█         | 42846/400000 [00:05<00:42, 8428.94it/s] 11%|█         | 43691/400000 [00:05<00:42, 8434.51it/s] 11%|█         | 44535/400000 [00:05<00:42, 8339.90it/s] 11%|█▏        | 45370/400000 [00:05<00:42, 8281.56it/s] 12%|█▏        | 46220/400000 [00:05<00:42, 8345.86it/s] 12%|█▏        | 47081/400000 [00:05<00:41, 8422.64it/s] 12%|█▏        | 47926/400000 [00:05<00:41, 8428.50it/s] 12%|█▏        | 48785/400000 [00:06<00:41, 8475.68it/s] 12%|█▏        | 49633/400000 [00:06<00:41, 8398.57it/s] 13%|█▎        | 50474/400000 [00:06<00:41, 8361.74it/s] 13%|█▎        | 51329/400000 [00:06<00:41, 8416.69it/s] 13%|█▎        | 52190/400000 [00:06<00:41, 8472.69it/s] 13%|█▎        | 53049/400000 [00:06<00:40, 8506.16it/s] 13%|█▎        | 53900/400000 [00:06<00:40, 8455.10it/s] 14%|█▎        | 54746/400000 [00:06<00:41, 8369.82it/s] 14%|█▍        | 55595/400000 [00:06<00:40, 8403.75it/s] 14%|█▍        | 56452/400000 [00:06<00:40, 8450.49it/s] 14%|█▍        | 57303/400000 [00:07<00:40, 8466.36it/s] 15%|█▍        | 58150/400000 [00:07<00:40, 8339.92it/s] 15%|█▍        | 58988/400000 [00:07<00:40, 8351.85it/s] 15%|█▍        | 59824/400000 [00:07<00:41, 8258.18it/s] 15%|█▌        | 60657/400000 [00:07<00:40, 8277.03it/s] 15%|█▌        | 61511/400000 [00:07<00:40, 8354.02it/s] 16%|█▌        | 62347/400000 [00:07<00:40, 8319.52it/s] 16%|█▌        | 63204/400000 [00:07<00:40, 8392.31it/s] 16%|█▌        | 64044/400000 [00:07<00:40, 8287.16it/s] 16%|█▌        | 64900/400000 [00:07<00:40, 8366.91it/s] 16%|█▋        | 65759/400000 [00:08<00:39, 8432.07it/s] 17%|█▋        | 66609/400000 [00:08<00:39, 8451.14it/s] 17%|█▋        | 67469/400000 [00:08<00:39, 8493.42it/s] 17%|█▋        | 68319/400000 [00:08<00:39, 8407.34it/s] 17%|█▋        | 69174/400000 [00:08<00:39, 8448.79it/s] 18%|█▊        | 70026/400000 [00:08<00:38, 8468.00it/s] 18%|█▊        | 70874/400000 [00:08<00:39, 8403.11it/s] 18%|█▊        | 71716/400000 [00:08<00:39, 8407.49it/s] 18%|█▊        | 72563/400000 [00:08<00:38, 8424.02it/s] 18%|█▊        | 73406/400000 [00:08<00:39, 8257.02it/s] 19%|█▊        | 74260/400000 [00:09<00:39, 8339.77it/s] 19%|█▉        | 75095/400000 [00:09<00:38, 8338.43it/s] 19%|█▉        | 75932/400000 [00:09<00:38, 8345.17it/s] 19%|█▉        | 76767/400000 [00:09<00:38, 8335.68it/s] 19%|█▉        | 77604/400000 [00:09<00:38, 8343.89it/s] 20%|█▉        | 78446/400000 [00:09<00:38, 8366.33it/s] 20%|█▉        | 79283/400000 [00:09<00:38, 8287.76it/s] 20%|██        | 80113/400000 [00:09<00:38, 8205.85it/s] 20%|██        | 80955/400000 [00:09<00:38, 8267.52it/s] 20%|██        | 81808/400000 [00:09<00:38, 8341.77it/s] 21%|██        | 82646/400000 [00:10<00:37, 8352.85it/s] 21%|██        | 83488/400000 [00:10<00:37, 8372.70it/s] 21%|██        | 84326/400000 [00:10<00:37, 8333.66it/s] 21%|██▏       | 85160/400000 [00:10<00:37, 8296.52it/s] 22%|██▏       | 86019/400000 [00:10<00:37, 8380.97it/s] 22%|██▏       | 86858/400000 [00:10<00:37, 8366.61it/s] 22%|██▏       | 87695/400000 [00:10<00:37, 8354.18it/s] 22%|██▏       | 88531/400000 [00:10<00:38, 8052.68it/s] 22%|██▏       | 89368/400000 [00:10<00:38, 8143.55it/s] 23%|██▎       | 90198/400000 [00:10<00:37, 8189.53it/s] 23%|██▎       | 91054/400000 [00:11<00:37, 8296.66it/s] 23%|██▎       | 91908/400000 [00:11<00:36, 8365.57it/s] 23%|██▎       | 92746/400000 [00:11<00:36, 8366.46it/s] 23%|██▎       | 93584/400000 [00:11<00:36, 8364.61it/s] 24%|██▎       | 94435/400000 [00:11<00:36, 8406.41it/s] 24%|██▍       | 95285/400000 [00:11<00:36, 8432.81it/s] 24%|██▍       | 96141/400000 [00:11<00:35, 8467.67it/s] 24%|██▍       | 96989/400000 [00:11<00:36, 8240.35it/s] 24%|██▍       | 97836/400000 [00:11<00:36, 8307.75it/s] 25%|██▍       | 98673/400000 [00:11<00:36, 8324.22it/s] 25%|██▍       | 99507/400000 [00:12<00:36, 8265.37it/s] 25%|██▌       | 100350/400000 [00:12<00:36, 8312.15it/s] 25%|██▌       | 101182/400000 [00:12<00:35, 8301.82it/s] 26%|██▌       | 102013/400000 [00:12<00:35, 8296.89it/s] 26%|██▌       | 102843/400000 [00:12<00:35, 8281.67it/s] 26%|██▌       | 103678/400000 [00:12<00:35, 8301.48it/s] 26%|██▌       | 104512/400000 [00:12<00:35, 8310.55it/s] 26%|██▋       | 105344/400000 [00:12<00:35, 8296.60it/s] 27%|██▋       | 106174/400000 [00:12<00:35, 8274.71it/s] 27%|██▋       | 107002/400000 [00:13<00:35, 8202.42it/s] 27%|██▋       | 107845/400000 [00:13<00:35, 8267.54it/s] 27%|██▋       | 108699/400000 [00:13<00:34, 8347.13it/s] 27%|██▋       | 109556/400000 [00:13<00:34, 8412.75it/s] 28%|██▊       | 110398/400000 [00:13<00:34, 8325.69it/s] 28%|██▊       | 111252/400000 [00:13<00:34, 8387.66it/s] 28%|██▊       | 112101/400000 [00:13<00:34, 8416.81it/s] 28%|██▊       | 112944/400000 [00:13<00:34, 8402.28it/s] 28%|██▊       | 113799/400000 [00:13<00:33, 8444.58it/s] 29%|██▊       | 114644/400000 [00:13<00:34, 8325.06it/s] 29%|██▉       | 115494/400000 [00:14<00:33, 8376.14it/s] 29%|██▉       | 116333/400000 [00:14<00:34, 8227.98it/s] 29%|██▉       | 117171/400000 [00:14<00:34, 8270.73it/s] 30%|██▉       | 118026/400000 [00:14<00:33, 8352.34it/s] 30%|██▉       | 118862/400000 [00:14<00:33, 8338.60it/s] 30%|██▉       | 119716/400000 [00:14<00:33, 8397.06it/s] 30%|███       | 120557/400000 [00:14<00:33, 8324.99it/s] 30%|███       | 121390/400000 [00:14<00:33, 8201.96it/s] 31%|███       | 122224/400000 [00:14<00:33, 8242.20it/s] 31%|███       | 123049/400000 [00:14<00:33, 8221.27it/s] 31%|███       | 123891/400000 [00:15<00:33, 8277.90it/s] 31%|███       | 124736/400000 [00:15<00:33, 8327.93it/s] 31%|███▏      | 125582/400000 [00:15<00:32, 8365.28it/s] 32%|███▏      | 126427/400000 [00:15<00:32, 8389.79it/s] 32%|███▏      | 127267/400000 [00:15<00:32, 8326.10it/s] 32%|███▏      | 128115/400000 [00:15<00:32, 8371.08it/s] 32%|███▏      | 128960/400000 [00:15<00:32, 8393.72it/s] 32%|███▏      | 129800/400000 [00:15<00:32, 8360.83it/s] 33%|███▎      | 130653/400000 [00:15<00:32, 8408.87it/s] 33%|███▎      | 131495/400000 [00:15<00:32, 8312.19it/s] 33%|███▎      | 132351/400000 [00:16<00:31, 8383.29it/s] 33%|███▎      | 133205/400000 [00:16<00:31, 8428.35it/s] 34%|███▎      | 134054/400000 [00:16<00:31, 8445.35it/s] 34%|███▎      | 134899/400000 [00:16<00:31, 8421.25it/s] 34%|███▍      | 135742/400000 [00:16<00:31, 8380.99it/s] 34%|███▍      | 136581/400000 [00:16<00:32, 8207.62it/s] 34%|███▍      | 137419/400000 [00:16<00:31, 8256.73it/s] 35%|███▍      | 138271/400000 [00:16<00:31, 8331.80it/s] 35%|███▍      | 139105/400000 [00:16<00:31, 8263.36it/s] 35%|███▍      | 139947/400000 [00:16<00:31, 8307.44it/s] 35%|███▌      | 140803/400000 [00:17<00:30, 8379.39it/s] 35%|███▌      | 141658/400000 [00:17<00:30, 8429.01it/s] 36%|███▌      | 142505/400000 [00:17<00:30, 8439.80it/s] 36%|███▌      | 143366/400000 [00:17<00:30, 8489.17it/s] 36%|███▌      | 144216/400000 [00:17<00:30, 8448.82it/s] 36%|███▋      | 145062/400000 [00:17<00:30, 8405.07it/s] 36%|███▋      | 145903/400000 [00:17<00:30, 8287.90it/s] 37%|███▋      | 146739/400000 [00:17<00:30, 8307.00it/s] 37%|███▋      | 147580/400000 [00:17<00:30, 8335.29it/s] 37%|███▋      | 148414/400000 [00:17<00:30, 8276.58it/s] 37%|███▋      | 149261/400000 [00:18<00:30, 8331.55it/s] 38%|███▊      | 150121/400000 [00:18<00:29, 8408.18it/s] 38%|███▊      | 150963/400000 [00:18<00:29, 8404.59it/s] 38%|███▊      | 151806/400000 [00:18<00:29, 8409.42it/s] 38%|███▊      | 152656/400000 [00:18<00:29, 8433.81it/s] 38%|███▊      | 153500/400000 [00:18<00:29, 8426.80it/s] 39%|███▊      | 154343/400000 [00:18<00:29, 8383.26it/s] 39%|███▉      | 155204/400000 [00:18<00:28, 8448.18it/s] 39%|███▉      | 156050/400000 [00:18<00:29, 8396.04it/s] 39%|███▉      | 156890/400000 [00:18<00:28, 8394.41it/s] 39%|███▉      | 157730/400000 [00:19<00:29, 8329.03it/s] 40%|███▉      | 158571/400000 [00:19<00:28, 8350.39it/s] 40%|███▉      | 159426/400000 [00:19<00:28, 8408.25it/s] 40%|████      | 160283/400000 [00:19<00:28, 8455.16it/s] 40%|████      | 161129/400000 [00:19<00:28, 8441.28it/s] 40%|████      | 161974/400000 [00:19<00:28, 8432.19it/s] 41%|████      | 162818/400000 [00:19<00:28, 8432.37it/s] 41%|████      | 163662/400000 [00:19<00:28, 8364.78it/s] 41%|████      | 164499/400000 [00:19<00:28, 8350.10it/s] 41%|████▏     | 165340/400000 [00:19<00:28, 8366.56it/s] 42%|████▏     | 166177/400000 [00:20<00:28, 8257.14it/s] 42%|████▏     | 167037/400000 [00:20<00:27, 8355.53it/s] 42%|████▏     | 167879/400000 [00:20<00:27, 8372.77it/s] 42%|████▏     | 168733/400000 [00:20<00:27, 8421.71it/s] 42%|████▏     | 169590/400000 [00:20<00:27, 8464.12it/s] 43%|████▎     | 170437/400000 [00:20<00:27, 8380.88it/s] 43%|████▎     | 171281/400000 [00:20<00:27, 8396.52it/s] 43%|████▎     | 172136/400000 [00:20<00:26, 8440.25it/s] 43%|████▎     | 172996/400000 [00:20<00:26, 8486.97it/s] 43%|████▎     | 173850/400000 [00:20<00:26, 8501.31it/s] 44%|████▎     | 174701/400000 [00:21<00:26, 8469.99it/s] 44%|████▍     | 175555/400000 [00:21<00:26, 8490.01it/s] 44%|████▍     | 176415/400000 [00:21<00:26, 8520.70it/s] 44%|████▍     | 177272/400000 [00:21<00:26, 8532.69it/s] 45%|████▍     | 178126/400000 [00:21<00:26, 8521.54it/s] 45%|████▍     | 178979/400000 [00:21<00:26, 8475.62it/s] 45%|████▍     | 179827/400000 [00:21<00:26, 8392.18it/s] 45%|████▌     | 180673/400000 [00:21<00:26, 8411.87it/s] 45%|████▌     | 181529/400000 [00:21<00:25, 8453.86it/s] 46%|████▌     | 182376/400000 [00:21<00:25, 8458.67it/s] 46%|████▌     | 183222/400000 [00:22<00:26, 8333.64it/s] 46%|████▌     | 184058/400000 [00:22<00:25, 8340.75it/s] 46%|████▌     | 184898/400000 [00:22<00:25, 8358.30it/s] 46%|████▋     | 185758/400000 [00:22<00:25, 8426.81it/s] 47%|████▋     | 186608/400000 [00:22<00:25, 8446.60it/s] 47%|████▋     | 187453/400000 [00:22<00:25, 8368.30it/s] 47%|████▋     | 188313/400000 [00:22<00:25, 8428.78it/s] 47%|████▋     | 189158/400000 [00:22<00:25, 8433.37it/s] 48%|████▊     | 190002/400000 [00:22<00:25, 8236.01it/s] 48%|████▊     | 190850/400000 [00:23<00:25, 8306.10it/s] 48%|████▊     | 191690/400000 [00:23<00:25, 8331.64it/s] 48%|████▊     | 192552/400000 [00:23<00:24, 8414.34it/s] 48%|████▊     | 193414/400000 [00:23<00:24, 8473.24it/s] 49%|████▊     | 194273/400000 [00:23<00:24, 8505.15it/s] 49%|████▉     | 195126/400000 [00:23<00:24, 8509.64it/s] 49%|████▉     | 195978/400000 [00:23<00:24, 8444.87it/s] 49%|████▉     | 196834/400000 [00:23<00:23, 8476.59it/s] 49%|████▉     | 197683/400000 [00:23<00:23, 8479.90it/s] 50%|████▉     | 198532/400000 [00:23<00:23, 8478.50it/s] 50%|████▉     | 199392/400000 [00:24<00:23, 8514.42it/s] 50%|█████     | 200244/400000 [00:24<00:23, 8387.17it/s] 50%|█████     | 201084/400000 [00:24<00:23, 8362.47it/s] 50%|█████     | 201945/400000 [00:24<00:23, 8434.76it/s] 51%|█████     | 202806/400000 [00:24<00:23, 8485.25it/s] 51%|█████     | 203655/400000 [00:24<00:23, 8483.86it/s] 51%|█████     | 204504/400000 [00:24<00:23, 8472.35it/s] 51%|█████▏    | 205352/400000 [00:24<00:23, 8437.67it/s] 52%|█████▏    | 206213/400000 [00:24<00:22, 8485.76it/s] 52%|█████▏    | 207063/400000 [00:24<00:22, 8489.35it/s] 52%|█████▏    | 207923/400000 [00:25<00:22, 8521.94it/s] 52%|█████▏    | 208776/400000 [00:25<00:22, 8449.20it/s] 52%|█████▏    | 209622/400000 [00:25<00:22, 8344.66it/s] 53%|█████▎    | 210457/400000 [00:25<00:22, 8318.07it/s] 53%|█████▎    | 211320/400000 [00:25<00:22, 8407.53it/s] 53%|█████▎    | 212171/400000 [00:25<00:22, 8435.66it/s] 53%|█████▎    | 213015/400000 [00:25<00:22, 8392.94it/s] 53%|█████▎    | 213870/400000 [00:25<00:22, 8439.42it/s] 54%|█████▎    | 214715/400000 [00:25<00:22, 8262.42it/s] 54%|█████▍    | 215571/400000 [00:25<00:22, 8348.98it/s] 54%|█████▍    | 216432/400000 [00:26<00:21, 8424.18it/s] 54%|█████▍    | 217276/400000 [00:26<00:21, 8354.95it/s] 55%|█████▍    | 218131/400000 [00:26<00:21, 8411.33it/s] 55%|█████▍    | 218988/400000 [00:26<00:21, 8456.76it/s] 55%|█████▍    | 219835/400000 [00:26<00:21, 8443.24it/s] 55%|█████▌    | 220696/400000 [00:26<00:21, 8492.47it/s] 55%|█████▌    | 221546/400000 [00:26<00:21, 8414.08it/s] 56%|█████▌    | 222398/400000 [00:26<00:21, 8443.64it/s] 56%|█████▌    | 223256/400000 [00:26<00:20, 8482.70it/s] 56%|█████▌    | 224108/400000 [00:26<00:20, 8490.89it/s] 56%|█████▌    | 224965/400000 [00:27<00:20, 8513.80it/s] 56%|█████▋    | 225817/400000 [00:27<00:20, 8489.17it/s] 57%|█████▋    | 226667/400000 [00:27<00:20, 8421.49it/s] 57%|█████▋    | 227527/400000 [00:27<00:20, 8472.47it/s] 57%|█████▋    | 228390/400000 [00:27<00:20, 8517.36it/s] 57%|█████▋    | 229242/400000 [00:27<00:20, 8481.23it/s] 58%|█████▊    | 230091/400000 [00:27<00:20, 8388.75it/s] 58%|█████▊    | 230931/400000 [00:27<00:20, 8347.01it/s] 58%|█████▊    | 231786/400000 [00:27<00:20, 8404.27it/s] 58%|█████▊    | 232643/400000 [00:27<00:19, 8451.92it/s] 58%|█████▊    | 233502/400000 [00:28<00:19, 8492.65it/s] 59%|█████▊    | 234352/400000 [00:28<00:19, 8366.08it/s] 59%|█████▉    | 235192/400000 [00:28<00:19, 8374.85it/s] 59%|█████▉    | 236043/400000 [00:28<00:19, 8412.44it/s] 59%|█████▉    | 236901/400000 [00:28<00:19, 8461.38it/s] 59%|█████▉    | 237748/400000 [00:28<00:19, 8442.92it/s] 60%|█████▉    | 238593/400000 [00:28<00:19, 8395.25it/s] 60%|█████▉    | 239440/400000 [00:28<00:19, 8414.92it/s] 60%|██████    | 240287/400000 [00:28<00:18, 8430.52it/s] 60%|██████    | 241148/400000 [00:28<00:18, 8482.36it/s] 61%|██████    | 242007/400000 [00:29<00:18, 8512.71it/s] 61%|██████    | 242859/400000 [00:29<00:18, 8479.91it/s] 61%|██████    | 243708/400000 [00:29<00:18, 8468.00it/s] 61%|██████    | 244555/400000 [00:29<00:18, 8457.68it/s] 61%|██████▏   | 245413/400000 [00:29<00:18, 8492.73it/s] 62%|██████▏   | 246263/400000 [00:29<00:18, 8484.59it/s] 62%|██████▏   | 247112/400000 [00:29<00:18, 8445.85it/s] 62%|██████▏   | 247957/400000 [00:29<00:18, 8336.51it/s] 62%|██████▏   | 248804/400000 [00:29<00:18, 8373.55it/s] 62%|██████▏   | 249665/400000 [00:29<00:17, 8441.76it/s] 63%|██████▎   | 250521/400000 [00:30<00:17, 8474.49it/s] 63%|██████▎   | 251369/400000 [00:30<00:17, 8381.40it/s] 63%|██████▎   | 252208/400000 [00:30<00:18, 8160.28it/s] 63%|██████▎   | 253067/400000 [00:30<00:17, 8284.52it/s] 63%|██████▎   | 253922/400000 [00:30<00:17, 8360.73it/s] 64%|██████▎   | 254782/400000 [00:30<00:17, 8429.56it/s] 64%|██████▍   | 255626/400000 [00:30<00:17, 8428.54it/s] 64%|██████▍   | 256470/400000 [00:30<00:17, 8412.25it/s] 64%|██████▍   | 257316/400000 [00:30<00:16, 8424.81it/s] 65%|██████▍   | 258159/400000 [00:30<00:17, 8273.71it/s] 65%|██████▍   | 259016/400000 [00:31<00:16, 8359.16it/s] 65%|██████▍   | 259853/400000 [00:31<00:16, 8285.63it/s] 65%|██████▌   | 260705/400000 [00:31<00:16, 8354.19it/s] 65%|██████▌   | 261550/400000 [00:31<00:16, 8382.10it/s] 66%|██████▌   | 262399/400000 [00:31<00:16, 8413.99it/s] 66%|██████▌   | 263256/400000 [00:31<00:16, 8459.33it/s] 66%|██████▌   | 264103/400000 [00:31<00:16, 8439.84it/s] 66%|██████▌   | 264948/400000 [00:31<00:16, 8356.32it/s] 66%|██████▋   | 265784/400000 [00:31<00:16, 8044.74it/s] 67%|██████▋   | 266630/400000 [00:32<00:16, 8164.73it/s] 67%|██████▋   | 267483/400000 [00:32<00:16, 8268.88it/s] 67%|██████▋   | 268312/400000 [00:32<00:16, 8209.34it/s] 67%|██████▋   | 269162/400000 [00:32<00:15, 8293.13it/s] 68%|██████▊   | 270024/400000 [00:32<00:15, 8386.20it/s] 68%|██████▊   | 270878/400000 [00:32<00:15, 8430.69it/s] 68%|██████▊   | 271722/400000 [00:32<00:15, 8224.00it/s] 68%|██████▊   | 272547/400000 [00:32<00:15, 8161.94it/s] 68%|██████▊   | 273390/400000 [00:32<00:15, 8238.15it/s] 69%|██████▊   | 274246/400000 [00:32<00:15, 8330.08it/s] 69%|██████▉   | 275104/400000 [00:33<00:14, 8401.81it/s] 69%|██████▉   | 275957/400000 [00:33<00:14, 8438.16it/s] 69%|██████▉   | 276802/400000 [00:33<00:14, 8300.51it/s] 69%|██████▉   | 277640/400000 [00:33<00:14, 8322.74it/s] 70%|██████▉   | 278473/400000 [00:33<00:14, 8312.78it/s] 70%|██████▉   | 279312/400000 [00:33<00:14, 8334.62it/s] 70%|███████   | 280153/400000 [00:33<00:14, 8354.64it/s] 70%|███████   | 280989/400000 [00:33<00:14, 8332.33it/s] 70%|███████   | 281823/400000 [00:33<00:14, 8326.73it/s] 71%|███████   | 282656/400000 [00:33<00:14, 8308.26it/s] 71%|███████   | 283510/400000 [00:34<00:13, 8373.61it/s] 71%|███████   | 284348/400000 [00:34<00:14, 8218.85it/s] 71%|███████▏  | 285194/400000 [00:34<00:13, 8288.92it/s] 72%|███████▏  | 286024/400000 [00:34<00:13, 8264.90it/s] 72%|███████▏  | 286883/400000 [00:34<00:13, 8359.64it/s] 72%|███████▏  | 287736/400000 [00:34<00:13, 8407.86it/s] 72%|███████▏  | 288593/400000 [00:34<00:13, 8454.35it/s] 72%|███████▏  | 289439/400000 [00:34<00:13, 8266.98it/s] 73%|███████▎  | 290267/400000 [00:34<00:13, 8240.69it/s] 73%|███████▎  | 291119/400000 [00:34<00:13, 8321.48it/s] 73%|███████▎  | 291975/400000 [00:35<00:12, 8388.85it/s] 73%|███████▎  | 292822/400000 [00:35<00:12, 8412.98it/s] 73%|███████▎  | 293671/400000 [00:35<00:12, 8434.05it/s] 74%|███████▎  | 294515/400000 [00:35<00:12, 8424.48it/s] 74%|███████▍  | 295372/400000 [00:35<00:12, 8465.18it/s] 74%|███████▍  | 296230/400000 [00:35<00:12, 8498.76it/s] 74%|███████▍  | 297081/400000 [00:35<00:12, 8447.32it/s] 74%|███████▍  | 297937/400000 [00:35<00:12, 8478.19it/s] 75%|███████▍  | 298785/400000 [00:35<00:12, 8431.49it/s] 75%|███████▍  | 299638/400000 [00:35<00:11, 8460.52it/s] 75%|███████▌  | 300495/400000 [00:36<00:11, 8492.12it/s] 75%|███████▌  | 301350/400000 [00:36<00:11, 8507.67it/s] 76%|███████▌  | 302203/400000 [00:36<00:11, 8512.71it/s] 76%|███████▌  | 303055/400000 [00:36<00:11, 8471.67it/s] 76%|███████▌  | 303903/400000 [00:36<00:11, 8451.63it/s] 76%|███████▌  | 304749/400000 [00:36<00:11, 8446.36it/s] 76%|███████▋  | 305594/400000 [00:36<00:11, 8427.79it/s] 77%|███████▋  | 306437/400000 [00:36<00:11, 8131.64it/s] 77%|███████▋  | 307256/400000 [00:36<00:11, 8148.14it/s] 77%|███████▋  | 308113/400000 [00:36<00:11, 8268.72it/s] 77%|███████▋  | 308972/400000 [00:37<00:10, 8361.35it/s] 77%|███████▋  | 309810/400000 [00:37<00:10, 8350.06it/s] 78%|███████▊  | 310655/400000 [00:37<00:10, 8377.15it/s] 78%|███████▊  | 311494/400000 [00:37<00:10, 8379.53it/s] 78%|███████▊  | 312333/400000 [00:37<00:10, 8381.53it/s] 78%|███████▊  | 313190/400000 [00:37<00:10, 8436.67it/s] 79%|███████▊  | 314035/400000 [00:37<00:10, 8438.89it/s] 79%|███████▊  | 314885/400000 [00:37<00:10, 8456.26it/s] 79%|███████▉  | 315731/400000 [00:37<00:09, 8455.25it/s] 79%|███████▉  | 316577/400000 [00:37<00:09, 8401.98it/s] 79%|███████▉  | 317433/400000 [00:38<00:09, 8448.00it/s] 80%|███████▉  | 318278/400000 [00:38<00:09, 8381.66it/s] 80%|███████▉  | 319120/400000 [00:38<00:09, 8390.41it/s] 80%|███████▉  | 319960/400000 [00:38<00:09, 8329.17it/s] 80%|████████  | 320820/400000 [00:38<00:09, 8406.09it/s] 80%|████████  | 321679/400000 [00:38<00:09, 8459.51it/s] 81%|████████  | 322537/400000 [00:38<00:09, 8493.48it/s] 81%|████████  | 323393/400000 [00:38<00:08, 8512.07it/s] 81%|████████  | 324245/400000 [00:38<00:08, 8481.48it/s] 81%|████████▏ | 325094/400000 [00:38<00:08, 8481.72it/s] 81%|████████▏ | 325955/400000 [00:39<00:08, 8518.57it/s] 82%|████████▏ | 326807/400000 [00:39<00:08, 8500.90it/s] 82%|████████▏ | 327658/400000 [00:39<00:08, 8468.76it/s] 82%|████████▏ | 328505/400000 [00:39<00:08, 8434.27it/s] 82%|████████▏ | 329352/400000 [00:39<00:08, 8443.84it/s] 83%|████████▎ | 330207/400000 [00:39<00:08, 8475.13it/s] 83%|████████▎ | 331055/400000 [00:39<00:08, 8463.90it/s] 83%|████████▎ | 331902/400000 [00:39<00:08, 8129.94it/s] 83%|████████▎ | 332718/400000 [00:39<00:08, 8074.82it/s] 83%|████████▎ | 333574/400000 [00:40<00:08, 8212.20it/s] 84%|████████▎ | 334432/400000 [00:40<00:07, 8319.16it/s] 84%|████████▍ | 335288/400000 [00:40<00:07, 8387.71it/s] 84%|████████▍ | 336135/400000 [00:40<00:07, 8412.17it/s] 84%|████████▍ | 336978/400000 [00:40<00:07, 8364.45it/s] 84%|████████▍ | 337831/400000 [00:40<00:07, 8411.96it/s] 85%|████████▍ | 338686/400000 [00:40<00:07, 8450.23it/s] 85%|████████▍ | 339544/400000 [00:40<00:07, 8488.64it/s] 85%|████████▌ | 340396/400000 [00:40<00:07, 8495.22it/s] 85%|████████▌ | 341246/400000 [00:40<00:06, 8401.72it/s] 86%|████████▌ | 342107/400000 [00:41<00:06, 8461.72it/s] 86%|████████▌ | 342966/400000 [00:41<00:06, 8497.69it/s] 86%|████████▌ | 343823/400000 [00:41<00:06, 8516.46it/s] 86%|████████▌ | 344675/400000 [00:41<00:06, 8512.84it/s] 86%|████████▋ | 345527/400000 [00:41<00:06, 8472.26it/s] 87%|████████▋ | 346386/400000 [00:41<00:06, 8505.73it/s] 87%|████████▋ | 347244/400000 [00:41<00:06, 8526.56it/s] 87%|████████▋ | 348102/400000 [00:41<00:06, 8540.28it/s] 87%|████████▋ | 348958/400000 [00:41<00:05, 8545.21it/s] 87%|████████▋ | 349813/400000 [00:41<00:05, 8448.71it/s] 88%|████████▊ | 350664/400000 [00:42<00:05, 8466.18it/s] 88%|████████▊ | 351524/400000 [00:42<00:05, 8503.59it/s] 88%|████████▊ | 352375/400000 [00:42<00:05, 8436.96it/s] 88%|████████▊ | 353228/400000 [00:42<00:05, 8462.25it/s] 89%|████████▊ | 354075/400000 [00:42<00:05, 8377.97it/s] 89%|████████▊ | 354933/400000 [00:42<00:05, 8434.63it/s] 89%|████████▉ | 355788/400000 [00:42<00:05, 8467.39it/s] 89%|████████▉ | 356635/400000 [00:42<00:05, 8413.20it/s] 89%|████████▉ | 357491/400000 [00:42<00:05, 8454.12it/s] 90%|████████▉ | 358337/400000 [00:42<00:04, 8419.11it/s] 90%|████████▉ | 359180/400000 [00:43<00:04, 8371.49it/s] 90%|█████████ | 360028/400000 [00:43<00:04, 8401.53it/s] 90%|█████████ | 360872/400000 [00:43<00:04, 8410.16it/s] 90%|█████████ | 361719/400000 [00:43<00:04, 8427.20it/s] 91%|█████████ | 362562/400000 [00:43<00:04, 8418.33it/s] 91%|█████████ | 363418/400000 [00:43<00:04, 8458.96it/s] 91%|█████████ | 364273/400000 [00:43<00:04, 8485.77it/s] 91%|█████████▏| 365132/400000 [00:43<00:04, 8515.05it/s] 91%|█████████▏| 365992/400000 [00:43<00:03, 8537.67it/s] 92%|█████████▏| 366846/400000 [00:43<00:03, 8425.33it/s] 92%|█████████▏| 367706/400000 [00:44<00:03, 8475.11it/s] 92%|█████████▏| 368564/400000 [00:44<00:03, 8505.81it/s] 92%|█████████▏| 369415/400000 [00:44<00:03, 8427.19it/s] 93%|█████████▎| 370259/400000 [00:44<00:03, 8424.19it/s] 93%|█████████▎| 371102/400000 [00:44<00:03, 8409.80it/s] 93%|█████████▎| 371948/400000 [00:44<00:03, 8423.75it/s] 93%|█████████▎| 372791/400000 [00:44<00:03, 8401.13it/s] 93%|█████████▎| 373633/400000 [00:44<00:03, 8405.44it/s] 94%|█████████▎| 374490/400000 [00:44<00:03, 8452.14it/s] 94%|█████████▍| 375336/400000 [00:44<00:02, 8331.89it/s] 94%|█████████▍| 376192/400000 [00:45<00:02, 8396.48it/s] 94%|█████████▍| 377050/400000 [00:45<00:02, 8450.16it/s] 94%|█████████▍| 377896/400000 [00:45<00:02, 8407.94it/s] 95%|█████████▍| 378748/400000 [00:45<00:02, 8439.13it/s] 95%|█████████▍| 379593/400000 [00:45<00:02, 8364.64it/s] 95%|█████████▌| 380451/400000 [00:45<00:02, 8427.73it/s] 95%|█████████▌| 381303/400000 [00:45<00:02, 8454.73it/s] 96%|█████████▌| 382149/400000 [00:45<00:02, 8444.75it/s] 96%|█████████▌| 382994/400000 [00:45<00:02, 8419.93it/s] 96%|█████████▌| 383837/400000 [00:45<00:01, 8407.10it/s] 96%|█████████▌| 384680/400000 [00:46<00:01, 8413.52it/s] 96%|█████████▋| 385541/400000 [00:46<00:01, 8468.59it/s] 97%|█████████▋| 386389/400000 [00:46<00:01, 8469.60it/s] 97%|█████████▋| 387243/400000 [00:46<00:01, 8487.34it/s] 97%|█████████▋| 388092/400000 [00:46<00:01, 8460.46it/s] 97%|█████████▋| 388939/400000 [00:46<00:01, 8332.80it/s] 97%|█████████▋| 389799/400000 [00:46<00:01, 8409.27it/s] 98%|█████████▊| 390656/400000 [00:46<00:01, 8454.89it/s] 98%|█████████▊| 391512/400000 [00:46<00:01, 8484.96it/s] 98%|█████████▊| 392361/400000 [00:46<00:00, 8392.94it/s] 98%|█████████▊| 393209/400000 [00:47<00:00, 8417.62it/s] 99%|█████████▊| 394057/400000 [00:47<00:00, 8433.45it/s] 99%|█████████▊| 394901/400000 [00:47<00:00, 8355.80it/s] 99%|█████████▉| 395759/400000 [00:47<00:00, 8419.06it/s] 99%|█████████▉| 396602/400000 [00:47<00:00, 8318.24it/s] 99%|█████████▉| 397458/400000 [00:47<00:00, 8389.14it/s]100%|█████████▉| 398298/400000 [00:47<00:00, 8381.91it/s]100%|█████████▉| 399158/400000 [00:47<00:00, 8443.69it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8355.15it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff7653762e8>, <torchtext.data.dataset.TabularDataset object at 0x7ff765376160>, <torchtext.vocab.Vocab object at 0x7ff765376080>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 43.25 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.50 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.50 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 19.50 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.53 file/s]2020-07-01 18:20:00.131038: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 18:20:00.135691: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-01 18:20:00.135865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c8229c21a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 18:20:00.135880: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 20%|██        | 2007040/9912422 [00:00<00:00, 20070275.59it/s]9920512it [00:00, 29409828.64it/s]                             
0it [00:00, ?it/s]32768it [00:00, 443649.14it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 394835.90it/s]1654784it [00:00, 8981031.85it/s]                          
0it [00:00, ?it/s]8192it [00:00, 247188.81it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
