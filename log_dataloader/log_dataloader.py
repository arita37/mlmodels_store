
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5f4e10e0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5f4e10e0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5fb94581e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5fb94581e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5fd779fe18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5fd779fe18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5f66785620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5f66785620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5f66785620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157605.89it/s]  1%|          | 98304/9912422 [00:00<00:48, 202010.98it/s]  4%|▍         | 434176/9912422 [00:00<00:34, 278661.70it/s] 18%|█▊        | 1753088/9912422 [00:00<00:20, 393067.29it/s] 55%|█████▌    | 5488640/9912422 [00:00<00:07, 558089.16it/s] 97%|█████████▋| 9641984/9912422 [00:01<00:00, 791035.34it/s]9920512it [00:01, 9835737.09it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 132345.25it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162005.84it/s]  6%|▌         | 98304/1648877 [00:00<00:07, 206208.56it/s] 26%|██▋       | 434176/1648877 [00:00<00:04, 284247.32it/s]1654784it [00:00, 2554019.18it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 49052.06it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f4d7c4940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f4d7b8be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5f66785268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5f66785268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5f66785268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:24,  1.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:24,  1.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:23,  1.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:22,  1.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<01:41,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:41,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:40,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:39,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<01:38,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<01:09,  2.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:09,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:09,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:08,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:08,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:07,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:07,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:47,  3.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:47,  3.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:46,  3.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:33,  4.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:23,  5.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:23,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:22,  5.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:16,  8.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:16,  8.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:16,  8.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:16,  8.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:15,  8.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:15,  8.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:06, 17.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 21.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 21.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 21.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 21.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 21.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 21.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 30.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 30.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 34.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 34.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 34.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 34.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 34.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 34.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 37.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 37.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 41.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 43.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 43.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 45.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 45.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 46.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 46.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 48.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 48.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 48.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 49.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 49.53 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 49.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 49.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 49.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 49.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 49.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 49.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 49.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 50.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 50.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 49.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 49.30 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 48.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.21s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.97 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.21s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 48.97 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.34 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ url]
0 examples [00:00, ? examples/s]2020-07-09 18:09:39.828402: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-09 18:09:39.851877: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-07-09 18:09:39.852065: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ac0b504440 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-09 18:09:39.852084: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
36 examples [00:00, 358.26 examples/s]139 examples [00:00, 444.83 examples/s]238 examples [00:00, 532.29 examples/s]346 examples [00:00, 627.13 examples/s]452 examples [00:00, 713.68 examples/s]561 examples [00:00, 795.20 examples/s]665 examples [00:00, 855.40 examples/s]761 examples [00:00, 883.89 examples/s]856 examples [00:00, 845.31 examples/s]958 examples [00:01, 891.02 examples/s]1066 examples [00:01, 940.13 examples/s]1172 examples [00:01, 972.19 examples/s]1272 examples [00:01, 954.57 examples/s]1371 examples [00:01, 964.11 examples/s]1477 examples [00:01, 990.26 examples/s]1581 examples [00:01, 1003.78 examples/s]1683 examples [00:01, 996.41 examples/s] 1784 examples [00:01, 992.83 examples/s]1892 examples [00:01, 1016.94 examples/s]1996 examples [00:02, 1021.58 examples/s]2102 examples [00:02, 1032.61 examples/s]2206 examples [00:02, 1028.10 examples/s]2309 examples [00:02, 1024.31 examples/s]2415 examples [00:02, 1033.13 examples/s]2520 examples [00:02, 1037.68 examples/s]2627 examples [00:02, 1045.13 examples/s]2734 examples [00:02, 1051.35 examples/s]2840 examples [00:02, 1021.67 examples/s]2945 examples [00:02, 1028.18 examples/s]3050 examples [00:03, 1034.12 examples/s]3158 examples [00:03, 1045.27 examples/s]3263 examples [00:03, 1046.20 examples/s]3368 examples [00:03, 1045.76 examples/s]3476 examples [00:03, 1052.89 examples/s]3582 examples [00:03, 1054.54 examples/s]3688 examples [00:03, 1050.12 examples/s]3794 examples [00:03, 1046.06 examples/s]3901 examples [00:03, 1051.46 examples/s]4007 examples [00:03, 1016.05 examples/s]4115 examples [00:04, 1032.54 examples/s]4221 examples [00:04, 1039.39 examples/s]4326 examples [00:04, 1018.88 examples/s]4429 examples [00:04, 1006.16 examples/s]4536 examples [00:04, 1022.96 examples/s]4646 examples [00:04, 1042.56 examples/s]4754 examples [00:04, 1051.15 examples/s]4860 examples [00:04, 1053.59 examples/s]4968 examples [00:04, 1059.35 examples/s]5080 examples [00:04, 1076.61 examples/s]5188 examples [00:05, 1077.61 examples/s]5296 examples [00:05, 1078.13 examples/s]5404 examples [00:05, 1068.28 examples/s]5511 examples [00:05, 1054.68 examples/s]5617 examples [00:05, 1044.67 examples/s]5722 examples [00:05, 1030.56 examples/s]5826 examples [00:05, 975.09 examples/s] 5932 examples [00:05, 996.91 examples/s]6033 examples [00:05, 978.36 examples/s]6139 examples [00:06, 1000.25 examples/s]6244 examples [00:06, 1013.52 examples/s]6352 examples [00:06, 1031.62 examples/s]6456 examples [00:06, 1013.60 examples/s]6559 examples [00:06, 1016.38 examples/s]6665 examples [00:06, 1027.57 examples/s]6772 examples [00:06, 1038.81 examples/s]6879 examples [00:06, 1045.68 examples/s]6988 examples [00:06, 1058.24 examples/s]7094 examples [00:06, 1044.38 examples/s]7199 examples [00:07, 1033.84 examples/s]7304 examples [00:07, 1037.58 examples/s]7411 examples [00:07, 1045.46 examples/s]7516 examples [00:07, 1035.86 examples/s]7624 examples [00:07, 1046.15 examples/s]7734 examples [00:07, 1059.20 examples/s]7841 examples [00:07, 1060.81 examples/s]7950 examples [00:07, 1068.61 examples/s]8058 examples [00:07, 1070.51 examples/s]8166 examples [00:07, 1066.02 examples/s]8276 examples [00:08, 1073.64 examples/s]8384 examples [00:08, 1063.24 examples/s]8491 examples [00:08, 1058.06 examples/s]8597 examples [00:08, 1055.05 examples/s]8703 examples [00:08, 1049.55 examples/s]8811 examples [00:08, 1055.90 examples/s]8920 examples [00:08, 1065.52 examples/s]9027 examples [00:08, 1062.74 examples/s]9134 examples [00:08, 1061.87 examples/s]9241 examples [00:08, 1038.98 examples/s]9346 examples [00:09, 1008.60 examples/s]9451 examples [00:09, 1018.63 examples/s]9554 examples [00:09, 1020.10 examples/s]9657 examples [00:09, 954.56 examples/s] 9755 examples [00:09, 959.29 examples/s]9857 examples [00:09, 975.36 examples/s]9956 examples [00:09, 972.76 examples/s]10054 examples [00:09, 903.09 examples/s]10157 examples [00:09, 937.47 examples/s]10252 examples [00:10, 925.09 examples/s]10353 examples [00:10, 948.28 examples/s]10456 examples [00:10, 969.01 examples/s]10554 examples [00:10, 966.83 examples/s]10653 examples [00:10, 972.39 examples/s]10760 examples [00:10, 999.13 examples/s]10865 examples [00:10, 1011.76 examples/s]10970 examples [00:10, 1022.45 examples/s]11074 examples [00:10, 1027.58 examples/s]11178 examples [00:10, 1029.05 examples/s]11282 examples [00:11, 1024.31 examples/s]11385 examples [00:11, 1013.18 examples/s]11487 examples [00:11, 966.74 examples/s] 11592 examples [00:11, 989.74 examples/s]11693 examples [00:11, 994.38 examples/s]11800 examples [00:11, 1015.28 examples/s]11908 examples [00:11, 1031.87 examples/s]12014 examples [00:11, 1037.46 examples/s]12120 examples [00:11, 1043.39 examples/s]12225 examples [00:12, 1030.91 examples/s]12329 examples [00:12, 1009.67 examples/s]12431 examples [00:12, 990.16 examples/s] 12531 examples [00:12, 982.47 examples/s]12630 examples [00:12, 975.22 examples/s]12728 examples [00:12, 969.96 examples/s]12836 examples [00:12, 998.47 examples/s]12945 examples [00:12, 1023.33 examples/s]13049 examples [00:12, 1026.82 examples/s]13152 examples [00:12, 983.46 examples/s] 13258 examples [00:13, 996.38 examples/s]13359 examples [00:13, 962.82 examples/s]13465 examples [00:13, 987.80 examples/s]13571 examples [00:13, 1006.72 examples/s]13676 examples [00:13, 1018.09 examples/s]13783 examples [00:13, 1031.00 examples/s]13887 examples [00:13, 1031.70 examples/s]13994 examples [00:13, 1041.51 examples/s]14101 examples [00:13, 1047.30 examples/s]14208 examples [00:13, 1053.42 examples/s]14315 examples [00:14, 1056.81 examples/s]14421 examples [00:14, 1057.37 examples/s]14527 examples [00:14, 1046.63 examples/s]14632 examples [00:14, 1025.85 examples/s]14739 examples [00:14, 1036.87 examples/s]14846 examples [00:14, 1045.34 examples/s]14955 examples [00:14, 1056.94 examples/s]15064 examples [00:14, 1064.27 examples/s]15173 examples [00:14, 1069.53 examples/s]15281 examples [00:14, 1065.34 examples/s]15388 examples [00:15, 1065.05 examples/s]15495 examples [00:15, 1065.35 examples/s]15602 examples [00:15, 1058.52 examples/s]15708 examples [00:15, 1050.94 examples/s]15814 examples [00:15, 1049.02 examples/s]15919 examples [00:15, 1026.60 examples/s]16022 examples [00:15, 1020.90 examples/s]16125 examples [00:15, 1015.10 examples/s]16227 examples [00:15, 1016.03 examples/s]16335 examples [00:16, 1031.99 examples/s]16439 examples [00:16, 1010.48 examples/s]16542 examples [00:16, 1015.02 examples/s]16644 examples [00:16, 987.86 examples/s] 16750 examples [00:16, 1005.97 examples/s]16857 examples [00:16, 1021.99 examples/s]16964 examples [00:16, 1034.50 examples/s]17072 examples [00:16, 1046.86 examples/s]17179 examples [00:16, 1050.75 examples/s]17285 examples [00:16, 1051.02 examples/s]17392 examples [00:17, 1053.88 examples/s]17498 examples [00:17, 1055.11 examples/s]17604 examples [00:17, 1055.56 examples/s]17710 examples [00:17, 1034.36 examples/s]17814 examples [00:17, 995.22 examples/s] 17923 examples [00:17, 1020.62 examples/s]18032 examples [00:17, 1038.23 examples/s]18139 examples [00:17, 1045.08 examples/s]18244 examples [00:17, 1036.94 examples/s]18352 examples [00:17, 1046.85 examples/s]18457 examples [00:18, 1045.53 examples/s]18562 examples [00:18, 1037.20 examples/s]18668 examples [00:18, 1043.44 examples/s]18773 examples [00:18, 1033.69 examples/s]18882 examples [00:18, 1047.43 examples/s]18987 examples [00:18, 1047.80 examples/s]19092 examples [00:18, 1046.97 examples/s]19197 examples [00:18, 1038.08 examples/s]19305 examples [00:18, 1048.27 examples/s]19413 examples [00:18, 1055.31 examples/s]19519 examples [00:19, 1044.23 examples/s]19624 examples [00:19, 1011.34 examples/s]19726 examples [00:19, 997.40 examples/s] 19832 examples [00:19, 1014.63 examples/s]19938 examples [00:19, 1025.94 examples/s]20041 examples [00:19, 978.62 examples/s] 20142 examples [00:19, 982.37 examples/s]20250 examples [00:19, 1007.57 examples/s]20357 examples [00:19, 1025.51 examples/s]20465 examples [00:20, 1039.75 examples/s]20571 examples [00:20, 1044.73 examples/s]20682 examples [00:20, 1062.96 examples/s]20792 examples [00:20, 1073.04 examples/s]20900 examples [00:20, 1060.35 examples/s]21007 examples [00:20, 1062.89 examples/s]21114 examples [00:20, 1064.83 examples/s]21221 examples [00:20, 1060.00 examples/s]21329 examples [00:20, 1065.20 examples/s]21436 examples [00:20, 1064.72 examples/s]21546 examples [00:21, 1072.85 examples/s]21654 examples [00:21, 1072.05 examples/s]21762 examples [00:21, 1070.32 examples/s]21870 examples [00:21, 1060.62 examples/s]21977 examples [00:21, 1027.11 examples/s]22080 examples [00:21, 1027.02 examples/s]22185 examples [00:21, 1031.67 examples/s]22289 examples [00:21, 1005.52 examples/s]22390 examples [00:21, 1003.97 examples/s]22495 examples [00:21, 1016.66 examples/s]22597 examples [00:22, 969.97 examples/s] 22703 examples [00:22, 994.30 examples/s]22803 examples [00:22, 978.87 examples/s]22902 examples [00:22, 946.55 examples/s]23010 examples [00:22, 980.95 examples/s]23116 examples [00:22, 1002.28 examples/s]23223 examples [00:22, 1020.64 examples/s]23332 examples [00:22, 1038.76 examples/s]23440 examples [00:22, 1049.56 examples/s]23548 examples [00:22, 1057.73 examples/s]23655 examples [00:23, 1057.31 examples/s]23761 examples [00:23, 1054.22 examples/s]23868 examples [00:23, 1056.01 examples/s]23974 examples [00:23, 1031.55 examples/s]24079 examples [00:23, 1035.41 examples/s]24183 examples [00:23, 1021.45 examples/s]24287 examples [00:23, 1025.28 examples/s]24390 examples [00:23, 1024.49 examples/s]24497 examples [00:23, 1037.22 examples/s]24601 examples [00:24, 1037.31 examples/s]24709 examples [00:24, 1047.51 examples/s]24814 examples [00:24, 1046.97 examples/s]24919 examples [00:24, 1015.32 examples/s]25021 examples [00:24, 998.34 examples/s] 25122 examples [00:24, 946.04 examples/s]25220 examples [00:24, 955.38 examples/s]25321 examples [00:24, 969.52 examples/s]25420 examples [00:24, 973.34 examples/s]25522 examples [00:24, 986.57 examples/s]25627 examples [00:25, 996.59 examples/s]25732 examples [00:25, 1009.83 examples/s]25834 examples [00:25, 1006.99 examples/s]25940 examples [00:25, 1020.85 examples/s]26043 examples [00:25, 960.22 examples/s] 26150 examples [00:25, 990.26 examples/s]26254 examples [00:25, 1002.03 examples/s]26360 examples [00:25, 1018.40 examples/s]26463 examples [00:25, 1016.19 examples/s]26565 examples [00:25, 1005.15 examples/s]26671 examples [00:26, 1018.43 examples/s]26778 examples [00:26, 1030.70 examples/s]26882 examples [00:26, 1030.72 examples/s]26986 examples [00:26, 988.47 examples/s] 27086 examples [00:26, 989.24 examples/s]27190 examples [00:26, 1001.68 examples/s]27291 examples [00:26, 992.90 examples/s] 27395 examples [00:26, 1004.72 examples/s]27501 examples [00:26, 1019.32 examples/s]27605 examples [00:27, 1024.75 examples/s]27709 examples [00:27, 1026.81 examples/s]27814 examples [00:27, 1031.22 examples/s]27920 examples [00:27, 1039.20 examples/s]28024 examples [00:27, 1030.32 examples/s]28130 examples [00:27, 1036.44 examples/s]28237 examples [00:27, 1043.96 examples/s]28342 examples [00:27, 1024.09 examples/s]28447 examples [00:27, 1030.95 examples/s]28551 examples [00:27, 990.07 examples/s] 28657 examples [00:28, 1007.58 examples/s]28759 examples [00:28, 1001.34 examples/s]28865 examples [00:28, 1015.79 examples/s]28968 examples [00:28, 1019.31 examples/s]29074 examples [00:28, 1031.16 examples/s]29178 examples [00:28, 976.00 examples/s] 29277 examples [00:28, 973.64 examples/s]29383 examples [00:28, 997.00 examples/s]29487 examples [00:28, 1007.37 examples/s]29591 examples [00:28, 1014.79 examples/s]29693 examples [00:29, 999.90 examples/s] 29798 examples [00:29, 1014.03 examples/s]29900 examples [00:29, 976.10 examples/s] 30000 examples [00:29, 980.47 examples/s]30099 examples [00:29, 944.93 examples/s]30194 examples [00:29, 942.07 examples/s]30289 examples [00:29, 938.41 examples/s]30393 examples [00:29, 965.23 examples/s]30496 examples [00:29, 982.43 examples/s]30604 examples [00:30, 1008.74 examples/s]30711 examples [00:30, 1025.27 examples/s]30817 examples [00:30, 1033.06 examples/s]30921 examples [00:30, 1016.96 examples/s]31028 examples [00:30, 1030.64 examples/s]31136 examples [00:30, 1043.35 examples/s]31243 examples [00:30, 1050.73 examples/s]31349 examples [00:30, 1045.80 examples/s]31454 examples [00:30, 1035.25 examples/s]31560 examples [00:30, 1040.80 examples/s]31665 examples [00:31, 1034.62 examples/s]31769 examples [00:31, 1021.76 examples/s]31874 examples [00:31, 1029.34 examples/s]31978 examples [00:31, 1018.87 examples/s]32080 examples [00:31, 1012.95 examples/s]32182 examples [00:31, 1000.56 examples/s]32288 examples [00:31, 1015.62 examples/s]32394 examples [00:31, 1026.77 examples/s]32499 examples [00:31, 1031.95 examples/s]32605 examples [00:31, 1038.25 examples/s]32709 examples [00:32, 1000.91 examples/s]32810 examples [00:32, 982.70 examples/s] 32910 examples [00:32, 986.37 examples/s]33012 examples [00:32, 991.85 examples/s]33114 examples [00:32, 997.89 examples/s]33218 examples [00:32, 1009.78 examples/s]33320 examples [00:32, 1001.11 examples/s]33426 examples [00:32, 1017.40 examples/s]33531 examples [00:32, 1025.31 examples/s]33636 examples [00:32, 1029.81 examples/s]33742 examples [00:33, 1038.20 examples/s]33848 examples [00:33, 1039.97 examples/s]33954 examples [00:33, 1045.51 examples/s]34059 examples [00:33, 1040.25 examples/s]34164 examples [00:33, 1042.49 examples/s]34271 examples [00:33, 1048.28 examples/s]34376 examples [00:33, 1046.85 examples/s]34483 examples [00:33, 1052.84 examples/s]34589 examples [00:33, 1050.25 examples/s]34695 examples [00:33, 1047.32 examples/s]34800 examples [00:34, 1047.68 examples/s]34905 examples [00:34, 1041.75 examples/s]35013 examples [00:34, 1050.35 examples/s]35119 examples [00:34, 1035.70 examples/s]35223 examples [00:34, 1031.19 examples/s]35327 examples [00:34, 1025.85 examples/s]35435 examples [00:34, 1038.54 examples/s]35543 examples [00:34, 1049.45 examples/s]35649 examples [00:34, 1049.24 examples/s]35754 examples [00:34, 1041.64 examples/s]35859 examples [00:35, 1026.46 examples/s]35962 examples [00:35, 1003.23 examples/s]36068 examples [00:35, 1018.23 examples/s]36172 examples [00:35, 1024.57 examples/s]36277 examples [00:35, 1031.06 examples/s]36382 examples [00:35, 1036.27 examples/s]36486 examples [00:35, 1036.36 examples/s]36592 examples [00:35, 1042.11 examples/s]36697 examples [00:35, 1038.93 examples/s]36803 examples [00:36, 1043.11 examples/s]36908 examples [00:36, 1043.60 examples/s]37015 examples [00:36, 1050.49 examples/s]37121 examples [00:36, 1003.72 examples/s]37227 examples [00:36, 1017.60 examples/s]37335 examples [00:36, 1033.04 examples/s]37444 examples [00:36, 1047.41 examples/s]37550 examples [00:36, 1048.62 examples/s]37656 examples [00:36, 1038.39 examples/s]37760 examples [00:36, 991.71 examples/s] 37863 examples [00:37, 1001.06 examples/s]37968 examples [00:37, 1005.96 examples/s]38073 examples [00:37, 1017.28 examples/s]38175 examples [00:37, 1000.55 examples/s]38276 examples [00:37, 988.87 examples/s] 38379 examples [00:37, 999.92 examples/s]38483 examples [00:37, 1009.28 examples/s]38585 examples [00:37, 1000.29 examples/s]38686 examples [00:37, 990.43 examples/s] 38789 examples [00:37, 1001.10 examples/s]38890 examples [00:38, 991.57 examples/s] 38990 examples [00:38, 948.78 examples/s]39087 examples [00:38, 955.03 examples/s]39186 examples [00:38, 964.82 examples/s]39288 examples [00:38, 978.58 examples/s]39391 examples [00:38, 991.04 examples/s]39493 examples [00:38, 998.96 examples/s]39594 examples [00:38, 1002.04 examples/s]39695 examples [00:38, 988.66 examples/s] 39797 examples [00:39, 997.77 examples/s]39899 examples [00:39, 1003.04 examples/s]40001 examples [00:39, 958.30 examples/s] 40102 examples [00:39, 972.50 examples/s]40200 examples [00:39, 957.75 examples/s]40307 examples [00:39, 987.97 examples/s]40413 examples [00:39, 1007.62 examples/s]40519 examples [00:39, 1020.41 examples/s]40626 examples [00:39, 1032.24 examples/s]40730 examples [00:39, 1019.35 examples/s]40833 examples [00:40, 1019.00 examples/s]40936 examples [00:40, 1009.37 examples/s]41038 examples [00:40, 992.70 examples/s] 41138 examples [00:40, 918.33 examples/s]41238 examples [00:40, 940.13 examples/s]41345 examples [00:40, 975.16 examples/s]41450 examples [00:40, 993.99 examples/s]41551 examples [00:40, 921.61 examples/s]41648 examples [00:40, 935.32 examples/s]41743 examples [00:41, 932.81 examples/s]41844 examples [00:41, 953.27 examples/s]41941 examples [00:41, 948.65 examples/s]42037 examples [00:41, 941.64 examples/s]42136 examples [00:41, 954.58 examples/s]42238 examples [00:41, 973.08 examples/s]42346 examples [00:41, 1001.24 examples/s]42451 examples [00:41, 1012.93 examples/s]42559 examples [00:41, 1031.90 examples/s]42665 examples [00:41, 1039.40 examples/s]42770 examples [00:42, 1037.37 examples/s]42877 examples [00:42, 1045.32 examples/s]42982 examples [00:42, 1006.86 examples/s]43089 examples [00:42, 1024.34 examples/s]43193 examples [00:42, 1026.77 examples/s]43301 examples [00:42, 1039.93 examples/s]43407 examples [00:42, 1043.69 examples/s]43513 examples [00:42, 1047.70 examples/s]43620 examples [00:42, 1053.94 examples/s]43726 examples [00:42, 1048.05 examples/s]43832 examples [00:43, 1050.36 examples/s]43939 examples [00:43, 1053.47 examples/s]44045 examples [00:43, 1024.78 examples/s]44148 examples [00:43, 1007.85 examples/s]44254 examples [00:43, 1020.50 examples/s]44359 examples [00:43, 1027.17 examples/s]44462 examples [00:43, 1014.50 examples/s]44568 examples [00:43, 1026.16 examples/s]44675 examples [00:43, 1036.23 examples/s]44781 examples [00:43, 1042.76 examples/s]44887 examples [00:44, 1047.59 examples/s]44994 examples [00:44, 1053.40 examples/s]45100 examples [00:44, 986.86 examples/s] 45204 examples [00:44, 1001.12 examples/s]45305 examples [00:44, 996.01 examples/s] 45412 examples [00:44, 1015.96 examples/s]45519 examples [00:44, 1030.20 examples/s]45628 examples [00:44, 1045.92 examples/s]45733 examples [00:44, 992.22 examples/s] 45837 examples [00:45, 1005.63 examples/s]45942 examples [00:45, 1017.91 examples/s]46047 examples [00:45, 1025.28 examples/s]46152 examples [00:45, 1031.11 examples/s]46260 examples [00:45, 1044.27 examples/s]46365 examples [00:45, 1045.25 examples/s]46472 examples [00:45, 1052.02 examples/s]46578 examples [00:45, 1053.44 examples/s]46685 examples [00:45, 1057.97 examples/s]46792 examples [00:45, 1059.47 examples/s]46898 examples [00:46, 1044.49 examples/s]47003 examples [00:46, 1040.91 examples/s]47110 examples [00:46, 1048.49 examples/s]47217 examples [00:46, 1052.46 examples/s]47323 examples [00:46, 1036.18 examples/s]47427 examples [00:46, 1031.60 examples/s]47531 examples [00:46, 1033.08 examples/s]47635 examples [00:46, 1013.84 examples/s]47737 examples [00:46, 1007.87 examples/s]47842 examples [00:46, 1017.75 examples/s]47949 examples [00:47, 1030.18 examples/s]48056 examples [00:47, 1039.75 examples/s]48161 examples [00:47, 1027.92 examples/s]48264 examples [00:47, 988.65 examples/s] 48365 examples [00:47, 993.98 examples/s]48467 examples [00:47, 999.21 examples/s]48568 examples [00:47, 1002.20 examples/s]48669 examples [00:47, 999.08 examples/s] 48776 examples [00:47, 1017.59 examples/s]48884 examples [00:47, 1034.53 examples/s]48988 examples [00:48, 1020.44 examples/s]49094 examples [00:48, 1030.16 examples/s]49198 examples [00:48, 1011.06 examples/s]49300 examples [00:48, 985.66 examples/s] 49401 examples [00:48, 990.53 examples/s]49501 examples [00:48, 989.56 examples/s]49601 examples [00:48, 990.27 examples/s]49701 examples [00:48, 984.43 examples/s]49803 examples [00:48, 992.97 examples/s]49903 examples [00:48, 992.79 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5552/50000 [00:00<00:00, 55515.29 examples/s] 34%|███▍      | 16938/50000 [00:00<00:00, 65599.14 examples/s] 60%|█████▉    | 29794/50000 [00:00<00:00, 76896.83 examples/s] 85%|████████▌ | 42709/50000 [00:00<00:00, 87519.22 examples/s]                                                               0 examples [00:00, ? examples/s]84 examples [00:00, 833.94 examples/s]191 examples [00:00, 891.43 examples/s]300 examples [00:00, 941.21 examples/s]406 examples [00:00, 973.95 examples/s]511 examples [00:00, 995.27 examples/s]620 examples [00:00, 1020.54 examples/s]727 examples [00:00, 1032.78 examples/s]825 examples [00:00, 985.97 examples/s] 931 examples [00:00, 1006.81 examples/s]1037 examples [00:01, 1022.01 examples/s]1142 examples [00:01, 1028.91 examples/s]1247 examples [00:01, 1034.83 examples/s]1351 examples [00:01, 1034.92 examples/s]1454 examples [00:01, 1028.13 examples/s]1557 examples [00:01, 1015.13 examples/s]1661 examples [00:01, 1020.11 examples/s]1763 examples [00:01, 1014.32 examples/s]1865 examples [00:01, 1001.40 examples/s]1971 examples [00:01, 1016.51 examples/s]2080 examples [00:02, 1035.83 examples/s]2189 examples [00:02, 1050.52 examples/s]2296 examples [00:02, 1055.25 examples/s]2404 examples [00:02, 1060.41 examples/s]2513 examples [00:02, 1068.00 examples/s]2620 examples [00:02, 1060.64 examples/s]2729 examples [00:02, 1067.27 examples/s]2837 examples [00:02, 1068.84 examples/s]2944 examples [00:02, 1059.69 examples/s]3051 examples [00:02, 1061.33 examples/s]3159 examples [00:03, 1065.60 examples/s]3266 examples [00:03, 1026.47 examples/s]3376 examples [00:03, 1045.90 examples/s]3484 examples [00:03, 1055.13 examples/s]3591 examples [00:03, 1058.34 examples/s]3698 examples [00:03, 1061.21 examples/s]3808 examples [00:03, 1069.90 examples/s]3916 examples [00:03, 1052.31 examples/s]4022 examples [00:03, 1027.39 examples/s]4130 examples [00:03, 1040.53 examples/s]4238 examples [00:04, 1050.35 examples/s]4347 examples [00:04, 1061.50 examples/s]4455 examples [00:04, 1066.78 examples/s]4562 examples [00:04, 1065.16 examples/s]4669 examples [00:04, 1052.02 examples/s]4777 examples [00:04, 1059.12 examples/s]4884 examples [00:04, 1061.21 examples/s]4992 examples [00:04, 1066.41 examples/s]5099 examples [00:04, 1058.99 examples/s]5205 examples [00:04, 1010.86 examples/s]5307 examples [00:05, 993.26 examples/s] 5413 examples [00:05, 1008.84 examples/s]5515 examples [00:05, 1010.87 examples/s]5622 examples [00:05, 1025.72 examples/s]5730 examples [00:05, 1040.06 examples/s]5835 examples [00:05, 1023.18 examples/s]5942 examples [00:05, 1034.22 examples/s]6050 examples [00:05, 1046.98 examples/s]6157 examples [00:05, 1051.55 examples/s]6263 examples [00:06, 1018.86 examples/s]6367 examples [00:06, 1024.57 examples/s]6471 examples [00:06, 1028.64 examples/s]6575 examples [00:06, 1015.77 examples/s]6682 examples [00:06, 1029.57 examples/s]6790 examples [00:06, 1042.49 examples/s]6898 examples [00:06, 1051.92 examples/s]7004 examples [00:06, 1053.40 examples/s]7112 examples [00:06, 1061.00 examples/s]7219 examples [00:06, 1031.63 examples/s]7323 examples [00:07, 1017.16 examples/s]7430 examples [00:07, 1030.10 examples/s]7539 examples [00:07, 1044.84 examples/s]7649 examples [00:07, 1058.60 examples/s]7756 examples [00:07, 1024.11 examples/s]7862 examples [00:07, 1033.02 examples/s]7966 examples [00:07, 1023.27 examples/s]8072 examples [00:07, 1032.00 examples/s]8176 examples [00:07, 1028.50 examples/s]8279 examples [00:07, 1028.94 examples/s]8382 examples [00:08, 1026.51 examples/s]8491 examples [00:08, 1042.19 examples/s]8598 examples [00:08, 1050.29 examples/s]8707 examples [00:08, 1061.67 examples/s]8816 examples [00:08, 1066.91 examples/s]8923 examples [00:08, 1062.83 examples/s]9032 examples [00:08, 1067.87 examples/s]9140 examples [00:08, 1070.70 examples/s]9248 examples [00:08, 1064.78 examples/s]9356 examples [00:08, 1067.52 examples/s]9464 examples [00:09, 1069.97 examples/s]9572 examples [00:09, 1053.53 examples/s]9679 examples [00:09, 1057.09 examples/s]9787 examples [00:09, 1063.16 examples/s]9895 examples [00:09, 1066.53 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteXOWEQU/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteXOWEQU/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5f66785620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5f66785620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5f66785620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ef0999048>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ef0990b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5f66785620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5f66785620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5f66785620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ef0993588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f4d7b8978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5ee838f378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5ee838f378> 

  function with postional parmater data_info <function split_train_valid at 0x7f5ee838f378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5ee838f488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5ee838f488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5ee838f488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=a19d964e6b02d93832448142b01819f4bd93387ecbfd273294e4b767056ed5bd
  Stored in directory: /tmp/pip-ephem-wheel-cache-qey89woi/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<10:20:45, 23.1kB/s].vector_cache/glove.6B.zip:   0%|          | 418k/862M [00:00<7:15:22, 33.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 4.20M/862M [00:00<5:03:32, 47.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.7M/862M [00:00<3:29:54, 67.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:00<2:25:06, 96.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:00<1:40:21, 137kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:00<1:09:23, 196kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:01<48:19, 279kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:01<34:28, 391kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:01<5:54:15, 38.1kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<4:07:54, 54.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<8:44:28, 25.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<6:06:45, 36.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<9:58:00, 22.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:58:08, 32.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<10:27:00, 21.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<7:18:20, 30.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<10:48:01, 20.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:33:01, 29.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<10:53:25, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:36:49, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<10:47:48, 20.5kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:32:51, 29.2kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<10:55:44, 20.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:38:24, 28.8kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<10:55:25, 20.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:38:12, 28.7kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<10:46:18, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:31:46, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<10:52:35, 20.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:36:16, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<10:28:05, 20.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:19:04, 29.7kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<10:40:37, 20.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:27:49, 29.1kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<10:41:15, 20.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:28:17, 29.0kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<10:34:11, 20.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:23:19, 29.2kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<10:39:19, 20.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:27:00, 28.9kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<10:08:02, 21.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:05:02, 30.3kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:18<10:28:02, 20.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:19:00, 29.3kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:35:22, 20.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:24:15, 28.9kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:20<10:03:48, 21.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:02:03, 30.3kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:23:37, 20.5kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:15:59, 29.2kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<10:05:57, 21.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:03:34, 30.0kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:21:11, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:14:17, 29.2kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:01:51, 21.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:00:41, 30.1kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:17:46, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:11:49, 29.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:19:05, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:12:42, 29.1kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:25:58, 20.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:17:31, 28.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:22:38, 20.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<7:15:18, 28.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<9:52:46, 21.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<6:54:19, 30.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:08:43, 20.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:05:28, 29.2kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:12:30, 20.3kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:09:07, 28.9kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<5:55:02, 34.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<4:08:21, 49.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<8:10:18, 25.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<5:42:52, 36.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<8:50:45, 23.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<6:11:01, 33.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<9:33:17, 21.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<6:40:42, 30.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<9:51:10, 20.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<6:53:13, 29.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<9:51:43, 20.7kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<6:53:33, 29.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:00:58, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<7:00:01, 28.9kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:00:46, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:59:54, 28.9kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:53:04, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:54:29, 29.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:58:27, 20.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:58:15, 28.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:53:54, 20.3kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:55:08, 28.9kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:37:07, 20.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:43:20, 29.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:46:49, 20.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:50:06, 29.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:49:11, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:51:47, 28.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:42:36, 20.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:47:13, 29.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:28:56, 20.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:37:41, 29.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:20:50, 21.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:31:56, 30.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:36:53, 20.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:43:13, 29.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:19:38, 21.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:31:06, 30.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:32:41, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<6:40:12, 29.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:36:46, 20.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:43:04, 28.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:31:24, 20.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:39:17, 29.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:36:19, 20.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:42:48, 28.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:16:08, 20.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:28:37, 29.7kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:27:28, 20.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:36:32, 29.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:28:45, 20.3kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:37:32, 28.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:03:20, 21.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:19:40, 30.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<9:18:40, 20.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:30:23, 29.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:21:54, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:32:40, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:16:16, 20.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:28:41, 29.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:20:50, 20.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:31:53, 28.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:20:17, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:31:30, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:14:25, 20.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:27:28, 29.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<8:57:35, 20.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:15:37, 29.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:09:34, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:24:00, 29.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<9:11:16, 20.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:25:17, 28.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<8:45:16, 21.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:07:01, 30.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:01:05, 20.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:18:04, 29.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<9:04:58, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:20:52, 29.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<8:41:47, 21.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:04:38, 30.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<8:37:30, 21.3kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:01:39, 30.3kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<8:36:38, 21.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:00:58, 30.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<8:52:40, 20.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:12:09, 29.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:56:16, 20.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:14:46, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:33:33, 21.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<5:58:48, 30.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:48:28, 20.5kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:09:13, 29.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:51:42, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<6:11:29, 29.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<8:46:24, 20.4kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<6:07:49, 29.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<8:33:19, 20.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<5:58:37, 29.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:43:41, 20.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<6:05:52, 29.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<8:45:09, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<6:06:52, 28.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<8:47:55, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<6:08:49, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<8:42:16, 20.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<6:04:51, 28.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<8:41:21, 20.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<6:04:12, 28.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<8:40:22, 20.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<6:03:32, 28.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<8:34:17, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<5:59:19, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<8:20:11, 20.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<5:49:26, 29.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<8:27:34, 20.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:54:35, 29.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:27:34, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:54:33, 29.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:33:02, 20.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:58:22, 28.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:30:22, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:56:32, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:22:54, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:51:17, 29.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:27:40, 20.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<5:54:37, 28.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<8:26:42, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:36<5:53:57, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<8:17:20, 20.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<5:47:27, 29.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<8:06:38, 20.8kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<5:39:55, 29.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<8:13:24, 20.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<5:44:42, 29.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<7:59:45, 21.0kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<5:35:06, 29.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<8:11:05, 20.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<5:43:00, 29.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<8:12:41, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<5:44:08, 28.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<8:07:09, 20.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<5:40:14, 29.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<8:12:39, 20.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<5:44:09, 28.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<7:52:25, 20.9kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<5:29:57, 29.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<8:02:04, 20.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<5:36:42, 29.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<8:03:21, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<5:37:39, 29.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<7:42:49, 21.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<5:23:14, 30.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<7:54:51, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<5:31:38, 29.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<7:57:03, 20.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<5:33:11, 29.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<7:51:54, 20.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<5:29:34, 29.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<7:56:53, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<5:33:02, 28.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<7:55:42, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<5:32:13, 28.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<7:49:58, 20.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<5:28:12, 29.0kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<7:53:10, 20.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:30:30, 28.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:33:07, 21.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<5:16:29, 29.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<7:28:16, 21.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<5:13:02, 30.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<7:42:01, 20.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<5:22:42, 29.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<7:26:57, 21.0kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<5:12:11, 30.0kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<7:22:04, 21.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<5:08:41, 30.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<7:35:18, 20.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<5:17:56, 29.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<7:36:44, 20.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<5:18:56, 29.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<7:32:24, 20.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<5:15:54, 29.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<7:34:53, 20.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<5:17:37, 28.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<7:35:32, 20.2kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<5:18:05, 28.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<7:29:42, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<5:13:59, 29.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<7:32:40, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<5:16:03, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:30:34, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<5:14:36, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<7:23:40, 20.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<5:09:45, 29.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<7:27:11, 20.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<5:12:13, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<7:25:29, 20.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<5:11:03, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<7:18:33, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<5:06:10, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<7:22:08, 20.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<5:08:44, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<7:04:55, 20.9kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<4:56:43, 29.8kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<6:58:06, 21.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<4:51:53, 30.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<7:09:38, 20.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<4:59:56, 29.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<7:11:54, 20.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<5:01:32, 28.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<7:05:47, 20.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<4:57:14, 29.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<7:10:52, 20.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<5:00:51, 28.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<6:53:41, 20.9kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<4:48:47, 29.9kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<7:02:14, 20.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<4:54:45, 29.1kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<7:04:28, 20.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<4:56:19, 28.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<6:58:18, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<4:51:59, 29.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<7:01:57, 20.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<4:54:32, 28.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<7:03:15, 20.0kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<4:55:27, 28.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<6:55:30, 20.3kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<4:50:01, 29.0kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<6:59:45, 20.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<4:52:59, 28.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<6:55:59, 20.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:28<4:50:21, 28.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<6:56:45, 20.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<4:50:56, 28.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<6:39:50, 20.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<4:39:04, 29.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<6:46:25, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<4:43:40, 29.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<6:47:39, 20.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<4:44:32, 28.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<6:42:36, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:33<4:40:59, 29.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<6:45:23, 20.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<4:42:59, 28.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<6:29:18, 20.9kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<4:31:41, 29.8kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<6:37:06, 20.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<4:37:08, 29.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<6:38:07, 20.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<4:37:51, 28.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<6:33:09, 20.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<4:34:25, 29.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<6:22:35, 20.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<4:27:02, 29.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<6:19:40, 21.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<4:24:57, 29.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<6:27:57, 20.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<4:30:43, 29.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<6:28:52, 20.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<4:31:22, 28.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<6:25:12, 20.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<4:28:47, 29.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<6:28:04, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<4:30:47, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<6:25:21, 20.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<4:28:53, 28.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<6:25:20, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<4:28:52, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<6:23:26, 20.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<4:27:33, 28.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<6:17:01, 20.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<4:23:03, 29.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<6:19:37, 20.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:49<4:24:51, 28.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<6:18:04, 20.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<4:23:47, 28.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<6:11:56, 20.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<4:19:29, 29.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<6:14:37, 20.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<4:21:22, 28.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<6:11:46, 20.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<4:19:24, 28.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<5:59:48, 20.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<4:11:00, 29.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<6:06:11, 20.3kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<4:15:27, 29.0kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<6:04:52, 20.3kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<4:14:31, 28.9kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<6:05:43, 20.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<4:15:10, 28.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<5:52:36, 20.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<4:05:57, 29.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<5:58:23, 20.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:59<4:10:00, 29.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<5:55:51, 20.4kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<4:08:16, 29.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<5:45:23, 20.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<4:00:54, 29.8kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<5:53:08, 20.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<4:06:21, 29.0kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<5:42:33, 20.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<3:58:56, 29.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<5:49:02, 20.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<4:03:26, 29.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<5:50:32, 20.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<4:04:30, 28.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<5:45:20, 20.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<4:00:50, 29.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<5:48:07, 20.2kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<4:02:47, 28.7kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<5:46:27, 20.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<4:01:38, 28.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<5:39:52, 20.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<3:57:03, 29.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<5:31:55, 20.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<3:51:31, 29.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<5:26:54, 21.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<3:47:58, 30.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<5:34:50, 20.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<3:53:29, 29.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<5:35:19, 20.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<3:53:50, 29.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<5:31:06, 20.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<3:50:54, 29.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<5:23:16, 20.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<3:45:27, 29.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:18:52, 21.0kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<3:42:20, 30.0kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<5:26:42, 20.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<3:47:47, 29.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<5:27:12, 20.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<3:48:09, 28.9kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<5:22:27, 20.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<3:44:48, 29.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<5:25:09, 20.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<3:46:41, 28.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<5:23:41, 20.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<3:45:40, 28.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<5:18:14, 20.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:22<3:41:53, 29.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<5:10:09, 20.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<3:36:13, 29.7kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<5:15:40, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<3:40:03, 29.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<5:15:29, 20.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<3:39:56, 28.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<5:11:00, 20.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<3:36:49, 29.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<5:03:30, 20.8kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<3:31:33, 29.7kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<5:09:15, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<3:35:33, 29.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<5:08:21, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:29<3:34:58, 28.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<4:53:25, 21.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:30<3:24:31, 30.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<5:00:45, 20.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<3:29:37, 29.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<5:01:39, 20.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<3:30:14, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<4:58:11, 20.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<3:27:48, 29.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<5:00:23, 20.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:34<3:29:20, 28.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<4:59:16, 20.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<3:28:34, 28.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<4:55:34, 20.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<3:25:59, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<4:45:13, 20.9kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<3:18:44, 29.8kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<4:51:32, 20.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<3:23:08, 29.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<4:52:00, 20.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<3:23:28, 28.8kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<4:46:56, 20.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<3:19:55, 29.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<4:48:43, 20.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<3:21:09, 28.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<4:47:07, 20.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<3:20:02, 28.8kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<4:42:04, 20.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<3:16:30, 29.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<4:43:44, 20.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<3:17:39, 28.8kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<4:44:24, 20.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<3:18:08, 28.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<4:37:58, 20.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<3:13:40, 29.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<4:30:06, 20.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<3:08:11, 29.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<4:25:25, 21.0kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<3:04:52, 30.0kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<4:32:07, 20.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<3:09:32, 29.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<4:32:14, 20.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<3:09:37, 28.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<4:28:19, 20.4kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<3:06:52, 29.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<4:29:35, 20.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<3:07:44, 28.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<4:28:37, 20.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<3:07:04, 28.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<4:24:49, 20.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<3:04:24, 28.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<4:25:21, 20.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<3:04:46, 28.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<4:23:18, 20.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<3:03:23, 28.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<4:09:39, 21.1kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<2:53:49, 30.1kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<4:15:40, 20.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<2:58:01, 29.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<4:16:17, 20.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<2:58:26, 28.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<4:12:40, 20.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<2:55:54, 29.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<4:13:46, 20.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:01<2:56:39, 28.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<4:12:41, 20.2kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<2:55:56, 28.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<4:00:07, 21.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<2:47:08, 30.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<4:05:44, 20.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<2:51:02, 29.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<4:06:00, 20.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:05<2:51:14, 28.9kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<4:02:46, 20.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<2:48:57, 29.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<4:04:04, 20.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<2:49:51, 28.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<4:02:16, 20.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:08<2:48:36, 28.8kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<3:58:22, 20.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:09<2:45:52, 29.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<3:59:19, 20.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<2:46:31, 28.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<3:57:45, 20.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<2:45:26, 28.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<3:53:08, 20.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<2:42:12, 29.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<3:54:06, 20.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:13<2:42:53, 28.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<3:44:06, 20.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<2:35:54, 29.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<3:47:15, 20.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<2:38:05, 29.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<3:47:53, 20.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<2:38:33, 28.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<3:35:56, 21.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<2:30:12, 30.2kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<3:41:26, 20.5kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<2:34:01, 29.2kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<3:42:00, 20.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<2:34:24, 28.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<3:38:53, 20.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<2:32:15, 29.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<3:30:34, 21.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<2:26:25, 30.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<3:35:22, 20.4kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<2:29:45, 29.1kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<3:35:33, 20.2kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<2:29:53, 28.8kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<3:31:32, 20.4kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<2:27:05, 29.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<3:25:33, 20.9kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<2:22:54, 29.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<3:29:24, 20.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<2:25:34, 29.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<3:28:41, 20.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<2:25:04, 28.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<3:24:19, 20.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<2:22:02, 29.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<3:18:16, 20.9kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<2:17:48, 29.9kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<3:21:37, 20.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<2:20:09, 29.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<3:14:41, 21.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<2:15:17, 29.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<3:18:11, 20.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<2:17:43, 29.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<3:16:32, 20.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<2:16:33, 29.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<3:17:11, 20.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<2:17:00, 28.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<3:14:30, 20.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<2:15:07, 28.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<3:14:00, 20.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<2:14:46, 28.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<3:11:51, 20.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<2:13:15, 28.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<3:11:51, 20.0kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<2:13:15, 28.5kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<3:09:04, 20.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<2:11:19, 28.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<3:05:21, 20.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<2:08:42, 29.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<3:05:16, 20.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<2:08:38, 28.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<3:03:53, 20.1kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<2:07:40, 28.7kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<3:00:09, 20.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:43<2:05:03, 29.0kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<3:00:08, 20.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<2:05:02, 28.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<2:58:06, 20.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<2:03:39, 28.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<2:48:33, 21.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<1:56:59, 30.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<2:51:32, 20.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<1:59:04, 29.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<2:45:25, 21.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<1:54:47, 30.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<2:48:48, 20.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<1:57:08, 29.2kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<2:42:33, 21.0kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<1:52:47, 30.0kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<2:39:43, 21.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<1:50:48, 30.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<2:43:26, 20.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<1:53:21, 29.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<2:42:53, 20.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<1:52:58, 29.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<2:40:18, 20.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<1:51:09, 29.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<2:44:21, 19.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<1:53:57, 28.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<2:39:50, 20.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<1:50:48, 28.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<2:38:21, 20.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<1:49:48, 28.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<2:28:58, 21.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<1:43:15, 30.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<2:31:40, 20.4kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<1:45:06, 29.2kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<2:31:15, 20.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<1:44:49, 28.9kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<2:28:32, 20.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<1:42:54, 29.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<2:29:43, 20.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<1:43:42, 28.5kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<2:27:39, 20.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:03<1:42:18, 28.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<2:18:56, 21.1kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<1:36:13, 30.0kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<2:21:11, 20.5kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<1:37:46, 29.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<2:20:38, 20.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<1:37:22, 29.0kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<2:17:51, 20.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:35:25, 29.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<2:18:00, 20.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<1:35:31, 28.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<2:16:27, 20.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<1:34:26, 28.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<2:13:10, 20.4kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<1:32:08, 29.1kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<2:13:08, 20.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<1:32:06, 28.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<2:11:27, 20.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<1:30:56, 28.7kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<2:07:53, 20.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<1:28:26, 29.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<2:07:33, 20.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<1:28:13, 28.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<2:00:14, 21.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<1:23:07, 30.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<2:01:47, 20.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:16<1:24:10, 29.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<2:01:34, 20.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<1:24:00, 29.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<1:59:22, 20.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<1:22:29, 29.1kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<1:54:18, 21.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<1:18:58, 30.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<1:52:00, 21.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<1:17:20, 30.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<1:54:10, 20.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<1:18:49, 29.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<1:52:54, 20.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<1:17:56, 29.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:52:25, 20.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<1:17:36, 28.7kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<1:46:44, 20.9kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:24<1:13:39, 29.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<1:47:15, 20.4kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<1:13:59, 29.2kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<1:46:20, 20.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<1:13:20, 28.9kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<1:44:04, 20.4kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<1:11:45, 29.1kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<1:43:37, 20.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<1:11:25, 28.7kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<1:42:00, 20.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<1:10:17, 28.7kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<1:38:49, 20.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<1:08:04, 29.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<1:35:03, 20.9kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<1:05:27, 29.7kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<1:35:24, 20.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<1:05:40, 29.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<1:34:10, 20.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<1:04:48, 29.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<1:33:08, 20.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<1:04:04, 28.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<1:31:36, 20.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:35<1:02:59, 28.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<1:28:41, 20.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<1:00:57, 29.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<1:27:56, 20.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<1:00:25, 28.8kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<1:26:09, 20.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<59:10, 28.8kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<1:23:41, 20.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<57:27, 29.0kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<1:22:36, 20.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<56:41, 28.8kB/s]  .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<1:20:59, 20.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<55:33, 28.8kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<1:18:20, 20.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<53:42, 29.1kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<1:17:38, 20.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:43<53:13, 28.7kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<1:13:14, 20.9kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<50:10, 29.8kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<1:13:08, 20.4kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<50:04, 29.1kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<1:11:42, 20.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<49:03, 29.0kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<1:10:51, 20.1kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<48:27, 28.6kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<1:08:27, 20.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<46:46, 28.9kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<1:07:16, 20.1kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<45:56, 28.7kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<1:05:11, 20.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<44:28, 28.8kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<1:03:57, 20.1kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<43:36, 28.6kB/s]  .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<1:01:56, 20.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<42:12, 28.7kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<57:28, 21.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<39:07, 30.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<57:22, 20.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<39:01, 29.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<54:18, 21.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<36:54, 30.0kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<54:06, 20.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<36:44, 29.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<52:57, 20.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<35:55, 28.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<49:02, 21.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<33:13, 30.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<48:49, 20.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<33:02, 29.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<45:59, 21.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:00<31:04, 30.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<45:32, 20.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<30:43, 29.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<44:09, 20.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<29:45, 29.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<41:57, 20.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<28:13, 29.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<40:52, 20.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:04<27:26, 28.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<39:20, 20.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<26:22, 28.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<35:42, 21.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:06<23:52, 30.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<35:12, 20.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<23:29, 29.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<33:33, 20.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<22:20, 29.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<32:20, 20.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<21:28, 28.8kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<29:29, 21.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<19:30, 29.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<28:30, 20.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<18:47, 29.2kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<26:41, 20.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<17:31, 29.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<25:25, 20.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<16:36, 28.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<23:43, 20.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<15:25, 28.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<21:49, 20.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<14:05, 29.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<20:19, 20.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:16<13:02, 28.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<18:32, 20.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<11:47, 28.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<16:53, 20.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:18<10:37, 28.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<15:09, 20.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<09:24, 28.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<13:14, 20.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<08:05, 29.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<11:40, 20.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<06:57, 28.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<09:53, 20.1kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<05:43, 28.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<08:04, 20.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<04:27, 29.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<06:23, 20.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:24<03:16, 28.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<04:40, 20.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<02:04, 28.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<02:54, 20.4kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:50, 29.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<01:10, 20.9kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<11:34:54,  9.59it/s]  0%|          | 790/400000 [00:00<8:05:43, 13.70it/s]  0%|          | 1615/400000 [00:00<5:39:32, 19.55it/s]  1%|          | 2482/400000 [00:00<3:57:23, 27.91it/s]  1%|          | 3339/400000 [00:00<2:46:02, 39.81it/s]  1%|          | 4183/400000 [00:00<1:56:13, 56.76it/s]  1%|▏         | 5029/400000 [00:00<1:21:24, 80.86it/s]  1%|▏         | 5874/400000 [00:00<57:06, 115.04it/s]   2%|▏         | 6722/400000 [00:00<40:07, 163.39it/s]  2%|▏         | 7569/400000 [00:01<28:15, 231.50it/s]  2%|▏         | 8404/400000 [00:01<19:58, 326.82it/s]  2%|▏         | 9223/400000 [00:01<14:11, 458.83it/s]  3%|▎         | 10055/400000 [00:01<10:08, 640.32it/s]  3%|▎         | 10896/400000 [00:01<07:19, 885.84it/s]  3%|▎         | 11754/400000 [00:01<05:20, 1211.81it/s]  3%|▎         | 12603/400000 [00:01<03:57, 1631.35it/s]  3%|▎         | 13472/400000 [00:01<02:59, 2156.91it/s]  4%|▎         | 14320/400000 [00:01<02:18, 2777.92it/s]  4%|▍         | 15178/400000 [00:01<01:50, 3484.46it/s]  4%|▍         | 16035/400000 [00:02<01:30, 4238.44it/s]  4%|▍         | 16888/400000 [00:02<01:17, 4973.67it/s]  4%|▍         | 17739/400000 [00:02<01:07, 5681.89it/s]  5%|▍         | 18588/400000 [00:02<01:01, 6222.09it/s]  5%|▍         | 19422/400000 [00:02<00:56, 6704.80it/s]  5%|▌         | 20258/400000 [00:02<00:53, 7126.04it/s]  5%|▌         | 21105/400000 [00:02<00:50, 7482.13it/s]  5%|▌         | 21945/400000 [00:02<00:48, 7735.09it/s]  6%|▌         | 22796/400000 [00:02<00:47, 7951.19it/s]  6%|▌         | 23641/400000 [00:02<00:46, 8093.91it/s]  6%|▌         | 24490/400000 [00:03<00:45, 8208.59it/s]  6%|▋         | 25335/400000 [00:03<00:45, 8266.10it/s]  7%|▋         | 26179/400000 [00:03<00:45, 8172.70it/s]  7%|▋         | 27009/400000 [00:03<00:46, 7981.56it/s]  7%|▋         | 27846/400000 [00:03<00:45, 8093.75it/s]  7%|▋         | 28667/400000 [00:03<00:45, 8127.48it/s]  7%|▋         | 29508/400000 [00:03<00:45, 8208.27it/s]  8%|▊         | 30333/400000 [00:03<00:45, 8112.51it/s]  8%|▊         | 31171/400000 [00:03<00:45, 8188.65it/s]  8%|▊         | 31992/400000 [00:03<00:44, 8191.89it/s]  8%|▊         | 32842/400000 [00:04<00:44, 8279.47it/s]  8%|▊         | 33688/400000 [00:04<00:43, 8331.71it/s]  9%|▊         | 34523/400000 [00:04<00:44, 8241.41it/s]  9%|▉         | 35353/400000 [00:04<00:44, 8258.14it/s]  9%|▉         | 36180/400000 [00:04<00:44, 8227.78it/s]  9%|▉         | 37008/400000 [00:04<00:44, 8240.58it/s]  9%|▉         | 37853/400000 [00:04<00:43, 8301.40it/s] 10%|▉         | 38684/400000 [00:04<00:44, 8205.02it/s] 10%|▉         | 39524/400000 [00:04<00:43, 8260.11it/s] 10%|█         | 40373/400000 [00:04<00:43, 8325.99it/s] 10%|█         | 41227/400000 [00:05<00:42, 8386.49it/s] 11%|█         | 42077/400000 [00:05<00:42, 8418.10it/s] 11%|█         | 42920/400000 [00:05<00:42, 8417.53it/s] 11%|█         | 43762/400000 [00:05<00:42, 8411.40it/s] 11%|█         | 44624/400000 [00:05<00:41, 8470.06it/s] 11%|█▏        | 45472/400000 [00:05<00:41, 8442.03it/s] 12%|█▏        | 46327/400000 [00:05<00:41, 8471.47it/s] 12%|█▏        | 47199/400000 [00:05<00:41, 8542.78it/s] 12%|█▏        | 48054/400000 [00:05<00:41, 8470.76it/s] 12%|█▏        | 48902/400000 [00:05<00:41, 8471.87it/s] 12%|█▏        | 49750/400000 [00:06<00:42, 8242.47it/s] 13%|█▎        | 50610/400000 [00:06<00:41, 8345.94it/s] 13%|█▎        | 51464/400000 [00:06<00:41, 8402.66it/s] 13%|█▎        | 52306/400000 [00:06<00:42, 8188.36it/s] 13%|█▎        | 53174/400000 [00:06<00:41, 8329.52it/s] 14%|█▎        | 54034/400000 [00:06<00:41, 8408.53it/s] 14%|█▎        | 54881/400000 [00:06<00:40, 8425.38it/s] 14%|█▍        | 55731/400000 [00:06<00:40, 8445.87it/s] 14%|█▍        | 56577/400000 [00:06<00:40, 8410.51it/s] 14%|█▍        | 57437/400000 [00:06<00:40, 8464.79it/s] 15%|█▍        | 58321/400000 [00:07<00:39, 8572.02it/s] 15%|█▍        | 59179/400000 [00:07<00:40, 8423.36it/s] 15%|█▌        | 60023/400000 [00:07<00:40, 8394.71it/s] 15%|█▌        | 60864/400000 [00:07<00:40, 8353.44it/s] 15%|█▌        | 61700/400000 [00:07<00:41, 8207.65it/s] 16%|█▌        | 62570/400000 [00:07<00:40, 8347.95it/s] 16%|█▌        | 63437/400000 [00:07<00:39, 8440.74it/s] 16%|█▌        | 64306/400000 [00:07<00:39, 8513.94it/s] 16%|█▋        | 65159/400000 [00:07<00:39, 8419.18it/s] 17%|█▋        | 66015/400000 [00:07<00:39, 8460.40it/s] 17%|█▋        | 66881/400000 [00:08<00:39, 8518.87it/s] 17%|█▋        | 67739/400000 [00:08<00:38, 8536.64it/s] 17%|█▋        | 68605/400000 [00:08<00:38, 8571.60it/s] 17%|█▋        | 69463/400000 [00:08<00:38, 8546.77it/s] 18%|█▊        | 70325/400000 [00:08<00:38, 8566.96it/s] 18%|█▊        | 71189/400000 [00:08<00:38, 8585.98it/s] 18%|█▊        | 72048/400000 [00:08<00:38, 8550.30it/s] 18%|█▊        | 72912/400000 [00:08<00:38, 8574.76it/s] 18%|█▊        | 73770/400000 [00:08<00:38, 8550.41it/s] 19%|█▊        | 74641/400000 [00:08<00:37, 8597.37it/s] 19%|█▉        | 75501/400000 [00:09<00:37, 8566.10it/s] 19%|█▉        | 76359/400000 [00:09<00:37, 8567.90it/s] 19%|█▉        | 77216/400000 [00:09<00:37, 8500.17it/s] 20%|█▉        | 78067/400000 [00:09<00:38, 8383.40it/s] 20%|█▉        | 78927/400000 [00:09<00:38, 8447.16it/s] 20%|█▉        | 79801/400000 [00:09<00:37, 8531.04it/s] 20%|██        | 80669/400000 [00:09<00:37, 8574.71it/s] 20%|██        | 81541/400000 [00:09<00:36, 8615.10it/s] 21%|██        | 82403/400000 [00:09<00:37, 8435.72it/s] 21%|██        | 83248/400000 [00:10<00:37, 8433.94it/s] 21%|██        | 84093/400000 [00:10<00:37, 8421.85it/s] 21%|██        | 84949/400000 [00:10<00:37, 8462.10it/s] 21%|██▏       | 85796/400000 [00:10<00:37, 8380.67it/s] 22%|██▏       | 86635/400000 [00:10<00:38, 8195.61it/s] 22%|██▏       | 87459/400000 [00:10<00:38, 8202.17it/s] 22%|██▏       | 88291/400000 [00:10<00:37, 8235.24it/s] 22%|██▏       | 89116/400000 [00:10<00:38, 8167.13it/s] 22%|██▏       | 89934/400000 [00:10<00:38, 8089.29it/s] 23%|██▎       | 90760/400000 [00:10<00:37, 8138.55it/s] 23%|██▎       | 91575/400000 [00:11<00:40, 7686.80it/s] 23%|██▎       | 92422/400000 [00:11<00:38, 7904.93it/s] 23%|██▎       | 93262/400000 [00:11<00:38, 8046.24it/s] 24%|██▎       | 94102/400000 [00:11<00:37, 8147.98it/s] 24%|██▎       | 94947/400000 [00:11<00:37, 8235.84it/s] 24%|██▍       | 95787/400000 [00:11<00:36, 8283.19it/s] 24%|██▍       | 96618/400000 [00:11<00:36, 8288.91it/s] 24%|██▍       | 97449/400000 [00:11<00:37, 8076.07it/s] 25%|██▍       | 98286/400000 [00:11<00:36, 8161.33it/s] 25%|██▍       | 99135/400000 [00:11<00:36, 8254.58it/s] 25%|██▍       | 99965/400000 [00:12<00:36, 8267.77it/s] 25%|██▌       | 100816/400000 [00:12<00:35, 8336.43it/s] 25%|██▌       | 101654/400000 [00:12<00:35, 8347.48it/s] 26%|██▌       | 102490/400000 [00:12<00:35, 8313.91it/s] 26%|██▌       | 103322/400000 [00:12<00:36, 8095.37it/s] 26%|██▌       | 104147/400000 [00:12<00:36, 8139.73it/s] 26%|██▋       | 105003/400000 [00:12<00:35, 8261.11it/s] 26%|██▋       | 105831/400000 [00:12<00:37, 7924.82it/s] 27%|██▋       | 106681/400000 [00:12<00:36, 8087.15it/s] 27%|██▋       | 107538/400000 [00:12<00:35, 8226.04it/s] 27%|██▋       | 108383/400000 [00:13<00:35, 8290.26it/s] 27%|██▋       | 109215/400000 [00:13<00:35, 8278.33it/s] 28%|██▊       | 110047/400000 [00:13<00:34, 8288.21it/s] 28%|██▊       | 110877/400000 [00:13<00:35, 8151.59it/s] 28%|██▊       | 111694/400000 [00:13<00:36, 7936.24it/s] 28%|██▊       | 112505/400000 [00:13<00:35, 7987.18it/s] 28%|██▊       | 113349/400000 [00:13<00:35, 8117.00it/s] 29%|██▊       | 114197/400000 [00:13<00:34, 8220.29it/s] 29%|██▉       | 115021/400000 [00:13<00:34, 8197.85it/s] 29%|██▉       | 115845/400000 [00:14<00:34, 8209.10it/s] 29%|██▉       | 116672/400000 [00:14<00:34, 8222.94it/s] 29%|██▉       | 117495/400000 [00:14<00:34, 8211.52it/s] 30%|██▉       | 118317/400000 [00:14<00:34, 8209.58it/s] 30%|██▉       | 119139/400000 [00:14<00:34, 8112.93it/s] 30%|██▉       | 119991/400000 [00:14<00:34, 8229.51it/s] 30%|███       | 120836/400000 [00:14<00:33, 8292.81it/s] 30%|███       | 121666/400000 [00:14<00:33, 8207.36it/s] 31%|███       | 122509/400000 [00:14<00:33, 8270.30it/s] 31%|███       | 123337/400000 [00:14<00:33, 8195.41it/s] 31%|███       | 124158/400000 [00:15<00:33, 8159.82it/s] 31%|███▏      | 125003/400000 [00:15<00:33, 8241.47it/s] 31%|███▏      | 125828/400000 [00:15<00:33, 8084.65it/s] 32%|███▏      | 126672/400000 [00:15<00:33, 8187.79it/s] 32%|███▏      | 127495/400000 [00:15<00:33, 8197.47it/s] 32%|███▏      | 128316/400000 [00:15<00:33, 8119.59it/s] 32%|███▏      | 129180/400000 [00:15<00:32, 8267.69it/s] 33%|███▎      | 130022/400000 [00:15<00:32, 8311.15it/s] 33%|███▎      | 130862/400000 [00:15<00:32, 8337.44it/s] 33%|███▎      | 131722/400000 [00:15<00:31, 8414.30it/s] 33%|███▎      | 132565/400000 [00:16<00:31, 8411.56it/s] 33%|███▎      | 133407/400000 [00:16<00:32, 8330.24it/s] 34%|███▎      | 134241/400000 [00:16<00:32, 8147.37it/s] 34%|███▍      | 135097/400000 [00:16<00:32, 8264.38it/s] 34%|███▍      | 135925/400000 [00:16<00:32, 8243.36it/s] 34%|███▍      | 136774/400000 [00:16<00:31, 8313.39it/s] 34%|███▍      | 137613/400000 [00:16<00:31, 8335.41it/s] 35%|███▍      | 138448/400000 [00:16<00:31, 8267.03it/s] 35%|███▍      | 139316/400000 [00:16<00:31, 8385.93it/s] 35%|███▌      | 140156/400000 [00:16<00:31, 8360.27it/s] 35%|███▌      | 141018/400000 [00:17<00:30, 8435.85it/s] 35%|███▌      | 141882/400000 [00:17<00:30, 8495.80it/s] 36%|███▌      | 142733/400000 [00:17<00:30, 8442.93it/s] 36%|███▌      | 143596/400000 [00:17<00:30, 8494.79it/s] 36%|███▌      | 144461/400000 [00:17<00:29, 8540.12it/s] 36%|███▋      | 145316/400000 [00:17<00:29, 8512.10it/s] 37%|███▋      | 146168/400000 [00:17<00:29, 8492.76it/s] 37%|███▋      | 147018/400000 [00:17<00:30, 8346.93it/s] 37%|███▋      | 147857/400000 [00:17<00:30, 8359.20it/s] 37%|███▋      | 148694/400000 [00:17<00:30, 8357.31it/s] 37%|███▋      | 149531/400000 [00:18<00:29, 8353.32it/s] 38%|███▊      | 150388/400000 [00:18<00:29, 8415.56it/s] 38%|███▊      | 151230/400000 [00:18<00:30, 8241.27it/s] 38%|███▊      | 152095/400000 [00:18<00:29, 8359.06it/s] 38%|███▊      | 152958/400000 [00:18<00:29, 8437.92it/s] 38%|███▊      | 153810/400000 [00:18<00:29, 8460.07it/s] 39%|███▊      | 154672/400000 [00:18<00:28, 8506.48it/s] 39%|███▉      | 155524/400000 [00:18<00:28, 8445.52it/s] 39%|███▉      | 156392/400000 [00:18<00:28, 8513.79it/s] 39%|███▉      | 157260/400000 [00:18<00:28, 8562.16it/s] 40%|███▉      | 158117/400000 [00:19<00:28, 8541.57it/s] 40%|███▉      | 158972/400000 [00:19<00:28, 8470.50it/s] 40%|███▉      | 159820/400000 [00:19<00:28, 8466.36it/s] 40%|████      | 160678/400000 [00:19<00:28, 8498.85it/s] 40%|████      | 161537/400000 [00:19<00:27, 8524.66it/s] 41%|████      | 162390/400000 [00:19<00:27, 8522.31it/s] 41%|████      | 163249/400000 [00:19<00:27, 8540.77it/s] 41%|████      | 164104/400000 [00:19<00:27, 8514.74it/s] 41%|████      | 164964/400000 [00:19<00:27, 8537.99it/s] 41%|████▏     | 165821/400000 [00:19<00:27, 8546.74it/s] 42%|████▏     | 166676/400000 [00:20<00:27, 8505.68it/s] 42%|████▏     | 167538/400000 [00:20<00:27, 8538.71it/s] 42%|████▏     | 168392/400000 [00:20<00:27, 8517.46it/s] 42%|████▏     | 169257/400000 [00:20<00:26, 8555.13it/s] 43%|████▎     | 170120/400000 [00:20<00:26, 8574.59it/s] 43%|████▎     | 170978/400000 [00:20<00:26, 8536.86it/s] 43%|████▎     | 171843/400000 [00:20<00:26, 8567.79it/s] 43%|████▎     | 172700/400000 [00:20<00:26, 8540.85it/s] 43%|████▎     | 173564/400000 [00:20<00:26, 8570.01it/s] 44%|████▎     | 174422/400000 [00:20<00:26, 8558.84it/s] 44%|████▍     | 175278/400000 [00:21<00:26, 8513.81it/s] 44%|████▍     | 176134/400000 [00:21<00:26, 8524.97it/s] 44%|████▍     | 176990/400000 [00:21<00:26, 8535.27it/s] 44%|████▍     | 177851/400000 [00:21<00:25, 8554.91it/s] 45%|████▍     | 178716/400000 [00:21<00:25, 8580.15it/s] 45%|████▍     | 179575/400000 [00:21<00:25, 8547.87it/s] 45%|████▌     | 180438/400000 [00:21<00:25, 8570.30it/s] 45%|████▌     | 181296/400000 [00:21<00:25, 8559.10it/s] 46%|████▌     | 182161/400000 [00:21<00:25, 8583.90it/s] 46%|████▌     | 183031/400000 [00:21<00:25, 8618.18it/s] 46%|████▌     | 183893/400000 [00:22<00:25, 8570.91it/s] 46%|████▌     | 184758/400000 [00:22<00:25, 8591.61it/s] 46%|████▋     | 185618/400000 [00:22<00:25, 8489.03it/s] 47%|████▋     | 186469/400000 [00:22<00:25, 8494.87it/s] 47%|████▋     | 187319/400000 [00:22<00:25, 8397.72it/s] 47%|████▋     | 188165/400000 [00:22<00:25, 8414.07it/s] 47%|████▋     | 189034/400000 [00:22<00:24, 8492.14it/s] 47%|████▋     | 189884/400000 [00:22<00:25, 8312.43it/s] 48%|████▊     | 190737/400000 [00:22<00:24, 8376.17it/s] 48%|████▊     | 191586/400000 [00:22<00:24, 8408.77it/s] 48%|████▊     | 192428/400000 [00:23<00:24, 8390.64it/s] 48%|████▊     | 193268/400000 [00:23<00:25, 8199.60it/s] 49%|████▊     | 194096/400000 [00:23<00:25, 8221.88it/s] 49%|████▊     | 194953/400000 [00:23<00:24, 8320.91it/s] 49%|████▉     | 195787/400000 [00:23<00:24, 8309.69it/s] 49%|████▉     | 196619/400000 [00:23<00:24, 8301.42it/s] 49%|████▉     | 197468/400000 [00:23<00:24, 8356.52it/s] 50%|████▉     | 198312/400000 [00:23<00:24, 8380.01it/s] 50%|████▉     | 199153/400000 [00:23<00:23, 8386.21it/s] 50%|████▉     | 199992/400000 [00:24<00:23, 8377.25it/s] 50%|█████     | 200830/400000 [00:24<00:23, 8302.34it/s] 50%|█████     | 201676/400000 [00:24<00:23, 8348.88it/s] 51%|█████     | 202512/400000 [00:24<00:23, 8347.21it/s] 51%|█████     | 203366/400000 [00:24<00:23, 8403.34it/s] 51%|█████     | 204221/400000 [00:24<00:23, 8446.61it/s] 51%|█████▏    | 205066/400000 [00:24<00:23, 8321.75it/s] 51%|█████▏    | 205934/400000 [00:24<00:23, 8425.52it/s] 52%|█████▏    | 206799/400000 [00:24<00:22, 8491.13it/s] 52%|█████▏    | 207666/400000 [00:24<00:22, 8542.28it/s] 52%|█████▏    | 208545/400000 [00:25<00:22, 8613.54it/s] 52%|█████▏    | 209407/400000 [00:25<00:22, 8527.80it/s] 53%|█████▎    | 210261/400000 [00:25<00:22, 8506.46it/s] 53%|█████▎    | 211113/400000 [00:25<00:22, 8474.00it/s] 53%|█████▎    | 211978/400000 [00:25<00:22, 8525.99it/s] 53%|█████▎    | 212831/400000 [00:25<00:22, 8326.00it/s] 53%|█████▎    | 213665/400000 [00:25<00:22, 8192.33it/s] 54%|█████▎    | 214510/400000 [00:25<00:22, 8266.39it/s] 54%|█████▍    | 215357/400000 [00:25<00:22, 8324.16it/s] 54%|█████▍    | 216211/400000 [00:25<00:21, 8385.63it/s] 54%|█████▍    | 217074/400000 [00:26<00:21, 8457.22it/s] 54%|█████▍    | 217921/400000 [00:26<00:21, 8390.29it/s] 55%|█████▍    | 218770/400000 [00:26<00:21, 8418.72it/s] 55%|█████▍    | 219613/400000 [00:26<00:21, 8415.63it/s] 55%|█████▌    | 220464/400000 [00:26<00:21, 8442.38it/s] 55%|█████▌    | 221322/400000 [00:26<00:21, 8481.05it/s] 56%|█████▌    | 222171/400000 [00:26<00:20, 8476.94it/s] 56%|█████▌    | 223022/400000 [00:26<00:20, 8486.42it/s] 56%|█████▌    | 223871/400000 [00:26<00:20, 8454.48it/s] 56%|█████▌    | 224717/400000 [00:26<00:20, 8408.21it/s] 56%|█████▋    | 225575/400000 [00:27<00:20, 8458.54it/s] 57%|█████▋    | 226422/400000 [00:27<00:20, 8417.28it/s] 57%|█████▋    | 227264/400000 [00:27<00:20, 8310.34it/s] 57%|█████▋    | 228120/400000 [00:27<00:20, 8381.90it/s] 57%|█████▋    | 228986/400000 [00:27<00:20, 8463.31it/s] 57%|█████▋    | 229851/400000 [00:27<00:19, 8516.59it/s] 58%|█████▊    | 230709/400000 [00:27<00:19, 8533.81it/s] 58%|█████▊    | 231563/400000 [00:27<00:19, 8451.85it/s] 58%|█████▊    | 232409/400000 [00:27<00:19, 8404.09it/s] 58%|█████▊    | 233255/400000 [00:27<00:19, 8419.00it/s] 59%|█████▊    | 234098/400000 [00:28<00:19, 8407.27it/s] 59%|█████▊    | 234939/400000 [00:28<00:19, 8402.09it/s] 59%|█████▉    | 235780/400000 [00:28<00:19, 8323.98it/s] 59%|█████▉    | 236613/400000 [00:28<00:19, 8256.80it/s] 59%|█████▉    | 237459/400000 [00:28<00:19, 8314.82it/s] 60%|█████▉    | 238291/400000 [00:28<00:19, 8201.08it/s] 60%|█████▉    | 239158/400000 [00:28<00:19, 8333.88it/s] 60%|██████    | 240011/400000 [00:28<00:19, 8389.41it/s] 60%|██████    | 240857/400000 [00:28<00:18, 8408.75it/s] 60%|██████    | 241720/400000 [00:28<00:18, 8472.26it/s] 61%|██████    | 242583/400000 [00:29<00:18, 8517.52it/s] 61%|██████    | 243436/400000 [00:29<00:18, 8501.10it/s] 61%|██████    | 244287/400000 [00:29<00:18, 8414.46it/s] 61%|██████▏   | 245129/400000 [00:29<00:18, 8399.40it/s] 61%|██████▏   | 245981/400000 [00:29<00:18, 8434.83it/s] 62%|██████▏   | 246828/400000 [00:29<00:18, 8443.75it/s] 62%|██████▏   | 247680/400000 [00:29<00:17, 8465.42it/s] 62%|██████▏   | 248527/400000 [00:29<00:18, 8211.95it/s] 62%|██████▏   | 249369/400000 [00:29<00:18, 8273.16it/s] 63%|██████▎   | 250206/400000 [00:29<00:18, 8301.19it/s] 63%|██████▎   | 251047/400000 [00:30<00:17, 8332.88it/s] 63%|██████▎   | 251881/400000 [00:30<00:17, 8272.95it/s] 63%|██████▎   | 252709/400000 [00:30<00:17, 8268.93it/s] 63%|██████▎   | 253554/400000 [00:30<00:17, 8320.36it/s] 64%|██████▎   | 254387/400000 [00:30<00:17, 8307.32it/s] 64%|██████▍   | 255250/400000 [00:30<00:17, 8401.27it/s] 64%|██████▍   | 256106/400000 [00:30<00:17, 8447.22it/s] 64%|██████▍   | 256952/400000 [00:30<00:16, 8437.00it/s] 64%|██████▍   | 257796/400000 [00:30<00:16, 8419.70it/s] 65%|██████▍   | 258639/400000 [00:30<00:17, 8167.44it/s] 65%|██████▍   | 259476/400000 [00:31<00:17, 8225.76it/s] 65%|██████▌   | 260300/400000 [00:31<00:17, 8214.57it/s] 65%|██████▌   | 261156/400000 [00:31<00:16, 8312.90it/s] 66%|██████▌   | 262020/400000 [00:31<00:16, 8406.55it/s] 66%|██████▌   | 262862/400000 [00:31<00:16, 8383.32it/s] 66%|██████▌   | 263716/400000 [00:31<00:16, 8429.57it/s] 66%|██████▌   | 264560/400000 [00:31<00:16, 8425.75it/s] 66%|██████▋   | 265418/400000 [00:31<00:15, 8469.36it/s] 67%|██████▋   | 266270/400000 [00:31<00:15, 8483.52it/s] 67%|██████▋   | 267119/400000 [00:32<00:15, 8419.79it/s] 67%|██████▋   | 267962/400000 [00:32<00:16, 8235.95it/s] 67%|██████▋   | 268810/400000 [00:32<00:15, 8305.01it/s] 67%|██████▋   | 269643/400000 [00:32<00:15, 8310.08it/s] 68%|██████▊   | 270475/400000 [00:32<00:15, 8260.41it/s] 68%|██████▊   | 271323/400000 [00:32<00:15, 8323.69it/s] 68%|██████▊   | 272196/400000 [00:32<00:15, 8441.42it/s] 68%|██████▊   | 273053/400000 [00:32<00:14, 8479.14it/s] 68%|██████▊   | 273906/400000 [00:32<00:14, 8491.80it/s] 69%|██████▊   | 274760/400000 [00:32<00:14, 8504.84it/s] 69%|██████▉   | 275611/400000 [00:33<00:14, 8497.17it/s] 69%|██████▉   | 276470/400000 [00:33<00:14, 8523.73it/s] 69%|██████▉   | 277327/400000 [00:33<00:14, 8534.50it/s] 70%|██████▉   | 278181/400000 [00:33<00:14, 8386.02it/s] 70%|██████▉   | 279039/400000 [00:33<00:14, 8441.83it/s] 70%|██████▉   | 279888/400000 [00:33<00:14, 8455.29it/s] 70%|███████   | 280746/400000 [00:33<00:14, 8491.77it/s] 70%|███████   | 281604/400000 [00:33<00:13, 8515.88it/s] 71%|███████   | 282456/400000 [00:33<00:13, 8449.80it/s] 71%|███████   | 283302/400000 [00:33<00:14, 7946.25it/s] 71%|███████   | 284132/400000 [00:34<00:14, 8046.76it/s] 71%|███████   | 284974/400000 [00:34<00:14, 8152.56it/s] 71%|███████▏  | 285793/400000 [00:34<00:14, 7937.28it/s] 72%|███████▏  | 286592/400000 [00:34<00:14, 7951.78it/s] 72%|███████▏  | 287425/400000 [00:34<00:13, 8059.52it/s] 72%|███████▏  | 288259/400000 [00:34<00:13, 8140.61it/s] 72%|███████▏  | 289078/400000 [00:34<00:13, 8151.58it/s] 72%|███████▏  | 289909/400000 [00:34<00:13, 8197.12it/s] 73%|███████▎  | 290730/400000 [00:34<00:13, 8020.81it/s] 73%|███████▎  | 291571/400000 [00:34<00:13, 8131.28it/s] 73%|███████▎  | 292386/400000 [00:35<00:13, 8108.18it/s] 73%|███████▎  | 293198/400000 [00:35<00:13, 8016.11it/s] 74%|███████▎  | 294024/400000 [00:35<00:13, 8087.46it/s] 74%|███████▎  | 294834/400000 [00:35<00:13, 8030.05it/s] 74%|███████▍  | 295638/400000 [00:35<00:13, 8010.93it/s] 74%|███████▍  | 296440/400000 [00:35<00:12, 8002.81it/s] 74%|███████▍  | 297285/400000 [00:35<00:12, 8129.35it/s] 75%|███████▍  | 298131/400000 [00:35<00:12, 8224.56it/s] 75%|███████▍  | 298955/400000 [00:35<00:12, 8079.96it/s] 75%|███████▍  | 299811/400000 [00:35<00:12, 8216.16it/s] 75%|███████▌  | 300647/400000 [00:36<00:12, 8258.13it/s] 75%|███████▌  | 301474/400000 [00:36<00:12, 8093.46it/s] 76%|███████▌  | 302312/400000 [00:36<00:11, 8174.85it/s] 76%|███████▌  | 303148/400000 [00:36<00:11, 8228.48it/s] 76%|███████▌  | 303977/400000 [00:36<00:11, 8245.77it/s] 76%|███████▌  | 304819/400000 [00:36<00:11, 8297.07it/s] 76%|███████▋  | 305652/400000 [00:36<00:11, 8304.44it/s] 77%|███████▋  | 306499/400000 [00:36<00:11, 8351.87it/s] 77%|███████▋  | 307335/400000 [00:36<00:11, 8338.69it/s] 77%|███████▋  | 308190/400000 [00:36<00:10, 8398.35it/s] 77%|███████▋  | 309052/400000 [00:37<00:10, 8462.09it/s] 77%|███████▋  | 309899/400000 [00:37<00:10, 8349.94it/s] 78%|███████▊  | 310744/400000 [00:37<00:10, 8379.28it/s] 78%|███████▊  | 311584/400000 [00:37<00:10, 8382.82it/s] 78%|███████▊  | 312434/400000 [00:37<00:10, 8416.47it/s] 78%|███████▊  | 313276/400000 [00:37<00:10, 8295.70it/s] 79%|███████▊  | 314120/400000 [00:37<00:10, 8337.72it/s] 79%|███████▊  | 314955/400000 [00:37<00:10, 8223.95it/s] 79%|███████▉  | 315790/400000 [00:37<00:10, 8260.69it/s] 79%|███████▉  | 316637/400000 [00:38<00:10, 8319.79it/s] 79%|███████▉  | 317470/400000 [00:38<00:09, 8257.08it/s] 80%|███████▉  | 318310/400000 [00:38<00:09, 8299.03it/s] 80%|███████▉  | 319169/400000 [00:38<00:09, 8382.87it/s] 80%|████████  | 320013/400000 [00:38<00:09, 8398.10it/s] 80%|████████  | 320874/400000 [00:38<00:09, 8459.21it/s] 80%|████████  | 321736/400000 [00:38<00:09, 8506.42it/s] 81%|████████  | 322591/400000 [00:38<00:09, 8519.09it/s] 81%|████████  | 323455/400000 [00:38<00:08, 8554.29it/s] 81%|████████  | 324311/400000 [00:38<00:08, 8516.13it/s] 81%|████████▏ | 325171/400000 [00:39<00:08, 8538.81it/s] 82%|████████▏ | 326026/400000 [00:39<00:08, 8514.68it/s] 82%|████████▏ | 326878/400000 [00:39<00:08, 8470.08it/s] 82%|████████▏ | 327734/400000 [00:39<00:08, 8496.80it/s] 82%|████████▏ | 328584/400000 [00:39<00:08, 8473.27it/s] 82%|████████▏ | 329444/400000 [00:39<00:08, 8510.83it/s] 83%|████████▎ | 330307/400000 [00:39<00:08, 8543.45it/s] 83%|████████▎ | 331162/400000 [00:39<00:08, 8507.34it/s] 83%|████████▎ | 332025/400000 [00:39<00:07, 8543.26it/s] 83%|████████▎ | 332881/400000 [00:39<00:07, 8547.24it/s] 83%|████████▎ | 333736/400000 [00:40<00:07, 8533.25it/s] 84%|████████▎ | 334590/400000 [00:40<00:07, 8503.08it/s] 84%|████████▍ | 335441/400000 [00:40<00:07, 8448.15it/s] 84%|████████▍ | 336301/400000 [00:40<00:07, 8490.79it/s] 84%|████████▍ | 337151/400000 [00:40<00:07, 8462.64it/s] 84%|████████▍ | 337998/400000 [00:40<00:07, 8420.25it/s] 85%|████████▍ | 338858/400000 [00:40<00:07, 8472.32it/s] 85%|████████▍ | 339718/400000 [00:40<00:07, 8508.31it/s] 85%|████████▌ | 340574/400000 [00:40<00:06, 8522.83it/s] 85%|████████▌ | 341427/400000 [00:40<00:06, 8524.74it/s] 86%|████████▌ | 342285/400000 [00:41<00:06, 8540.43it/s] 86%|████████▌ | 343140/400000 [00:41<00:06, 8367.53it/s] 86%|████████▌ | 343978/400000 [00:41<00:06, 8231.66it/s] 86%|████████▌ | 344805/400000 [00:41<00:06, 8241.57it/s] 86%|████████▋ | 345630/400000 [00:41<00:06, 8220.60it/s] 87%|████████▋ | 346453/400000 [00:41<00:06, 8195.24it/s] 87%|████████▋ | 347300/400000 [00:41<00:06, 8275.56it/s] 87%|████████▋ | 348141/400000 [00:41<00:06, 8312.70it/s] 87%|████████▋ | 348985/400000 [00:41<00:06, 8349.55it/s] 87%|████████▋ | 349833/400000 [00:41<00:05, 8387.22it/s] 88%|████████▊ | 350683/400000 [00:42<00:05, 8418.15it/s] 88%|████████▊ | 351526/400000 [00:42<00:05, 8318.61it/s] 88%|████████▊ | 352371/400000 [00:42<00:05, 8355.31it/s] 88%|████████▊ | 353224/400000 [00:42<00:05, 8404.95it/s] 89%|████████▊ | 354069/400000 [00:42<00:05, 8415.71it/s] 89%|████████▊ | 354925/400000 [00:42<00:05, 8457.30it/s] 89%|████████▉ | 355804/400000 [00:42<00:05, 8553.44it/s] 89%|████████▉ | 356660/400000 [00:42<00:05, 8469.03it/s] 89%|████████▉ | 357522/400000 [00:42<00:04, 8512.25it/s] 90%|████████▉ | 358374/400000 [00:42<00:04, 8492.25it/s] 90%|████████▉ | 359224/400000 [00:43<00:04, 8403.73it/s] 90%|█████████ | 360065/400000 [00:43<00:04, 8372.77it/s] 90%|█████████ | 360903/400000 [00:43<00:04, 8226.08it/s] 90%|█████████ | 361759/400000 [00:43<00:04, 8320.90it/s] 91%|█████████ | 362593/400000 [00:43<00:04, 8324.05it/s] 91%|█████████ | 363434/400000 [00:43<00:04, 8348.52it/s] 91%|█████████ | 364270/400000 [00:43<00:04, 8318.33it/s] 91%|█████████▏| 365107/400000 [00:43<00:04, 8332.64it/s] 91%|█████████▏| 365966/400000 [00:43<00:04, 8407.72it/s] 92%|█████████▏| 366808/400000 [00:43<00:03, 8367.48it/s] 92%|█████████▏| 367659/400000 [00:44<00:03, 8408.12it/s] 92%|█████████▏| 368510/400000 [00:44<00:03, 8436.05it/s] 92%|█████████▏| 369354/400000 [00:44<00:03, 8431.76it/s] 93%|█████████▎| 370220/400000 [00:44<00:03, 8497.26it/s] 93%|█████████▎| 371070/400000 [00:44<00:03, 8437.36it/s] 93%|█████████▎| 371914/400000 [00:44<00:03, 8288.13it/s] 93%|█████████▎| 372769/400000 [00:44<00:03, 8363.57it/s] 93%|█████████▎| 373615/400000 [00:44<00:03, 8391.60it/s] 94%|█████████▎| 374468/400000 [00:44<00:03, 8432.37it/s] 94%|█████████▍| 375312/400000 [00:44<00:02, 8406.90it/s] 94%|█████████▍| 376153/400000 [00:45<00:02, 8208.18it/s] 94%|█████████▍| 376976/400000 [00:45<00:02, 8119.55it/s] 94%|█████████▍| 377790/400000 [00:45<00:02, 8008.05it/s] 95%|█████████▍| 378592/400000 [00:45<00:02, 7917.25it/s] 95%|█████████▍| 379436/400000 [00:45<00:02, 8065.85it/s] 95%|█████████▌| 380290/400000 [00:45<00:02, 8202.09it/s] 95%|█████████▌| 381145/400000 [00:45<00:02, 8300.64it/s] 95%|█████████▌| 381980/400000 [00:45<00:02, 8315.15it/s] 96%|█████████▌| 382833/400000 [00:45<00:02, 8378.11it/s] 96%|█████████▌| 383674/400000 [00:45<00:01, 8385.28it/s] 96%|█████████▌| 384532/400000 [00:46<00:01, 8440.31it/s] 96%|█████████▋| 385383/400000 [00:46<00:01, 8461.04it/s] 97%|█████████▋| 386230/400000 [00:46<00:01, 8437.24it/s] 97%|█████████▋| 387082/400000 [00:46<00:01, 8461.69it/s] 97%|█████████▋| 387936/400000 [00:46<00:01, 8484.30it/s] 97%|█████████▋| 388785/400000 [00:46<00:01, 8422.10it/s] 97%|█████████▋| 389646/400000 [00:46<00:01, 8476.25it/s] 98%|█████████▊| 390494/400000 [00:46<00:01, 8411.04it/s] 98%|█████████▊| 391353/400000 [00:46<00:01, 8462.82it/s] 98%|█████████▊| 392200/400000 [00:46<00:00, 8402.35it/s] 98%|█████████▊| 393041/400000 [00:47<00:00, 8392.17it/s] 98%|█████████▊| 393882/400000 [00:47<00:00, 8396.76it/s] 99%|█████████▊| 394722/400000 [00:47<00:00, 8334.13it/s] 99%|█████████▉| 395583/400000 [00:47<00:00, 8412.71it/s] 99%|█████████▉| 396432/400000 [00:47<00:00, 8435.69it/s] 99%|█████████▉| 397276/400000 [00:47<00:00, 8279.05it/s]100%|█████████▉| 398105/400000 [00:47<00:00, 8273.73it/s]100%|█████████▉| 398936/400000 [00:47<00:00, 8281.97it/s]100%|█████████▉| 399765/400000 [00:47<00:00, 8275.56it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8344.47it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5ee83accc0>, <torchtext.data.dataset.TabularDataset object at 0x7f5ee83ace10>, <torchtext.vocab.Vocab object at 0x7f5ee83acd30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 25.07 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 47.07 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 23.27 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 23.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.81 file/s]2020-07-09 18:18:49.478857: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-09 18:18:49.483368: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-07-09 18:18:49.483570: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561ffe6c6b90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-09 18:18:49.483585: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:34, 289713.97it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 373300.80it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 516361.46it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 729185.25it/s] 76%|███████▌  | 7520256/9912422 [00:00<00:02, 1029990.26it/s]9920512it [00:01, 9781109.81it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 132181.18it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 163061.45it/s]  6%|▌         | 98304/1648877 [00:00<00:07, 207659.53it/s] 26%|██▋       | 434176/1648877 [00:00<00:04, 286127.42it/s]1654784it [00:00, 2589693.92it/s]                           
0it [00:00, ?it/s]8192it [00:00, 90231.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
