
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff6d036bea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff6d036bea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff73b62a1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff73b62a1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff759972e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff759972e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6e8952488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6e8952488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6e8952488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132812.56it/s] 50%|████▉     | 4931584/9912422 [00:00<00:26, 189512.72it/s]9920512it [00:00, 36243723.64it/s]                           
0it [00:00, ?it/s]32768it [00:00, 681801.72it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478284.76it/s]1654784it [00:00, 11664026.24it/s]                         
0it [00:00, ?it/s]8192it [00:00, 261798.46it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6cf9943c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6cf99f630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff6e89520d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff6e89520d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff6e89520d0> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:49,  3.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:49,  3.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:49,  3.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:48,  3.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:34,  4.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:34,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:34,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:34,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:33,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:33,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:33,  4.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  6.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  6.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:16,  8.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:11, 11.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:11, 11.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 15.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:08, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:08, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:08, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:08, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:08, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 19.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 43.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 43.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 53.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 57.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 57.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 60.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.09 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.87s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.87s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.87s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.62 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.87s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.62 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.52 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ url]
0 examples [00:00, ? examples/s]2020-06-22 18:10:11.111876: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-22 18:10:11.126926: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-22 18:10:11.128451: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5645325d6510 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 18:10:11.128476: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
30 examples [00:00, 297.81 examples/s]119 examples [00:00, 371.76 examples/s]201 examples [00:00, 444.10 examples/s]287 examples [00:00, 519.31 examples/s]374 examples [00:00, 589.83 examples/s]461 examples [00:00, 652.67 examples/s]549 examples [00:00, 706.61 examples/s]635 examples [00:00, 745.48 examples/s]729 examples [00:00, 793.71 examples/s]818 examples [00:01, 819.46 examples/s]907 examples [00:01, 837.09 examples/s]994 examples [00:01, 834.79 examples/s]1080 examples [00:01, 812.64 examples/s]1163 examples [00:01, 802.28 examples/s]1257 examples [00:01, 837.90 examples/s]1348 examples [00:01, 856.63 examples/s]1444 examples [00:01, 883.73 examples/s]1538 examples [00:01, 899.65 examples/s]1633 examples [00:01, 913.22 examples/s]1725 examples [00:02, 911.49 examples/s]1817 examples [00:02, 896.66 examples/s]1907 examples [00:02, 881.70 examples/s]2000 examples [00:02, 891.99 examples/s]2096 examples [00:02, 908.57 examples/s]2188 examples [00:02, 908.53 examples/s]2284 examples [00:02, 923.08 examples/s]2377 examples [00:02, 898.28 examples/s]2472 examples [00:02, 912.08 examples/s]2564 examples [00:02, 909.38 examples/s]2658 examples [00:03, 917.85 examples/s]2752 examples [00:03, 923.86 examples/s]2846 examples [00:03, 928.26 examples/s]2939 examples [00:03, 916.62 examples/s]3034 examples [00:03, 924.17 examples/s]3129 examples [00:03, 929.17 examples/s]3222 examples [00:03, 925.75 examples/s]3315 examples [00:03, 905.04 examples/s]3409 examples [00:03, 914.11 examples/s]3505 examples [00:03, 926.87 examples/s]3602 examples [00:04, 939.01 examples/s]3697 examples [00:04, 936.94 examples/s]3791 examples [00:04, 936.25 examples/s]3885 examples [00:04, 920.80 examples/s]3978 examples [00:04, 886.28 examples/s]4069 examples [00:04, 890.98 examples/s]4163 examples [00:04, 903.85 examples/s]4254 examples [00:04, 890.71 examples/s]4349 examples [00:04, 907.53 examples/s]4440 examples [00:05, 899.33 examples/s]4531 examples [00:05, 890.91 examples/s]4621 examples [00:05, 886.16 examples/s]4710 examples [00:05, 878.71 examples/s]4798 examples [00:05, 854.65 examples/s]4892 examples [00:05, 876.27 examples/s]4981 examples [00:05, 878.37 examples/s]5070 examples [00:05, 881.40 examples/s]5159 examples [00:05, 874.15 examples/s]5250 examples [00:05, 882.79 examples/s]5344 examples [00:06, 898.36 examples/s]5434 examples [00:06, 816.87 examples/s]5527 examples [00:06, 847.28 examples/s]5614 examples [00:06, 853.21 examples/s]5702 examples [00:06, 858.95 examples/s]5794 examples [00:06, 874.77 examples/s]5886 examples [00:06, 885.93 examples/s]5981 examples [00:06, 903.13 examples/s]6072 examples [00:06, 900.75 examples/s]6163 examples [00:06, 900.53 examples/s]6256 examples [00:07, 908.24 examples/s]6350 examples [00:07, 915.53 examples/s]6443 examples [00:07, 918.10 examples/s]6535 examples [00:07, 917.62 examples/s]6627 examples [00:07, 912.78 examples/s]6720 examples [00:07, 915.90 examples/s]6814 examples [00:07, 921.13 examples/s]6907 examples [00:07, 904.49 examples/s]7001 examples [00:07, 912.58 examples/s]7098 examples [00:07, 928.25 examples/s]7196 examples [00:08, 941.15 examples/s]7291 examples [00:08, 937.16 examples/s]7385 examples [00:08, 934.36 examples/s]7479 examples [00:08, 924.56 examples/s]7572 examples [00:08, 907.78 examples/s]7663 examples [00:08, 895.03 examples/s]7753 examples [00:08, 886.01 examples/s]7847 examples [00:08, 900.85 examples/s]7938 examples [00:08, 902.63 examples/s]8029 examples [00:09, 900.07 examples/s]8120 examples [00:09, 892.24 examples/s]8215 examples [00:09, 907.93 examples/s]8309 examples [00:09, 915.65 examples/s]8403 examples [00:09, 920.80 examples/s]8496 examples [00:09, 909.10 examples/s]8587 examples [00:09, 901.63 examples/s]8679 examples [00:09, 906.17 examples/s]8771 examples [00:09, 907.83 examples/s]8862 examples [00:09, 876.45 examples/s]8955 examples [00:10, 889.83 examples/s]9045 examples [00:10, 891.25 examples/s]9135 examples [00:10, 882.32 examples/s]9224 examples [00:10, 876.91 examples/s]9312 examples [00:10, 860.57 examples/s]9400 examples [00:10, 866.07 examples/s]9489 examples [00:10, 871.14 examples/s]9583 examples [00:10, 889.86 examples/s]9673 examples [00:10, 881.51 examples/s]9762 examples [00:10, 880.32 examples/s]9854 examples [00:11, 889.79 examples/s]9945 examples [00:11, 894.14 examples/s]10035 examples [00:11, 848.07 examples/s]10127 examples [00:11, 865.90 examples/s]10215 examples [00:11, 852.83 examples/s]10301 examples [00:11, 843.11 examples/s]10387 examples [00:11, 847.90 examples/s]10482 examples [00:11, 873.80 examples/s]10570 examples [00:11, 867.80 examples/s]10660 examples [00:12, 875.40 examples/s]10756 examples [00:12, 896.48 examples/s]10846 examples [00:12, 893.74 examples/s]10936 examples [00:12, 890.19 examples/s]11029 examples [00:12, 900.57 examples/s]11120 examples [00:12, 901.66 examples/s]11213 examples [00:12, 908.42 examples/s]11305 examples [00:12, 909.96 examples/s]11397 examples [00:12, 912.92 examples/s]11494 examples [00:12, 927.75 examples/s]11587 examples [00:13, 926.44 examples/s]11680 examples [00:13, 916.02 examples/s]11772 examples [00:13, 898.19 examples/s]11862 examples [00:13, 878.82 examples/s]11951 examples [00:13, 870.30 examples/s]12039 examples [00:13, 860.08 examples/s]12126 examples [00:13, 828.74 examples/s]12210 examples [00:13, 831.39 examples/s]12303 examples [00:13, 858.63 examples/s]12395 examples [00:13, 875.24 examples/s]12483 examples [00:14, 867.65 examples/s]12575 examples [00:14, 880.14 examples/s]12664 examples [00:14, 872.14 examples/s]12753 examples [00:14, 875.14 examples/s]12841 examples [00:14, 866.57 examples/s]12928 examples [00:14, 850.16 examples/s]13014 examples [00:14, 809.81 examples/s]13097 examples [00:14, 815.40 examples/s]13181 examples [00:14, 821.35 examples/s]13264 examples [00:14, 819.28 examples/s]13347 examples [00:15, 818.97 examples/s]13436 examples [00:15, 837.41 examples/s]13520 examples [00:15, 831.60 examples/s]13614 examples [00:15, 858.10 examples/s]13701 examples [00:15, 855.26 examples/s]13792 examples [00:15, 870.18 examples/s]13881 examples [00:15, 874.40 examples/s]13974 examples [00:15, 889.01 examples/s]14069 examples [00:15, 902.06 examples/s]14160 examples [00:16, 897.78 examples/s]14252 examples [00:16, 901.46 examples/s]14343 examples [00:16, 849.93 examples/s]14429 examples [00:16, 832.20 examples/s]14516 examples [00:16, 841.88 examples/s]14607 examples [00:16, 860.20 examples/s]14700 examples [00:16, 877.82 examples/s]14789 examples [00:16, 874.64 examples/s]14877 examples [00:16, 870.97 examples/s]14965 examples [00:16, 866.59 examples/s]15052 examples [00:17, 850.13 examples/s]15139 examples [00:17, 853.73 examples/s]15225 examples [00:17, 843.12 examples/s]15310 examples [00:17, 841.17 examples/s]15401 examples [00:17, 859.20 examples/s]15491 examples [00:17, 868.87 examples/s]15586 examples [00:17, 890.56 examples/s]15676 examples [00:17, 886.73 examples/s]15765 examples [00:17, 876.61 examples/s]15859 examples [00:17, 894.27 examples/s]15949 examples [00:18, 877.29 examples/s]16041 examples [00:18, 889.49 examples/s]16135 examples [00:18, 902.64 examples/s]16228 examples [00:18, 908.76 examples/s]16320 examples [00:18, 904.61 examples/s]16415 examples [00:18, 916.69 examples/s]16508 examples [00:18, 919.66 examples/s]16602 examples [00:18, 923.53 examples/s]16695 examples [00:18, 897.40 examples/s]16785 examples [00:19, 890.44 examples/s]16875 examples [00:19, 889.08 examples/s]16968 examples [00:19, 899.55 examples/s]17059 examples [00:19, 876.90 examples/s]17149 examples [00:19, 882.23 examples/s]17241 examples [00:19, 892.70 examples/s]17336 examples [00:19, 908.90 examples/s]17432 examples [00:19, 923.56 examples/s]17525 examples [00:19, 894.50 examples/s]17619 examples [00:19, 905.68 examples/s]17710 examples [00:20, 905.67 examples/s]17801 examples [00:20, 872.03 examples/s]17889 examples [00:20, 827.48 examples/s]17979 examples [00:20, 847.48 examples/s]18065 examples [00:20, 844.09 examples/s]18155 examples [00:20, 859.43 examples/s]18244 examples [00:20, 865.78 examples/s]18331 examples [00:20, 855.82 examples/s]18421 examples [00:20, 867.56 examples/s]18516 examples [00:20, 888.62 examples/s]18611 examples [00:21, 903.80 examples/s]18705 examples [00:21, 912.18 examples/s]18803 examples [00:21, 929.44 examples/s]18897 examples [00:21, 904.15 examples/s]18988 examples [00:21, 894.53 examples/s]19078 examples [00:21, 869.19 examples/s]19167 examples [00:21, 874.33 examples/s]19255 examples [00:21, 864.00 examples/s]19347 examples [00:21, 878.37 examples/s]19436 examples [00:22, 880.03 examples/s]19531 examples [00:22, 898.30 examples/s]19624 examples [00:22, 904.54 examples/s]19715 examples [00:22, 904.59 examples/s]19806 examples [00:22, 890.76 examples/s]19896 examples [00:22, 884.69 examples/s]19985 examples [00:22, 880.77 examples/s]20074 examples [00:22, 843.28 examples/s]20161 examples [00:22, 850.00 examples/s]20254 examples [00:22, 869.97 examples/s]20347 examples [00:23, 881.09 examples/s]20438 examples [00:23, 888.99 examples/s]20532 examples [00:23, 902.83 examples/s]20623 examples [00:23, 901.96 examples/s]20714 examples [00:23, 873.99 examples/s]20807 examples [00:23, 887.68 examples/s]20900 examples [00:23, 899.96 examples/s]20995 examples [00:23, 914.40 examples/s]21087 examples [00:23, 902.10 examples/s]21178 examples [00:23, 898.62 examples/s]21271 examples [00:24, 906.33 examples/s]21364 examples [00:24, 910.47 examples/s]21459 examples [00:24, 919.73 examples/s]21554 examples [00:24, 926.49 examples/s]21647 examples [00:24, 914.66 examples/s]21741 examples [00:24, 921.08 examples/s]21834 examples [00:24, 862.38 examples/s]21922 examples [00:24, 853.97 examples/s]22014 examples [00:24, 871.94 examples/s]22106 examples [00:24, 883.87 examples/s]22199 examples [00:25, 894.72 examples/s]22289 examples [00:25, 895.92 examples/s]22379 examples [00:25, 882.32 examples/s]22468 examples [00:25, 870.96 examples/s]22556 examples [00:25, 860.75 examples/s]22643 examples [00:25, 848.95 examples/s]22732 examples [00:25, 858.91 examples/s]22819 examples [00:25, 858.28 examples/s]22905 examples [00:25, 803.66 examples/s]22987 examples [00:26, 807.14 examples/s]23070 examples [00:26, 812.07 examples/s]23155 examples [00:26, 821.81 examples/s]23247 examples [00:26, 847.90 examples/s]23333 examples [00:26, 819.27 examples/s]23416 examples [00:26, 810.85 examples/s]23498 examples [00:26, 804.32 examples/s]23580 examples [00:26, 807.32 examples/s]23666 examples [00:26, 821.47 examples/s]23759 examples [00:26, 850.49 examples/s]23845 examples [00:27, 846.32 examples/s]23930 examples [00:27, 842.45 examples/s]24021 examples [00:27, 859.70 examples/s]24115 examples [00:27, 880.46 examples/s]24204 examples [00:27, 875.15 examples/s]24292 examples [00:27, 860.96 examples/s]24383 examples [00:27, 873.04 examples/s]24471 examples [00:27, 874.67 examples/s]24568 examples [00:27, 900.17 examples/s]24662 examples [00:27, 909.85 examples/s]24755 examples [00:28, 914.45 examples/s]24847 examples [00:28, 907.65 examples/s]24938 examples [00:28, 899.34 examples/s]25029 examples [00:28, 886.85 examples/s]25121 examples [00:28, 892.65 examples/s]25211 examples [00:28, 872.47 examples/s]25299 examples [00:28, 858.39 examples/s]25388 examples [00:28, 865.63 examples/s]25478 examples [00:28, 873.25 examples/s]25566 examples [00:29, 823.75 examples/s]25655 examples [00:29, 841.57 examples/s]25742 examples [00:29, 849.65 examples/s]25835 examples [00:29, 870.41 examples/s]25927 examples [00:29, 883.95 examples/s]26016 examples [00:29, 871.12 examples/s]26105 examples [00:29, 874.44 examples/s]26193 examples [00:29, 813.68 examples/s]26276 examples [00:29, 809.02 examples/s]26360 examples [00:29, 816.31 examples/s]26451 examples [00:30, 840.87 examples/s]26536 examples [00:30, 831.98 examples/s]26621 examples [00:30, 837.26 examples/s]26714 examples [00:30, 862.51 examples/s]26801 examples [00:30, 801.00 examples/s]26885 examples [00:30, 811.51 examples/s]26968 examples [00:30, 814.57 examples/s]27052 examples [00:30, 819.42 examples/s]27138 examples [00:30, 829.86 examples/s]27223 examples [00:31, 834.36 examples/s]27311 examples [00:31, 845.78 examples/s]27399 examples [00:31, 855.62 examples/s]27485 examples [00:31, 853.26 examples/s]27572 examples [00:31, 857.08 examples/s]27658 examples [00:31, 847.22 examples/s]27746 examples [00:31, 855.23 examples/s]27838 examples [00:31, 873.61 examples/s]27926 examples [00:31, 867.80 examples/s]28017 examples [00:31, 878.50 examples/s]28105 examples [00:32, 858.85 examples/s]28195 examples [00:32, 868.53 examples/s]28284 examples [00:32, 872.71 examples/s]28374 examples [00:32, 879.88 examples/s]28463 examples [00:32, 877.82 examples/s]28552 examples [00:32, 880.53 examples/s]28648 examples [00:32, 902.53 examples/s]28740 examples [00:32, 907.20 examples/s]28831 examples [00:32, 903.99 examples/s]28923 examples [00:32, 906.55 examples/s]29018 examples [00:33, 919.06 examples/s]29111 examples [00:33, 920.78 examples/s]29204 examples [00:33, 918.99 examples/s]29298 examples [00:33, 923.19 examples/s]29392 examples [00:33, 927.68 examples/s]29485 examples [00:33, 890.61 examples/s]29575 examples [00:33, 884.81 examples/s]29664 examples [00:33, 871.51 examples/s]29752 examples [00:33, 860.30 examples/s]29839 examples [00:33, 850.64 examples/s]29928 examples [00:34, 860.06 examples/s]30015 examples [00:34, 797.61 examples/s]30099 examples [00:34, 808.87 examples/s]30188 examples [00:34, 831.47 examples/s]30278 examples [00:34, 850.58 examples/s]30371 examples [00:34, 872.74 examples/s]30460 examples [00:34, 876.39 examples/s]30549 examples [00:34, 877.85 examples/s]30638 examples [00:34, 859.95 examples/s]30725 examples [00:35, 838.56 examples/s]30810 examples [00:35, 835.37 examples/s]30902 examples [00:35, 857.46 examples/s]30990 examples [00:35, 863.99 examples/s]31077 examples [00:35, 863.76 examples/s]31171 examples [00:35, 884.62 examples/s]31260 examples [00:35, 883.60 examples/s]31352 examples [00:35, 894.06 examples/s]31442 examples [00:35, 877.15 examples/s]31531 examples [00:35, 880.53 examples/s]31625 examples [00:36, 896.15 examples/s]31720 examples [00:36, 905.42 examples/s]31811 examples [00:36, 900.84 examples/s]31903 examples [00:36, 906.11 examples/s]31994 examples [00:36, 891.45 examples/s]32084 examples [00:36, 857.49 examples/s]32174 examples [00:36, 869.81 examples/s]32264 examples [00:36, 875.72 examples/s]32352 examples [00:36, 854.79 examples/s]32438 examples [00:36, 856.32 examples/s]32526 examples [00:37, 862.94 examples/s]32613 examples [00:37, 852.72 examples/s]32700 examples [00:37, 855.21 examples/s]32786 examples [00:37, 846.69 examples/s]32875 examples [00:37, 857.87 examples/s]32969 examples [00:37, 880.82 examples/s]33063 examples [00:37, 897.76 examples/s]33154 examples [00:37, 897.23 examples/s]33245 examples [00:37, 900.17 examples/s]33342 examples [00:37, 917.01 examples/s]33434 examples [00:38, 886.17 examples/s]33523 examples [00:38, 878.31 examples/s]33616 examples [00:38, 892.88 examples/s]33709 examples [00:38, 902.02 examples/s]33800 examples [00:38, 903.26 examples/s]33892 examples [00:38, 906.11 examples/s]33983 examples [00:38, 904.91 examples/s]34074 examples [00:38, 901.22 examples/s]34165 examples [00:38, 889.88 examples/s]34255 examples [00:39, 882.21 examples/s]34344 examples [00:39, 876.62 examples/s]34433 examples [00:39, 876.73 examples/s]34521 examples [00:39, 870.09 examples/s]34609 examples [00:39, 861.42 examples/s]34703 examples [00:39, 881.82 examples/s]34793 examples [00:39, 885.01 examples/s]34882 examples [00:39, 865.80 examples/s]34969 examples [00:39, 866.99 examples/s]35056 examples [00:39, 860.55 examples/s]35143 examples [00:40, 862.51 examples/s]35230 examples [00:40, 860.94 examples/s]35321 examples [00:40, 873.27 examples/s]35412 examples [00:40, 881.22 examples/s]35502 examples [00:40, 886.18 examples/s]35592 examples [00:40, 887.89 examples/s]35681 examples [00:40, 888.32 examples/s]35772 examples [00:40, 892.10 examples/s]35864 examples [00:40, 898.99 examples/s]35954 examples [00:40, 886.66 examples/s]36047 examples [00:41, 898.93 examples/s]36138 examples [00:41, 902.00 examples/s]36229 examples [00:41, 895.20 examples/s]36319 examples [00:41, 882.33 examples/s]36408 examples [00:41, 868.44 examples/s]36495 examples [00:41, 841.21 examples/s]36582 examples [00:41, 849.22 examples/s]36677 examples [00:41, 874.89 examples/s]36769 examples [00:41, 886.38 examples/s]36858 examples [00:41, 880.09 examples/s]36947 examples [00:42, 876.71 examples/s]37039 examples [00:42, 886.73 examples/s]37128 examples [00:42, 880.70 examples/s]37223 examples [00:42, 900.06 examples/s]37314 examples [00:42, 883.32 examples/s]37404 examples [00:42, 887.02 examples/s]37494 examples [00:42, 890.76 examples/s]37584 examples [00:42, 875.32 examples/s]37672 examples [00:42, 846.75 examples/s]37757 examples [00:43, 825.34 examples/s]37842 examples [00:43, 830.94 examples/s]37934 examples [00:43, 854.62 examples/s]38030 examples [00:43, 881.31 examples/s]38121 examples [00:43, 889.38 examples/s]38211 examples [00:43, 876.98 examples/s]38299 examples [00:43, 868.84 examples/s]38387 examples [00:43, 869.35 examples/s]38475 examples [00:43, 869.49 examples/s]38564 examples [00:43, 874.07 examples/s]38653 examples [00:44, 876.76 examples/s]38744 examples [00:44, 884.44 examples/s]38833 examples [00:44, 859.92 examples/s]38920 examples [00:44, 841.43 examples/s]39007 examples [00:44, 846.26 examples/s]39096 examples [00:44, 858.50 examples/s]39183 examples [00:44, 818.08 examples/s]39266 examples [00:44, 771.32 examples/s]39356 examples [00:44, 804.60 examples/s]39445 examples [00:45, 826.97 examples/s]39539 examples [00:45, 857.05 examples/s]39626 examples [00:45, 847.47 examples/s]39719 examples [00:45, 869.19 examples/s]39813 examples [00:45, 888.02 examples/s]39905 examples [00:45, 896.20 examples/s]39995 examples [00:45, 864.12 examples/s]40082 examples [00:45, 845.10 examples/s]40172 examples [00:45, 859.01 examples/s]40267 examples [00:45, 884.01 examples/s]40362 examples [00:46, 899.80 examples/s]40453 examples [00:46, 892.50 examples/s]40543 examples [00:46, 870.23 examples/s]40632 examples [00:46, 874.06 examples/s]40726 examples [00:46, 891.75 examples/s]40816 examples [00:46, 859.77 examples/s]40907 examples [00:46, 871.93 examples/s]40995 examples [00:46, 841.18 examples/s]41086 examples [00:46, 860.67 examples/s]41179 examples [00:46, 877.49 examples/s]41268 examples [00:47, 864.95 examples/s]41355 examples [00:47, 840.03 examples/s]41440 examples [00:47, 825.17 examples/s]41531 examples [00:47, 846.93 examples/s]41617 examples [00:47, 831.29 examples/s]41701 examples [00:47, 832.86 examples/s]41788 examples [00:47, 842.20 examples/s]41873 examples [00:47, 840.77 examples/s]41958 examples [00:47, 819.39 examples/s]42043 examples [00:48, 828.15 examples/s]42126 examples [00:48, 824.56 examples/s]42213 examples [00:48, 836.95 examples/s]42300 examples [00:48, 845.15 examples/s]42388 examples [00:48, 853.14 examples/s]42476 examples [00:48, 858.94 examples/s]42562 examples [00:48, 833.27 examples/s]42649 examples [00:48, 842.07 examples/s]42740 examples [00:48, 861.16 examples/s]42827 examples [00:48, 855.13 examples/s]42921 examples [00:49, 877.86 examples/s]43011 examples [00:49, 882.06 examples/s]43103 examples [00:49, 891.24 examples/s]43201 examples [00:49, 915.09 examples/s]43295 examples [00:49, 920.88 examples/s]43388 examples [00:49, 872.66 examples/s]43476 examples [00:49, 870.64 examples/s]43564 examples [00:49, 868.02 examples/s]43652 examples [00:49, 867.07 examples/s]43742 examples [00:49, 875.09 examples/s]43830 examples [00:50, 868.91 examples/s]43918 examples [00:50, 837.66 examples/s]44003 examples [00:50, 836.76 examples/s]44087 examples [00:50, 800.77 examples/s]44171 examples [00:50, 809.55 examples/s]44257 examples [00:50, 823.23 examples/s]44344 examples [00:50, 835.47 examples/s]44430 examples [00:50, 841.66 examples/s]44518 examples [00:50, 851.21 examples/s]44614 examples [00:51, 880.25 examples/s]44703 examples [00:51, 877.98 examples/s]44794 examples [00:51, 886.58 examples/s]44883 examples [00:51, 887.58 examples/s]44972 examples [00:51, 888.16 examples/s]45061 examples [00:51, 859.22 examples/s]45148 examples [00:51, 844.88 examples/s]45237 examples [00:51, 857.06 examples/s]45327 examples [00:51, 869.50 examples/s]45415 examples [00:51, 863.70 examples/s]45505 examples [00:52, 873.25 examples/s]45593 examples [00:52, 865.66 examples/s]45685 examples [00:52, 877.33 examples/s]45774 examples [00:52, 879.52 examples/s]45863 examples [00:52, 872.33 examples/s]45951 examples [00:52, 864.46 examples/s]46038 examples [00:52, 857.65 examples/s]46124 examples [00:52, 840.37 examples/s]46210 examples [00:52, 845.48 examples/s]46295 examples [00:52, 846.01 examples/s]46384 examples [00:53, 858.62 examples/s]46471 examples [00:53, 860.62 examples/s]46559 examples [00:53, 865.56 examples/s]46647 examples [00:53, 867.69 examples/s]46734 examples [00:53, 863.53 examples/s]46829 examples [00:53, 885.59 examples/s]46918 examples [00:53, 882.19 examples/s]47007 examples [00:53, 878.59 examples/s]47095 examples [00:53, 865.98 examples/s]47182 examples [00:53, 859.46 examples/s]47272 examples [00:54, 870.58 examples/s]47360 examples [00:54, 862.20 examples/s]47453 examples [00:54, 880.76 examples/s]47543 examples [00:54, 884.20 examples/s]47632 examples [00:54, 836.64 examples/s]47718 examples [00:54, 842.50 examples/s]47805 examples [00:54, 848.98 examples/s]47891 examples [00:54, 823.43 examples/s]47974 examples [00:54, 743.53 examples/s]48068 examples [00:55, 791.90 examples/s]48157 examples [00:55, 816.67 examples/s]48246 examples [00:55, 836.66 examples/s]48332 examples [00:55, 843.05 examples/s]48418 examples [00:55, 847.98 examples/s]48505 examples [00:55, 852.04 examples/s]48596 examples [00:55, 867.98 examples/s]48684 examples [00:55, 869.05 examples/s]48775 examples [00:55, 878.59 examples/s]48868 examples [00:55, 890.29 examples/s]48958 examples [00:56, 892.52 examples/s]49048 examples [00:56, 886.10 examples/s]49137 examples [00:56, 887.25 examples/s]49226 examples [00:56, 876.21 examples/s]49314 examples [00:56, 834.04 examples/s]49398 examples [00:56, 824.49 examples/s]49486 examples [00:56, 838.30 examples/s]49579 examples [00:56, 863.05 examples/s]49672 examples [00:56, 880.73 examples/s]49761 examples [00:56, 881.35 examples/s]49850 examples [00:57, 862.37 examples/s]49937 examples [00:57, 847.35 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6319/50000 [00:00<00:00, 63185.09 examples/s] 35%|███▍      | 17449/50000 [00:00<00:00, 72600.60 examples/s] 56%|█████▌    | 28113/50000 [00:00<00:00, 80288.25 examples/s] 75%|███████▌  | 37667/50000 [00:00<00:00, 84174.17 examples/s] 96%|█████████▌| 47753/50000 [00:00<00:00, 88568.47 examples/s]                                                               0 examples [00:00, ? examples/s]60 examples [00:00, 595.32 examples/s]149 examples [00:00, 660.42 examples/s]236 examples [00:00, 710.10 examples/s]323 examples [00:00, 751.29 examples/s]414 examples [00:00, 792.61 examples/s]505 examples [00:00, 824.06 examples/s]594 examples [00:00, 841.49 examples/s]689 examples [00:00, 869.70 examples/s]782 examples [00:00, 885.40 examples/s]870 examples [00:01, 858.47 examples/s]963 examples [00:01, 876.63 examples/s]1058 examples [00:01, 896.83 examples/s]1148 examples [00:01, 894.40 examples/s]1238 examples [00:01, 887.76 examples/s]1327 examples [00:01, 862.69 examples/s]1420 examples [00:01, 879.72 examples/s]1509 examples [00:01, 873.57 examples/s]1606 examples [00:01, 899.18 examples/s]1698 examples [00:01, 903.68 examples/s]1796 examples [00:02, 924.40 examples/s]1889 examples [00:02, 889.51 examples/s]1979 examples [00:02, 871.22 examples/s]2067 examples [00:02, 864.53 examples/s]2157 examples [00:02, 873.20 examples/s]2252 examples [00:02, 892.88 examples/s]2342 examples [00:02, 894.20 examples/s]2432 examples [00:02, 890.97 examples/s]2524 examples [00:02, 898.79 examples/s]2614 examples [00:02, 888.37 examples/s]2703 examples [00:03, 879.91 examples/s]2792 examples [00:03, 770.50 examples/s]2883 examples [00:03, 805.46 examples/s]2968 examples [00:03, 818.25 examples/s]3061 examples [00:03, 847.16 examples/s]3153 examples [00:03, 866.99 examples/s]3243 examples [00:03, 874.29 examples/s]3332 examples [00:03, 858.72 examples/s]3419 examples [00:03, 831.90 examples/s]3503 examples [00:04, 834.29 examples/s]3590 examples [00:04, 843.28 examples/s]3676 examples [00:04, 845.26 examples/s]3763 examples [00:04, 851.22 examples/s]3852 examples [00:04, 861.09 examples/s]3939 examples [00:04, 838.51 examples/s]4027 examples [00:04, 848.91 examples/s]4113 examples [00:04, 828.99 examples/s]4197 examples [00:04, 828.66 examples/s]4284 examples [00:04, 839.72 examples/s]4370 examples [00:05, 844.59 examples/s]4457 examples [00:05, 849.55 examples/s]4550 examples [00:05, 871.34 examples/s]4643 examples [00:05, 886.17 examples/s]4738 examples [00:05, 901.75 examples/s]4833 examples [00:05, 915.07 examples/s]4925 examples [00:05, 911.00 examples/s]5017 examples [00:05, 892.89 examples/s]5110 examples [00:05, 901.61 examples/s]5202 examples [00:05, 906.15 examples/s]5294 examples [00:06, 908.08 examples/s]5388 examples [00:06, 917.05 examples/s]5480 examples [00:06, 913.41 examples/s]5572 examples [00:06, 908.66 examples/s]5663 examples [00:06, 859.69 examples/s]5750 examples [00:06, 862.37 examples/s]5837 examples [00:06, 856.72 examples/s]5923 examples [00:06, 850.17 examples/s]6015 examples [00:06, 869.67 examples/s]6104 examples [00:07, 875.33 examples/s]6192 examples [00:07, 875.71 examples/s]6280 examples [00:07, 848.08 examples/s]6367 examples [00:07, 854.02 examples/s]6454 examples [00:07, 856.53 examples/s]6540 examples [00:07, 854.42 examples/s]6631 examples [00:07, 868.13 examples/s]6721 examples [00:07, 875.44 examples/s]6809 examples [00:07, 872.94 examples/s]6902 examples [00:07, 887.67 examples/s]6991 examples [00:08, 855.46 examples/s]7078 examples [00:08, 859.26 examples/s]7169 examples [00:08, 873.63 examples/s]7257 examples [00:08, 872.78 examples/s]7348 examples [00:08, 881.04 examples/s]7437 examples [00:08, 877.00 examples/s]7526 examples [00:08, 880.26 examples/s]7615 examples [00:08, 861.30 examples/s]7702 examples [00:08, 841.99 examples/s]7789 examples [00:08, 849.54 examples/s]7875 examples [00:09, 847.82 examples/s]7960 examples [00:09, 844.61 examples/s]8045 examples [00:09, 818.83 examples/s]8130 examples [00:09, 826.89 examples/s]8218 examples [00:09, 840.11 examples/s]8307 examples [00:09, 853.42 examples/s]8393 examples [00:09, 847.04 examples/s]8483 examples [00:09, 859.84 examples/s]8576 examples [00:09, 878.15 examples/s]8665 examples [00:09, 864.04 examples/s]8754 examples [00:10, 870.13 examples/s]8843 examples [00:10, 870.73 examples/s]8931 examples [00:10, 806.90 examples/s]9017 examples [00:10, 821.48 examples/s]9111 examples [00:10, 851.53 examples/s]9199 examples [00:10, 858.44 examples/s]9294 examples [00:10, 882.82 examples/s]9389 examples [00:10, 900.81 examples/s]9480 examples [00:10, 885.13 examples/s]9572 examples [00:11, 894.91 examples/s]9662 examples [00:11, 870.67 examples/s]9750 examples [00:11, 868.50 examples/s]9838 examples [00:11, 842.22 examples/s]9930 examples [00:11, 861.78 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0TWAH5/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0TWAH5/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6e8952488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6e8952488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6e8952488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff671e13278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff671de05c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff6e8952488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff6e8952488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff6e8952488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff671daeba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff6cf99f0b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff6700cc1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff6700cc1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7ff6700cc1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff6700cc2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff6700cc2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff6700cc2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=21ad41aeda4fa18cdd919786079c737d70b846316c0c920046bc168c1f27db4d
  Stored in directory: /tmp/pip-ephem-wheel-cache-5rkvlsw1/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:10:48, 9.90kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:09:23, 14.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:03:46, 19.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:27:05, 28.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:54:02, 40.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.37M/862M [00:01<4:06:16, 57.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.2M/862M [00:01<2:51:29, 82.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:59:38, 118kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.9M/862M [00:01<1:23:20, 168kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<58:12, 239kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.9M/862M [00:02<40:37, 341kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:02<28:24, 485kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<19:51, 690kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:02<13:56, 978kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<09:48, 1.38MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<07:17, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<07:00, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<08:43, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:55, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<05:07, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:46, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:25, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:07<04:54, 2.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:06, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:05, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<05:32, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<04:05, 3.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<07:50, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:50, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<05:06, 2.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:39, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:19, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<05:47, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:09, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<05:39, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<06:02, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<05:32, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<04:12, 3.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:00, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:49, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<05:22, 2.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<03:53, 3.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<42:08, 306kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<9:32:04, 22.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<6:40:49, 32.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<4:39:44, 45.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<3:26:08, 62.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<2:26:58, 87.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<1:43:21, 124kB/s] .vector_cache/glove.6B.zip:  11%|█         | 96.3M/862M [00:22<1:12:18, 177kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<57:17, 223kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<41:23, 308kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<29:15, 435kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<23:21, 543kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<17:40, 717kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<12:39, 999kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<11:48, 1.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:48, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:11, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:45, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:42, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:00, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:25, 1.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:00, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:28, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:57, 3.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<14:59, 828kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:46, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<08:32, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:50, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:41, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:37, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<04:49, 2.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:15, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:02, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:02, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:28, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<03:56, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<22:04, 550kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:41, 726kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<11:58, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:11, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:18, 1.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:43, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<09:31, 1.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:54, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:49, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:50, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:12, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:33, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:02, 2.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<09:15, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:41, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:37, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:42, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:04, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:32, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:59, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<11:21:59, 17.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<7:58:20, 24.5kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<5:34:23, 35.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<3:56:05, 49.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<2:46:21, 70.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<1:56:29, 99.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<1:24:00, 138kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<1:01:07, 189kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<43:20, 267kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<32:04, 359kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<23:35, 488kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<16:44, 686kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<14:22, 797kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<12:22, 925kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:13, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:16, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:58, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:09, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<04:13, 2.68MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<8:30:43, 22.2kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<5:57:45, 31.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<4:09:31, 45.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<3:04:27, 61.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<2:11:28, 85.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:32:26, 121kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<1:04:36, 173kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<53:19, 210kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<38:28, 290kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<27:09, 410kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<21:31, 516kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<17:25, 637kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<12:45, 869kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<09:01, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:25:19, 129kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:00:49, 181kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<42:43, 258kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<32:22, 339kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<24:53, 441kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<17:56, 610kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<14:16, 764kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<10:56, 996kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:58, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<05:42, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<39:52, 272kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<28:49, 376kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<20:24, 529kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<16:47, 641kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<13:54, 773kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<10:14, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<08:53, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:17, 1.47MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:22, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:14, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:31, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:38, 2.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:58, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:04, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:54, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:35, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:43, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:14, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:20, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:37, 2.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:33, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:24, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:44, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:23, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:41, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:19, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:16, 3.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:20, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:09, 2.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:03, 3.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:12, 1.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:46, 1.74MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:17, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:18, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:23, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:09, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:56, 832kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<09:31, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:52, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:09, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:49, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:19, 2.96MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:58, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:12, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:06, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:00, 3.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:29, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:23, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:34, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:19, 2.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<14:31, 665kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<11:10, 865kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<08:03, 1.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:49, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:19, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:19, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<6:45:44, 23.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<4:44:14, 33.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<3:18:10, 47.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<2:24:48, 65.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:43:14, 91.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:12:33, 130kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<50:45, 185kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<39:22, 239kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<28:29, 329kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<20:07, 465kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<16:13, 575kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<13:15, 703kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<09:44, 956kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<08:16, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:43, 1.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:53, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:35, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:46, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:26, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:13, 2.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<07:08, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:56, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:22, 2.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:10, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:33, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:22, 2.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:28, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:56, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:51, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:46, 3.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<13:09, 677kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:06, 881kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:14, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<07:08, 1.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:48, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:09, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:44, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:03, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:09, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:49, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:42, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:11, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:08, 2.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:13, 2.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:53, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:44, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:31, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:33, 3.31MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:20, 815kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<08:05, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:51, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:03, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<05:56, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:31, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:15, 2.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<07:28, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:04, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:25, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:01, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:11, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:02, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:54, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<11:57, 683kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<09:12, 886kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<06:36, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:29, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:11, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:44, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:36, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:54, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:53, 2.76MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<02:07, 3.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<33:36, 237kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<25:08, 316kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<17:54, 443kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<12:35, 628kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<09:49, 803kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<5:44:46, 22.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<4:01:25, 32.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:43<2:48:05, 46.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<2:04:21, 62.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<1:28:37, 88.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<1:02:18, 125kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<44:35, 174kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<31:59, 242kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<22:31, 343kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<17:28, 440kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<13:46, 557kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<10:01, 765kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<08:12, 928kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:30, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:44, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:05, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:56, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<02:49, 2.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<7:14:11, 17.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<5:04:26, 24.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<3:32:32, 35.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<2:29:47, 49.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:46:18, 69.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<1:14:36, 99.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<52:03, 141kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<40:25, 182kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<29:00, 253kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<20:25, 358kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<15:56, 456kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<11:53, 611kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<08:28, 854kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:35, 948kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<06:02, 1.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:24, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:45, 1.50MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:02, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:46, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:03, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:12, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<03:21, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:03, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:17, 3.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:13, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:39, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:51, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:04, 3.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:56, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:50, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:44, 1.83MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:41, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:18, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:27, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:16, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:39, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:51, 2.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:11, 3.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:05, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<02:51, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:08, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:05, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:30, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:44, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<01:59, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:39, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:56, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:53, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:30, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:46, 1.71MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:57, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:07, 3.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<24:44, 258kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<17:58, 354kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<12:41, 500kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<10:20, 610kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:31, 740kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<06:14, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:26, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:49, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:42, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:26, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:50, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:57, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:04, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:11, 2.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<08:24, 727kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<06:30, 937kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<04:40, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:40, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:29, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:26, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:21, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:57, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:11, 2.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:54, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:11, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:29, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<01:48, 3.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:32, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:46, 1.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:46, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:17, 1.75MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:27, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:42, 2.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:33, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:54, 2.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:24, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:49, 3.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:35, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:06, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:25, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:44, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<1:20:55, 67.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<57:12, 95.8kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<39:59, 136kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<29:04, 186kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<20:52, 259kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<14:41, 367kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<11:28, 466kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<08:33, 624kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<06:06, 871kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<05:29, 961kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:54, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:39, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:36, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:14, 994kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:12, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:03, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:20, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:22, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:37, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:53, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<37:36, 135kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<26:43, 190kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<18:44, 269kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<14:12, 352kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<10:57, 457kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<07:53, 632kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<05:31, 894kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<4:49:18, 17.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<3:22:45, 24.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<2:21:18, 34.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<1:39:19, 49.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<1:10:28, 69.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<49:25, 98.2kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<34:18, 140kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<30:25, 158kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<21:45, 220kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<15:16, 312kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<11:43, 404kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<08:40, 545kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<06:08, 765kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<05:22, 867kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<04:42, 991kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:29, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:01, 572kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:05, 753kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:20, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<04:04, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:14, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:42, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<18:01, 247kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<13:02, 341kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<09:11, 481kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<07:25, 591kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<05:34, 787kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:03, 1.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:51, 1.52MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<20:37, 209kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<15:18, 282kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<10:54, 395kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<07:35, 560kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<36:07, 118kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<25:40, 165kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<17:58, 235kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<13:30, 310kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<10:17, 406kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<07:23, 564kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<05:10, 798kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<05:48, 709kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:34, 899kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<03:18, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<02:22, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:16, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:42, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:00, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:27, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:44, 839kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:42, 1.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:41, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:55, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:33, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:21, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:31, 1.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:49, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:39, 1.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:14, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:39, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:02, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:48, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:20, 2.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:48, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:08, 3.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<27:06, 134kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<19:18, 188kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<13:30, 266kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<10:12, 349kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<07:51, 453kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<05:38, 629kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<03:57, 887kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:08, 845kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:15, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:20, 1.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:25, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:23, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:48, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:12, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:36, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:27, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:05, 2.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:30, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:44, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:21, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<00:58, 3.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:05, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:28, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:47, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:58, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:59, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:32, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:05, 2.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<05:12, 580kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:56, 763kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:48, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:37, 1.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:08, 1.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:33, 1.87MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:31, 1.89MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:07, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:26, 1.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:17, 2.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:57, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:11, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:53, 3.04MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:15, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:25, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:07, 2.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:48, 3.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:24, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:57, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:25, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:34, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:21, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:59, 2.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:16, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:50, 2.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:09, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:17, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:00, 2.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:44, 3.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:15, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:06, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:49, 2.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:42, 3.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<1:45:19, 21.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<1:13:25, 30.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<50:14, 43.7kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<40:08, 54.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<28:30, 76.8kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<19:56, 109kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<13:58, 152kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<09:58, 212kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<06:56, 301kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<05:15, 391kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<04:05, 502kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:56, 695kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<02:01, 983kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:58, 400kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<03:39, 540kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<02:34, 757kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:13, 862kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:44, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:15, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:17, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:17, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:58, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:41, 2.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:14, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:02, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:53, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:40, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:46, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:41, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:32, 3.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:38, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:28, 3.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:46, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:39, 2.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:35, 2.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:26, 3.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:37, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:42, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:33, 2.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:23, 3.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:49, 709kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:24, 920kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:59, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:58, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:47, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:34, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:40, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:34, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:25, 2.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:29, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:21, 2.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:29, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:26, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:24, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:17, 3.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:27, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:21, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:20, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 3.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:23, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:17, 2.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:12, 3.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:19, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:09, 3.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:28, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:23, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:16, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:09, 2.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<17:58, 22.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<12:13, 32.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:17<07:46, 46.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<05:17, 64.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<03:44, 90.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<02:31, 128kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:31, 177kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:04, 247kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:41, 350kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:27, 448kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:21, 563kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:14, 771kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:07, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:29, 274kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:20, 376kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:11, 530kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:06, 643kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 774kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.48MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 480/400000 [00:00<01:23, 4794.67it/s]  0%|          | 1221/400000 [00:00<01:14, 5361.02it/s]  0%|          | 1982/400000 [00:00<01:07, 5882.51it/s]  1%|          | 2795/400000 [00:00<01:01, 6413.09it/s]  1%|          | 3581/400000 [00:00<00:58, 6786.69it/s]  1%|          | 4394/400000 [00:00<00:55, 7140.03it/s]  1%|▏         | 5152/400000 [00:00<00:54, 7263.71it/s]  1%|▏         | 5858/400000 [00:00<00:55, 7123.41it/s]  2%|▏         | 6557/400000 [00:00<00:56, 6970.89it/s]  2%|▏         | 7246/400000 [00:01<00:56, 6923.06it/s]  2%|▏         | 8053/400000 [00:01<00:54, 7230.01it/s]  2%|▏         | 8803/400000 [00:01<00:53, 7307.33it/s]  2%|▏         | 9534/400000 [00:01<00:54, 7166.83it/s]  3%|▎         | 10252/400000 [00:01<00:55, 7055.80it/s]  3%|▎         | 10959/400000 [00:01<00:55, 7056.98it/s]  3%|▎         | 11755/400000 [00:01<00:53, 7304.00it/s]  3%|▎         | 12508/400000 [00:01<00:52, 7367.96it/s]  3%|▎         | 13259/400000 [00:01<00:52, 7409.53it/s]  4%|▎         | 14005/400000 [00:01<00:51, 7423.51it/s]  4%|▎         | 14749/400000 [00:02<00:52, 7362.38it/s]  4%|▍         | 15487/400000 [00:02<00:53, 7195.09it/s]  4%|▍         | 16215/400000 [00:02<00:53, 7219.67it/s]  4%|▍         | 16960/400000 [00:02<00:52, 7285.63it/s]  4%|▍         | 17712/400000 [00:02<00:51, 7352.96it/s]  5%|▍         | 18495/400000 [00:02<00:50, 7487.67it/s]  5%|▍         | 19252/400000 [00:02<00:50, 7511.53it/s]  5%|▌         | 20004/400000 [00:02<00:51, 7421.04it/s]  5%|▌         | 20747/400000 [00:02<00:51, 7413.38it/s]  5%|▌         | 21577/400000 [00:02<00:49, 7658.69it/s]  6%|▌         | 22394/400000 [00:03<00:48, 7800.15it/s]  6%|▌         | 23177/400000 [00:03<00:50, 7424.75it/s]  6%|▌         | 23925/400000 [00:03<00:52, 7125.33it/s]  6%|▌         | 24673/400000 [00:03<00:51, 7225.80it/s]  6%|▋         | 25446/400000 [00:03<00:50, 7369.00it/s]  7%|▋         | 26237/400000 [00:03<00:49, 7522.84it/s]  7%|▋         | 26993/400000 [00:03<00:49, 7521.30it/s]  7%|▋         | 27748/400000 [00:03<00:50, 7400.69it/s]  7%|▋         | 28499/400000 [00:03<00:49, 7432.06it/s]  7%|▋         | 29244/400000 [00:03<00:49, 7430.60it/s]  7%|▋         | 29989/400000 [00:04<00:49, 7417.79it/s]  8%|▊         | 30748/400000 [00:04<00:49, 7468.47it/s]  8%|▊         | 31497/400000 [00:04<00:49, 7474.11it/s]  8%|▊         | 32245/400000 [00:04<00:50, 7345.24it/s]  8%|▊         | 32999/400000 [00:04<00:49, 7402.59it/s]  8%|▊         | 33806/400000 [00:04<00:48, 7590.85it/s]  9%|▊         | 34567/400000 [00:04<00:51, 7071.41it/s]  9%|▉         | 35302/400000 [00:04<00:51, 7149.99it/s]  9%|▉         | 36073/400000 [00:04<00:49, 7307.49it/s]  9%|▉         | 36822/400000 [00:05<00:49, 7358.55it/s]  9%|▉         | 37624/400000 [00:05<00:48, 7544.19it/s] 10%|▉         | 38383/400000 [00:05<00:48, 7458.30it/s] 10%|▉         | 39188/400000 [00:05<00:47, 7625.54it/s] 10%|▉         | 39954/400000 [00:05<00:47, 7589.51it/s] 10%|█         | 40716/400000 [00:05<00:47, 7575.41it/s] 10%|█         | 41508/400000 [00:05<00:46, 7674.89it/s] 11%|█         | 42277/400000 [00:05<00:47, 7472.23it/s] 11%|█         | 43027/400000 [00:05<00:47, 7467.84it/s] 11%|█         | 43784/400000 [00:05<00:47, 7495.50it/s] 11%|█         | 44535/400000 [00:06<00:47, 7420.71it/s] 11%|█▏        | 45302/400000 [00:06<00:47, 7491.50it/s] 12%|█▏        | 46096/400000 [00:06<00:46, 7617.25it/s] 12%|█▏        | 46859/400000 [00:06<00:46, 7569.61it/s] 12%|█▏        | 47617/400000 [00:06<00:47, 7377.55it/s] 12%|█▏        | 48371/400000 [00:06<00:47, 7423.05it/s] 12%|█▏        | 49163/400000 [00:06<00:46, 7564.41it/s] 12%|█▏        | 49921/400000 [00:06<00:47, 7373.03it/s] 13%|█▎        | 50661/400000 [00:06<00:48, 7181.28it/s] 13%|█▎        | 51382/400000 [00:06<00:49, 7059.89it/s] 13%|█▎        | 52113/400000 [00:07<00:48, 7131.17it/s] 13%|█▎        | 52828/400000 [00:07<00:50, 6926.05it/s] 13%|█▎        | 53524/400000 [00:07<00:50, 6815.21it/s] 14%|█▎        | 54241/400000 [00:07<00:49, 6917.64it/s] 14%|█▎        | 54935/400000 [00:07<00:51, 6704.11it/s] 14%|█▍        | 55609/400000 [00:07<00:51, 6682.94it/s] 14%|█▍        | 56287/400000 [00:07<00:51, 6710.34it/s] 14%|█▍        | 56960/400000 [00:07<00:51, 6614.47it/s] 14%|█▍        | 57632/400000 [00:07<00:51, 6644.70it/s] 15%|█▍        | 58301/400000 [00:08<00:51, 6655.45it/s] 15%|█▍        | 58968/400000 [00:08<00:51, 6643.55it/s] 15%|█▍        | 59647/400000 [00:08<00:50, 6686.27it/s] 15%|█▌        | 60320/400000 [00:08<00:50, 6697.16it/s] 15%|█▌        | 61094/400000 [00:08<00:48, 6978.97it/s] 15%|█▌        | 61913/400000 [00:08<00:46, 7300.88it/s] 16%|█▌        | 62663/400000 [00:08<00:45, 7358.89it/s] 16%|█▌        | 63404/400000 [00:08<00:46, 7186.98it/s] 16%|█▌        | 64127/400000 [00:08<00:48, 6934.47it/s] 16%|█▌        | 64826/400000 [00:08<00:49, 6811.67it/s] 16%|█▋        | 65513/400000 [00:09<00:48, 6826.82it/s] 17%|█▋        | 66199/400000 [00:09<00:48, 6829.45it/s] 17%|█▋        | 66884/400000 [00:09<00:49, 6764.71it/s] 17%|█▋        | 67624/400000 [00:09<00:47, 6941.61it/s] 17%|█▋        | 68417/400000 [00:09<00:45, 7209.12it/s] 17%|█▋        | 69211/400000 [00:09<00:44, 7411.46it/s] 17%|█▋        | 69963/400000 [00:09<00:44, 7441.32it/s] 18%|█▊        | 70739/400000 [00:09<00:43, 7533.98it/s] 18%|█▊        | 71516/400000 [00:09<00:43, 7601.24it/s] 18%|█▊        | 72336/400000 [00:09<00:42, 7770.66it/s] 18%|█▊        | 73143/400000 [00:10<00:41, 7856.62it/s] 18%|█▊        | 73931/400000 [00:10<00:41, 7863.33it/s] 19%|█▊        | 74719/400000 [00:10<00:41, 7842.31it/s] 19%|█▉        | 75505/400000 [00:10<00:43, 7439.99it/s] 19%|█▉        | 76254/400000 [00:10<00:43, 7360.35it/s] 19%|█▉        | 76994/400000 [00:10<00:44, 7319.08it/s] 19%|█▉        | 77790/400000 [00:10<00:42, 7498.99it/s] 20%|█▉        | 78598/400000 [00:10<00:41, 7664.19it/s] 20%|█▉        | 79415/400000 [00:10<00:41, 7807.83it/s] 20%|██        | 80199/400000 [00:10<00:41, 7749.86it/s] 20%|██        | 80976/400000 [00:11<00:41, 7661.04it/s] 20%|██        | 81744/400000 [00:11<00:43, 7346.64it/s] 21%|██        | 82483/400000 [00:11<00:45, 7051.95it/s] 21%|██        | 83194/400000 [00:11<00:46, 6882.57it/s] 21%|██        | 83887/400000 [00:11<00:45, 6873.94it/s] 21%|██        | 84669/400000 [00:11<00:44, 7130.95it/s] 21%|██▏       | 85501/400000 [00:11<00:42, 7449.14it/s] 22%|██▏       | 86253/400000 [00:11<00:43, 7194.33it/s] 22%|██▏       | 86980/400000 [00:11<00:44, 7067.44it/s] 22%|██▏       | 87692/400000 [00:12<00:45, 6924.98it/s] 22%|██▏       | 88389/400000 [00:12<00:45, 6894.03it/s] 22%|██▏       | 89150/400000 [00:12<00:43, 7092.76it/s] 22%|██▏       | 89870/400000 [00:12<00:43, 7123.34it/s] 23%|██▎       | 90585/400000 [00:12<00:44, 7009.77it/s] 23%|██▎       | 91300/400000 [00:12<00:43, 7049.77it/s] 23%|██▎       | 92007/400000 [00:12<00:44, 6983.90it/s] 23%|██▎       | 92707/400000 [00:12<00:44, 6901.37it/s] 23%|██▎       | 93399/400000 [00:12<00:44, 6857.47it/s] 24%|██▎       | 94165/400000 [00:12<00:43, 7078.60it/s] 24%|██▎       | 94899/400000 [00:13<00:42, 7152.49it/s] 24%|██▍       | 95617/400000 [00:13<00:43, 7011.10it/s] 24%|██▍       | 96321/400000 [00:13<00:43, 6965.70it/s] 24%|██▍       | 97024/400000 [00:13<00:43, 6981.50it/s] 24%|██▍       | 97724/400000 [00:13<00:43, 6893.62it/s] 25%|██▍       | 98415/400000 [00:13<00:44, 6798.27it/s] 25%|██▍       | 99190/400000 [00:13<00:42, 7055.83it/s] 25%|██▍       | 99931/400000 [00:13<00:41, 7155.93it/s] 25%|██▌       | 100662/400000 [00:13<00:41, 7201.36it/s] 25%|██▌       | 101429/400000 [00:13<00:40, 7333.76it/s] 26%|██▌       | 102237/400000 [00:14<00:39, 7541.10it/s] 26%|██▌       | 103059/400000 [00:14<00:38, 7730.62it/s] 26%|██▌       | 103858/400000 [00:14<00:37, 7805.26it/s] 26%|██▌       | 104641/400000 [00:14<00:38, 7703.97it/s] 26%|██▋       | 105459/400000 [00:14<00:37, 7839.16it/s] 27%|██▋       | 106245/400000 [00:14<00:37, 7792.63it/s] 27%|██▋       | 107026/400000 [00:14<00:37, 7791.42it/s] 27%|██▋       | 107807/400000 [00:14<00:37, 7758.00it/s] 27%|██▋       | 108584/400000 [00:14<00:38, 7590.66it/s] 27%|██▋       | 109411/400000 [00:14<00:37, 7780.04it/s] 28%|██▊       | 110212/400000 [00:15<00:36, 7845.25it/s] 28%|██▊       | 111024/400000 [00:15<00:36, 7924.42it/s] 28%|██▊       | 111818/400000 [00:15<00:37, 7772.71it/s] 28%|██▊       | 112597/400000 [00:15<00:38, 7483.22it/s] 28%|██▊       | 113360/400000 [00:15<00:38, 7526.28it/s] 29%|██▊       | 114164/400000 [00:15<00:37, 7673.05it/s] 29%|██▊       | 114957/400000 [00:15<00:36, 7739.32it/s] 29%|██▉       | 115755/400000 [00:15<00:36, 7807.86it/s] 29%|██▉       | 116538/400000 [00:15<00:36, 7781.39it/s] 29%|██▉       | 117318/400000 [00:16<00:36, 7701.60it/s] 30%|██▉       | 118101/400000 [00:16<00:36, 7739.18it/s] 30%|██▉       | 118876/400000 [00:16<00:36, 7683.22it/s] 30%|██▉       | 119645/400000 [00:16<00:36, 7598.61it/s] 30%|███       | 120406/400000 [00:16<00:36, 7600.92it/s] 30%|███       | 121180/400000 [00:16<00:36, 7639.33it/s] 30%|███       | 121945/400000 [00:16<00:36, 7629.02it/s] 31%|███       | 122709/400000 [00:16<00:37, 7484.72it/s] 31%|███       | 123459/400000 [00:16<00:38, 7185.83it/s] 31%|███       | 124181/400000 [00:16<00:39, 7054.97it/s] 31%|███       | 124890/400000 [00:17<00:39, 7036.98it/s] 31%|███▏      | 125646/400000 [00:17<00:38, 7184.91it/s] 32%|███▏      | 126471/400000 [00:17<00:36, 7474.31it/s] 32%|███▏      | 127223/400000 [00:17<00:36, 7484.66it/s] 32%|███▏      | 127991/400000 [00:17<00:36, 7541.20it/s] 32%|███▏      | 128748/400000 [00:17<00:37, 7321.01it/s] 32%|███▏      | 129484/400000 [00:17<00:37, 7173.27it/s] 33%|███▎      | 130205/400000 [00:17<00:37, 7109.38it/s] 33%|███▎      | 130919/400000 [00:17<00:38, 6982.83it/s] 33%|███▎      | 131620/400000 [00:17<00:39, 6800.01it/s] 33%|███▎      | 132303/400000 [00:18<00:39, 6782.61it/s] 33%|███▎      | 133099/400000 [00:18<00:37, 7095.03it/s] 33%|███▎      | 133915/400000 [00:18<00:36, 7383.44it/s] 34%|███▎      | 134660/400000 [00:18<00:36, 7288.33it/s] 34%|███▍      | 135394/400000 [00:18<00:37, 7096.72it/s] 34%|███▍      | 136109/400000 [00:18<00:38, 6944.09it/s] 34%|███▍      | 136872/400000 [00:18<00:36, 7135.81it/s] 34%|███▍      | 137697/400000 [00:18<00:35, 7435.36it/s] 35%|███▍      | 138528/400000 [00:18<00:34, 7677.49it/s] 35%|███▍      | 139328/400000 [00:19<00:33, 7769.42it/s] 35%|███▌      | 140110/400000 [00:19<00:34, 7549.24it/s] 35%|███▌      | 140870/400000 [00:19<00:35, 7280.79it/s] 35%|███▌      | 141604/400000 [00:19<00:36, 7072.12it/s] 36%|███▌      | 142317/400000 [00:19<00:37, 6860.99it/s] 36%|███▌      | 143008/400000 [00:19<00:37, 6816.13it/s] 36%|███▌      | 143694/400000 [00:19<00:38, 6677.05it/s] 36%|███▌      | 144365/400000 [00:19<00:38, 6671.43it/s] 36%|███▋      | 145035/400000 [00:19<00:38, 6669.63it/s] 36%|███▋      | 145718/400000 [00:19<00:37, 6715.13it/s] 37%|███▋      | 146502/400000 [00:20<00:36, 7016.12it/s] 37%|███▋      | 147220/400000 [00:20<00:35, 7063.51it/s] 37%|███▋      | 147930/400000 [00:20<00:36, 6982.41it/s] 37%|███▋      | 148631/400000 [00:20<00:36, 6939.52it/s] 37%|███▋      | 149327/400000 [00:20<00:37, 6694.79it/s] 38%|███▊      | 150028/400000 [00:20<00:36, 6786.02it/s] 38%|███▊      | 150835/400000 [00:20<00:34, 7124.80it/s] 38%|███▊      | 151640/400000 [00:20<00:33, 7377.82it/s] 38%|███▊      | 152459/400000 [00:20<00:32, 7601.59it/s] 38%|███▊      | 153226/400000 [00:20<00:32, 7523.41it/s] 39%|███▊      | 154014/400000 [00:21<00:32, 7626.00it/s] 39%|███▊      | 154781/400000 [00:21<00:32, 7618.78it/s] 39%|███▉      | 155546/400000 [00:21<00:32, 7482.00it/s] 39%|███▉      | 156349/400000 [00:21<00:31, 7638.26it/s] 39%|███▉      | 157134/400000 [00:21<00:31, 7699.60it/s] 39%|███▉      | 157906/400000 [00:21<00:31, 7694.44it/s] 40%|███▉      | 158677/400000 [00:21<00:31, 7681.25it/s] 40%|███▉      | 159475/400000 [00:21<00:30, 7767.14it/s] 40%|████      | 160253/400000 [00:21<00:30, 7757.41it/s] 40%|████      | 161030/400000 [00:22<00:31, 7653.88it/s] 40%|████      | 161797/400000 [00:22<00:32, 7253.84it/s] 41%|████      | 162528/400000 [00:22<00:33, 7160.80it/s] 41%|████      | 163308/400000 [00:22<00:32, 7340.24it/s] 41%|████      | 164060/400000 [00:22<00:31, 7390.81it/s] 41%|████      | 164802/400000 [00:22<00:32, 7144.90it/s] 41%|████▏     | 165521/400000 [00:22<00:33, 7016.01it/s] 42%|████▏     | 166226/400000 [00:22<00:34, 6799.10it/s] 42%|████▏     | 167010/400000 [00:22<00:32, 7077.59it/s] 42%|████▏     | 167754/400000 [00:22<00:32, 7180.39it/s] 42%|████▏     | 168510/400000 [00:23<00:31, 7290.02it/s] 42%|████▏     | 169297/400000 [00:23<00:30, 7449.61it/s] 43%|████▎     | 170073/400000 [00:23<00:30, 7538.46it/s] 43%|████▎     | 170844/400000 [00:23<00:30, 7586.64it/s] 43%|████▎     | 171605/400000 [00:23<00:31, 7322.57it/s] 43%|████▎     | 172341/400000 [00:23<00:31, 7118.33it/s] 43%|████▎     | 173057/400000 [00:23<00:32, 7030.51it/s] 43%|████▎     | 173763/400000 [00:23<00:32, 7024.64it/s] 44%|████▎     | 174568/400000 [00:23<00:30, 7303.63it/s] 44%|████▍     | 175378/400000 [00:23<00:29, 7522.96it/s] 44%|████▍     | 176158/400000 [00:24<00:29, 7602.33it/s] 44%|████▍     | 176957/400000 [00:24<00:28, 7714.28it/s] 44%|████▍     | 177754/400000 [00:24<00:28, 7786.46it/s] 45%|████▍     | 178558/400000 [00:24<00:28, 7859.45it/s] 45%|████▍     | 179346/400000 [00:24<00:28, 7785.32it/s] 45%|████▌     | 180147/400000 [00:24<00:28, 7849.27it/s] 45%|████▌     | 180954/400000 [00:24<00:27, 7911.37it/s] 45%|████▌     | 181746/400000 [00:24<00:27, 7865.25it/s] 46%|████▌     | 182534/400000 [00:24<00:28, 7550.88it/s] 46%|████▌     | 183293/400000 [00:25<00:30, 7215.19it/s] 46%|████▌     | 184020/400000 [00:25<00:30, 7128.36it/s] 46%|████▌     | 184778/400000 [00:25<00:29, 7256.05it/s] 46%|████▋     | 185507/400000 [00:25<00:30, 7135.71it/s] 47%|████▋     | 186224/400000 [00:25<00:30, 6996.16it/s] 47%|████▋     | 186943/400000 [00:25<00:30, 7053.10it/s] 47%|████▋     | 187715/400000 [00:25<00:29, 7239.51it/s] 47%|████▋     | 188543/400000 [00:25<00:28, 7522.64it/s] 47%|████▋     | 189320/400000 [00:25<00:27, 7592.99it/s] 48%|████▊     | 190107/400000 [00:25<00:27, 7672.88it/s] 48%|████▊     | 190877/400000 [00:26<00:27, 7627.28it/s] 48%|████▊     | 191642/400000 [00:26<00:28, 7300.86it/s] 48%|████▊     | 192377/400000 [00:26<00:29, 7111.91it/s] 48%|████▊     | 193093/400000 [00:26<00:29, 6982.96it/s] 48%|████▊     | 193795/400000 [00:26<00:29, 6894.36it/s] 49%|████▊     | 194488/400000 [00:26<00:30, 6781.25it/s] 49%|████▉     | 195172/400000 [00:26<00:30, 6797.06it/s] 49%|████▉     | 195854/400000 [00:26<00:30, 6716.88it/s] 49%|████▉     | 196537/400000 [00:26<00:30, 6748.74it/s] 49%|████▉     | 197301/400000 [00:26<00:28, 6991.84it/s] 50%|████▉     | 198065/400000 [00:27<00:28, 7173.59it/s] 50%|████▉     | 198867/400000 [00:27<00:27, 7405.78it/s] 50%|████▉     | 199649/400000 [00:27<00:26, 7524.57it/s] 50%|█████     | 200405/400000 [00:27<00:26, 7393.99it/s] 50%|█████     | 201148/400000 [00:27<00:26, 7393.67it/s] 50%|█████     | 201890/400000 [00:27<00:26, 7387.28it/s] 51%|█████     | 202656/400000 [00:27<00:26, 7464.65it/s] 51%|█████     | 203414/400000 [00:27<00:26, 7487.33it/s] 51%|█████     | 204181/400000 [00:27<00:25, 7539.04it/s] 51%|█████     | 204936/400000 [00:28<00:27, 7188.46it/s] 51%|█████▏    | 205659/400000 [00:28<00:27, 6989.91it/s] 52%|█████▏    | 206376/400000 [00:28<00:27, 7042.57it/s] 52%|█████▏    | 207125/400000 [00:28<00:26, 7171.06it/s] 52%|█████▏    | 207906/400000 [00:28<00:26, 7351.44it/s] 52%|█████▏    | 208688/400000 [00:28<00:25, 7484.47it/s] 52%|█████▏    | 209451/400000 [00:28<00:25, 7523.54it/s] 53%|█████▎    | 210290/400000 [00:28<00:24, 7764.06it/s] 53%|█████▎    | 211102/400000 [00:28<00:24, 7865.88it/s] 53%|█████▎    | 211892/400000 [00:28<00:25, 7275.08it/s] 53%|█████▎    | 212631/400000 [00:29<00:26, 7111.35it/s] 53%|█████▎    | 213351/400000 [00:29<00:26, 7129.77it/s] 54%|█████▎    | 214145/400000 [00:29<00:25, 7354.12it/s] 54%|█████▎    | 214910/400000 [00:29<00:24, 7439.50it/s] 54%|█████▍    | 215709/400000 [00:29<00:24, 7596.41it/s] 54%|█████▍    | 216499/400000 [00:29<00:23, 7683.12it/s] 54%|█████▍    | 217271/400000 [00:29<00:24, 7606.60it/s] 55%|█████▍    | 218034/400000 [00:29<00:24, 7436.52it/s] 55%|█████▍    | 218781/400000 [00:29<00:25, 7188.19it/s] 55%|█████▍    | 219504/400000 [00:29<00:25, 7098.63it/s] 55%|█████▌    | 220217/400000 [00:30<00:25, 6957.82it/s] 55%|█████▌    | 220916/400000 [00:30<00:26, 6877.29it/s] 55%|█████▌    | 221606/400000 [00:30<00:26, 6783.60it/s] 56%|█████▌    | 222293/400000 [00:30<00:26, 6806.06it/s] 56%|█████▌    | 223028/400000 [00:30<00:25, 6958.83it/s] 56%|█████▌    | 223832/400000 [00:30<00:24, 7250.26it/s] 56%|█████▌    | 224632/400000 [00:30<00:23, 7457.55it/s] 56%|█████▋    | 225396/400000 [00:30<00:23, 7511.08it/s] 57%|█████▋    | 226151/400000 [00:30<00:23, 7450.37it/s] 57%|█████▋    | 226899/400000 [00:31<00:23, 7431.19it/s] 57%|█████▋    | 227676/400000 [00:31<00:22, 7527.96it/s] 57%|█████▋    | 228435/400000 [00:31<00:22, 7543.97it/s] 57%|█████▋    | 229210/400000 [00:31<00:22, 7604.33it/s] 57%|█████▋    | 229978/400000 [00:31<00:22, 7623.52it/s] 58%|█████▊    | 230741/400000 [00:31<00:22, 7478.00it/s] 58%|█████▊    | 231517/400000 [00:31<00:22, 7559.41it/s] 58%|█████▊    | 232274/400000 [00:31<00:22, 7459.81it/s] 58%|█████▊    | 233066/400000 [00:31<00:21, 7591.34it/s] 58%|█████▊    | 233831/400000 [00:31<00:21, 7606.39it/s] 59%|█████▊    | 234593/400000 [00:32<00:22, 7507.89it/s] 59%|█████▉    | 235355/400000 [00:32<00:21, 7540.18it/s] 59%|█████▉    | 236114/400000 [00:32<00:21, 7552.62it/s] 59%|█████▉    | 236874/400000 [00:32<00:21, 7561.73it/s] 59%|█████▉    | 237631/400000 [00:32<00:21, 7508.98it/s] 60%|█████▉    | 238383/400000 [00:32<00:21, 7448.94it/s] 60%|█████▉    | 239129/400000 [00:32<00:21, 7398.79it/s] 60%|█████▉    | 239870/400000 [00:32<00:21, 7385.23it/s] 60%|██████    | 240666/400000 [00:32<00:21, 7547.54it/s] 60%|██████    | 241503/400000 [00:32<00:20, 7775.90it/s] 61%|██████    | 242346/400000 [00:33<00:19, 7959.24it/s] 61%|██████    | 243145/400000 [00:33<00:20, 7823.75it/s] 61%|██████    | 243930/400000 [00:33<00:21, 7363.47it/s] 61%|██████    | 244674/400000 [00:33<00:21, 7375.26it/s] 61%|██████▏   | 245446/400000 [00:33<00:20, 7472.80it/s] 62%|██████▏   | 246198/400000 [00:33<00:21, 7298.19it/s] 62%|██████▏   | 246932/400000 [00:33<00:21, 7125.40it/s] 62%|██████▏   | 247648/400000 [00:33<00:22, 6815.86it/s] 62%|██████▏   | 248346/400000 [00:33<00:22, 6861.82it/s] 62%|██████▏   | 249040/400000 [00:34<00:21, 6884.10it/s] 62%|██████▏   | 249732/400000 [00:34<00:21, 6875.92it/s] 63%|██████▎   | 250422/400000 [00:34<00:21, 6850.89it/s] 63%|██████▎   | 251109/400000 [00:34<00:21, 6791.53it/s] 63%|██████▎   | 251790/400000 [00:34<00:21, 6737.35it/s] 63%|██████▎   | 252534/400000 [00:34<00:21, 6931.39it/s] 63%|██████▎   | 253287/400000 [00:34<00:20, 7099.89it/s] 64%|██████▎   | 254087/400000 [00:34<00:19, 7347.28it/s] 64%|██████▎   | 254826/400000 [00:34<00:19, 7327.50it/s] 64%|██████▍   | 255595/400000 [00:34<00:19, 7429.92it/s] 64%|██████▍   | 256341/400000 [00:35<00:20, 7073.58it/s] 64%|██████▍   | 257076/400000 [00:35<00:19, 7153.82it/s] 64%|██████▍   | 257818/400000 [00:35<00:19, 7228.86it/s] 65%|██████▍   | 258581/400000 [00:35<00:19, 7343.46it/s] 65%|██████▍   | 259348/400000 [00:35<00:18, 7437.43it/s] 65%|██████▌   | 260117/400000 [00:35<00:18, 7508.71it/s] 65%|██████▌   | 260870/400000 [00:35<00:18, 7400.88it/s] 65%|██████▌   | 261612/400000 [00:35<00:19, 7102.36it/s] 66%|██████▌   | 262326/400000 [00:35<00:19, 7010.99it/s] 66%|██████▌   | 263112/400000 [00:35<00:18, 7245.34it/s] 66%|██████▌   | 263910/400000 [00:36<00:18, 7449.49it/s] 66%|██████▌   | 264681/400000 [00:36<00:17, 7524.23it/s] 66%|██████▋   | 265437/400000 [00:36<00:18, 7447.57it/s] 67%|██████▋   | 266218/400000 [00:36<00:17, 7551.15it/s] 67%|██████▋   | 266981/400000 [00:36<00:17, 7573.42it/s] 67%|██████▋   | 267740/400000 [00:36<00:17, 7557.85it/s] 67%|██████▋   | 268503/400000 [00:36<00:17, 7576.30it/s] 67%|██████▋   | 269262/400000 [00:36<00:17, 7488.82it/s] 68%|██████▊   | 270019/400000 [00:36<00:17, 7510.99it/s] 68%|██████▊   | 270788/400000 [00:36<00:17, 7562.69it/s] 68%|██████▊   | 271584/400000 [00:37<00:16, 7676.46it/s] 68%|██████▊   | 272372/400000 [00:37<00:16, 7736.20it/s] 68%|██████▊   | 273148/400000 [00:37<00:16, 7742.61it/s] 68%|██████▊   | 273931/400000 [00:37<00:16, 7768.05it/s] 69%|██████▊   | 274709/400000 [00:37<00:16, 7593.33it/s] 69%|██████▉   | 275470/400000 [00:37<00:17, 7207.48it/s] 69%|██████▉   | 276209/400000 [00:37<00:17, 7259.40it/s] 69%|██████▉   | 276951/400000 [00:37<00:16, 7305.38it/s] 69%|██████▉   | 277685/400000 [00:37<00:17, 7095.56it/s] 70%|██████▉   | 278398/400000 [00:38<00:17, 6984.13it/s] 70%|██████▉   | 279099/400000 [00:38<00:17, 6929.73it/s] 70%|██████▉   | 279794/400000 [00:38<00:17, 6882.12it/s] 70%|███████   | 280484/400000 [00:38<00:17, 6818.26it/s] 70%|███████   | 281167/400000 [00:38<00:17, 6813.36it/s] 70%|███████   | 281850/400000 [00:38<00:17, 6733.98it/s] 71%|███████   | 282535/400000 [00:38<00:17, 6766.80it/s] 71%|███████   | 283213/400000 [00:38<00:17, 6696.93it/s] 71%|███████   | 283898/400000 [00:38<00:17, 6741.94it/s] 71%|███████   | 284607/400000 [00:38<00:16, 6841.95it/s] 71%|███████▏  | 285410/400000 [00:39<00:16, 7159.50it/s] 72%|███████▏  | 286159/400000 [00:39<00:15, 7254.99it/s] 72%|███████▏  | 286922/400000 [00:39<00:15, 7360.89it/s] 72%|███████▏  | 287661/400000 [00:39<00:15, 7276.95it/s] 72%|███████▏  | 288391/400000 [00:39<00:15, 7120.67it/s] 72%|███████▏  | 289106/400000 [00:39<00:15, 7035.82it/s] 72%|███████▏  | 289889/400000 [00:39<00:15, 7254.81it/s] 73%|███████▎  | 290627/400000 [00:39<00:15, 7289.80it/s] 73%|███████▎  | 291391/400000 [00:39<00:14, 7391.30it/s] 73%|███████▎  | 292132/400000 [00:39<00:14, 7372.06it/s] 73%|███████▎  | 292871/400000 [00:40<00:15, 7137.35it/s] 73%|███████▎  | 293588/400000 [00:40<00:15, 6978.38it/s] 74%|███████▎  | 294289/400000 [00:40<00:15, 6841.10it/s] 74%|███████▍  | 295033/400000 [00:40<00:14, 7009.08it/s] 74%|███████▍  | 295828/400000 [00:40<00:14, 7266.88it/s] 74%|███████▍  | 296616/400000 [00:40<00:13, 7438.66it/s] 74%|███████▍  | 297364/400000 [00:40<00:13, 7443.91it/s] 75%|███████▍  | 298112/400000 [00:40<00:13, 7450.95it/s] 75%|███████▍  | 298860/400000 [00:40<00:13, 7416.68it/s] 75%|███████▍  | 299655/400000 [00:40<00:13, 7566.70it/s] 75%|███████▌  | 300445/400000 [00:41<00:12, 7660.07it/s] 75%|███████▌  | 301213/400000 [00:41<00:12, 7608.34it/s] 75%|███████▌  | 301975/400000 [00:41<00:12, 7609.59it/s] 76%|███████▌  | 302737/400000 [00:41<00:12, 7525.66it/s] 76%|███████▌  | 303491/400000 [00:41<00:12, 7517.28it/s] 76%|███████▌  | 304244/400000 [00:41<00:12, 7433.89it/s] 76%|███████▌  | 304996/400000 [00:41<00:12, 7457.63it/s] 76%|███████▋  | 305755/400000 [00:41<00:12, 7496.66it/s] 77%|███████▋  | 306520/400000 [00:41<00:12, 7541.29it/s] 77%|███████▋  | 307293/400000 [00:41<00:12, 7595.84it/s] 77%|███████▋  | 308053/400000 [00:42<00:12, 7336.18it/s] 77%|███████▋  | 308869/400000 [00:42<00:12, 7563.67it/s] 77%|███████▋  | 309630/400000 [00:42<00:11, 7576.08it/s] 78%|███████▊  | 310390/400000 [00:42<00:12, 7455.82it/s] 78%|███████▊  | 311138/400000 [00:42<00:12, 7299.20it/s] 78%|███████▊  | 311871/400000 [00:42<00:12, 7069.12it/s] 78%|███████▊  | 312581/400000 [00:42<00:12, 6956.91it/s] 78%|███████▊  | 313289/400000 [00:42<00:12, 6991.17it/s] 79%|███████▊  | 314129/400000 [00:42<00:11, 7360.43it/s] 79%|███████▊  | 314948/400000 [00:43<00:11, 7591.01it/s] 79%|███████▉  | 315714/400000 [00:43<00:11, 7601.16it/s] 79%|███████▉  | 316499/400000 [00:43<00:10, 7671.85it/s] 79%|███████▉  | 317270/400000 [00:43<00:10, 7611.12it/s] 80%|███████▉  | 318070/400000 [00:43<00:10, 7721.36it/s] 80%|███████▉  | 318893/400000 [00:43<00:10, 7865.43it/s] 80%|███████▉  | 319682/400000 [00:43<00:10, 7760.32it/s] 80%|████████  | 320460/400000 [00:43<00:10, 7726.01it/s] 80%|████████  | 321234/400000 [00:43<00:10, 7503.85it/s] 81%|████████  | 322026/400000 [00:43<00:10, 7621.32it/s] 81%|████████  | 322791/400000 [00:44<00:10, 7589.66it/s] 81%|████████  | 323552/400000 [00:44<00:10, 7497.67it/s] 81%|████████  | 324335/400000 [00:44<00:09, 7591.90it/s] 81%|████████▏ | 325106/400000 [00:44<00:09, 7626.01it/s] 81%|████████▏ | 325879/400000 [00:44<00:09, 7656.13it/s] 82%|████████▏ | 326661/400000 [00:44<00:09, 7704.15it/s] 82%|████████▏ | 327450/400000 [00:44<00:09, 7756.91it/s] 82%|████████▏ | 328258/400000 [00:44<00:09, 7850.11it/s] 82%|████████▏ | 329044/400000 [00:44<00:09, 7335.18it/s] 82%|████████▏ | 329787/400000 [00:44<00:09, 7360.95it/s] 83%|████████▎ | 330529/400000 [00:45<00:09, 7198.33it/s] 83%|████████▎ | 331254/400000 [00:45<00:09, 7085.66it/s] 83%|████████▎ | 331967/400000 [00:45<00:09, 6978.76it/s] 83%|████████▎ | 332668/400000 [00:45<00:09, 6746.50it/s] 83%|████████▎ | 333398/400000 [00:45<00:09, 6902.90it/s] 84%|████████▎ | 334168/400000 [00:45<00:09, 7121.65it/s] 84%|████████▎ | 334960/400000 [00:45<00:08, 7342.93it/s] 84%|████████▍ | 335776/400000 [00:45<00:08, 7568.85it/s] 84%|████████▍ | 336582/400000 [00:45<00:08, 7707.30it/s] 84%|████████▍ | 337357/400000 [00:46<00:08, 7641.47it/s] 85%|████████▍ | 338125/400000 [00:46<00:08, 7363.83it/s] 85%|████████▍ | 338919/400000 [00:46<00:08, 7525.88it/s] 85%|████████▍ | 339734/400000 [00:46<00:07, 7701.20it/s] 85%|████████▌ | 340513/400000 [00:46<00:07, 7724.88it/s] 85%|████████▌ | 341321/400000 [00:46<00:07, 7824.91it/s] 86%|████████▌ | 342106/400000 [00:46<00:07, 7700.21it/s] 86%|████████▌ | 342893/400000 [00:46<00:07, 7749.05it/s] 86%|████████▌ | 343670/400000 [00:46<00:07, 7521.33it/s] 86%|████████▌ | 344425/400000 [00:46<00:07, 7316.74it/s] 86%|████████▋ | 345168/400000 [00:47<00:07, 7337.81it/s] 86%|████████▋ | 345904/400000 [00:47<00:07, 7272.69it/s] 87%|████████▋ | 346633/400000 [00:47<00:07, 7204.05it/s] 87%|████████▋ | 347374/400000 [00:47<00:07, 7262.74it/s] 87%|████████▋ | 348102/400000 [00:47<00:07, 7143.87it/s] 87%|████████▋ | 348818/400000 [00:47<00:07, 7064.52it/s] 87%|████████▋ | 349585/400000 [00:47<00:06, 7234.34it/s] 88%|████████▊ | 350311/400000 [00:47<00:06, 7235.76it/s] 88%|████████▊ | 351036/400000 [00:47<00:06, 7091.16it/s] 88%|████████▊ | 351747/400000 [00:47<00:06, 7041.66it/s] 88%|████████▊ | 352453/400000 [00:48<00:06, 6955.86it/s] 88%|████████▊ | 353193/400000 [00:48<00:06, 7081.48it/s] 88%|████████▊ | 353977/400000 [00:48<00:06, 7291.66it/s] 89%|████████▊ | 354748/400000 [00:48<00:06, 7412.00it/s] 89%|████████▉ | 355505/400000 [00:48<00:05, 7458.63it/s] 89%|████████▉ | 356300/400000 [00:48<00:05, 7597.97it/s] 89%|████████▉ | 357062/400000 [00:48<00:05, 7465.23it/s] 89%|████████▉ | 357811/400000 [00:48<00:05, 7257.60it/s] 90%|████████▉ | 358540/400000 [00:48<00:05, 6997.11it/s] 90%|████████▉ | 359305/400000 [00:49<00:05, 7179.98it/s] 90%|█████████ | 360056/400000 [00:49<00:05, 7273.17it/s] 90%|█████████ | 360823/400000 [00:49<00:05, 7387.58it/s] 90%|█████████ | 361585/400000 [00:49<00:05, 7454.67it/s] 91%|█████████ | 362333/400000 [00:49<00:05, 7431.20it/s] 91%|█████████ | 363078/400000 [00:49<00:04, 7387.44it/s] 91%|█████████ | 363864/400000 [00:49<00:04, 7522.54it/s] 91%|█████████ | 364649/400000 [00:49<00:04, 7615.89it/s] 91%|█████████▏| 365416/400000 [00:49<00:04, 7631.29it/s] 92%|█████████▏| 366186/400000 [00:49<00:04, 7650.06it/s] 92%|█████████▏| 366952/400000 [00:50<00:04, 7608.76it/s] 92%|█████████▏| 367719/400000 [00:50<00:04, 7625.24it/s] 92%|█████████▏| 368482/400000 [00:50<00:04, 7304.61it/s] 92%|█████████▏| 369232/400000 [00:50<00:04, 7361.80it/s] 92%|█████████▏| 369971/400000 [00:50<00:04, 7316.41it/s] 93%|█████████▎| 370705/400000 [00:50<00:04, 7253.00it/s] 93%|█████████▎| 371441/400000 [00:50<00:03, 7284.43it/s] 93%|█████████▎| 372222/400000 [00:50<00:03, 7433.48it/s] 93%|█████████▎| 373040/400000 [00:50<00:03, 7641.46it/s] 93%|█████████▎| 373863/400000 [00:50<00:03, 7808.72it/s] 94%|█████████▎| 374683/400000 [00:51<00:03, 7919.96it/s] 94%|█████████▍| 375480/400000 [00:51<00:03, 7932.35it/s] 94%|█████████▍| 376289/400000 [00:51<00:02, 7976.56it/s] 94%|█████████▍| 377088/400000 [00:51<00:02, 7721.66it/s] 94%|█████████▍| 377863/400000 [00:51<00:02, 7459.19it/s] 95%|█████████▍| 378698/400000 [00:51<00:02, 7703.62it/s] 95%|█████████▍| 379473/400000 [00:51<00:02, 7346.94it/s] 95%|█████████▌| 380257/400000 [00:51<00:02, 7485.49it/s] 95%|█████████▌| 381044/400000 [00:51<00:02, 7595.72it/s] 95%|█████████▌| 381821/400000 [00:51<00:02, 7645.88it/s] 96%|█████████▌| 382613/400000 [00:52<00:02, 7724.11it/s] 96%|█████████▌| 383388/400000 [00:52<00:02, 7701.78it/s] 96%|█████████▌| 384160/400000 [00:52<00:02, 7594.31it/s] 96%|█████████▌| 384921/400000 [00:52<00:01, 7573.06it/s] 96%|█████████▋| 385680/400000 [00:52<00:01, 7565.93it/s] 97%|█████████▋| 386438/400000 [00:52<00:01, 7461.57it/s] 97%|█████████▋| 387185/400000 [00:52<00:01, 7157.26it/s] 97%|█████████▋| 387904/400000 [00:52<00:01, 7052.80it/s] 97%|█████████▋| 388721/400000 [00:52<00:01, 7351.62it/s] 97%|█████████▋| 389510/400000 [00:53<00:01, 7504.66it/s] 98%|█████████▊| 390265/400000 [00:53<00:01, 7436.16it/s] 98%|█████████▊| 391012/400000 [00:53<00:01, 7266.88it/s] 98%|█████████▊| 391742/400000 [00:53<00:01, 7127.25it/s] 98%|█████████▊| 392458/400000 [00:53<00:01, 6966.78it/s] 98%|█████████▊| 393158/400000 [00:53<00:00, 6883.74it/s] 98%|█████████▊| 393938/400000 [00:53<00:00, 7133.83it/s] 99%|█████████▊| 394725/400000 [00:53<00:00, 7338.49it/s] 99%|█████████▉| 395463/400000 [00:53<00:00, 7228.25it/s] 99%|█████████▉| 396255/400000 [00:53<00:00, 7418.57it/s] 99%|█████████▉| 397048/400000 [00:54<00:00, 7562.94it/s] 99%|█████████▉| 397820/400000 [00:54<00:00, 7607.22it/s]100%|█████████▉| 398628/400000 [00:54<00:00, 7743.08it/s]100%|█████████▉| 399454/400000 [00:54<00:00, 7890.88it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7350.07it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff6700e0710>, <torchtext.data.dataset.TabularDataset object at 0x7ff6700e0860>, <torchtext.vocab.Vocab object at 0x7ff6700e0780>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.08 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.69 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.69 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.89 file/s]2020-06-22 18:19:38.564197: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-22 18:19:38.569038: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-22 18:19:38.569226: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1f4191420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 18:19:38.569247: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134642.96it/s] 71%|███████   | 7028736/9912422 [00:00<00:15, 192188.75it/s]9920512it [00:00, 39973833.93it/s]                           
0it [00:00, ?it/s]32768it [00:00, 587581.99it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1006944.60it/s]1654784it [00:00, 12327852.32it/s]                           
0it [00:00, ?it/s]8192it [00:00, 204708.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
