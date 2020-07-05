
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f23865990d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f23865990d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f23f18e21e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f23f18e21e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f240fc2ae18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f240fc2ae18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f239ec10620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f239ec10620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f239ec10620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2564096/9912422 [00:00<00:00, 24475031.60it/s]9920512it [00:00, 32284443.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 606593.64it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463374.21it/s]1654784it [00:00, 11583358.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 162262.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f239c0b9978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2385c44be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f239ec10268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f239ec10268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f239ec10268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:35,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:33,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:32,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:32,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:02,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:01,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:41,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:40,  3.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:27,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:26,  4.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 29.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 36.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 44.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 51.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 59.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 66.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.87 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.87 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
0 examples [00:00, ? examples/s]2020-07-05 18:08:56.065108: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-05 18:08:56.079616: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-05 18:08:56.079783: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561b036bda20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-05 18:08:56.079799: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.38 examples/s]108 examples [00:00,  4.82 examples/s]220 examples [00:00,  6.87 examples/s]331 examples [00:00,  9.78 examples/s]435 examples [00:00, 13.92 examples/s]544 examples [00:00, 19.77 examples/s]653 examples [00:00, 28.03 examples/s]761 examples [00:00, 39.60 examples/s]874 examples [00:01, 55.73 examples/s]984 examples [00:01, 77.91 examples/s]1089 examples [00:01, 107.36 examples/s]1193 examples [00:01, 146.87 examples/s]1299 examples [00:01, 198.01 examples/s]1411 examples [00:01, 262.83 examples/s]1521 examples [00:01, 340.52 examples/s]1630 examples [00:01, 428.71 examples/s]1740 examples [00:01, 524.51 examples/s]1848 examples [00:02, 618.98 examples/s]1961 examples [00:02, 715.41 examples/s]2071 examples [00:02, 775.23 examples/s]2177 examples [00:02, 840.98 examples/s]2289 examples [00:02, 907.49 examples/s]2400 examples [00:02, 958.53 examples/s]2511 examples [00:02, 998.89 examples/s]2620 examples [00:02, 1022.25 examples/s]2730 examples [00:02, 1041.70 examples/s]2839 examples [00:02, 1042.97 examples/s]2950 examples [00:03, 1061.03 examples/s]3061 examples [00:03, 1073.37 examples/s]3170 examples [00:03, 1052.56 examples/s]3277 examples [00:03, 1053.64 examples/s]3388 examples [00:03, 1066.70 examples/s]3496 examples [00:03, 1068.24 examples/s]3606 examples [00:03, 1077.50 examples/s]3715 examples [00:03, 1068.04 examples/s]3823 examples [00:03, 1065.36 examples/s]3931 examples [00:03, 1069.36 examples/s]4039 examples [00:04, 1062.69 examples/s]4149 examples [00:04, 1071.95 examples/s]4257 examples [00:04, 1068.87 examples/s]4364 examples [00:04, 1022.63 examples/s]4470 examples [00:04, 1033.10 examples/s]4574 examples [00:04, 1027.03 examples/s]4681 examples [00:04, 1039.09 examples/s]4788 examples [00:04, 1043.15 examples/s]4893 examples [00:04, 1029.21 examples/s]5001 examples [00:04, 1043.48 examples/s]5110 examples [00:05, 1056.28 examples/s]5218 examples [00:05, 1060.80 examples/s]5325 examples [00:05, 1035.33 examples/s]5431 examples [00:05, 1040.22 examples/s]5536 examples [00:05, 1025.86 examples/s]5639 examples [00:05, 1000.40 examples/s]5740 examples [00:05, 1003.10 examples/s]5841 examples [00:05, 995.68 examples/s] 5944 examples [00:05, 1004.89 examples/s]6049 examples [00:06, 1017.63 examples/s]6152 examples [00:06, 1020.32 examples/s]6257 examples [00:06, 1027.23 examples/s]6360 examples [00:06, 1021.29 examples/s]6466 examples [00:06, 1031.95 examples/s]6574 examples [00:06, 1045.38 examples/s]6683 examples [00:06, 1056.38 examples/s]6793 examples [00:06, 1067.90 examples/s]6900 examples [00:06, 1063.63 examples/s]7010 examples [00:06, 1073.89 examples/s]7120 examples [00:07, 1081.53 examples/s]7229 examples [00:07, 1073.14 examples/s]7337 examples [00:07, 1063.32 examples/s]7444 examples [00:07, 1009.01 examples/s]7551 examples [00:07, 1024.39 examples/s]7660 examples [00:07, 1041.43 examples/s]7765 examples [00:07, 1032.03 examples/s]7871 examples [00:07, 1039.29 examples/s]7976 examples [00:07, 1032.84 examples/s]8085 examples [00:07, 1047.46 examples/s]8196 examples [00:08, 1063.22 examples/s]8303 examples [00:08, 1064.30 examples/s]8410 examples [00:08, 1056.30 examples/s]8517 examples [00:08, 1058.34 examples/s]8626 examples [00:08, 1067.33 examples/s]8737 examples [00:08, 1077.67 examples/s]8848 examples [00:08, 1086.46 examples/s]8959 examples [00:08, 1092.43 examples/s]9069 examples [00:08, 1078.34 examples/s]9182 examples [00:08, 1091.13 examples/s]9293 examples [00:09, 1093.69 examples/s]9403 examples [00:09, 1075.49 examples/s]9511 examples [00:09, 1072.12 examples/s]9619 examples [00:09, 1066.10 examples/s]9728 examples [00:09, 1072.77 examples/s]9836 examples [00:09, 1053.16 examples/s]9945 examples [00:09, 1063.09 examples/s]10052 examples [00:09, 1014.27 examples/s]10156 examples [00:09, 1019.61 examples/s]10264 examples [00:10, 1036.76 examples/s]10373 examples [00:10, 1052.07 examples/s]10479 examples [00:10, 1054.00 examples/s]10586 examples [00:10, 1056.80 examples/s]10695 examples [00:10, 1064.78 examples/s]10805 examples [00:10, 1074.34 examples/s]10913 examples [00:10, 1069.12 examples/s]11020 examples [00:10, 1069.22 examples/s]11129 examples [00:10, 1074.11 examples/s]11237 examples [00:10, 1059.50 examples/s]11344 examples [00:11, 1060.77 examples/s]11456 examples [00:11, 1076.18 examples/s]11565 examples [00:11, 1079.06 examples/s]11675 examples [00:11, 1083.28 examples/s]11785 examples [00:11, 1086.42 examples/s]11896 examples [00:11, 1091.29 examples/s]12006 examples [00:11, 1087.69 examples/s]12115 examples [00:11, 1086.17 examples/s]12224 examples [00:11, 1072.68 examples/s]12332 examples [00:11, 1051.63 examples/s]12438 examples [00:12, 1015.48 examples/s]12540 examples [00:12, 1014.44 examples/s]12642 examples [00:12, 1002.00 examples/s]12743 examples [00:12, 948.82 examples/s] 12839 examples [00:12, 942.59 examples/s]12934 examples [00:12, 925.10 examples/s]13027 examples [00:12, 892.67 examples/s]13117 examples [00:12, 892.65 examples/s]13211 examples [00:12, 903.72 examples/s]13304 examples [00:13, 909.18 examples/s]13398 examples [00:13, 918.11 examples/s]13490 examples [00:13, 909.37 examples/s]13582 examples [00:13, 901.82 examples/s]13675 examples [00:13, 908.62 examples/s]13784 examples [00:13, 955.45 examples/s]13889 examples [00:13, 979.54 examples/s]14000 examples [00:13, 1014.46 examples/s]14111 examples [00:13, 1039.34 examples/s]14219 examples [00:13, 1049.18 examples/s]14328 examples [00:14, 1060.11 examples/s]14435 examples [00:14, 1056.61 examples/s]14541 examples [00:14, 1056.24 examples/s]14647 examples [00:14, 1001.79 examples/s]14753 examples [00:14, 1016.66 examples/s]14865 examples [00:14, 1043.83 examples/s]14975 examples [00:14, 1058.60 examples/s]15087 examples [00:14, 1075.27 examples/s]15195 examples [00:14, 1060.45 examples/s]15302 examples [00:14, 1035.63 examples/s]15412 examples [00:15, 1051.73 examples/s]15518 examples [00:15, 1038.65 examples/s]15623 examples [00:15, 1019.80 examples/s]15730 examples [00:15, 1033.45 examples/s]15838 examples [00:15, 1046.68 examples/s]15947 examples [00:15, 1058.70 examples/s]16054 examples [00:15, 1048.05 examples/s]16159 examples [00:15, 1030.28 examples/s]16268 examples [00:15, 1046.33 examples/s]16375 examples [00:15, 1048.86 examples/s]16481 examples [00:16, 1050.55 examples/s]16589 examples [00:16, 1057.43 examples/s]16697 examples [00:16, 1063.60 examples/s]16805 examples [00:16, 1067.41 examples/s]16913 examples [00:16, 1069.44 examples/s]17025 examples [00:16, 1081.95 examples/s]17134 examples [00:16, 1075.10 examples/s]17242 examples [00:16, 1066.18 examples/s]17349 examples [00:16, 1058.42 examples/s]17455 examples [00:16, 1056.44 examples/s]17563 examples [00:17, 1062.74 examples/s]17674 examples [00:17, 1075.06 examples/s]17782 examples [00:17, 1032.13 examples/s]17886 examples [00:17, 999.35 examples/s] 17993 examples [00:17, 1019.38 examples/s]18100 examples [00:17, 1032.37 examples/s]18204 examples [00:17, 1004.80 examples/s]18311 examples [00:17, 1022.61 examples/s]18421 examples [00:17, 1044.55 examples/s]18529 examples [00:18, 1054.44 examples/s]18635 examples [00:18, 1039.82 examples/s]18740 examples [00:18, 1032.61 examples/s]18844 examples [00:18, 1026.01 examples/s]18952 examples [00:18, 1039.44 examples/s]19060 examples [00:18, 1049.90 examples/s]19172 examples [00:18, 1067.37 examples/s]19284 examples [00:18, 1082.31 examples/s]19393 examples [00:18, 1041.96 examples/s]19498 examples [00:18, 1042.87 examples/s]19605 examples [00:19, 1050.33 examples/s]19711 examples [00:19, 1047.48 examples/s]19821 examples [00:19, 1061.95 examples/s]19930 examples [00:19, 1067.82 examples/s]20037 examples [00:19, 1020.73 examples/s]20147 examples [00:19, 1042.40 examples/s]20257 examples [00:19, 1056.34 examples/s]20367 examples [00:19, 1067.36 examples/s]20475 examples [00:19, 1070.63 examples/s]20583 examples [00:19, 1051.48 examples/s]20690 examples [00:20, 1054.44 examples/s]20797 examples [00:20, 1057.75 examples/s]20903 examples [00:20, 1037.21 examples/s]21010 examples [00:20, 1045.33 examples/s]21115 examples [00:20, 1041.55 examples/s]21220 examples [00:20, 1016.35 examples/s]21329 examples [00:20, 1037.06 examples/s]21440 examples [00:20, 1056.46 examples/s]21551 examples [00:20, 1069.15 examples/s]21659 examples [00:21, 1061.37 examples/s]21768 examples [00:21, 1068.32 examples/s]21875 examples [00:21, 1049.41 examples/s]21981 examples [00:21, 1042.99 examples/s]22086 examples [00:21, 1039.99 examples/s]22192 examples [00:21, 1045.47 examples/s]22297 examples [00:21, 1027.05 examples/s]22400 examples [00:21, 1018.72 examples/s]22502 examples [00:21, 992.24 examples/s] 22602 examples [00:21, 954.80 examples/s]22707 examples [00:22, 981.01 examples/s]22817 examples [00:22, 1013.21 examples/s]22919 examples [00:22, 1011.40 examples/s]23021 examples [00:22, 945.54 examples/s] 23124 examples [00:22, 969.38 examples/s]23233 examples [00:22, 1000.52 examples/s]23341 examples [00:22, 1022.97 examples/s]23450 examples [00:22, 1041.62 examples/s]23558 examples [00:22, 1052.56 examples/s]23668 examples [00:22, 1063.90 examples/s]23778 examples [00:23, 1071.91 examples/s]23886 examples [00:23, 1071.52 examples/s]23995 examples [00:23, 1075.61 examples/s]24103 examples [00:23, 1075.88 examples/s]24211 examples [00:23, 1071.01 examples/s]24319 examples [00:23, 1062.06 examples/s]24426 examples [00:23, 1052.08 examples/s]24532 examples [00:23, 1052.92 examples/s]24638 examples [00:23, 1053.97 examples/s]24747 examples [00:23, 1062.32 examples/s]24854 examples [00:24, 1051.36 examples/s]24960 examples [00:24, 1053.40 examples/s]25066 examples [00:24, 1017.61 examples/s]25169 examples [00:24, 987.35 examples/s] 25277 examples [00:24, 1010.62 examples/s]25379 examples [00:24, 1011.56 examples/s]25489 examples [00:24, 1036.46 examples/s]25600 examples [00:24, 1056.09 examples/s]25706 examples [00:24, 1040.86 examples/s]25811 examples [00:25, 994.38 examples/s] 25912 examples [00:25, 996.89 examples/s]26016 examples [00:25, 1008.88 examples/s]26118 examples [00:25, 1002.57 examples/s]26219 examples [00:25, 997.07 examples/s] 26323 examples [00:25, 1007.40 examples/s]26428 examples [00:25, 1019.48 examples/s]26534 examples [00:25, 1031.11 examples/s]26641 examples [00:25, 1041.96 examples/s]26753 examples [00:25, 1062.70 examples/s]26860 examples [00:26, 1060.34 examples/s]26969 examples [00:26, 1068.23 examples/s]27076 examples [00:26, 1032.80 examples/s]27180 examples [00:26, 1002.48 examples/s]27291 examples [00:26, 1030.70 examples/s]27405 examples [00:26, 1060.47 examples/s]27520 examples [00:26, 1084.37 examples/s]27636 examples [00:26, 1103.37 examples/s]27747 examples [00:26, 1100.05 examples/s]27863 examples [00:26, 1117.00 examples/s]27978 examples [00:27, 1125.12 examples/s]28094 examples [00:27, 1132.79 examples/s]28208 examples [00:27, 1084.72 examples/s]28322 examples [00:27, 1099.49 examples/s]28434 examples [00:27, 1103.99 examples/s]28550 examples [00:27, 1118.29 examples/s]28664 examples [00:27, 1123.14 examples/s]28777 examples [00:27, 1103.86 examples/s]28891 examples [00:27, 1113.49 examples/s]29003 examples [00:28, 1086.02 examples/s]29114 examples [00:28, 1092.66 examples/s]29225 examples [00:28, 1096.56 examples/s]29335 examples [00:28, 1094.28 examples/s]29447 examples [00:28, 1099.79 examples/s]29558 examples [00:28, 1085.36 examples/s]29673 examples [00:28, 1102.01 examples/s]29787 examples [00:28, 1112.07 examples/s]29903 examples [00:28, 1124.32 examples/s]30016 examples [00:28, 1039.71 examples/s]30127 examples [00:29, 1058.44 examples/s]30236 examples [00:29, 1067.13 examples/s]30348 examples [00:29, 1079.97 examples/s]30457 examples [00:29, 1057.38 examples/s]30564 examples [00:29, 1059.03 examples/s]30675 examples [00:29, 1073.72 examples/s]30787 examples [00:29, 1087.16 examples/s]30904 examples [00:29, 1110.18 examples/s]31016 examples [00:29, 1100.12 examples/s]31130 examples [00:29, 1111.70 examples/s]31246 examples [00:30, 1122.84 examples/s]31359 examples [00:30, 1122.73 examples/s]31472 examples [00:30, 1093.33 examples/s]31583 examples [00:30, 1095.80 examples/s]31694 examples [00:30, 1099.78 examples/s]31806 examples [00:30, 1105.30 examples/s]31917 examples [00:30, 1103.66 examples/s]32028 examples [00:30, 1097.05 examples/s]32140 examples [00:30, 1102.28 examples/s]32251 examples [00:30, 1098.71 examples/s]32361 examples [00:31, 1057.01 examples/s]32471 examples [00:31, 1067.98 examples/s]32579 examples [00:31, 1029.32 examples/s]32687 examples [00:31, 1042.97 examples/s]32792 examples [00:31, 1026.60 examples/s]32907 examples [00:31, 1058.96 examples/s]33021 examples [00:31, 1079.92 examples/s]33130 examples [00:31, 1064.28 examples/s]33240 examples [00:31, 1071.98 examples/s]33349 examples [00:32, 1074.64 examples/s]33457 examples [00:32, 1074.72 examples/s]33565 examples [00:32, 1045.55 examples/s]33670 examples [00:32, 988.03 examples/s] 33780 examples [00:32, 1018.36 examples/s]33885 examples [00:32, 1026.85 examples/s]33989 examples [00:32, 1017.61 examples/s]34097 examples [00:32, 1035.28 examples/s]34210 examples [00:32, 1061.79 examples/s]34327 examples [00:32, 1090.38 examples/s]34437 examples [00:33, 1083.99 examples/s]34548 examples [00:33, 1090.21 examples/s]34661 examples [00:33, 1100.56 examples/s]34777 examples [00:33, 1115.17 examples/s]34892 examples [00:33, 1122.59 examples/s]35005 examples [00:33, 1112.55 examples/s]35117 examples [00:33, 1109.28 examples/s]35229 examples [00:33, 1103.64 examples/s]35340 examples [00:33, 1104.09 examples/s]35454 examples [00:33, 1113.06 examples/s]35569 examples [00:34, 1122.27 examples/s]35682 examples [00:34, 1083.32 examples/s]35794 examples [00:34, 1093.54 examples/s]35908 examples [00:34, 1104.98 examples/s]36019 examples [00:34, 1060.37 examples/s]36126 examples [00:34, 1058.44 examples/s]36239 examples [00:34, 1077.99 examples/s]36352 examples [00:34, 1090.41 examples/s]36463 examples [00:34, 1094.33 examples/s]36573 examples [00:35, 1088.06 examples/s]36684 examples [00:35, 1094.53 examples/s]36798 examples [00:35, 1105.20 examples/s]36911 examples [00:35, 1110.00 examples/s]37028 examples [00:35, 1127.29 examples/s]37141 examples [00:35, 1122.75 examples/s]37254 examples [00:35, 1103.41 examples/s]37365 examples [00:35, 1084.87 examples/s]37474 examples [00:35, 1071.48 examples/s]37584 examples [00:35, 1077.58 examples/s]37697 examples [00:36, 1092.50 examples/s]37809 examples [00:36, 1100.02 examples/s]37920 examples [00:36, 1097.48 examples/s]38035 examples [00:36, 1111.94 examples/s]38147 examples [00:36, 1110.75 examples/s]38263 examples [00:36, 1124.46 examples/s]38376 examples [00:36, 1115.83 examples/s]38490 examples [00:36, 1120.73 examples/s]38603 examples [00:36, 1119.98 examples/s]38716 examples [00:36, 1109.43 examples/s]38831 examples [00:37, 1121.01 examples/s]38944 examples [00:37, 1079.49 examples/s]39053 examples [00:37, 1041.57 examples/s]39158 examples [00:37, 981.62 examples/s] 39271 examples [00:37, 1019.49 examples/s]39384 examples [00:37, 1050.06 examples/s]39496 examples [00:37, 1069.38 examples/s]39604 examples [00:37, 1062.60 examples/s]39718 examples [00:37, 1083.21 examples/s]39827 examples [00:38, 1073.49 examples/s]39941 examples [00:38, 1089.67 examples/s]40051 examples [00:38, 1046.35 examples/s]40161 examples [00:38, 1056.60 examples/s]40273 examples [00:38, 1073.62 examples/s]40386 examples [00:38, 1087.28 examples/s]40498 examples [00:38, 1096.87 examples/s]40608 examples [00:38, 1010.34 examples/s]40711 examples [00:38, 900.24 examples/s] 40805 examples [00:39, 837.97 examples/s]40906 examples [00:39, 882.29 examples/s]41010 examples [00:39, 922.83 examples/s]41105 examples [00:39, 919.91 examples/s]41206 examples [00:39, 934.34 examples/s]41320 examples [00:39, 985.89 examples/s]41435 examples [00:39, 1026.98 examples/s]41545 examples [00:39, 1047.49 examples/s]41659 examples [00:39, 1072.27 examples/s]41768 examples [00:39, 1073.19 examples/s]41877 examples [00:40, 1077.34 examples/s]41986 examples [00:40, 1062.04 examples/s]42093 examples [00:40, 1046.57 examples/s]42198 examples [00:40, 1038.51 examples/s]42311 examples [00:40, 1063.67 examples/s]42420 examples [00:40, 1070.52 examples/s]42528 examples [00:40, 1071.63 examples/s]42636 examples [00:40, 1054.87 examples/s]42749 examples [00:40, 1075.32 examples/s]42863 examples [00:40, 1092.79 examples/s]42973 examples [00:41, 1077.51 examples/s]43081 examples [00:41, 1065.90 examples/s]43189 examples [00:41, 1068.63 examples/s]43297 examples [00:41, 1069.51 examples/s]43408 examples [00:41, 1079.78 examples/s]43517 examples [00:41, 1070.19 examples/s]43625 examples [00:41, 1067.77 examples/s]43732 examples [00:41, 1052.87 examples/s]43839 examples [00:41, 1056.98 examples/s]43950 examples [00:41, 1070.64 examples/s]44058 examples [00:42, 1070.93 examples/s]44166 examples [00:42, 1059.69 examples/s]44273 examples [00:42, 1047.90 examples/s]44378 examples [00:42, 1012.02 examples/s]44484 examples [00:42, 1024.31 examples/s]44592 examples [00:42, 1039.49 examples/s]44698 examples [00:42, 1044.33 examples/s]44803 examples [00:42, 1038.82 examples/s]44909 examples [00:42, 1044.45 examples/s]45018 examples [00:43, 1055.53 examples/s]45128 examples [00:43, 1067.64 examples/s]45238 examples [00:43, 1075.12 examples/s]45346 examples [00:43, 1012.56 examples/s]45449 examples [00:43, 978.97 examples/s] 45553 examples [00:43, 996.05 examples/s]45656 examples [00:43, 1005.27 examples/s]45770 examples [00:43, 1039.88 examples/s]45875 examples [00:43, 1036.13 examples/s]45989 examples [00:43, 1063.59 examples/s]46100 examples [00:44, 1074.88 examples/s]46208 examples [00:44, 1070.81 examples/s]46320 examples [00:44, 1083.75 examples/s]46429 examples [00:44, 1068.89 examples/s]46539 examples [00:44, 1076.66 examples/s]46650 examples [00:44, 1085.71 examples/s]46762 examples [00:44, 1094.86 examples/s]46876 examples [00:44, 1106.16 examples/s]46987 examples [00:44, 1097.95 examples/s]47100 examples [00:44, 1104.91 examples/s]47211 examples [00:45, 1096.77 examples/s]47324 examples [00:45, 1105.23 examples/s]47435 examples [00:45, 1091.76 examples/s]47545 examples [00:45, 1094.18 examples/s]47655 examples [00:45, 1082.66 examples/s]47769 examples [00:45, 1097.50 examples/s]47884 examples [00:45, 1112.11 examples/s]47996 examples [00:45, 1111.05 examples/s]48108 examples [00:45, 1093.92 examples/s]48221 examples [00:45, 1103.00 examples/s]48333 examples [00:46, 1107.56 examples/s]48448 examples [00:46, 1119.13 examples/s]48560 examples [00:46, 1116.13 examples/s]48672 examples [00:46, 1064.04 examples/s]48786 examples [00:46, 1084.56 examples/s]48902 examples [00:46, 1105.11 examples/s]49017 examples [00:46, 1117.00 examples/s]49130 examples [00:46, 1096.46 examples/s]49240 examples [00:46, 1085.36 examples/s]49349 examples [00:47, 1082.81 examples/s]49465 examples [00:47, 1103.07 examples/s]49576 examples [00:47, 1104.64 examples/s]49687 examples [00:47, 1079.52 examples/s]49796 examples [00:47, 1061.67 examples/s]49906 examples [00:47, 1071.04 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7450/50000 [00:00<00:00, 74497.94 examples/s] 42%|████▏     | 20844/50000 [00:00<00:00, 85939.26 examples/s] 69%|██████▉   | 34488/50000 [00:00<00:00, 96672.03 examples/s] 96%|█████████▌| 47888/50000 [00:00<00:00, 105487.19 examples/s]                                                                0 examples [00:00, ? examples/s]91 examples [00:00, 906.01 examples/s]199 examples [00:00, 950.03 examples/s]305 examples [00:00, 979.56 examples/s]403 examples [00:00, 977.98 examples/s]512 examples [00:00, 1008.26 examples/s]622 examples [00:00, 1032.76 examples/s]735 examples [00:00, 1058.31 examples/s]843 examples [00:00, 1063.86 examples/s]951 examples [00:00, 1067.86 examples/s]1062 examples [00:01, 1078.68 examples/s]1171 examples [00:01, 1082.00 examples/s]1284 examples [00:01, 1094.10 examples/s]1395 examples [00:01, 1096.34 examples/s]1504 examples [00:01, 1074.12 examples/s]1618 examples [00:01, 1090.53 examples/s]1730 examples [00:01, 1097.84 examples/s]1845 examples [00:01, 1110.34 examples/s]1957 examples [00:01, 1112.74 examples/s]2069 examples [00:01, 1094.02 examples/s]2179 examples [00:02, 1076.50 examples/s]2288 examples [00:02, 1079.22 examples/s]2396 examples [00:02, 1040.65 examples/s]2501 examples [00:02, 979.64 examples/s] 2610 examples [00:02, 1008.90 examples/s]2718 examples [00:02, 1027.55 examples/s]2830 examples [00:02, 1052.45 examples/s]2946 examples [00:02, 1082.24 examples/s]3059 examples [00:02, 1094.40 examples/s]3170 examples [00:02, 1096.49 examples/s]3286 examples [00:03, 1113.88 examples/s]3402 examples [00:03, 1124.33 examples/s]3516 examples [00:03, 1128.72 examples/s]3631 examples [00:03, 1131.92 examples/s]3745 examples [00:03, 1132.97 examples/s]3860 examples [00:03, 1137.11 examples/s]3974 examples [00:03, 1128.38 examples/s]4087 examples [00:03, 1108.42 examples/s]4198 examples [00:03, 1073.12 examples/s]4307 examples [00:03, 1077.83 examples/s]4420 examples [00:04, 1092.86 examples/s]4534 examples [00:04, 1103.95 examples/s]4645 examples [00:04, 1100.90 examples/s]4756 examples [00:04, 1049.63 examples/s]4866 examples [00:04, 1062.44 examples/s]4973 examples [00:04, 1051.31 examples/s]5085 examples [00:04, 1069.42 examples/s]5197 examples [00:04, 1083.71 examples/s]5309 examples [00:04, 1091.77 examples/s]5422 examples [00:05, 1100.14 examples/s]5540 examples [00:05, 1120.94 examples/s]5655 examples [00:05, 1128.41 examples/s]5768 examples [00:05, 1086.32 examples/s]5878 examples [00:05, 1078.44 examples/s]5988 examples [00:05, 1082.86 examples/s]6106 examples [00:05, 1109.01 examples/s]6220 examples [00:05, 1117.43 examples/s]6332 examples [00:05, 1114.39 examples/s]6444 examples [00:05, 1113.48 examples/s]6556 examples [00:06, 1084.49 examples/s]6665 examples [00:06, 1061.52 examples/s]6773 examples [00:06, 1065.68 examples/s]6888 examples [00:06, 1088.16 examples/s]6999 examples [00:06, 1091.48 examples/s]7109 examples [00:06, 1084.78 examples/s]7218 examples [00:06, 1078.14 examples/s]7326 examples [00:06, 1067.00 examples/s]7435 examples [00:06, 1068.22 examples/s]7542 examples [00:06, 1059.04 examples/s]7651 examples [00:07, 1067.43 examples/s]7764 examples [00:07, 1084.31 examples/s]7874 examples [00:07, 1086.97 examples/s]7985 examples [00:07, 1092.27 examples/s]8095 examples [00:07, 1054.59 examples/s]8204 examples [00:07, 1063.86 examples/s]8318 examples [00:07, 1084.96 examples/s]8430 examples [00:07, 1093.21 examples/s]8540 examples [00:07, 1085.92 examples/s]8651 examples [00:07, 1089.17 examples/s]8762 examples [00:08, 1093.97 examples/s]8872 examples [00:08, 1084.00 examples/s]8981 examples [00:08, 1061.43 examples/s]9092 examples [00:08, 1075.47 examples/s]9202 examples [00:08, 1080.43 examples/s]9311 examples [00:08, 1080.37 examples/s]9420 examples [00:08, 1060.31 examples/s]9529 examples [00:08, 1067.26 examples/s]9640 examples [00:08, 1078.81 examples/s]9750 examples [00:09, 1082.60 examples/s]9859 examples [00:09, 1065.45 examples/s]9966 examples [00:09, 1054.79 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteA2KE1Z/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteA2KE1Z/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f239ec10620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f239ec10620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f239ec10620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f23280a1208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f23280d7c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f239ec10620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f239ec10620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f239ec10620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2385c445c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2385c44320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f23201b9158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f23201b9158> 

  function with postional parmater data_info <function split_train_valid at 0x7f23201b9158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f23201b9268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f23201b9268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f23201b9268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=664ac45354aa0e4100fa76e0304eb3dfe3a1cccb1fdfc63d4222cc8e62df742a
  Stored in directory: /tmp/pip-ephem-wheel-cache-sk1pxjgl/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:54:39, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:34:08, 15.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:57:05, 21.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:40:26, 31.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:21:29, 44.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.24M/862M [00:01<3:43:40, 63.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:35:36, 90.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:48:16, 130kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<1:15:22, 185kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<52:29, 263kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:02<36:35, 375kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<25:31, 534kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<17:50, 759kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<13:17, 1.02MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:03<09:22, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<15:15, 880kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<12:52, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<09:32, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<09:00, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<09:34, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:07<07:32, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:07<05:26, 2.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<10:55, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<09:17, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<06:54, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<07:29, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<08:01, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<06:12, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<04:31, 2.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<08:17, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<07:10, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<05:17, 2.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:47, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<07:30, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<05:55, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:15<04:17, 3.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<1:35:58, 136kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<1:08:27, 190kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:17<48:07, 270kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<36:38, 353kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<28:17, 457kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:19<20:26, 632kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<16:20, 788kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<14:03, 916kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<10:24, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<07:24, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<16:28, 777kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<12:52, 994kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<09:16, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<09:25, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<09:14, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<07:07, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:07, 2.48MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:34:39, 134kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:07:31, 188kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<47:30, 266kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<36:06, 349kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<27:50, 453kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<20:00, 629kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<14:08, 888kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<16:45, 748kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:01, 962kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<09:24, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:30, 1.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:14, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:06, 1.75MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:07, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:32:05, 135kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:05:42, 189kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<46:13, 267kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<35:07, 351kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<27:05, 455kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<19:29, 632kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<13:47, 890kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:27, 848kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:21, 1.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<08:15, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:37, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:31, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:35, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:45, 2.55MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:42:21, 118kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:12:50, 166kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<51:08, 236kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<38:31, 313kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<29:24, 410kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<21:04, 571kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<14:52, 807kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<16:05, 745kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<12:29, 959kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:59, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:05, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:46, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:44, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:50, 2.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<14:43, 804kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:38, 1.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:27, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:26, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:38, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:42, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:50, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<14:56, 783kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<11:47, 993kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:31, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:26, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:36, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:40, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:48, 2.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<14:08, 818kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<11:13, 1.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:07, 1.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:08, 1.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<08:15, 1.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:26, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:39, 2.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<14:30, 788kB/s] .vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<11:25, 1.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:15, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:12, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:16, 1.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:19, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<04:35, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:17, 1.55MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:22, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:44, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:34, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<03:48, 2.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<13:39, 817kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:49, 1.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:52, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:52, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:47, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:00, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:52, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:39, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:16, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:48, 2.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<12:43, 860kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:08, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:22, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:27, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:43, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:55, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:16, 2.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:38, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:35, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:54, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:44, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:30, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:06, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:42, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:52, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:01, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:28, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:24, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:53, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:33, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<12:52, 817kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<10:11, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:21, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<3:33:11, 49.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<2:31:37, 68.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<1:46:30, 98.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<1:14:37, 140kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<54:45, 190kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<39:26, 263kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<27:50, 372kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<21:38, 477kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<17:30, 589kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<12:44, 809kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<09:06, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:09, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<07:34, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:35, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:03, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:37, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:07, 1.98MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:45, 2.69MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:36, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:03, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<03:46, 2.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<02:45, 3.64MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<3:23:12, 49.4kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<2:24:24, 69.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<1:41:31, 98.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<1:10:52, 141kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<1:01:23, 162kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<44:04, 226kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<31:03, 320kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<23:48, 416kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<17:46, 557kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<12:39, 780kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<10:55, 900kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:54, 992kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:28, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:21, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<13:22, 730kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<10:24, 937kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:32, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:23, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:18, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:34, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:01, 2.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:50, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:51, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:21, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:05, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:39, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:28, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:14, 2.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<17:12, 551kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<13:06, 723kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<09:25, 1.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:35, 1.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:05, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:09, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:25, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<17:48, 525kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<13:30, 691kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<09:39, 965kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<08:43, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<08:15, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<06:12, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:29, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:03, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:13, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:53, 2.36MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:45, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:20, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:03, 2.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:45, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:02, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:46, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:34, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:10, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:03, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:57, 3.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:36, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:55, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:41, 2.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:28, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:11, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<02:57, 2.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:50, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:04, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:45, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:03, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<02:55, 2.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<10:34, 817kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<08:24, 1.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<06:06, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:05, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:09, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:42, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:23, 2.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:32, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:31, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:05, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<04:40, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:12, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:02, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:56, 2.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:26, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:45, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:33, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:15, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:48, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:46, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:43, 3.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:19, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:59, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:31, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:58, 1.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:52, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<02:48, 2.89MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:32, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:48, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:32, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:11, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:47, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:43, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<02:41, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<07:47, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<06:20, 1.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<04:37, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:55, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:10, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:03, 1.94MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<02:56, 2.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<09:45, 801kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<07:44, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:35, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:33, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:40, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:24, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<03:10, 2.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<09:48, 783kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:43, 993kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:34, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:31, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:38, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:22, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<03:08, 2.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<09:38, 783kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<07:36, 991kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:31, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:27, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:10, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:01, 2.45MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:43, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:09, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:06, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:44, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:18, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:21, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:45, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:27, 2.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:36, 2.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:21, 2.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:00, 1.79MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:12, 2.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:19, 3.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:16, 861kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:35, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:47, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:51, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:02, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:52, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:49, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:11, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:42, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:47, 2.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:26, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:08, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:16, 3.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<08:20, 821kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:36, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:47, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:48, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:52, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:43, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:43, 2.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:54, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:29, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:35, 2.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:14, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:48, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:01, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<02:12, 2.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<08:00, 820kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:19, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:34, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:35, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:36, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:36, 2.47MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:12, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:39, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:42, 2.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:14, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:44, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:57, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:08, 2.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<07:43, 815kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<06:06, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:24, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:24, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:27, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:28, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:29, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<07:49, 786kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:09, 997kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:28, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:24, 1.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:29, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:29, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:30, 2.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:40, 784kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<06:02, 994kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:22, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:19, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:24, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:24, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:27, 2.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:30, 782kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:53, 997kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:15, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:15, 1.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:11, 1.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:43<02:17, 2.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<05:44, 1.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:39, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:24, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:34, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:48, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:58, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:08, 2.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:08, 911kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:55, 1.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:35, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:40, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:09, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:20, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:51, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:15, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:35, 2.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<01:52, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<06:37, 813kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<05:14, 1.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:47, 1.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:47, 1.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:52, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:57, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:07, 2.47MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:34, 1.47MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:05, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:16, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:41, 1.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:01, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:22, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:42, 2.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:35, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:05, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:16, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:01, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:21, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:41, 2.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:20, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:35, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:38, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:52, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:08, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:25, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:45, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:41, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:07, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:17, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:52, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:14, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:36, 2.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:52, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:13, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:21, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:37, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:54, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:15, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:37, 2.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:03, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:38, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:58, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:19, 1.93MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:08, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:36, 2.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:03, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:27, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:55, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:25, 3.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:18, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:06, 2.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:34, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:00, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:23, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:52, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:21, 3.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:22, 1.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:49, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<02:04, 2.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:19, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:05, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:33, 2.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:57, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:18, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:48, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:18, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:46, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:23, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:45, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:03, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:18, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:50, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:19, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<04:31, 853kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:35, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:36, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:37, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:39, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:29, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<04:42, 789kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:42, 1.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:40, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:38, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:41, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:03, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:28, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:19, 829kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:25, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<02:28, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:27, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:06, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:33, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:49, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:04, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:38, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:10, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:08, 813kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:16, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:21, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:20, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:23, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:50, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:18, 2.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:24, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:02, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:30, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:43, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:54, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:28, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:04, 2.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:39, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:30, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:08, 2.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:25, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:41, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:20, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:57, 3.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:23, 872kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:42, 1.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:57, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:58, 1.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:02, 1.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:35, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:07, 2.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:22, 836kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:40, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:55, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<01:22, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<05:25, 506kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:24, 623kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:12, 851kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<02:16, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:29, 1.07MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:20, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:46, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:15, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:27, 756kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:42, 960kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:56, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:52, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:54, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:01, 2.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:46, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:30, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:07, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:17, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:27, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:07, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:49, 2.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:11, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:05, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:49, 2.80MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:02, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:59, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:45, 2.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:11, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:40, 3.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:35, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:20, 1.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:59, 2.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:06, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:59, 2.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:42, 2.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:24, 821kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:54, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:22, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:21, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:09, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:58, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:07, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:52, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:37, 2.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:03, 862kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:38, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:10, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:10, 1.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:11, 1.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:55, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:39, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:04, 791kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:37, 1.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:10, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:07, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:08, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:36, 2.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:05, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:54, 1.65MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:39, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<00:28, 3.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:32, 921kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:24, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:03, 1.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:44, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:46, 765kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:23, 973kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:59, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:56, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:56, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:43, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:30, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:43, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:31, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:40, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:23, 2.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:29, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:21, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:33, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:18, 3.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:52, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:42, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:30, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:33, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:17, 2.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:59, 802kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:47, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:33, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:31, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:31, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:16, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:47, 830kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:37, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:26, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:24, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:13, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:19, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:13, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 3.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:08, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:08, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:05, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:18, 817kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:13, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:01, 2.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.17MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<24:19:44,  4.57it/s]  0%|          | 845/400000 [00:00<16:59:54,  6.52it/s]  0%|          | 1797/400000 [00:00<11:52:26,  9.32it/s]  1%|          | 2737/400000 [00:00<8:17:44, 13.30it/s]   1%|          | 3701/400000 [00:00<5:47:46, 18.99it/s]  1%|          | 4589/400000 [00:00<4:03:07, 27.11it/s]  1%|▏         | 5484/400000 [00:00<2:50:01, 38.67it/s]  2%|▏         | 6343/400000 [00:00<1:58:59, 55.14it/s]  2%|▏         | 7191/400000 [00:01<1:23:20, 78.55it/s]  2%|▏         | 8142/400000 [00:01<58:24, 111.82it/s]   2%|▏         | 9113/400000 [00:01<40:58, 158.96it/s]  3%|▎         | 10068/400000 [00:01<28:49, 225.48it/s]  3%|▎         | 11044/400000 [00:01<20:19, 318.95it/s]  3%|▎         | 12059/400000 [00:01<14:22, 449.59it/s]  3%|▎         | 13039/400000 [00:01<10:14, 629.88it/s]  4%|▎         | 14014/400000 [00:01<07:20, 875.56it/s]  4%|▎         | 14981/400000 [00:01<05:20, 1201.85it/s]  4%|▍         | 15935/400000 [00:01<03:56, 1626.37it/s]  4%|▍         | 16880/400000 [00:02<02:57, 2155.38it/s]  4%|▍         | 17855/400000 [00:02<02:15, 2812.39it/s]  5%|▍         | 18868/400000 [00:02<01:46, 3590.29it/s]  5%|▍         | 19833/400000 [00:02<01:25, 4423.06it/s]  5%|▌         | 20797/400000 [00:02<01:13, 5138.73it/s]  5%|▌         | 21733/400000 [00:02<01:03, 5941.90it/s]  6%|▌         | 22677/400000 [00:02<00:56, 6683.72it/s]  6%|▌         | 23609/400000 [00:02<00:51, 7238.56it/s]  6%|▌         | 24534/400000 [00:02<00:48, 7742.61it/s]  6%|▋         | 25457/400000 [00:02<00:46, 8116.02it/s]  7%|▋         | 26377/400000 [00:03<00:44, 8406.56it/s]  7%|▋         | 27328/400000 [00:03<00:42, 8709.28it/s]  7%|▋         | 28370/400000 [00:03<00:40, 9158.75it/s]  7%|▋         | 29333/400000 [00:03<00:40, 9264.05it/s]  8%|▊         | 30325/400000 [00:03<00:39, 9451.40it/s]  8%|▊         | 31301/400000 [00:03<00:38, 9541.73it/s]  8%|▊         | 32273/400000 [00:03<00:39, 9394.14it/s]  8%|▊         | 33248/400000 [00:03<00:38, 9494.46it/s]  9%|▊         | 34207/400000 [00:03<00:39, 9291.26it/s]  9%|▉         | 35164/400000 [00:03<00:38, 9371.95it/s]  9%|▉         | 36107/400000 [00:04<00:39, 9209.67it/s]  9%|▉         | 37125/400000 [00:04<00:38, 9480.77it/s] 10%|▉         | 38107/400000 [00:04<00:37, 9578.63it/s] 10%|▉         | 39069/400000 [00:04<00:37, 9498.76it/s] 10%|█         | 40100/400000 [00:04<00:37, 9725.90it/s] 10%|█         | 41089/400000 [00:04<00:36, 9773.67it/s] 11%|█         | 42069/400000 [00:04<00:36, 9736.01it/s] 11%|█         | 43062/400000 [00:04<00:36, 9791.73it/s] 11%|█         | 44043/400000 [00:04<00:36, 9725.38it/s] 11%|█▏        | 45017/400000 [00:04<00:36, 9717.84it/s] 11%|█▏        | 45990/400000 [00:05<00:36, 9663.87it/s] 12%|█▏        | 46957/400000 [00:05<00:36, 9587.03it/s] 12%|█▏        | 47917/400000 [00:05<00:36, 9526.00it/s] 12%|█▏        | 48871/400000 [00:05<00:38, 9185.05it/s] 12%|█▏        | 49876/400000 [00:05<00:37, 9426.25it/s] 13%|█▎        | 50874/400000 [00:05<00:36, 9584.34it/s] 13%|█▎        | 51836/400000 [00:05<00:37, 9351.79it/s] 13%|█▎        | 52806/400000 [00:05<00:36, 9452.93it/s] 13%|█▎        | 53755/400000 [00:05<00:36, 9461.69it/s] 14%|█▎        | 54704/400000 [00:06<00:37, 9305.50it/s] 14%|█▍        | 55637/400000 [00:06<00:37, 9226.70it/s] 14%|█▍        | 56634/400000 [00:06<00:36, 9435.08it/s] 14%|█▍        | 57603/400000 [00:06<00:36, 9509.57it/s] 15%|█▍        | 58556/400000 [00:06<00:36, 9435.32it/s] 15%|█▍        | 59557/400000 [00:06<00:35, 9599.12it/s] 15%|█▌        | 60519/400000 [00:06<00:35, 9569.24it/s] 15%|█▌        | 61478/400000 [00:06<00:36, 9401.61it/s] 16%|█▌        | 62475/400000 [00:06<00:35, 9562.15it/s] 16%|█▌        | 63433/400000 [00:06<00:35, 9426.95it/s] 16%|█▌        | 64378/400000 [00:07<00:36, 9252.14it/s] 16%|█▋        | 65324/400000 [00:07<00:35, 9311.77it/s] 17%|█▋        | 66294/400000 [00:07<00:35, 9422.21it/s] 17%|█▋        | 67271/400000 [00:07<00:34, 9523.91it/s] 17%|█▋        | 68289/400000 [00:07<00:34, 9710.27it/s] 17%|█▋        | 69287/400000 [00:07<00:33, 9786.61it/s] 18%|█▊        | 70267/400000 [00:07<00:34, 9613.91it/s] 18%|█▊        | 71231/400000 [00:07<00:35, 9387.56it/s] 18%|█▊        | 72173/400000 [00:07<00:35, 9283.99it/s] 18%|█▊        | 73115/400000 [00:07<00:35, 9322.47it/s] 19%|█▊        | 74061/400000 [00:08<00:34, 9361.01it/s] 19%|█▊        | 74999/400000 [00:08<00:35, 9211.88it/s] 19%|█▉        | 75922/400000 [00:08<00:36, 8987.28it/s] 19%|█▉        | 76823/400000 [00:08<00:36, 8897.02it/s] 19%|█▉        | 77787/400000 [00:08<00:35, 9105.44it/s] 20%|█▉        | 78727/400000 [00:08<00:34, 9189.76it/s] 20%|█▉        | 79648/400000 [00:08<00:34, 9187.84it/s] 20%|██        | 80569/400000 [00:08<00:35, 9109.52it/s] 20%|██        | 81510/400000 [00:08<00:34, 9195.78it/s] 21%|██        | 82434/400000 [00:08<00:34, 9207.23it/s] 21%|██        | 83356/400000 [00:09<00:34, 9115.54it/s] 21%|██        | 84269/400000 [00:09<00:34, 9100.39it/s] 21%|██▏       | 85198/400000 [00:09<00:34, 9154.36it/s] 22%|██▏       | 86129/400000 [00:09<00:34, 9197.84it/s] 22%|██▏       | 87050/400000 [00:09<00:34, 9079.36it/s] 22%|██▏       | 87959/400000 [00:09<00:34, 9013.05it/s] 22%|██▏       | 88866/400000 [00:09<00:34, 9028.19it/s] 22%|██▏       | 89779/400000 [00:09<00:34, 9056.54it/s] 23%|██▎       | 90763/400000 [00:09<00:33, 9276.36it/s] 23%|██▎       | 91693/400000 [00:10<00:33, 9197.97it/s] 23%|██▎       | 92616/400000 [00:10<00:33, 9202.30it/s] 23%|██▎       | 93538/400000 [00:10<00:34, 8984.88it/s] 24%|██▎       | 94505/400000 [00:10<00:33, 9178.11it/s] 24%|██▍       | 95486/400000 [00:10<00:32, 9357.62it/s] 24%|██▍       | 96478/400000 [00:10<00:31, 9518.95it/s] 24%|██▍       | 97433/400000 [00:10<00:32, 9405.13it/s] 25%|██▍       | 98391/400000 [00:10<00:31, 9453.46it/s] 25%|██▍       | 99338/400000 [00:10<00:31, 9396.53it/s] 25%|██▌       | 100279/400000 [00:10<00:32, 9301.03it/s] 25%|██▌       | 101220/400000 [00:11<00:32, 9333.43it/s] 26%|██▌       | 102155/400000 [00:11<00:32, 9272.89it/s] 26%|██▌       | 103092/400000 [00:11<00:31, 9299.86it/s] 26%|██▌       | 104023/400000 [00:11<00:31, 9293.01it/s] 26%|██▋       | 105028/400000 [00:11<00:31, 9506.29it/s] 27%|██▋       | 106012/400000 [00:11<00:30, 9602.86it/s] 27%|██▋       | 107025/400000 [00:11<00:30, 9752.49it/s] 27%|██▋       | 108002/400000 [00:11<00:31, 9160.26it/s] 27%|██▋       | 108971/400000 [00:11<00:31, 9312.57it/s] 27%|██▋       | 109991/400000 [00:11<00:30, 9560.08it/s] 28%|██▊       | 110954/400000 [00:12<00:30, 9559.03it/s] 28%|██▊       | 111915/400000 [00:12<00:30, 9516.59it/s] 28%|██▊       | 112870/400000 [00:12<00:30, 9284.23it/s] 28%|██▊       | 113802/400000 [00:12<00:31, 9008.18it/s] 29%|██▊       | 114707/400000 [00:12<00:31, 8941.33it/s] 29%|██▉       | 115619/400000 [00:12<00:31, 8993.29it/s] 29%|██▉       | 116608/400000 [00:12<00:30, 9243.90it/s] 29%|██▉       | 117536/400000 [00:12<00:30, 9129.34it/s] 30%|██▉       | 118452/400000 [00:12<00:31, 9068.31it/s] 30%|██▉       | 119401/400000 [00:12<00:30, 9190.19it/s] 30%|███       | 120322/400000 [00:13<00:30, 9144.64it/s] 30%|███       | 121343/400000 [00:13<00:29, 9440.06it/s] 31%|███       | 122291/400000 [00:13<00:30, 9181.73it/s] 31%|███       | 123241/400000 [00:13<00:29, 9272.59it/s] 31%|███       | 124217/400000 [00:13<00:29, 9411.13it/s] 31%|███▏      | 125163/400000 [00:13<00:29, 9423.55it/s] 32%|███▏      | 126166/400000 [00:13<00:28, 9596.17it/s] 32%|███▏      | 127128/400000 [00:13<00:29, 9398.06it/s] 32%|███▏      | 128098/400000 [00:13<00:28, 9485.52it/s] 32%|███▏      | 129052/400000 [00:13<00:28, 9499.77it/s] 33%|███▎      | 130004/400000 [00:14<00:29, 9302.19it/s] 33%|███▎      | 130937/400000 [00:14<00:29, 9200.22it/s] 33%|███▎      | 131859/400000 [00:14<00:29, 9105.75it/s] 33%|███▎      | 132771/400000 [00:14<00:29, 9024.23it/s] 33%|███▎      | 133803/400000 [00:14<00:28, 9376.60it/s] 34%|███▎      | 134745/400000 [00:14<00:28, 9373.16it/s] 34%|███▍      | 135686/400000 [00:14<00:28, 9189.74it/s] 34%|███▍      | 136608/400000 [00:14<00:29, 8951.10it/s] 34%|███▍      | 137507/400000 [00:14<00:29, 8803.98it/s] 35%|███▍      | 138496/400000 [00:15<00:28, 9102.31it/s] 35%|███▍      | 139411/400000 [00:15<00:28, 9102.13it/s] 35%|███▌      | 140341/400000 [00:15<00:28, 9158.57it/s] 35%|███▌      | 141260/400000 [00:15<00:28, 8943.02it/s] 36%|███▌      | 142178/400000 [00:15<00:28, 9012.57it/s] 36%|███▌      | 143098/400000 [00:15<00:28, 9066.78it/s] 36%|███▌      | 144007/400000 [00:15<00:28, 9072.38it/s] 36%|███▌      | 144990/400000 [00:15<00:27, 9284.23it/s] 36%|███▋      | 145921/400000 [00:15<00:27, 9179.86it/s] 37%|███▋      | 146894/400000 [00:15<00:27, 9337.60it/s] 37%|███▋      | 147832/400000 [00:16<00:26, 9348.11it/s] 37%|███▋      | 148769/400000 [00:16<00:27, 9203.12it/s] 37%|███▋      | 149700/400000 [00:16<00:27, 9234.06it/s] 38%|███▊      | 150625/400000 [00:16<00:27, 9095.63it/s] 38%|███▊      | 151536/400000 [00:16<00:27, 9065.70it/s] 38%|███▊      | 152466/400000 [00:16<00:27, 9134.03it/s] 38%|███▊      | 153426/400000 [00:16<00:26, 9266.50it/s] 39%|███▊      | 154404/400000 [00:16<00:26, 9413.18it/s] 39%|███▉      | 155347/400000 [00:16<00:26, 9332.60it/s] 39%|███▉      | 156282/400000 [00:16<00:26, 9331.72it/s] 39%|███▉      | 157216/400000 [00:17<00:26, 9230.60it/s] 40%|███▉      | 158147/400000 [00:17<00:26, 9252.88it/s] 40%|███▉      | 159077/400000 [00:17<00:26, 9265.04it/s] 40%|████      | 160004/400000 [00:17<00:27, 8752.12it/s] 40%|████      | 160886/400000 [00:17<00:27, 8568.51it/s] 40%|████      | 161749/400000 [00:17<00:27, 8566.42it/s] 41%|████      | 162642/400000 [00:17<00:27, 8670.30it/s] 41%|████      | 163553/400000 [00:17<00:26, 8795.56it/s] 41%|████      | 164435/400000 [00:17<00:27, 8644.31it/s] 41%|████▏     | 165340/400000 [00:18<00:26, 8761.50it/s] 42%|████▏     | 166272/400000 [00:18<00:26, 8920.22it/s] 42%|████▏     | 167240/400000 [00:18<00:25, 9132.72it/s] 42%|████▏     | 168183/400000 [00:18<00:25, 9219.63it/s] 42%|████▏     | 169108/400000 [00:18<00:25, 9194.27it/s] 43%|████▎     | 170029/400000 [00:18<00:25, 9184.04it/s] 43%|████▎     | 170996/400000 [00:18<00:24, 9322.59it/s] 43%|████▎     | 171930/400000 [00:18<00:24, 9317.49it/s] 43%|████▎     | 172863/400000 [00:18<00:24, 9294.08it/s] 43%|████▎     | 173826/400000 [00:18<00:24, 9390.12it/s] 44%|████▎     | 174766/400000 [00:19<00:24, 9245.54it/s] 44%|████▍     | 175692/400000 [00:19<00:24, 9231.61it/s] 44%|████▍     | 176637/400000 [00:19<00:24, 9295.56it/s] 44%|████▍     | 177603/400000 [00:19<00:23, 9401.72it/s] 45%|████▍     | 178544/400000 [00:19<00:23, 9248.01it/s] 45%|████▍     | 179472/400000 [00:19<00:23, 9256.53it/s] 45%|████▌     | 180399/400000 [00:19<00:24, 9111.82it/s] 45%|████▌     | 181379/400000 [00:19<00:23, 9307.79it/s] 46%|████▌     | 182403/400000 [00:19<00:22, 9568.62it/s] 46%|████▌     | 183363/400000 [00:19<00:23, 9388.06it/s] 46%|████▌     | 184305/400000 [00:20<00:23, 9347.07it/s] 46%|████▋     | 185260/400000 [00:20<00:22, 9405.45it/s] 47%|████▋     | 186203/400000 [00:20<00:22, 9371.74it/s] 47%|████▋     | 187142/400000 [00:20<00:22, 9281.58it/s] 47%|████▋     | 188072/400000 [00:20<00:23, 8868.71it/s] 47%|████▋     | 188978/400000 [00:20<00:23, 8924.73it/s] 47%|████▋     | 189885/400000 [00:20<00:23, 8966.41it/s] 48%|████▊     | 190834/400000 [00:20<00:22, 9115.61it/s] 48%|████▊     | 191763/400000 [00:20<00:22, 9165.02it/s] 48%|████▊     | 192682/400000 [00:20<00:22, 9159.81it/s] 48%|████▊     | 193600/400000 [00:21<00:22, 9146.32it/s] 49%|████▊     | 194535/400000 [00:21<00:22, 9206.45it/s] 49%|████▉     | 195457/400000 [00:21<00:22, 9194.01it/s] 49%|████▉     | 196377/400000 [00:21<00:22, 9142.71it/s] 49%|████▉     | 197292/400000 [00:21<00:22, 8989.54it/s] 50%|████▉     | 198192/400000 [00:21<00:22, 8859.72it/s] 50%|████▉     | 199079/400000 [00:21<00:22, 8806.64it/s] 50%|████▉     | 199961/400000 [00:21<00:22, 8718.95it/s] 50%|█████     | 200834/400000 [00:21<00:22, 8715.48it/s] 50%|█████     | 201707/400000 [00:21<00:22, 8653.66it/s] 51%|█████     | 202576/400000 [00:22<00:22, 8664.40it/s] 51%|█████     | 203580/400000 [00:22<00:21, 9034.48it/s] 51%|█████     | 204488/400000 [00:22<00:21, 9018.15it/s] 51%|█████▏    | 205479/400000 [00:22<00:20, 9267.22it/s] 52%|█████▏    | 206410/400000 [00:22<00:20, 9235.54it/s] 52%|█████▏    | 207426/400000 [00:22<00:20, 9492.25it/s] 52%|█████▏    | 208379/400000 [00:22<00:20, 9498.54it/s] 52%|█████▏    | 209332/400000 [00:22<00:20, 9429.97it/s] 53%|█████▎    | 210277/400000 [00:22<00:20, 9412.08it/s] 53%|█████▎    | 211234/400000 [00:22<00:19, 9457.07it/s] 53%|█████▎    | 212211/400000 [00:23<00:19, 9548.27it/s] 53%|█████▎    | 213209/400000 [00:23<00:19, 9673.14it/s] 54%|█████▎    | 214178/400000 [00:23<00:19, 9492.61it/s] 54%|█████▍    | 215129/400000 [00:23<00:19, 9485.76it/s] 54%|█████▍    | 216079/400000 [00:23<00:19, 9481.74it/s] 54%|█████▍    | 217028/400000 [00:23<00:20, 9126.07it/s] 54%|█████▍    | 217985/400000 [00:23<00:19, 9253.00it/s] 55%|█████▍    | 218966/400000 [00:23<00:19, 9412.49it/s] 55%|█████▍    | 219969/400000 [00:23<00:18, 9587.84it/s] 55%|█████▌    | 220931/400000 [00:24<00:18, 9573.16it/s] 55%|█████▌    | 221891/400000 [00:24<00:18, 9408.45it/s] 56%|█████▌    | 222885/400000 [00:24<00:18, 9561.75it/s] 56%|█████▌    | 223844/400000 [00:24<00:18, 9403.93it/s] 56%|█████▌    | 224835/400000 [00:24<00:18, 9550.11it/s] 56%|█████▋    | 225792/400000 [00:24<00:18, 9471.71it/s] 57%|█████▋    | 226767/400000 [00:24<00:18, 9550.73it/s] 57%|█████▋    | 227819/400000 [00:24<00:17, 9820.61it/s] 57%|█████▋    | 228804/400000 [00:24<00:17, 9722.77it/s] 57%|█████▋    | 229779/400000 [00:24<00:18, 9443.63it/s] 58%|█████▊    | 230727/400000 [00:25<00:18, 9258.00it/s] 58%|█████▊    | 231656/400000 [00:25<00:18, 9212.30it/s] 58%|█████▊    | 232658/400000 [00:25<00:17, 9439.16it/s] 58%|█████▊    | 233605/400000 [00:25<00:18, 9185.76it/s] 59%|█████▊    | 234536/400000 [00:25<00:17, 9219.99it/s] 59%|█████▉    | 235461/400000 [00:25<00:18, 8836.78it/s] 59%|█████▉    | 236443/400000 [00:25<00:17, 9109.42it/s] 59%|█████▉    | 237375/400000 [00:25<00:17, 9170.45it/s] 60%|█████▉    | 238297/400000 [00:25<00:17, 9105.35it/s] 60%|█████▉    | 239211/400000 [00:25<00:17, 9035.65it/s] 60%|██████    | 240143/400000 [00:26<00:17, 9117.22it/s] 60%|██████    | 241100/400000 [00:26<00:17, 9247.74it/s] 61%|██████    | 242110/400000 [00:26<00:16, 9486.47it/s] 61%|██████    | 243062/400000 [00:26<00:16, 9263.88it/s] 61%|██████    | 244019/400000 [00:26<00:16, 9351.11it/s] 61%|██████    | 244957/400000 [00:26<00:16, 9243.53it/s] 61%|██████▏   | 245884/400000 [00:26<00:16, 9210.91it/s] 62%|██████▏   | 246821/400000 [00:26<00:16, 9255.63it/s] 62%|██████▏   | 247795/400000 [00:26<00:16, 9393.76it/s] 62%|██████▏   | 248803/400000 [00:26<00:15, 9586.79it/s] 62%|██████▏   | 249764/400000 [00:27<00:16, 9337.37it/s] 63%|██████▎   | 250701/400000 [00:27<00:15, 9344.61it/s] 63%|██████▎   | 251638/400000 [00:27<00:16, 8774.22it/s] 63%|██████▎   | 252524/400000 [00:27<00:17, 8567.94it/s] 63%|██████▎   | 253388/400000 [00:27<00:17, 8507.43it/s] 64%|██████▎   | 254331/400000 [00:27<00:16, 8763.74it/s] 64%|██████▍   | 255262/400000 [00:27<00:16, 8918.98it/s] 64%|██████▍   | 256159/400000 [00:27<00:16, 8726.67it/s] 64%|██████▍   | 257036/400000 [00:27<00:16, 8537.34it/s] 64%|██████▍   | 257952/400000 [00:28<00:16, 8713.16it/s] 65%|██████▍   | 258827/400000 [00:28<00:16, 8564.95it/s] 65%|██████▍   | 259687/400000 [00:28<00:16, 8408.34it/s] 65%|██████▌   | 260531/400000 [00:28<00:17, 8140.15it/s] 65%|██████▌   | 261349/400000 [00:28<00:17, 8087.33it/s] 66%|██████▌   | 262238/400000 [00:28<00:16, 8309.85it/s] 66%|██████▌   | 263178/400000 [00:28<00:15, 8608.78it/s] 66%|██████▌   | 264180/400000 [00:28<00:15, 8988.17it/s] 66%|██████▋   | 265141/400000 [00:28<00:14, 9164.68it/s] 67%|██████▋   | 266090/400000 [00:28<00:14, 9259.26it/s] 67%|██████▋   | 267021/400000 [00:29<00:14, 9213.33it/s] 67%|██████▋   | 267946/400000 [00:29<00:14, 9141.35it/s] 67%|██████▋   | 268869/400000 [00:29<00:14, 9166.56it/s] 67%|██████▋   | 269788/400000 [00:29<00:14, 9158.77it/s] 68%|██████▊   | 270706/400000 [00:29<00:14, 9013.22it/s] 68%|██████▊   | 271609/400000 [00:29<00:14, 8998.19it/s] 68%|██████▊   | 272540/400000 [00:29<00:14, 9089.38it/s] 68%|██████▊   | 273534/400000 [00:29<00:13, 9327.37it/s] 69%|██████▊   | 274469/400000 [00:29<00:13, 8977.23it/s] 69%|██████▉   | 275393/400000 [00:30<00:13, 9052.45it/s] 69%|██████▉   | 276302/400000 [00:30<00:13, 9002.49it/s] 69%|██████▉   | 277205/400000 [00:30<00:13, 8982.47it/s] 70%|██████▉   | 278105/400000 [00:30<00:13, 8984.14it/s] 70%|██████▉   | 279029/400000 [00:30<00:13, 9058.33it/s] 70%|██████▉   | 279997/400000 [00:30<00:12, 9236.12it/s] 70%|███████   | 280923/400000 [00:30<00:13, 9152.51it/s] 70%|███████   | 281854/400000 [00:30<00:12, 9196.82it/s] 71%|███████   | 282831/400000 [00:30<00:12, 9360.09it/s] 71%|███████   | 283819/400000 [00:30<00:12, 9508.47it/s] 71%|███████   | 284772/400000 [00:31<00:12, 9419.93it/s] 71%|███████▏  | 285716/400000 [00:31<00:12, 9184.79it/s] 72%|███████▏  | 286642/400000 [00:31<00:12, 9204.84it/s] 72%|███████▏  | 287565/400000 [00:31<00:12, 9172.94it/s] 72%|███████▏  | 288484/400000 [00:31<00:12, 9067.69it/s] 72%|███████▏  | 289392/400000 [00:31<00:12, 9050.31it/s] 73%|███████▎  | 290298/400000 [00:31<00:12, 8770.05it/s] 73%|███████▎  | 291178/400000 [00:31<00:12, 8727.21it/s] 73%|███████▎  | 292137/400000 [00:31<00:12, 8967.69it/s] 73%|███████▎  | 293139/400000 [00:31<00:11, 9256.91it/s] 74%|███████▎  | 294070/400000 [00:32<00:11, 9234.56it/s] 74%|███████▍  | 295021/400000 [00:32<00:11, 9314.19it/s] 74%|███████▍  | 295970/400000 [00:32<00:11, 9364.44it/s] 74%|███████▍  | 296950/400000 [00:32<00:10, 9488.44it/s] 74%|███████▍  | 297901/400000 [00:32<00:10, 9467.07it/s] 75%|███████▍  | 298858/400000 [00:32<00:10, 9495.80it/s] 75%|███████▍  | 299809/400000 [00:32<00:10, 9328.14it/s] 75%|███████▌  | 300744/400000 [00:32<00:10, 9102.02it/s] 75%|███████▌  | 301657/400000 [00:32<00:10, 9105.84it/s] 76%|███████▌  | 302570/400000 [00:32<00:10, 8867.51it/s] 76%|███████▌  | 303460/400000 [00:33<00:11, 8633.72it/s] 76%|███████▌  | 304402/400000 [00:33<00:10, 8855.35it/s] 76%|███████▋  | 305292/400000 [00:33<00:10, 8866.85it/s] 77%|███████▋  | 306250/400000 [00:33<00:10, 9067.89it/s] 77%|███████▋  | 307174/400000 [00:33<00:10, 9117.82it/s] 77%|███████▋  | 308185/400000 [00:33<00:09, 9392.83it/s] 77%|███████▋  | 309128/400000 [00:33<00:09, 9335.05it/s] 78%|███████▊  | 310130/400000 [00:33<00:09, 9529.70it/s] 78%|███████▊  | 311091/400000 [00:33<00:09, 9551.68it/s] 78%|███████▊  | 312053/400000 [00:33<00:09, 9570.73it/s] 78%|███████▊  | 313012/400000 [00:34<00:09, 9459.57it/s] 78%|███████▊  | 313960/400000 [00:34<00:09, 9424.83it/s] 79%|███████▊  | 314904/400000 [00:34<00:09, 9421.45it/s] 79%|███████▉  | 315847/400000 [00:34<00:08, 9369.22it/s] 79%|███████▉  | 316785/400000 [00:34<00:08, 9299.13it/s] 79%|███████▉  | 317741/400000 [00:34<00:08, 9374.36it/s] 80%|███████▉  | 318679/400000 [00:34<00:08, 9321.26it/s] 80%|███████▉  | 319612/400000 [00:34<00:08, 9283.71it/s] 80%|████████  | 320569/400000 [00:34<00:08, 9365.64it/s] 80%|████████  | 321529/400000 [00:34<00:08, 9432.49it/s] 81%|████████  | 322523/400000 [00:35<00:08, 9578.12it/s] 81%|████████  | 323482/400000 [00:35<00:08, 9178.96it/s] 81%|████████  | 324405/400000 [00:35<00:08, 9018.46it/s] 81%|████████▏ | 325318/400000 [00:35<00:08, 9050.31it/s] 82%|████████▏ | 326238/400000 [00:35<00:08, 9093.12it/s] 82%|████████▏ | 327221/400000 [00:35<00:07, 9301.51it/s] 82%|████████▏ | 328154/400000 [00:35<00:08, 8711.63it/s] 82%|████████▏ | 329035/400000 [00:35<00:08, 8227.60it/s] 82%|████████▏ | 329870/400000 [00:35<00:09, 7484.58it/s] 83%|████████▎ | 330640/400000 [00:36<00:10, 6758.92it/s] 83%|████████▎ | 331407/400000 [00:36<00:09, 7008.51it/s] 83%|████████▎ | 332339/400000 [00:36<00:08, 7569.94it/s] 83%|████████▎ | 333125/400000 [00:36<00:08, 7512.87it/s] 84%|████████▎ | 334024/400000 [00:36<00:08, 7901.62it/s] 84%|████████▎ | 334938/400000 [00:36<00:07, 8235.75it/s] 84%|████████▍ | 335838/400000 [00:36<00:07, 8450.68it/s] 84%|████████▍ | 336810/400000 [00:36<00:07, 8794.42it/s] 84%|████████▍ | 337845/400000 [00:36<00:06, 9206.80it/s] 85%|████████▍ | 338847/400000 [00:37<00:06, 9435.09it/s] 85%|████████▍ | 339857/400000 [00:37<00:06, 9623.70it/s] 85%|████████▌ | 340829/400000 [00:37<00:06, 9459.67it/s] 85%|████████▌ | 341802/400000 [00:37<00:06, 9537.94it/s] 86%|████████▌ | 342761/400000 [00:37<00:06, 9463.84it/s] 86%|████████▌ | 343730/400000 [00:37<00:05, 9530.36it/s] 86%|████████▌ | 344746/400000 [00:37<00:05, 9710.21it/s] 86%|████████▋ | 345720/400000 [00:37<00:05, 9440.52it/s] 87%|████████▋ | 346669/400000 [00:37<00:05, 9454.72it/s] 87%|████████▋ | 347629/400000 [00:37<00:05, 9495.64it/s] 87%|████████▋ | 348581/400000 [00:38<00:05, 9483.81it/s] 87%|████████▋ | 349531/400000 [00:38<00:05, 9409.01it/s] 88%|████████▊ | 350473/400000 [00:38<00:05, 9364.19it/s] 88%|████████▊ | 351411/400000 [00:38<00:05, 9268.96it/s] 88%|████████▊ | 352374/400000 [00:38<00:05, 9371.65it/s] 88%|████████▊ | 353350/400000 [00:38<00:04, 9482.62it/s] 89%|████████▊ | 354329/400000 [00:38<00:04, 9570.78it/s] 89%|████████▉ | 355287/400000 [00:38<00:04, 9007.92it/s] 89%|████████▉ | 356291/400000 [00:38<00:04, 9293.59it/s] 89%|████████▉ | 357335/400000 [00:38<00:04, 9608.24it/s] 90%|████████▉ | 358304/400000 [00:39<00:04, 9548.51it/s] 90%|████████▉ | 359272/400000 [00:39<00:04, 9585.16it/s] 90%|█████████ | 360235/400000 [00:39<00:04, 9559.28it/s] 90%|█████████ | 361194/400000 [00:39<00:04, 9443.42it/s] 91%|█████████ | 362141/400000 [00:39<00:04, 9398.66it/s] 91%|█████████ | 363083/400000 [00:39<00:03, 9325.24it/s] 91%|█████████ | 364053/400000 [00:39<00:03, 9433.79it/s] 91%|█████████ | 364998/400000 [00:39<00:03, 9320.08it/s] 91%|█████████▏| 365999/400000 [00:39<00:03, 9516.04it/s] 92%|█████████▏| 366998/400000 [00:40<00:03, 9651.19it/s] 92%|█████████▏| 367965/400000 [00:40<00:03, 9549.08it/s] 92%|█████████▏| 368922/400000 [00:40<00:03, 9316.80it/s] 92%|█████████▏| 369856/400000 [00:40<00:03, 9074.62it/s] 93%|█████████▎| 370767/400000 [00:40<00:03, 8965.96it/s] 93%|█████████▎| 371706/400000 [00:40<00:03, 9087.44it/s] 93%|█████████▎| 372660/400000 [00:40<00:02, 9216.54it/s] 93%|█████████▎| 373615/400000 [00:40<00:02, 9312.56it/s] 94%|█████████▎| 374559/400000 [00:40<00:02, 9346.12it/s] 94%|█████████▍| 375495/400000 [00:40<00:02, 9289.82it/s] 94%|█████████▍| 376425/400000 [00:41<00:02, 9283.58it/s] 94%|█████████▍| 377442/400000 [00:41<00:02, 9531.97it/s] 95%|█████████▍| 378398/400000 [00:41<00:02, 9509.68it/s] 95%|█████████▍| 379351/400000 [00:41<00:02, 9275.47it/s] 95%|█████████▌| 380321/400000 [00:41<00:02, 9396.86it/s] 95%|█████████▌| 381324/400000 [00:41<00:01, 9577.50it/s] 96%|█████████▌| 382284/400000 [00:41<00:01, 9522.07it/s] 96%|█████████▌| 383259/400000 [00:41<00:01, 9588.43it/s] 96%|█████████▌| 384220/400000 [00:41<00:01, 9363.68it/s] 96%|█████████▋| 385159/400000 [00:41<00:01, 9217.67it/s] 97%|█████████▋| 386100/400000 [00:42<00:01, 9273.18it/s] 97%|█████████▋| 387029/400000 [00:42<00:01, 9186.32it/s] 97%|█████████▋| 387990/400000 [00:42<00:01, 9248.58it/s] 97%|█████████▋| 388953/400000 [00:42<00:01, 9357.16it/s] 97%|█████████▋| 389890/400000 [00:42<00:01, 8709.00it/s] 98%|█████████▊| 390837/400000 [00:42<00:01, 8922.93it/s] 98%|█████████▊| 391779/400000 [00:42<00:00, 9064.33it/s] 98%|█████████▊| 392731/400000 [00:42<00:00, 9196.19it/s] 98%|█████████▊| 393656/400000 [00:42<00:00, 9130.55it/s] 99%|█████████▊| 394686/400000 [00:42<00:00, 9452.03it/s] 99%|█████████▉| 395637/400000 [00:43<00:00, 9371.89it/s] 99%|█████████▉| 396639/400000 [00:43<00:00, 9555.02it/s] 99%|█████████▉| 397599/400000 [00:43<00:00, 9437.11it/s]100%|█████████▉| 398582/400000 [00:43<00:00, 9548.65it/s]100%|█████████▉| 399540/400000 [00:43<00:00, 9482.27it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9182.75it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f23201d3b38>, <torchtext.data.dataset.TabularDataset object at 0x7f23201d3c88>, <torchtext.vocab.Vocab object at 0x7f23201d3ba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.31 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.94 file/s]2020-07-05 18:17:57.370861: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-05 18:17:57.374808: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-05 18:17:57.374973: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ae3bf67e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-05 18:17:57.374987: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:23, 119096.02it/s] 87%|████████▋ | 8658944/9912422 [00:00<00:07, 170036.61it/s]9920512it [00:00, 39827482.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 556018.81it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466305.74it/s]1654784it [00:00, 11539928.89it/s]                         
0it [00:00, ?it/s]8192it [00:00, 187851.55it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
