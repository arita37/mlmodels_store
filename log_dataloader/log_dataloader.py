
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0087e15048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0087e15048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f00f39bc1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f00f39bc1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0111d04ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0111d04ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00a0cea620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00a0cea620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00a0cea620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 55%|█████▌    | 5472256/9912422 [00:00<00:00, 54597258.35it/s]9920512it [00:01, 9567091.31it/s]                              
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 57654.85it/s]            
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 3178490.90it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 19075.69it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f009e2d1978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0087d1ebe0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f00a0cea268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f00a0cea268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f00a0cea268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:20,  1.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:20,  1.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:01<01:46,  1.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:01<01:21,  1.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<01:21,  1.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:01<01:01,  2.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<01:01,  2.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:01,  2.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:47,  3.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:47,  3.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:46,  3.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:35,  4.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:35,  4.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:35,  4.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:27,  5.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  7.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  7.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:20,  7.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:20,  7.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:02<00:16,  8.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:02<00:16,  8.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:02<00:16,  8.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:02<00:16,  8.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:02<00:12, 11.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:02<00:12, 11.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:12, 11.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:12, 11.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:02<00:10, 13.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:10, 13.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:10, 13.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:10, 13.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:09, 15.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:09, 15.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:08, 15.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:08, 15.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:07, 17.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:07, 17.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:07, 17.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:06, 18.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:06, 18.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:06, 18.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:06, 18.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 20.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:06, 20.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:06, 20.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:06, 20.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 21.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 21.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:05, 21.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:05, 21.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:03<00:05, 22.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:05, 22.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:05, 22.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:05, 22.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:05, 22.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:05, 22.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:05, 22.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:05, 22.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:03<00:05, 23.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:05, 23.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:04, 23.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:04, 23.08 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:04, 23.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:04, 23.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:04, 23.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:04, 23.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:03<00:04, 23.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:04, 23.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:04, 23.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:03<00:04, 23.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 23.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:03<00:04, 23.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:03<00:04, 23.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:03<00:04, 23.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:03<00:04, 24.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:03<00:04, 24.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:03<00:04, 24.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:03<00:04, 24.11 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:03<00:04, 24.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:03<00:04, 24.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:03<00:04, 24.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:03<00:04, 24.73 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:04<00:03, 24.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:04<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:04<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:04<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:04<00:03, 24.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:04<00:03, 24.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:04<00:03, 24.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:04<00:03, 24.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:04<00:03, 25.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:04<00:03, 25.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:04<00:03, 25.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:04<00:03, 25.10 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 25.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:04<00:03, 25.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:04<00:03, 25.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:04<00:03, 25.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 25.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:04<00:03, 25.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:04<00:03, 25.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:04<00:03, 25.67 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 25.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:04<00:03, 25.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:04<00:03, 25.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:04<00:03, 25.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 25.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:04<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:04<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:04<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:02, 26.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:04<00:02, 26.03 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:04<00:02, 26.03 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:04<00:02, 26.03 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:02, 25.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:04<00:02, 25.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:04<00:02, 25.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:05<00:02, 25.90 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:05<00:02, 26.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:05<00:02, 26.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:05<00:02, 26.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:05<00:02, 26.13 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:02, 26.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:05<00:02, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:05<00:02, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:05<00:02, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 26.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:05<00:02, 26.49 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:05<00:02, 26.49 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:05<00:02, 26.49 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 26.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:05<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:05<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:05<00:02, 26.58 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 26.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:05<00:02, 26.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:05<00:02, 26.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:05<00:02, 26.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 27.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:05<00:02, 27.02 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:05<00:02, 27.02 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:05<00:01, 27.02 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:01, 26.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:05<00:01, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:05<00:01, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:05<00:01, 26.86 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:01, 26.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:05<00:01, 26.99 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:05<00:01, 26.99 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:05<00:01, 26.99 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:05<00:01, 27.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:05<00:01, 27.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:05<00:01, 27.00 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:06<00:01, 27.00 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:01, 27.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:06<00:01, 27.62 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:06<00:01, 27.62 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:06<00:01, 27.62 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:01, 27.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:06<00:01, 27.33 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:06<00:01, 27.33 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:06<00:01, 27.33 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:01, 27.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:06<00:01, 27.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:06<00:01, 27.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:06<00:01, 27.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:06<00:01, 28.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:06<00:01, 28.12 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:06<00:01, 28.12 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:06<00:01, 28.12 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 27.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:06<00:01, 27.71 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:06<00:01, 27.71 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:06<00:01, 27.71 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:01, 28.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:01, 28.27 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:06<00:00, 28.27 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:06<00:00, 28.27 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:00, 28.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:00, 28.45 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:00, 28.45 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:06<00:00, 28.45 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:00, 27.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:00, 27.94 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:00, 27.94 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:00, 27.94 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:06<00:00, 27.94 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 28.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:06<00:00, 28.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:06<00:00, 28.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:06<00:00, 28.48 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 28.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:07<00:00, 28.35 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:07<00:00, 28.35 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:00, 28.35 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:00, 28.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:00, 28.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:00, 28.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 28.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 28.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 28.56 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00, 28.56 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:07<00:00, 28.56 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00, 28.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:07<00:00, 28.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:07<00:00, 28.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:07<00:00, 28.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 29.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:07<00:00, 29.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:07<00:00, 29.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:07<00:00, 29.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 28.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:07<00:00, 28.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.59s/ url]Dl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.99 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:07<00:00,  7.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:07<00:00, 28.99 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:07<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.79s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  7.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 28.99 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.79s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:09<00:00,  9.79s/ file]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 16.55 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.79s/ url]
0 examples [00:00, ? examples/s]2020-07-12 06:09:07.539255: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-12 06:09:07.551940: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-12 06:09:07.552162: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55774fe92240 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 06:09:07.552182: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
29 examples [00:00, 287.37 examples/s]123 examples [00:00, 362.55 examples/s]217 examples [00:00, 443.94 examples/s]312 examples [00:00, 527.65 examples/s]407 examples [00:00, 608.30 examples/s]503 examples [00:00, 681.88 examples/s]598 examples [00:00, 743.26 examples/s]685 examples [00:00, 777.09 examples/s]780 examples [00:00, 820.04 examples/s]876 examples [00:01, 856.22 examples/s]969 examples [00:01, 876.42 examples/s]1064 examples [00:01, 895.75 examples/s]1156 examples [00:01, 897.51 examples/s]1258 examples [00:01, 930.22 examples/s]1354 examples [00:01, 936.76 examples/s]1451 examples [00:01, 946.40 examples/s]1547 examples [00:01, 948.42 examples/s]1643 examples [00:01, 948.03 examples/s]1744 examples [00:01, 965.46 examples/s]1841 examples [00:02, 933.83 examples/s]1939 examples [00:02, 944.51 examples/s]2036 examples [00:02, 950.32 examples/s]2132 examples [00:02, 931.72 examples/s]2226 examples [00:02, 919.88 examples/s]2324 examples [00:02, 936.71 examples/s]2422 examples [00:02, 947.68 examples/s]2517 examples [00:02, 947.02 examples/s]2612 examples [00:02, 940.50 examples/s]2711 examples [00:02, 952.46 examples/s]2807 examples [00:03, 948.79 examples/s]2904 examples [00:03, 951.02 examples/s]3000 examples [00:03, 949.11 examples/s]3095 examples [00:03, 944.36 examples/s]3193 examples [00:03, 954.77 examples/s]3294 examples [00:03, 967.72 examples/s]3391 examples [00:03, 967.82 examples/s]3488 examples [00:03, 953.97 examples/s]3584 examples [00:03, 931.98 examples/s]3678 examples [00:03, 933.63 examples/s]3777 examples [00:04, 949.79 examples/s]3875 examples [00:04, 956.38 examples/s]3972 examples [00:04, 958.29 examples/s]4068 examples [00:04, 940.83 examples/s]4167 examples [00:04, 954.79 examples/s]4265 examples [00:04, 962.17 examples/s]4364 examples [00:04, 968.68 examples/s]4461 examples [00:04, 916.61 examples/s]4554 examples [00:04, 862.54 examples/s]4642 examples [00:05, 863.02 examples/s]4740 examples [00:05, 894.58 examples/s]4836 examples [00:05, 912.22 examples/s]4928 examples [00:05, 860.66 examples/s]5018 examples [00:05, 870.80 examples/s]5108 examples [00:05, 878.84 examples/s]5209 examples [00:05, 914.43 examples/s]5309 examples [00:05, 937.78 examples/s]5404 examples [00:05, 885.24 examples/s]5494 examples [00:05, 873.43 examples/s]5590 examples [00:06, 895.76 examples/s]5690 examples [00:06, 923.95 examples/s]5790 examples [00:06, 945.07 examples/s]5887 examples [00:06, 950.63 examples/s]5989 examples [00:06, 968.85 examples/s]6087 examples [00:06, 971.93 examples/s]6185 examples [00:06, 967.89 examples/s]6286 examples [00:06, 977.84 examples/s]6384 examples [00:06, 968.46 examples/s]6482 examples [00:06, 969.87 examples/s]6583 examples [00:07, 980.98 examples/s]6687 examples [00:07, 997.19 examples/s]6787 examples [00:07, 979.49 examples/s]6886 examples [00:07, 961.61 examples/s]6983 examples [00:07, 963.00 examples/s]7086 examples [00:07, 980.63 examples/s]7188 examples [00:07, 990.46 examples/s]7288 examples [00:07, 992.27 examples/s]7388 examples [00:07, 990.77 examples/s]7492 examples [00:07, 1003.34 examples/s]7593 examples [00:08, 975.81 examples/s] 7691 examples [00:08, 974.61 examples/s]7793 examples [00:08, 986.94 examples/s]7892 examples [00:08, 984.32 examples/s]7998 examples [00:08, 1004.68 examples/s]8106 examples [00:08, 1023.63 examples/s]8217 examples [00:08, 1047.39 examples/s]8323 examples [00:08, 1047.18 examples/s]8428 examples [00:08, 1014.24 examples/s]8530 examples [00:09, 1008.43 examples/s]8632 examples [00:09, 988.69 examples/s] 8732 examples [00:09, 979.69 examples/s]8831 examples [00:09, 980.33 examples/s]8930 examples [00:09, 967.04 examples/s]9030 examples [00:09, 974.99 examples/s]9128 examples [00:09, 927.76 examples/s]9225 examples [00:09, 938.98 examples/s]9321 examples [00:09, 943.74 examples/s]9416 examples [00:09, 941.76 examples/s]9520 examples [00:10, 969.16 examples/s]9627 examples [00:10, 995.49 examples/s]9737 examples [00:10, 1023.16 examples/s]9848 examples [00:10, 1047.27 examples/s]9954 examples [00:10, 1031.89 examples/s]10058 examples [00:10, 950.72 examples/s]10159 examples [00:10, 967.22 examples/s]10259 examples [00:10, 976.42 examples/s]10361 examples [00:10, 988.96 examples/s]10461 examples [00:11, 980.93 examples/s]10560 examples [00:11, 956.29 examples/s]10661 examples [00:11, 970.89 examples/s]10765 examples [00:11, 989.05 examples/s]10871 examples [00:11, 1007.84 examples/s]10973 examples [00:11, 998.22 examples/s] 11074 examples [00:11, 986.59 examples/s]11178 examples [00:11, 999.45 examples/s]11281 examples [00:11, 1007.21 examples/s]11382 examples [00:11, 979.21 examples/s] 11484 examples [00:12, 990.06 examples/s]11590 examples [00:12, 1008.99 examples/s]11694 examples [00:12, 1016.42 examples/s]11803 examples [00:12, 1034.71 examples/s]11907 examples [00:12, 1030.20 examples/s]12011 examples [00:12, 990.90 examples/s] 12116 examples [00:12, 1005.64 examples/s]12219 examples [00:12, 1011.10 examples/s]12321 examples [00:12, 984.37 examples/s] 12420 examples [00:12, 979.72 examples/s]12519 examples [00:13, 967.16 examples/s]12616 examples [00:13, 939.83 examples/s]12721 examples [00:13, 969.87 examples/s]12824 examples [00:13, 985.81 examples/s]12923 examples [00:13, 985.15 examples/s]13025 examples [00:13, 994.29 examples/s]13125 examples [00:13, 988.23 examples/s]13226 examples [00:13, 993.55 examples/s]13331 examples [00:13, 1008.19 examples/s]13433 examples [00:13, 1009.65 examples/s]13536 examples [00:14, 1013.11 examples/s]13638 examples [00:14, 967.43 examples/s] 13736 examples [00:14, 964.93 examples/s]13846 examples [00:14, 1001.75 examples/s]13951 examples [00:14, 1014.27 examples/s]14054 examples [00:14, 1016.50 examples/s]14162 examples [00:14, 1032.13 examples/s]14266 examples [00:14, 981.02 examples/s] 14372 examples [00:14, 1001.26 examples/s]14473 examples [00:15, 999.34 examples/s] 14583 examples [00:15, 1025.35 examples/s]14687 examples [00:15, 1021.78 examples/s]14790 examples [00:15, 1021.75 examples/s]14893 examples [00:15, 994.86 examples/s] 14993 examples [00:15, 989.67 examples/s]15104 examples [00:15, 1022.50 examples/s]15208 examples [00:15, 1026.33 examples/s]15314 examples [00:15, 1034.72 examples/s]15418 examples [00:15, 1034.84 examples/s]15522 examples [00:16, 1005.15 examples/s]15623 examples [00:16, 1002.17 examples/s]15726 examples [00:16, 1009.71 examples/s]15832 examples [00:16, 1021.96 examples/s]15935 examples [00:16, 1018.71 examples/s]16037 examples [00:16, 991.19 examples/s] 16143 examples [00:16, 1010.41 examples/s]16245 examples [00:16, 998.24 examples/s] 16350 examples [00:16, 1012.82 examples/s]16454 examples [00:16, 1018.67 examples/s]16557 examples [00:17, 999.03 examples/s] 16665 examples [00:17, 1018.97 examples/s]16768 examples [00:17, 1001.35 examples/s]16875 examples [00:17, 1020.50 examples/s]16981 examples [00:17, 1030.38 examples/s]17085 examples [00:17, 994.81 examples/s] 17185 examples [00:17, 987.67 examples/s]17287 examples [00:17, 996.01 examples/s]17390 examples [00:17, 1005.39 examples/s]17491 examples [00:18, 996.74 examples/s] 17593 examples [00:18, 1001.18 examples/s]17694 examples [00:18, 998.43 examples/s] 17795 examples [00:18, 1000.32 examples/s]17896 examples [00:18, 992.77 examples/s] 17996 examples [00:18, 991.41 examples/s]18096 examples [00:18, 979.21 examples/s]18200 examples [00:18, 993.95 examples/s]18300 examples [00:18, 957.74 examples/s]18410 examples [00:18, 994.80 examples/s]18512 examples [00:19, 1000.75 examples/s]18613 examples [00:19, 997.00 examples/s] 18720 examples [00:19, 1015.99 examples/s]18830 examples [00:19, 1038.67 examples/s]18941 examples [00:19, 1056.96 examples/s]19051 examples [00:19, 1069.35 examples/s]19159 examples [00:19, 1054.83 examples/s]19265 examples [00:19, 1033.99 examples/s]19369 examples [00:19, 1032.17 examples/s]19473 examples [00:19, 1022.10 examples/s]19576 examples [00:20, 1001.27 examples/s]19677 examples [00:20, 995.55 examples/s] 19777 examples [00:20, 965.06 examples/s]19880 examples [00:20, 982.41 examples/s]19979 examples [00:20, 961.67 examples/s]20076 examples [00:20, 895.58 examples/s]20179 examples [00:20, 930.14 examples/s]20278 examples [00:20, 944.85 examples/s]20378 examples [00:20, 958.63 examples/s]20476 examples [00:21, 963.17 examples/s]20573 examples [00:21, 915.19 examples/s]20668 examples [00:21, 924.44 examples/s]20770 examples [00:21, 950.56 examples/s]20872 examples [00:21, 969.51 examples/s]20977 examples [00:21, 991.09 examples/s]21077 examples [00:21, 964.24 examples/s]21179 examples [00:21, 979.72 examples/s]21279 examples [00:21, 984.31 examples/s]21385 examples [00:21, 1003.94 examples/s]21488 examples [00:22, 1009.64 examples/s]21590 examples [00:22, 981.49 examples/s] 21693 examples [00:22, 993.14 examples/s]21796 examples [00:22, 1001.59 examples/s]21900 examples [00:22, 1010.09 examples/s]22010 examples [00:22, 1034.46 examples/s]22114 examples [00:22, 1017.75 examples/s]22217 examples [00:22, 1007.90 examples/s]22334 examples [00:22, 1050.10 examples/s]22440 examples [00:22, 1051.23 examples/s]22546 examples [00:23, 1031.13 examples/s]22650 examples [00:23, 989.55 examples/s] 22751 examples [00:23, 991.80 examples/s]22851 examples [00:23, 981.29 examples/s]22950 examples [00:23, 961.36 examples/s]23052 examples [00:23, 977.09 examples/s]23152 examples [00:23, 981.96 examples/s]23252 examples [00:23, 984.58 examples/s]23353 examples [00:23, 989.30 examples/s]23454 examples [00:24, 994.48 examples/s]23554 examples [00:24, 985.08 examples/s]23653 examples [00:24, 977.70 examples/s]23751 examples [00:24, 974.54 examples/s]23852 examples [00:24, 981.76 examples/s]23957 examples [00:24, 1001.00 examples/s]24058 examples [00:24, 987.81 examples/s] 24157 examples [00:24, 977.00 examples/s]24255 examples [00:24, 967.31 examples/s]24352 examples [00:24, 960.79 examples/s]24449 examples [00:25, 931.86 examples/s]24543 examples [00:25, 884.70 examples/s]24633 examples [00:25, 879.42 examples/s]24724 examples [00:25, 887.98 examples/s]24814 examples [00:25, 888.02 examples/s]24904 examples [00:25, 890.30 examples/s]24995 examples [00:25, 894.09 examples/s]25085 examples [00:25, 886.26 examples/s]25174 examples [00:25, 844.12 examples/s]25259 examples [00:26, 820.66 examples/s]25346 examples [00:26, 834.05 examples/s]25430 examples [00:26, 835.76 examples/s]25520 examples [00:26, 852.18 examples/s]25610 examples [00:26, 865.74 examples/s]25703 examples [00:26, 882.27 examples/s]25799 examples [00:26, 902.21 examples/s]25890 examples [00:26, 882.72 examples/s]25982 examples [00:26, 890.57 examples/s]26072 examples [00:26, 893.21 examples/s]26164 examples [00:27, 899.67 examples/s]26255 examples [00:27, 898.06 examples/s]26345 examples [00:27, 893.44 examples/s]26439 examples [00:27, 906.88 examples/s]26533 examples [00:27, 915.49 examples/s]26625 examples [00:27, 911.27 examples/s]26719 examples [00:27, 919.27 examples/s]26811 examples [00:27, 902.24 examples/s]26902 examples [00:27, 898.00 examples/s]26992 examples [00:27, 892.88 examples/s]27084 examples [00:28, 900.44 examples/s]27179 examples [00:28, 912.12 examples/s]27271 examples [00:28, 896.15 examples/s]27362 examples [00:28, 899.96 examples/s]27453 examples [00:28, 889.20 examples/s]27543 examples [00:28, 864.35 examples/s]27633 examples [00:28, 873.00 examples/s]27721 examples [00:28, 871.79 examples/s]27809 examples [00:28, 871.97 examples/s]27900 examples [00:28, 882.01 examples/s]27994 examples [00:29, 896.62 examples/s]28088 examples [00:29, 907.07 examples/s]28179 examples [00:29, 902.06 examples/s]28272 examples [00:29, 907.62 examples/s]28365 examples [00:29, 912.19 examples/s]28457 examples [00:29, 905.62 examples/s]28556 examples [00:29, 926.63 examples/s]28649 examples [00:29, 892.93 examples/s]28739 examples [00:29, 886.79 examples/s]28830 examples [00:30, 893.06 examples/s]28920 examples [00:30, 893.93 examples/s]29014 examples [00:30, 904.96 examples/s]29105 examples [00:30, 900.91 examples/s]29197 examples [00:30, 906.49 examples/s]29288 examples [00:30, 903.62 examples/s]29385 examples [00:30, 922.32 examples/s]29478 examples [00:30, 921.57 examples/s]29571 examples [00:30, 915.38 examples/s]29671 examples [00:30, 937.82 examples/s]29765 examples [00:31, 908.57 examples/s]29857 examples [00:31, 870.01 examples/s]29950 examples [00:31, 884.79 examples/s]30039 examples [00:31, 831.39 examples/s]30139 examples [00:31, 873.91 examples/s]30235 examples [00:31, 895.87 examples/s]30328 examples [00:31, 904.24 examples/s]30422 examples [00:31, 910.62 examples/s]30515 examples [00:31, 915.85 examples/s]30610 examples [00:31, 925.22 examples/s]30706 examples [00:32, 935.12 examples/s]30800 examples [00:32, 935.47 examples/s]30897 examples [00:32, 944.29 examples/s]30992 examples [00:32, 934.16 examples/s]31086 examples [00:32, 932.16 examples/s]31180 examples [00:32, 896.75 examples/s]31271 examples [00:32, 879.90 examples/s]31363 examples [00:32, 889.71 examples/s]31455 examples [00:32, 897.44 examples/s]31549 examples [00:32, 908.95 examples/s]31645 examples [00:33, 920.68 examples/s]31738 examples [00:33, 916.44 examples/s]31831 examples [00:33, 920.29 examples/s]31924 examples [00:33, 915.25 examples/s]32016 examples [00:33, 904.43 examples/s]32107 examples [00:33, 880.15 examples/s]32203 examples [00:33, 901.20 examples/s]32294 examples [00:33, 887.10 examples/s]32389 examples [00:33, 902.53 examples/s]32486 examples [00:34, 919.63 examples/s]32582 examples [00:34, 929.41 examples/s]32676 examples [00:34, 925.08 examples/s]32772 examples [00:34, 933.00 examples/s]32866 examples [00:34, 919.88 examples/s]32965 examples [00:34, 937.68 examples/s]33062 examples [00:34, 945.30 examples/s]33158 examples [00:34, 947.55 examples/s]33253 examples [00:34, 940.52 examples/s]33349 examples [00:34, 944.28 examples/s]33444 examples [00:35, 933.07 examples/s]33544 examples [00:35, 950.23 examples/s]33642 examples [00:35, 956.52 examples/s]33738 examples [00:35, 918.98 examples/s]33837 examples [00:35, 938.65 examples/s]33932 examples [00:35, 921.86 examples/s]34027 examples [00:35, 927.81 examples/s]34121 examples [00:35, 917.64 examples/s]34213 examples [00:35, 913.23 examples/s]34309 examples [00:35, 924.84 examples/s]34402 examples [00:36, 878.15 examples/s]34493 examples [00:36, 886.67 examples/s]34585 examples [00:36, 894.65 examples/s]34675 examples [00:36, 894.59 examples/s]34765 examples [00:36, 889.88 examples/s]34862 examples [00:36, 911.48 examples/s]34958 examples [00:36, 924.19 examples/s]35055 examples [00:36, 937.35 examples/s]35149 examples [00:36, 933.64 examples/s]35243 examples [00:37, 931.13 examples/s]35337 examples [00:37, 930.86 examples/s]35432 examples [00:37, 934.44 examples/s]35527 examples [00:37, 937.13 examples/s]35621 examples [00:37, 930.93 examples/s]35715 examples [00:37, 922.36 examples/s]35812 examples [00:37, 932.62 examples/s]35908 examples [00:37, 939.36 examples/s]36007 examples [00:37, 951.86 examples/s]36103 examples [00:37, 948.53 examples/s]36205 examples [00:38, 966.36 examples/s]36302 examples [00:38, 963.82 examples/s]36399 examples [00:38, 965.41 examples/s]36496 examples [00:38, 958.58 examples/s]36592 examples [00:38, 941.06 examples/s]36687 examples [00:38, 908.78 examples/s]36779 examples [00:38, 851.84 examples/s]36867 examples [00:38, 859.76 examples/s]36962 examples [00:38, 884.65 examples/s]37061 examples [00:38, 912.78 examples/s]37163 examples [00:39, 940.60 examples/s]37262 examples [00:39, 953.90 examples/s]37361 examples [00:39, 963.28 examples/s]37462 examples [00:39, 976.52 examples/s]37561 examples [00:39, 979.42 examples/s]37670 examples [00:39, 1008.79 examples/s]37775 examples [00:39, 1019.37 examples/s]37879 examples [00:39, 1023.23 examples/s]37982 examples [00:39, 1015.84 examples/s]38084 examples [00:39, 988.61 examples/s] 38186 examples [00:40, 996.70 examples/s]38287 examples [00:40, 999.44 examples/s]38392 examples [00:40, 1013.92 examples/s]38500 examples [00:40, 1030.57 examples/s]38604 examples [00:40, 1011.38 examples/s]38706 examples [00:40, 998.54 examples/s] 38811 examples [00:40, 1013.01 examples/s]38916 examples [00:40, 1022.14 examples/s]39019 examples [00:40, 979.51 examples/s] 39118 examples [00:41, 979.24 examples/s]39217 examples [00:41, 969.31 examples/s]39317 examples [00:41, 977.68 examples/s]39417 examples [00:41, 984.19 examples/s]39520 examples [00:41, 997.06 examples/s]39620 examples [00:41, 973.39 examples/s]39718 examples [00:41, 969.99 examples/s]39816 examples [00:41, 964.88 examples/s]39913 examples [00:41, 948.50 examples/s]40008 examples [00:41, 904.90 examples/s]40105 examples [00:42, 923.24 examples/s]40203 examples [00:42, 936.82 examples/s]40301 examples [00:42, 947.67 examples/s]40404 examples [00:42, 969.94 examples/s]40504 examples [00:42, 974.90 examples/s]40603 examples [00:42, 979.22 examples/s]40706 examples [00:42, 990.66 examples/s]40806 examples [00:42, 986.60 examples/s]40906 examples [00:42, 987.98 examples/s]41008 examples [00:42, 996.12 examples/s]41108 examples [00:43, 985.08 examples/s]41212 examples [00:43, 999.83 examples/s]41313 examples [00:43, 984.20 examples/s]41420 examples [00:43, 1007.75 examples/s]41527 examples [00:43, 1022.49 examples/s]41630 examples [00:43, 1007.96 examples/s]41731 examples [00:43, 1005.93 examples/s]41834 examples [00:43, 1012.86 examples/s]41940 examples [00:43, 1024.52 examples/s]42048 examples [00:43, 1040.28 examples/s]42153 examples [00:44, 1003.84 examples/s]42254 examples [00:44, 999.02 examples/s] 42356 examples [00:44, 1002.73 examples/s]42462 examples [00:44, 1018.97 examples/s]42567 examples [00:44, 1026.90 examples/s]42670 examples [00:44, 1020.17 examples/s]42773 examples [00:44, 1014.40 examples/s]42875 examples [00:44, 986.44 examples/s] 42974 examples [00:44, 966.04 examples/s]43071 examples [00:45, 958.36 examples/s]43171 examples [00:45, 970.26 examples/s]43273 examples [00:45, 984.50 examples/s]43376 examples [00:45, 997.43 examples/s]43481 examples [00:45, 1010.87 examples/s]43583 examples [00:45, 980.79 examples/s] 43682 examples [00:45, 946.68 examples/s]43782 examples [00:45, 960.18 examples/s]43889 examples [00:45, 988.72 examples/s]43996 examples [00:45, 1011.10 examples/s]44098 examples [00:46, 1006.89 examples/s]44200 examples [00:46, 993.04 examples/s] 44306 examples [00:46, 1011.82 examples/s]44411 examples [00:46, 1016.73 examples/s]44518 examples [00:46, 1030.09 examples/s]44622 examples [00:46, 1021.05 examples/s]44725 examples [00:46, 991.04 examples/s] 44828 examples [00:46, 1001.81 examples/s]44935 examples [00:46, 1020.67 examples/s]45041 examples [00:46, 1029.55 examples/s]45145 examples [00:47, 1010.17 examples/s]45248 examples [00:47, 1015.40 examples/s]45354 examples [00:47, 1026.82 examples/s]45457 examples [00:47, 1019.20 examples/s]45562 examples [00:47, 1026.28 examples/s]45665 examples [00:47, 1015.78 examples/s]45768 examples [00:47, 1018.08 examples/s]45874 examples [00:47, 1028.42 examples/s]45977 examples [00:47, 978.74 examples/s] 46080 examples [00:48, 991.17 examples/s]46180 examples [00:48, 981.11 examples/s]46279 examples [00:48, 978.24 examples/s]46381 examples [00:48, 990.27 examples/s]46486 examples [00:48, 1005.42 examples/s]46588 examples [00:48, 1008.12 examples/s]46689 examples [00:48, 998.85 examples/s] 46792 examples [00:48, 1007.23 examples/s]46894 examples [00:48, 1008.60 examples/s]46995 examples [00:48, 1008.28 examples/s]47096 examples [00:49, 994.30 examples/s] 47196 examples [00:49, 980.29 examples/s]47295 examples [00:49, 972.48 examples/s]47393 examples [00:49, 974.18 examples/s]47494 examples [00:49, 983.94 examples/s]47595 examples [00:49, 991.28 examples/s]47695 examples [00:49, 955.18 examples/s]47791 examples [00:49, 933.10 examples/s]47890 examples [00:49, 946.97 examples/s]47988 examples [00:49, 955.07 examples/s]48085 examples [00:50, 958.76 examples/s]48182 examples [00:50, 932.30 examples/s]48280 examples [00:50, 944.36 examples/s]48384 examples [00:50, 969.49 examples/s]48490 examples [00:50, 994.16 examples/s]48590 examples [00:50, 968.92 examples/s]48690 examples [00:50, 977.78 examples/s]48790 examples [00:50, 984.10 examples/s]48890 examples [00:50, 987.93 examples/s]48989 examples [00:50, 963.26 examples/s]49093 examples [00:51, 983.71 examples/s]49192 examples [00:51, 958.21 examples/s]49289 examples [00:51, 955.41 examples/s]49396 examples [00:51, 984.83 examples/s]49500 examples [00:51, 999.25 examples/s]49608 examples [00:51, 1021.83 examples/s]49711 examples [00:51, 1013.15 examples/s]49814 examples [00:51, 1016.58 examples/s]49919 examples [00:51, 1025.25 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8847/50000 [00:00<00:00, 88467.34 examples/s] 43%|████▎     | 21686/50000 [00:00<00:00, 97568.88 examples/s] 70%|███████   | 35219/50000 [00:00<00:00, 106481.54 examples/s] 98%|█████████▊| 48788/50000 [00:00<00:00, 113832.40 examples/s]                                                                0 examples [00:00, ? examples/s]84 examples [00:00, 837.86 examples/s]194 examples [00:00, 900.62 examples/s]292 examples [00:00, 922.93 examples/s]398 examples [00:00, 958.63 examples/s]501 examples [00:00, 977.14 examples/s]612 examples [00:00, 1013.38 examples/s]722 examples [00:00, 1037.63 examples/s]824 examples [00:00, 1029.97 examples/s]930 examples [00:00, 1037.40 examples/s]1035 examples [00:01, 1040.19 examples/s]1138 examples [00:01, 1036.27 examples/s]1246 examples [00:01, 1047.29 examples/s]1350 examples [00:01, 1019.80 examples/s]1452 examples [00:01, 1009.34 examples/s]1561 examples [00:01, 1030.60 examples/s]1664 examples [00:01, 998.30 examples/s] 1764 examples [00:01, 972.33 examples/s]1862 examples [00:01, 960.30 examples/s]1961 examples [00:01, 967.83 examples/s]2066 examples [00:02, 989.08 examples/s]2170 examples [00:02, 1001.40 examples/s]2275 examples [00:02, 1014.46 examples/s]2377 examples [00:02, 960.04 examples/s] 2474 examples [00:02, 962.95 examples/s]2579 examples [00:02, 984.42 examples/s]2678 examples [00:02, 976.00 examples/s]2777 examples [00:02, 974.72 examples/s]2875 examples [00:02, 972.87 examples/s]2974 examples [00:02, 976.35 examples/s]3075 examples [00:03, 983.79 examples/s]3174 examples [00:03, 982.94 examples/s]3277 examples [00:03, 993.33 examples/s]3377 examples [00:03, 947.96 examples/s]3482 examples [00:03, 976.24 examples/s]3581 examples [00:03, 956.57 examples/s]3678 examples [00:03, 921.42 examples/s]3779 examples [00:03, 944.27 examples/s]3886 examples [00:03, 978.00 examples/s]3994 examples [00:04, 1005.38 examples/s]4099 examples [00:04, 1017.23 examples/s]4203 examples [00:04, 1021.85 examples/s]4306 examples [00:04, 1019.20 examples/s]4410 examples [00:04, 1024.44 examples/s]4517 examples [00:04, 1036.59 examples/s]4627 examples [00:04, 1053.42 examples/s]4733 examples [00:04, 1029.46 examples/s]4837 examples [00:04, 1029.42 examples/s]4941 examples [00:04, 1027.73 examples/s]5044 examples [00:05, 999.97 examples/s] 5145 examples [00:05, 985.22 examples/s]5244 examples [00:05, 984.84 examples/s]5343 examples [00:05, 969.21 examples/s]5441 examples [00:05, 965.88 examples/s]5539 examples [00:05, 968.78 examples/s]5636 examples [00:05, 959.14 examples/s]5732 examples [00:05, 943.31 examples/s]5827 examples [00:05, 941.93 examples/s]5922 examples [00:05, 933.87 examples/s]6016 examples [00:06, 920.32 examples/s]6109 examples [00:06, 917.26 examples/s]6204 examples [00:06, 924.89 examples/s]6297 examples [00:06, 908.34 examples/s]6388 examples [00:06, 901.47 examples/s]6479 examples [00:06, 896.22 examples/s]6574 examples [00:06, 910.91 examples/s]6670 examples [00:06, 923.50 examples/s]6763 examples [00:06, 916.14 examples/s]6860 examples [00:06, 930.06 examples/s]6954 examples [00:07, 930.17 examples/s]7048 examples [00:07, 925.91 examples/s]7143 examples [00:07, 930.20 examples/s]7237 examples [00:07, 926.88 examples/s]7338 examples [00:07, 950.20 examples/s]7437 examples [00:07, 960.27 examples/s]7534 examples [00:07, 931.45 examples/s]7630 examples [00:07, 938.01 examples/s]7725 examples [00:07, 927.53 examples/s]7822 examples [00:08, 937.00 examples/s]7916 examples [00:08, 921.45 examples/s]8009 examples [00:08, 920.77 examples/s]8102 examples [00:08, 911.01 examples/s]8194 examples [00:08, 899.70 examples/s]8287 examples [00:08, 908.37 examples/s]8383 examples [00:08, 922.96 examples/s]8478 examples [00:08, 930.73 examples/s]8573 examples [00:08, 935.35 examples/s]8667 examples [00:08, 930.57 examples/s]8762 examples [00:09, 935.14 examples/s]8856 examples [00:09, 903.69 examples/s]8947 examples [00:09, 865.78 examples/s]9035 examples [00:09, 863.30 examples/s]9125 examples [00:09, 873.18 examples/s]9223 examples [00:09, 902.22 examples/s]9320 examples [00:09, 920.79 examples/s]9415 examples [00:09, 927.14 examples/s]9511 examples [00:09, 935.20 examples/s]9605 examples [00:09, 923.25 examples/s]9698 examples [00:10, 913.23 examples/s]9791 examples [00:10, 917.39 examples/s]9886 examples [00:10, 926.03 examples/s]9979 examples [00:10, 916.02 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1YDET6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1YDET6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00a0cea620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00a0cea620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00a0cea620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f009e2d1940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f002a1a34e0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00a0cea620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00a0cea620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00a0cea620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0087d1e5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0087d1e2b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f001876b400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f001876b400> 

  function with postional parmater data_info <function split_train_valid at 0x7f001876b400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f001876b510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f001876b510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f001876b510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=74295c37dd5cbeb4ba0571d005bba2d2435a0484e2ed25c442d9c12330c053af
  Stored in directory: /tmp/pip-ephem-wheel-cache-nzqqridw/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:33:33, 12.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:55:22, 17.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:47:57, 24.4kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:52:04, 34.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:47:44, 49.7kB/s].vector_cache/glove.6B.zip:   1%|          | 9.51M/862M [00:01<3:20:07, 71.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:19:42, 101kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:01<1:37:23, 145kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:07:56, 206kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<47:22, 294kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:01<33:04, 419kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:01<23:08, 596kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<16:12, 846kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:02<11:23, 1.20MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<08:01, 1.69MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<06:01, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<06:07, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<08:07, 1.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:36, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<04:48, 2.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<09:07, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<08:03, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<06:02, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:54, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:45, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<06:09, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:09<04:29, 2.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<19:11, 689kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<14:45, 895kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<10:39, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<10:32, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<08:41, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<06:24, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<07:32, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<07:55, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<06:10, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<06:25, 2.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<05:50, 2.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<04:21, 2.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:06, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:34, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:18<04:13, 3.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<06:42, 1.92MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:20, 2.41MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:48, 2.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<05:23, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<04:02, 3.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<06:39, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<05:18, 2.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<03:49, 3.32MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:35:17, 133kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:07:57, 186kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<47:47, 265kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<36:19, 347kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<26:41, 472kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<18:57, 663kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<16:12, 774kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:53, 903kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<10:16, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<07:17, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<34:32, 361kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<25:27, 490kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<18:05, 687kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<15:32, 798kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<12:07, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<08:47, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:04, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:34, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:33, 2.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:48, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:00, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:30, 2.72MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:02, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:41, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:17, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<03:49, 3.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<1:43:29, 117kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:13:37, 165kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<51:44, 234kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<38:56, 310kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<29:40, 406kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<21:21, 564kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<15:01, 798kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<11:43:42, 17.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:13:34, 24.3kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<5:45:03, 34.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<4:03:36, 48.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:52:53, 68.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<2:01:29, 98.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<1:26:36, 137kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<1:01:47, 192kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<43:27, 272kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<33:06, 356kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<24:20, 484kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<17:18, 679kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:49, 790kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<12:44, 919kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:30, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:30, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:24, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:36, 2.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:49, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<08:03, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<05:55, 1.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:49, 1.69MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:55, 1.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<04:26, 2.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:48, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<05:02, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:54, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:43, 3.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:56, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:38, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:22, 3.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<10:06, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:13, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<06:01, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:50, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:03, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<03:54, 2.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<16:25, 676kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<12:37, 879kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<09:07, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:56, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<10:05, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:52, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<05:43, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:00, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:49, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:30, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:56, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:31, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:48, 2.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:11, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:12, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:49, 2.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:11, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:21, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:06, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:30, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:56, 1.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:37, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:20, 3.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<28:30, 373kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<23:37, 449kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<17:19, 612kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<12:20, 858kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<11:20, 930kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:45, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:12, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<05:08, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<17:08, 612kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<13:48, 759kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:03, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<07:06, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<5:13:20, 33.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<3:41:03, 47.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<2:34:58, 67.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:49:45, 94.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:20:23, 129kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<57:10, 181kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<40:07, 257kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<30:55, 332kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<22:11, 462kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<15:40, 652kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<14:31, 703kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<13:41, 745kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:23, 981kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<07:28, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:49, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:13, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:53, 2.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<32:34, 309kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<24:32, 410kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<17:31, 573kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<12:20, 811kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<20:44, 482kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<18:15, 548kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<13:39, 732kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<09:43, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<09:23, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:17, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:13, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:53, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:23, 1.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:40, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:49, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:46, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:27, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:46, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:57, 1.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:35, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:04, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:23, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:26, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:11, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:00, 3.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<45:44, 209kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<35:36, 269kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<25:36, 374kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<18:05, 528kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<15:05, 631kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<12:11, 780kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<08:56, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:50, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<09:01, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:03, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:08, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<06:01, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:50, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:27, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:41, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:57, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<02:45, 3.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<11:23, 811kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<11:11, 826kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<08:33, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:10, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:46, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:32, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:04, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:20, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:30, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:09, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:46, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:31, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:25, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:11, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:26, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:26, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:17, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:54, 2.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:02, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:04, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:52, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<02:48, 3.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<10:50, 814kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:10, 962kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<06:47, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:13, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:22, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:49, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<04:13, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:11, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:58, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:14, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:27, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:26, 2.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<02:29, 3.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<22:51, 374kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<17:31, 488kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<12:37, 676kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<10:13, 829kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:37, 983kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:20, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<04:33, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:43, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:09, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:40, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:40, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:42, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:39, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:57, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:12, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:14, 2.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:21, 3.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:59, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:39, 947kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:47, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:53, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:31, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:16, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<03:59, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<02:51, 2.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<31:32, 256kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<23:11, 348kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<16:34, 486kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<12:50, 623kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<10:22, 771kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:35, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:38, 1.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<06:04, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:32, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:14, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<11:47, 667kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<09:36, 817kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:00, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<04:59, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<08:48, 884kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<07:33, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:37, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<05:13, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:59, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:49, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:58, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:06, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:09, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:30, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:16, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:21, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:13, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:13, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:17, 2.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:21, 3.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:29:19, 83.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<1:03:47, 117kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<44:53, 166kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<32:28, 227kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<25:16, 292kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<18:20, 402kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<12:56, 567kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<10:56, 668kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<08:55, 819kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:30, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<04:38, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:57, 1.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:08, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:33, 1.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<03:16, 2.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<06:09, 1.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:32, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:11, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:07, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:07, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:08, 2.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:16, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<08:14, 855kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<07:01, 1.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:13, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<04:48, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<05:46, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:33, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:18, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:03, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:04, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:06, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:13, 3.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<2:16:29, 50.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<1:37:52, 69.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<1:09:04, 98.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<48:16, 141kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<35:23, 191kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<25:58, 260kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<18:26, 366kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<13:57, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<10:56, 611kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<07:54, 844kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:37, 999kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<05:49, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<04:21, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:08, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:02, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<03:03, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:12, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:15, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:29, 999kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:05, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:41, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:20, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:10, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:17, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:27, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:39, 2.38MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:55, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:14, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:32, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:36, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:23, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:29, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:40, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<01:56, 3.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:07, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:42, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:33, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<03:17, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:52, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:49, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:06, 2.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:37, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<05:19, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:16, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:05, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:40, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:38, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:48, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:05, 1.90MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:22, 2.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<01:43, 3.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:49, 741kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<07:29, 774kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:45, 1.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<04:08, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:22, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:03, 1.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:03, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:45<02:12, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:11, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:53, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:50, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:50, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:04, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:07, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:25, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:37, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:48, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:12, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:27, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:45, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:02, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:14, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:57, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:59, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:39, 3.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<06:25, 827kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<05:27, 973kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:00, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:50, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<08:05, 649kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<07:28, 702kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:41, 919kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<04:04, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<04:11, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:51, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:55, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:53, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:54, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:13, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:37, 3.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:12, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:09, 1.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:25, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:31, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:37, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:03, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:16, 1.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:39, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:58, 2.46MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:20, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:28, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<01:54, 2.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:23, 3.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:56, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:24, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:30, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:32, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:00, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:47, 1.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<02:09, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:34, 2.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:14, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:04, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:18, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:39, 2.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:12, 876kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:20, 854kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:08, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:57, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:13, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:02, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:17, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:38, 2.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:44, 769kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:47, 923kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:32, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:11, 1.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:44, 1.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:00, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:11, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:38, 1.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:35, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:57, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:25, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:39, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:06, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:16, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:24, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:24, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:51, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:58, 2.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:05, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:37, 2.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:49, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:57, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:32, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:44, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:54, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:30, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<01:04, 3.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<46:05, 83.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<32:52, 118kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<23:05, 167kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<16:36, 229kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<12:16, 309kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<08:41, 435kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<06:04, 616kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<06:54, 540kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<06:07, 609kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<04:33, 815kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:14, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<03:09, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:49, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:06, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:30, 2.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<03:16, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:55, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:11, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:06, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:05, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:35, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:09, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:14, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:08, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:38, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:43, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:23, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:55, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:24, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:52, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:53, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:33, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:13, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:47, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:17, 2.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:42, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:44, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:21, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:30, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:08, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:43, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:15, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:33, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:36, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:14, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:22, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:28, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:09, 2.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:17, 2.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:24, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:04, 2.67MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<00:47, 3.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:32, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:15, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:41, 1.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:38, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:37, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:15, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:19, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:51, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:06, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:29, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:30, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:08, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:48, 3.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<05:28, 468kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<04:43, 542kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:30, 725kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:29, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:24, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<02:05, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:33, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:29, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:27, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:07, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:10, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:38, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:20, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:58, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:18, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:17, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:59, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:41, 3.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<07:24, 299kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<05:57, 372kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<04:20, 508kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:02, 714kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<02:41, 797kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:15, 951kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:09, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:24, 862kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:24, 862kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:51, 1.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:19, 1.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:28, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:22, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:01, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:43, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<04:13, 459kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<03:17, 589kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<02:21, 812kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:56, 967kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:59, 935kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:33, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:06, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:15, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:11, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:54, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:14, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:43, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:57, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:57, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:43, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:46, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:05, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:53, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:39, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:51, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:52, 1.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:39, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:42, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:34, 2.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:37, 2.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:55, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:44, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:32, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:43, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:44, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:33, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:23, 3.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:56, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:05, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:36, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:45, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:56, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:45, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:40, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:39, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:30, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:43, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:35, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:25, 2.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:24, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:17, 3.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:40, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:47, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:36, 1.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:26, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:31, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:15, 2.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:46, 998kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:48, 955kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:36, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:26, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:17, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:11, 3.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<02:38, 238kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<02:02, 306kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:28, 421kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:59, 594kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:48, 687kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:39, 839kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:28, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:23, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:21, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:09, 2.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<02:11, 192kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<01:35, 262kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:05, 369kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:43, 483kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:33, 616kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:23, 848kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:16, 1.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:17, 958kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:13, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:08, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:08, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.51MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<50:23:28,  2.20it/s]  0%|          | 889/400000 [00:00<35:11:57,  3.15it/s]  0%|          | 1800/400000 [00:00<24:35:13,  4.50it/s]  1%|          | 2728/400000 [00:00<17:10:27,  6.43it/s]  1%|          | 3701/400000 [00:00<11:59:45,  9.18it/s]  1%|          | 4675/400000 [00:00<8:22:47, 13.10it/s]   1%|▏         | 5656/400000 [00:01<5:51:17, 18.71it/s]  2%|▏         | 6572/400000 [00:01<4:05:32, 26.70it/s]  2%|▏         | 7534/400000 [00:01<2:51:39, 38.10it/s]  2%|▏         | 8447/400000 [00:01<2:00:06, 54.34it/s]  2%|▏         | 9431/400000 [00:01<1:24:03, 77.44it/s]  3%|▎         | 10393/400000 [00:01<58:53, 110.25it/s]  3%|▎         | 11400/400000 [00:01<41:18, 156.76it/s]  3%|▎         | 12358/400000 [00:01<29:03, 222.33it/s]  3%|▎         | 13340/400000 [00:01<20:29, 314.56it/s]  4%|▎         | 14313/400000 [00:01<14:30, 443.23it/s]  4%|▍         | 15275/400000 [00:02<10:19, 620.78it/s]  4%|▍         | 16234/400000 [00:02<07:24, 862.69it/s]  4%|▍         | 17191/400000 [00:02<05:23, 1182.54it/s]  5%|▍         | 18156/400000 [00:02<03:57, 1604.99it/s]  5%|▍         | 19098/400000 [00:02<02:58, 2135.66it/s]  5%|▌         | 20065/400000 [00:02<02:16, 2787.03it/s]  5%|▌         | 21013/400000 [00:02<01:47, 3516.16it/s]  5%|▌         | 21948/400000 [00:02<01:27, 4298.26it/s]  6%|▌         | 22911/400000 [00:02<01:13, 5154.16it/s]  6%|▌         | 23855/400000 [00:02<01:03, 5966.73it/s]  6%|▌         | 24813/400000 [00:03<00:55, 6726.61it/s]  6%|▋         | 25757/400000 [00:03<00:51, 7285.42it/s]  7%|▋         | 26689/400000 [00:03<00:49, 7508.09it/s]  7%|▋         | 27605/400000 [00:03<00:46, 7937.05it/s]  7%|▋         | 28573/400000 [00:03<00:44, 8389.38it/s]  7%|▋         | 29544/400000 [00:03<00:42, 8741.96it/s]  8%|▊         | 30480/400000 [00:03<00:42, 8718.86it/s]  8%|▊         | 31479/400000 [00:03<00:40, 9063.79it/s]  8%|▊         | 32451/400000 [00:03<00:39, 9249.20it/s]  8%|▊         | 33430/400000 [00:04<00:38, 9403.56it/s]  9%|▊         | 34409/400000 [00:04<00:38, 9513.72it/s]  9%|▉         | 35374/400000 [00:04<00:38, 9417.20it/s]  9%|▉         | 36325/400000 [00:04<00:39, 9167.21it/s]  9%|▉         | 37318/400000 [00:04<00:38, 9383.14it/s] 10%|▉         | 38313/400000 [00:04<00:37, 9543.81it/s] 10%|▉         | 39273/400000 [00:04<00:38, 9475.45it/s] 10%|█         | 40225/400000 [00:04<00:38, 9446.10it/s] 10%|█         | 41207/400000 [00:04<00:37, 9553.65it/s] 11%|█         | 42208/400000 [00:04<00:36, 9683.57it/s] 11%|█         | 43179/400000 [00:05<00:39, 9097.60it/s] 11%|█         | 44136/400000 [00:05<00:38, 9232.22it/s] 11%|█▏        | 45066/400000 [00:05<00:38, 9166.40it/s] 12%|█▏        | 46045/400000 [00:05<00:37, 9342.33it/s] 12%|█▏        | 46984/400000 [00:05<00:38, 9271.86it/s] 12%|█▏        | 47962/400000 [00:05<00:37, 9417.08it/s] 12%|█▏        | 48940/400000 [00:05<00:36, 9521.53it/s] 12%|█▏        | 49895/400000 [00:05<00:37, 9385.85it/s] 13%|█▎        | 50859/400000 [00:05<00:36, 9460.24it/s] 13%|█▎        | 51825/400000 [00:05<00:36, 9518.10it/s] 13%|█▎        | 52784/400000 [00:06<00:36, 9539.31it/s] 13%|█▎        | 53739/400000 [00:06<00:36, 9434.74it/s] 14%|█▎        | 54684/400000 [00:06<00:37, 9250.39it/s] 14%|█▍        | 55675/400000 [00:06<00:36, 9436.57it/s] 14%|█▍        | 56627/400000 [00:06<00:36, 9460.99it/s] 14%|█▍        | 57625/400000 [00:06<00:35, 9609.06it/s] 15%|█▍        | 58595/400000 [00:06<00:35, 9635.50it/s] 15%|█▍        | 59560/400000 [00:06<00:35, 9526.46it/s] 15%|█▌        | 60525/400000 [00:06<00:35, 9560.52it/s] 15%|█▌        | 61510/400000 [00:06<00:35, 9645.35it/s] 16%|█▌        | 62480/400000 [00:07<00:34, 9654.01it/s] 16%|█▌        | 63446/400000 [00:07<00:34, 9634.04it/s] 16%|█▌        | 64410/400000 [00:07<00:35, 9546.20it/s] 16%|█▋        | 65366/400000 [00:07<00:35, 9448.66it/s] 17%|█▋        | 66326/400000 [00:07<00:35, 9491.07it/s] 17%|█▋        | 67276/400000 [00:07<00:35, 9347.36it/s] 17%|█▋        | 68212/400000 [00:07<00:35, 9232.24it/s] 17%|█▋        | 69137/400000 [00:07<00:36, 9072.01it/s] 18%|█▊        | 70046/400000 [00:07<00:36, 9023.38it/s] 18%|█▊        | 70979/400000 [00:08<00:36, 9113.06it/s] 18%|█▊        | 71892/400000 [00:08<00:36, 9094.00it/s] 18%|█▊        | 72803/400000 [00:08<00:37, 8679.04it/s] 18%|█▊        | 73676/400000 [00:08<00:39, 8268.27it/s] 19%|█▊        | 74523/400000 [00:08<00:39, 8327.73it/s] 19%|█▉        | 75370/400000 [00:08<00:38, 8367.95it/s] 19%|█▉        | 76235/400000 [00:08<00:38, 8449.02it/s] 19%|█▉        | 77144/400000 [00:08<00:37, 8629.40it/s] 20%|█▉        | 78010/400000 [00:08<00:38, 8446.31it/s] 20%|█▉        | 78898/400000 [00:08<00:37, 8569.65it/s] 20%|█▉        | 79768/400000 [00:09<00:37, 8605.75it/s] 20%|██        | 80631/400000 [00:09<00:37, 8549.16it/s] 20%|██        | 81488/400000 [00:09<00:37, 8421.98it/s] 21%|██        | 82332/400000 [00:09<00:38, 8310.10it/s] 21%|██        | 83240/400000 [00:09<00:37, 8526.86it/s] 21%|██        | 84127/400000 [00:09<00:36, 8626.03it/s] 21%|██        | 84992/400000 [00:09<00:37, 8507.13it/s] 21%|██▏       | 85849/400000 [00:09<00:36, 8524.86it/s] 22%|██▏       | 86703/400000 [00:09<00:37, 8262.55it/s] 22%|██▏       | 87540/400000 [00:09<00:37, 8294.16it/s] 22%|██▏       | 88379/400000 [00:10<00:37, 8320.89it/s] 22%|██▏       | 89286/400000 [00:10<00:36, 8530.97it/s] 23%|██▎       | 90153/400000 [00:10<00:36, 8571.31it/s] 23%|██▎       | 91012/400000 [00:10<00:36, 8457.65it/s] 23%|██▎       | 91860/400000 [00:10<00:36, 8462.48it/s] 23%|██▎       | 92737/400000 [00:10<00:35, 8549.79it/s] 23%|██▎       | 93593/400000 [00:10<00:35, 8537.95it/s] 24%|██▎       | 94448/400000 [00:10<00:37, 8243.81it/s] 24%|██▍       | 95276/400000 [00:10<00:38, 7956.90it/s] 24%|██▍       | 96083/400000 [00:11<00:38, 7986.16it/s] 24%|██▍       | 96923/400000 [00:11<00:37, 8105.88it/s] 24%|██▍       | 97842/400000 [00:11<00:35, 8401.34it/s] 25%|██▍       | 98687/400000 [00:11<00:35, 8371.08it/s] 25%|██▍       | 99528/400000 [00:11<00:36, 8246.61it/s] 25%|██▌       | 100375/400000 [00:11<00:36, 8310.66it/s] 25%|██▌       | 101209/400000 [00:11<00:36, 8236.64it/s] 26%|██▌       | 102035/400000 [00:11<00:36, 8194.65it/s] 26%|██▌       | 102856/400000 [00:11<00:36, 8085.54it/s] 26%|██▌       | 103666/400000 [00:11<00:37, 7887.22it/s] 26%|██▌       | 104457/400000 [00:12<00:37, 7887.93it/s] 26%|██▋       | 105450/400000 [00:12<00:35, 8404.96it/s] 27%|██▋       | 106397/400000 [00:12<00:33, 8698.25it/s] 27%|██▋       | 107311/400000 [00:12<00:33, 8825.02it/s] 27%|██▋       | 108201/400000 [00:12<00:33, 8611.80it/s] 27%|██▋       | 109096/400000 [00:12<00:33, 8706.65it/s] 27%|██▋       | 109972/400000 [00:12<00:33, 8531.89it/s] 28%|██▊       | 110830/400000 [00:12<00:34, 8355.84it/s] 28%|██▊       | 111670/400000 [00:12<00:34, 8322.31it/s] 28%|██▊       | 112505/400000 [00:12<00:35, 8117.14it/s] 28%|██▊       | 113320/400000 [00:13<00:35, 8083.17it/s] 29%|██▊       | 114131/400000 [00:13<00:35, 8029.15it/s] 29%|██▊       | 114936/400000 [00:13<00:35, 7948.75it/s] 29%|██▉       | 115733/400000 [00:13<00:35, 7931.30it/s] 29%|██▉       | 116530/400000 [00:13<00:35, 7942.25it/s] 29%|██▉       | 117475/400000 [00:13<00:33, 8341.18it/s] 30%|██▉       | 118409/400000 [00:13<00:32, 8616.07it/s] 30%|██▉       | 119277/400000 [00:13<00:32, 8513.67it/s] 30%|███       | 120148/400000 [00:13<00:32, 8569.55it/s] 30%|███       | 121009/400000 [00:13<00:32, 8540.76it/s] 30%|███       | 121866/400000 [00:14<00:32, 8538.11it/s] 31%|███       | 122722/400000 [00:14<00:32, 8453.86it/s] 31%|███       | 123569/400000 [00:14<00:32, 8455.20it/s] 31%|███       | 124416/400000 [00:14<00:33, 8283.08it/s] 31%|███▏      | 125246/400000 [00:14<00:34, 8062.93it/s] 32%|███▏      | 126111/400000 [00:14<00:33, 8229.74it/s] 32%|███▏      | 126993/400000 [00:14<00:32, 8397.36it/s] 32%|███▏      | 127836/400000 [00:14<00:32, 8401.77it/s] 32%|███▏      | 128709/400000 [00:14<00:31, 8495.20it/s] 32%|███▏      | 129561/400000 [00:15<00:31, 8455.70it/s] 33%|███▎      | 130488/400000 [00:15<00:31, 8684.17it/s] 33%|███▎      | 131444/400000 [00:15<00:30, 8928.93it/s] 33%|███▎      | 132341/400000 [00:15<00:30, 8868.95it/s] 33%|███▎      | 133284/400000 [00:15<00:29, 9026.78it/s] 34%|███▎      | 134190/400000 [00:15<00:30, 8761.84it/s] 34%|███▍      | 135110/400000 [00:15<00:29, 8886.63it/s] 34%|███▍      | 136025/400000 [00:15<00:29, 8962.52it/s] 34%|███▍      | 136941/400000 [00:15<00:29, 9020.64it/s] 34%|███▍      | 137845/400000 [00:15<00:29, 8990.51it/s] 35%|███▍      | 138746/400000 [00:16<00:29, 8987.32it/s] 35%|███▍      | 139668/400000 [00:16<00:28, 9055.80it/s] 35%|███▌      | 140583/400000 [00:16<00:28, 9081.90it/s] 35%|███▌      | 141499/400000 [00:16<00:28, 9102.60it/s] 36%|███▌      | 142410/400000 [00:16<00:28, 9016.73it/s] 36%|███▌      | 143313/400000 [00:16<00:28, 8872.73it/s] 36%|███▌      | 144202/400000 [00:16<00:29, 8763.83it/s] 36%|███▋      | 145134/400000 [00:16<00:28, 8921.14it/s] 37%|███▋      | 146028/400000 [00:16<00:28, 8861.44it/s] 37%|███▋      | 146916/400000 [00:16<00:29, 8708.10it/s] 37%|███▋      | 147789/400000 [00:17<00:29, 8682.91it/s] 37%|███▋      | 148674/400000 [00:17<00:28, 8730.07it/s] 37%|███▋      | 149567/400000 [00:17<00:28, 8788.87it/s] 38%|███▊      | 150489/400000 [00:17<00:27, 8911.62it/s] 38%|███▊      | 151381/400000 [00:17<00:28, 8733.85it/s] 38%|███▊      | 152256/400000 [00:17<00:28, 8618.16it/s] 38%|███▊      | 153174/400000 [00:17<00:28, 8777.94it/s] 39%|███▊      | 154054/400000 [00:17<00:28, 8712.22it/s] 39%|███▊      | 154944/400000 [00:17<00:27, 8764.71it/s] 39%|███▉      | 155832/400000 [00:17<00:27, 8798.70it/s] 39%|███▉      | 156713/400000 [00:18<00:27, 8745.84it/s] 39%|███▉      | 157623/400000 [00:18<00:27, 8847.78it/s] 40%|███▉      | 158509/400000 [00:18<00:27, 8729.52it/s] 40%|███▉      | 159400/400000 [00:18<00:27, 8781.24it/s] 40%|████      | 160279/400000 [00:18<00:27, 8771.89it/s] 40%|████      | 161172/400000 [00:18<00:27, 8817.54it/s] 41%|████      | 162088/400000 [00:18<00:26, 8915.93it/s] 41%|████      | 163008/400000 [00:18<00:26, 8998.30it/s] 41%|████      | 163909/400000 [00:18<00:26, 8968.43it/s] 41%|████      | 164807/400000 [00:18<00:27, 8636.12it/s] 41%|████▏     | 165742/400000 [00:19<00:26, 8836.97it/s] 42%|████▏     | 166689/400000 [00:19<00:25, 9017.58it/s] 42%|████▏     | 167594/400000 [00:19<00:26, 8912.34it/s] 42%|████▏     | 168488/400000 [00:19<00:26, 8843.96it/s] 42%|████▏     | 169375/400000 [00:19<00:26, 8847.93it/s] 43%|████▎     | 170262/400000 [00:19<00:26, 8772.58it/s] 43%|████▎     | 171193/400000 [00:19<00:25, 8925.02it/s] 43%|████▎     | 172087/400000 [00:19<00:25, 8879.03it/s] 43%|████▎     | 172976/400000 [00:19<00:26, 8647.83it/s] 43%|████▎     | 173843/400000 [00:20<00:26, 8440.67it/s] 44%|████▎     | 174703/400000 [00:20<00:26, 8487.84it/s] 44%|████▍     | 175607/400000 [00:20<00:25, 8645.69it/s] 44%|████▍     | 176529/400000 [00:20<00:25, 8808.05it/s] 44%|████▍     | 177427/400000 [00:20<00:25, 8857.12it/s] 45%|████▍     | 178315/400000 [00:20<00:25, 8797.21it/s] 45%|████▍     | 179225/400000 [00:20<00:24, 8885.37it/s] 45%|████▌     | 180115/400000 [00:20<00:24, 8874.99it/s] 45%|████▌     | 181004/400000 [00:20<00:24, 8760.63it/s] 45%|████▌     | 181881/400000 [00:20<00:25, 8622.52it/s] 46%|████▌     | 182745/400000 [00:21<00:25, 8538.82it/s] 46%|████▌     | 183644/400000 [00:21<00:24, 8668.23it/s] 46%|████▌     | 184536/400000 [00:21<00:24, 8740.13it/s] 46%|████▋     | 185411/400000 [00:21<00:24, 8672.41it/s] 47%|████▋     | 186301/400000 [00:21<00:24, 8738.83it/s] 47%|████▋     | 187192/400000 [00:21<00:24, 8788.68it/s] 47%|████▋     | 188134/400000 [00:21<00:23, 8967.04it/s] 47%|████▋     | 189069/400000 [00:21<00:23, 9078.06it/s] 47%|████▋     | 189991/400000 [00:21<00:23, 9118.23it/s] 48%|████▊     | 190937/400000 [00:21<00:22, 9216.33it/s] 48%|████▊     | 191860/400000 [00:22<00:23, 8891.30it/s] 48%|████▊     | 192806/400000 [00:22<00:22, 9051.47it/s] 48%|████▊     | 193759/400000 [00:22<00:22, 9187.68it/s] 49%|████▊     | 194751/400000 [00:22<00:21, 9394.35it/s] 49%|████▉     | 195741/400000 [00:22<00:21, 9538.67it/s] 49%|████▉     | 196698/400000 [00:22<00:21, 9397.45it/s] 49%|████▉     | 197641/400000 [00:22<00:21, 9368.85it/s] 50%|████▉     | 198586/400000 [00:22<00:21, 9392.12it/s] 50%|████▉     | 199570/400000 [00:22<00:21, 9520.03it/s] 50%|█████     | 200541/400000 [00:22<00:20, 9576.00it/s] 50%|█████     | 201500/400000 [00:23<00:21, 9349.48it/s] 51%|█████     | 202437/400000 [00:23<00:21, 9335.70it/s] 51%|█████     | 203372/400000 [00:23<00:21, 9274.89it/s] 51%|█████     | 204360/400000 [00:23<00:20, 9447.54it/s] 51%|█████▏    | 205337/400000 [00:23<00:20, 9541.40it/s] 52%|█████▏    | 206293/400000 [00:23<00:20, 9428.22it/s] 52%|█████▏    | 207279/400000 [00:23<00:20, 9553.65it/s] 52%|█████▏    | 208236/400000 [00:23<00:20, 9308.87it/s] 52%|█████▏    | 209209/400000 [00:23<00:20, 9429.27it/s] 53%|█████▎    | 210169/400000 [00:23<00:20, 9479.27it/s] 53%|█████▎    | 211119/400000 [00:24<00:20, 9285.58it/s] 53%|█████▎    | 212082/400000 [00:24<00:20, 9385.90it/s] 53%|█████▎    | 213023/400000 [00:24<00:20, 9232.66it/s] 53%|█████▎    | 213948/400000 [00:24<00:20, 9234.62it/s] 54%|█████▎    | 214896/400000 [00:24<00:19, 9304.51it/s] 54%|█████▍    | 215828/400000 [00:24<00:20, 9006.68it/s] 54%|█████▍    | 216766/400000 [00:24<00:20, 9114.09it/s] 54%|█████▍    | 217694/400000 [00:24<00:19, 9163.07it/s] 55%|█████▍    | 218612/400000 [00:24<00:20, 8986.75it/s] 55%|█████▍    | 219540/400000 [00:25<00:19, 9071.60it/s] 55%|█████▌    | 220449/400000 [00:25<00:20, 8778.75it/s] 55%|█████▌    | 221416/400000 [00:25<00:19, 9027.94it/s] 56%|█████▌    | 222410/400000 [00:25<00:19, 9281.45it/s] 56%|█████▌    | 223404/400000 [00:25<00:18, 9468.28it/s] 56%|█████▌    | 224356/400000 [00:25<00:18, 9480.22it/s] 56%|█████▋    | 225307/400000 [00:25<00:19, 9046.34it/s] 57%|█████▋    | 226252/400000 [00:25<00:18, 9161.28it/s] 57%|█████▋    | 227267/400000 [00:25<00:18, 9435.80it/s] 57%|█████▋    | 228266/400000 [00:25<00:17, 9593.45it/s] 57%|█████▋    | 229234/400000 [00:26<00:17, 9617.07it/s] 58%|█████▊    | 230204/400000 [00:26<00:17, 9640.16it/s] 58%|█████▊    | 231171/400000 [00:26<00:17, 9630.94it/s] 58%|█████▊    | 232164/400000 [00:26<00:17, 9717.43it/s] 58%|█████▊    | 233137/400000 [00:26<00:17, 9652.60it/s] 59%|█████▊    | 234104/400000 [00:26<00:17, 9623.57it/s] 59%|█████▉    | 235073/400000 [00:26<00:17, 9641.33it/s] 59%|█████▉    | 236038/400000 [00:26<00:17, 9494.20it/s] 59%|█████▉    | 237024/400000 [00:26<00:16, 9599.61it/s] 59%|█████▉    | 237985/400000 [00:26<00:17, 9370.03it/s] 60%|█████▉    | 238924/400000 [00:27<00:17, 9142.98it/s] 60%|█████▉    | 239841/400000 [00:27<00:17, 8978.25it/s] 60%|██████    | 240798/400000 [00:27<00:17, 9144.99it/s] 60%|██████    | 241794/400000 [00:27<00:16, 9374.68it/s] 61%|██████    | 242808/400000 [00:27<00:16, 9591.51it/s] 61%|██████    | 243771/400000 [00:27<00:16, 9458.80it/s] 61%|██████    | 244784/400000 [00:27<00:16, 9650.03it/s] 61%|██████▏   | 245752/400000 [00:27<00:16, 9614.57it/s] 62%|██████▏   | 246727/400000 [00:27<00:15, 9653.80it/s] 62%|██████▏   | 247712/400000 [00:27<00:15, 9710.25it/s] 62%|██████▏   | 248685/400000 [00:28<00:16, 9456.98it/s] 62%|██████▏   | 249633/400000 [00:28<00:16, 9343.48it/s] 63%|██████▎   | 250595/400000 [00:28<00:15, 9423.10it/s] 63%|██████▎   | 251591/400000 [00:28<00:15, 9575.81it/s] 63%|██████▎   | 252581/400000 [00:28<00:15, 9670.48it/s] 63%|██████▎   | 253550/400000 [00:28<00:15, 9590.46it/s] 64%|██████▎   | 254511/400000 [00:28<00:15, 9518.74it/s] 64%|██████▍   | 255468/400000 [00:28<00:15, 9533.78it/s] 64%|██████▍   | 256463/400000 [00:28<00:14, 9654.35it/s] 64%|██████▍   | 257468/400000 [00:28<00:14, 9768.12it/s] 65%|██████▍   | 258446/400000 [00:29<00:14, 9559.68it/s] 65%|██████▍   | 259419/400000 [00:29<00:14, 9610.01it/s] 65%|██████▌   | 260382/400000 [00:29<00:14, 9512.57it/s] 65%|██████▌   | 261389/400000 [00:29<00:14, 9672.13it/s] 66%|██████▌   | 262358/400000 [00:29<00:14, 9659.99it/s] 66%|██████▌   | 263325/400000 [00:29<00:14, 9598.91it/s] 66%|██████▌   | 264291/400000 [00:29<00:14, 9614.25it/s] 66%|██████▋   | 265253/400000 [00:29<00:14, 9498.87it/s] 67%|██████▋   | 266232/400000 [00:29<00:13, 9582.72it/s] 67%|██████▋   | 267191/400000 [00:30<00:14, 9361.39it/s] 67%|██████▋   | 268129/400000 [00:30<00:14, 9161.07it/s] 67%|██████▋   | 269071/400000 [00:30<00:14, 9235.05it/s] 68%|██████▊   | 270016/400000 [00:30<00:13, 9296.62it/s] 68%|██████▊   | 270985/400000 [00:30<00:13, 9410.31it/s] 68%|██████▊   | 271981/400000 [00:30<00:13, 9567.24it/s] 68%|██████▊   | 272940/400000 [00:30<00:13, 9538.68it/s] 68%|██████▊   | 273930/400000 [00:30<00:13, 9643.21it/s] 69%|██████▊   | 274931/400000 [00:30<00:12, 9748.14it/s] 69%|██████▉   | 275907/400000 [00:30<00:12, 9688.11it/s] 69%|██████▉   | 276877/400000 [00:31<00:12, 9676.29it/s] 69%|██████▉   | 277846/400000 [00:31<00:12, 9589.80it/s] 70%|██████▉   | 278806/400000 [00:31<00:12, 9571.35it/s] 70%|██████▉   | 279764/400000 [00:31<00:12, 9348.70it/s] 70%|███████   | 280735/400000 [00:31<00:12, 9454.09it/s] 70%|███████   | 281699/400000 [00:31<00:12, 9508.55it/s] 71%|███████   | 282660/400000 [00:31<00:12, 9538.50it/s] 71%|███████   | 283615/400000 [00:31<00:12, 9301.42it/s] 71%|███████   | 284547/400000 [00:31<00:12, 9152.94it/s] 71%|███████▏  | 285494/400000 [00:31<00:12, 9243.97it/s] 72%|███████▏  | 286480/400000 [00:32<00:12, 9419.53it/s] 72%|███████▏  | 287424/400000 [00:32<00:12, 9347.02it/s] 72%|███████▏  | 288361/400000 [00:32<00:12, 9211.39it/s] 72%|███████▏  | 289328/400000 [00:32<00:11, 9343.55it/s] 73%|███████▎  | 290264/400000 [00:32<00:11, 9317.01it/s] 73%|███████▎  | 291213/400000 [00:32<00:11, 9366.22it/s] 73%|███████▎  | 292153/400000 [00:32<00:11, 9372.41it/s] 73%|███████▎  | 293091/400000 [00:32<00:11, 9168.20it/s] 74%|███████▎  | 294054/400000 [00:32<00:11, 9300.15it/s] 74%|███████▍  | 295008/400000 [00:32<00:11, 9369.16it/s] 74%|███████▍  | 295951/400000 [00:33<00:11, 9384.65it/s] 74%|███████▍  | 296891/400000 [00:33<00:11, 9133.87it/s] 74%|███████▍  | 297807/400000 [00:33<00:11, 8863.90it/s] 75%|███████▍  | 298713/400000 [00:33<00:11, 8919.65it/s] 75%|███████▍  | 299690/400000 [00:33<00:10, 9158.36it/s] 75%|███████▌  | 300665/400000 [00:33<00:10, 9326.93it/s] 75%|███████▌  | 301601/400000 [00:33<00:10, 9297.79it/s] 76%|███████▌  | 302533/400000 [00:33<00:10, 9293.30it/s] 76%|███████▌  | 303505/400000 [00:33<00:10, 9415.93it/s] 76%|███████▌  | 304449/400000 [00:33<00:10, 9404.42it/s] 76%|███████▋  | 305391/400000 [00:34<00:10, 9347.87it/s] 77%|███████▋  | 306347/400000 [00:34<00:09, 9410.16it/s] 77%|███████▋  | 307289/400000 [00:34<00:09, 9321.51it/s] 77%|███████▋  | 308279/400000 [00:34<00:09, 9486.40it/s] 77%|███████▋  | 309256/400000 [00:34<00:09, 9566.20it/s] 78%|███████▊  | 310220/400000 [00:34<00:09, 9585.96it/s] 78%|███████▊  | 311203/400000 [00:34<00:09, 9657.10it/s] 78%|███████▊  | 312170/400000 [00:34<00:09, 9485.44it/s] 78%|███████▊  | 313121/400000 [00:34<00:09, 9491.02it/s] 79%|███████▊  | 314071/400000 [00:35<00:09, 9468.48it/s] 79%|███████▉  | 315019/400000 [00:35<00:09, 9348.57it/s] 79%|███████▉  | 315966/400000 [00:35<00:08, 9383.88it/s] 79%|███████▉  | 316905/400000 [00:35<00:08, 9249.22it/s] 79%|███████▉  | 317831/400000 [00:35<00:08, 9171.27it/s] 80%|███████▉  | 318806/400000 [00:35<00:08, 9336.54it/s] 80%|███████▉  | 319777/400000 [00:35<00:08, 9443.79it/s] 80%|████████  | 320723/400000 [00:35<00:08, 9239.95it/s] 80%|████████  | 321649/400000 [00:35<00:08, 9171.51it/s] 81%|████████  | 322635/400000 [00:35<00:08, 9366.21it/s] 81%|████████  | 323596/400000 [00:36<00:08, 9436.26it/s] 81%|████████  | 324561/400000 [00:36<00:07, 9497.82it/s] 81%|████████▏ | 325512/400000 [00:36<00:07, 9431.12it/s] 82%|████████▏ | 326457/400000 [00:36<00:08, 8914.12it/s] 82%|████████▏ | 327355/400000 [00:36<00:08, 8893.88it/s] 82%|████████▏ | 328288/400000 [00:36<00:07, 9019.74it/s] 82%|████████▏ | 329281/400000 [00:36<00:07, 9273.02it/s] 83%|████████▎ | 330213/400000 [00:36<00:07, 9286.25it/s] 83%|████████▎ | 331145/400000 [00:36<00:07, 9274.11it/s] 83%|████████▎ | 332143/400000 [00:36<00:07, 9473.52it/s] 83%|████████▎ | 333117/400000 [00:37<00:07, 9550.41it/s] 84%|████████▎ | 334093/400000 [00:37<00:06, 9608.20it/s] 84%|████████▍ | 335056/400000 [00:37<00:06, 9557.28it/s] 84%|████████▍ | 336013/400000 [00:37<00:06, 9454.13it/s] 84%|████████▍ | 336993/400000 [00:37<00:06, 9553.22it/s] 84%|████████▍ | 337958/400000 [00:37<00:06, 9581.14it/s] 85%|████████▍ | 338934/400000 [00:37<00:06, 9630.98it/s] 85%|████████▍ | 339910/400000 [00:37<00:06, 9667.00it/s] 85%|████████▌ | 340878/400000 [00:37<00:06, 9287.05it/s] 85%|████████▌ | 341852/400000 [00:37<00:06, 9416.35it/s] 86%|████████▌ | 342846/400000 [00:38<00:05, 9567.33it/s] 86%|████████▌ | 343806/400000 [00:38<00:05, 9512.91it/s] 86%|████████▌ | 344760/400000 [00:38<00:05, 9306.73it/s] 86%|████████▋ | 345693/400000 [00:38<00:05, 9303.06it/s] 87%|████████▋ | 346625/400000 [00:38<00:05, 9283.73it/s] 87%|████████▋ | 347587/400000 [00:38<00:05, 9379.96it/s] 87%|████████▋ | 348539/400000 [00:38<00:05, 9419.31it/s] 87%|████████▋ | 349525/400000 [00:38<00:05, 9545.13it/s] 88%|████████▊ | 350481/400000 [00:38<00:05, 9490.84it/s] 88%|████████▊ | 351499/400000 [00:38<00:05, 9687.44it/s] 88%|████████▊ | 352470/400000 [00:39<00:04, 9540.61it/s] 88%|████████▊ | 353426/400000 [00:39<00:04, 9497.29it/s] 89%|████████▊ | 354377/400000 [00:39<00:04, 9312.54it/s] 89%|████████▉ | 355310/400000 [00:39<00:04, 9082.08it/s] 89%|████████▉ | 356233/400000 [00:39<00:04, 9123.49it/s] 89%|████████▉ | 357191/400000 [00:39<00:04, 9254.55it/s] 90%|████████▉ | 358170/400000 [00:39<00:04, 9406.68it/s] 90%|████████▉ | 359157/400000 [00:39<00:04, 9539.61it/s] 90%|█████████ | 360113/400000 [00:39<00:04, 9402.61it/s] 90%|█████████ | 361065/400000 [00:40<00:04, 9435.59it/s] 91%|█████████ | 362010/400000 [00:40<00:04, 9389.48it/s] 91%|█████████ | 363031/400000 [00:40<00:03, 9619.88it/s] 91%|█████████ | 364033/400000 [00:40<00:03, 9736.12it/s] 91%|█████████▏| 365009/400000 [00:40<00:03, 9524.03it/s] 91%|█████████▏| 365993/400000 [00:40<00:03, 9615.35it/s] 92%|█████████▏| 366957/400000 [00:40<00:03, 9580.65it/s] 92%|█████████▏| 367931/400000 [00:40<00:03, 9627.38it/s] 92%|█████████▏| 368895/400000 [00:40<00:03, 9418.17it/s] 92%|█████████▏| 369839/400000 [00:40<00:03, 9105.48it/s] 93%|█████████▎| 370762/400000 [00:41<00:03, 9140.25it/s] 93%|█████████▎| 371714/400000 [00:41<00:03, 9250.16it/s] 93%|█████████▎| 372642/400000 [00:41<00:02, 9214.78it/s] 93%|█████████▎| 373565/400000 [00:41<00:02, 9041.08it/s] 94%|█████████▎| 374471/400000 [00:41<00:02, 8817.39it/s] 94%|█████████▍| 375356/400000 [00:41<00:02, 8703.47it/s] 94%|█████████▍| 376229/400000 [00:41<00:02, 8702.42it/s] 94%|█████████▍| 377101/400000 [00:41<00:02, 8610.10it/s] 94%|█████████▍| 377964/400000 [00:41<00:02, 8569.27it/s] 95%|█████████▍| 378846/400000 [00:41<00:02, 8640.51it/s] 95%|█████████▍| 379727/400000 [00:42<00:02, 8684.33it/s] 95%|█████████▌| 380597/400000 [00:42<00:02, 8673.04it/s] 95%|█████████▌| 381465/400000 [00:42<00:02, 8397.37it/s] 96%|█████████▌| 382307/400000 [00:42<00:02, 8375.31it/s] 96%|█████████▌| 383147/400000 [00:42<00:02, 8323.96it/s] 96%|█████████▌| 384005/400000 [00:42<00:01, 8398.79it/s] 96%|█████████▌| 384857/400000 [00:42<00:01, 8432.35it/s] 96%|█████████▋| 385701/400000 [00:42<00:01, 8064.01it/s] 97%|█████████▋| 386512/400000 [00:42<00:01, 7849.67it/s] 97%|█████████▋| 387309/400000 [00:43<00:01, 7883.26it/s] 97%|█████████▋| 388114/400000 [00:43<00:01, 7931.36it/s] 97%|█████████▋| 389000/400000 [00:43<00:01, 8187.06it/s] 97%|█████████▋| 389871/400000 [00:43<00:01, 8335.81it/s] 98%|█████████▊| 390708/400000 [00:43<00:01, 8307.78it/s] 98%|█████████▊| 391562/400000 [00:43<00:01, 8374.45it/s] 98%|█████████▊| 392402/400000 [00:43<00:00, 8373.81it/s] 98%|█████████▊| 393241/400000 [00:43<00:00, 8324.77it/s] 99%|█████████▊| 394075/400000 [00:43<00:00, 8315.14it/s] 99%|█████████▊| 394908/400000 [00:43<00:00, 8082.66it/s] 99%|█████████▉| 395719/400000 [00:44<00:00, 7913.59it/s] 99%|█████████▉| 396513/400000 [00:44<00:00, 7903.27it/s] 99%|█████████▉| 397305/400000 [00:44<00:00, 7847.44it/s]100%|█████████▉| 398146/400000 [00:44<00:00, 8007.85it/s]100%|█████████▉| 399007/400000 [00:44<00:00, 8177.20it/s]100%|█████████▉| 399878/400000 [00:44<00:00, 8330.06it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8977.30it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f0018786c50>, <torchtext.data.dataset.TabularDataset object at 0x7f0018786da0>, <torchtext.vocab.Vocab object at 0x7f0018786cc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.03 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.91 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.91 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.31 file/s]2020-07-12 06:18:16.640804: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-12 06:18:16.644225: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-12 06:18:16.644374: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560a7bf38420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 06:18:16.644391: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 149892.63it/s] 84%|████████▍ | 8364032/9912422 [00:00<00:07, 213964.95it/s]9920512it [00:00, 44635721.93it/s]                           
0it [00:00, ?it/s]32768it [00:00, 500710.24it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466172.88it/s]1654784it [00:00, 11886841.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 154360.58it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
