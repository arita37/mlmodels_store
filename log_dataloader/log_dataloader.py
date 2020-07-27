
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f83f0745ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f83f0745ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f845b3b5510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f845b3b5510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f847a5e2ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f847a5e2ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f840875da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f840875da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f840875da60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 2998272/9912422 [00:00<00:00, 29769959.01it/s]9920512it [00:00, 35447487.40it/s]                             
0it [00:00, ?it/s]32768it [00:00, 559242.81it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472177.65it/s]1654784it [00:00, 11471759.95it/s]                         
0it [00:00, ?it/s]8192it [00:00, 145683.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83f0601630>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83f06128d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f840875d6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f840875d6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f840875d6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:08,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:08,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:07,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:07,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:07,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:06,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:46,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:31,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:10, 11.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 55.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 63.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 63.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 68.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 68.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 73.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 73.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 77.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 77.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 80.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 80.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.65s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.91 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.65s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.91 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.14 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.89s/ url]
0 examples [00:00, ? examples/s]2020-07-27 18:12:10.329432: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-27 18:12:10.334087: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-27 18:12:10.334236: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5644c59a22f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-27 18:12:10.334252: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
68 examples [00:00, 679.91 examples/s]169 examples [00:00, 753.21 examples/s]273 examples [00:00, 819.37 examples/s]379 examples [00:00, 878.09 examples/s]486 examples [00:00, 927.10 examples/s]593 examples [00:00, 963.65 examples/s]698 examples [00:00, 987.86 examples/s]805 examples [00:00, 1008.88 examples/s]911 examples [00:00, 1023.32 examples/s]1019 examples [00:01, 1036.95 examples/s]1126 examples [00:01, 1045.40 examples/s]1231 examples [00:01, 1045.34 examples/s]1338 examples [00:01, 1050.98 examples/s]1443 examples [00:01, 1039.29 examples/s]1549 examples [00:01, 1042.77 examples/s]1655 examples [00:01, 1046.88 examples/s]1760 examples [00:01, 1046.05 examples/s]1865 examples [00:01, 1045.91 examples/s]1972 examples [00:01, 1050.67 examples/s]2078 examples [00:02, 1038.40 examples/s]2184 examples [00:02, 1043.24 examples/s]2289 examples [00:02, 1043.35 examples/s]2395 examples [00:02, 1046.47 examples/s]2500 examples [00:02, 1042.62 examples/s]2605 examples [00:02, 1036.18 examples/s]2710 examples [00:02, 1038.38 examples/s]2814 examples [00:02, 1025.13 examples/s]2917 examples [00:02, 1026.39 examples/s]3023 examples [00:02, 1034.22 examples/s]3127 examples [00:03, 1035.30 examples/s]3233 examples [00:03, 1041.02 examples/s]3338 examples [00:03, 1040.53 examples/s]3444 examples [00:03, 1045.39 examples/s]3550 examples [00:03, 1046.19 examples/s]3655 examples [00:03, 1044.90 examples/s]3760 examples [00:03, 1045.66 examples/s]3865 examples [00:03, 1038.63 examples/s]3969 examples [00:03, 1030.10 examples/s]4074 examples [00:03, 1035.33 examples/s]4178 examples [00:04, 1027.95 examples/s]4281 examples [00:04, 1017.71 examples/s]4383 examples [00:04, 1011.61 examples/s]4485 examples [00:04, 1013.28 examples/s]4588 examples [00:04, 1015.91 examples/s]4690 examples [00:04, 1012.14 examples/s]4795 examples [00:04, 1021.65 examples/s]4898 examples [00:04, 1019.52 examples/s]5001 examples [00:04, 1022.10 examples/s]5105 examples [00:04, 1026.49 examples/s]5210 examples [00:05, 1031.88 examples/s]5316 examples [00:05, 1038.90 examples/s]5420 examples [00:05, 1036.11 examples/s]5525 examples [00:05, 1039.81 examples/s]5629 examples [00:05, 1021.09 examples/s]5733 examples [00:05, 1025.80 examples/s]5838 examples [00:05, 1030.55 examples/s]5943 examples [00:05, 1034.34 examples/s]6048 examples [00:05, 1036.75 examples/s]6153 examples [00:05, 1038.74 examples/s]6259 examples [00:06, 1043.16 examples/s]6366 examples [00:06, 1049.44 examples/s]6471 examples [00:06, 1028.49 examples/s]6577 examples [00:06, 1035.16 examples/s]6682 examples [00:06, 1038.19 examples/s]6786 examples [00:06, 1035.29 examples/s]6892 examples [00:06, 1041.37 examples/s]6997 examples [00:06, 1039.00 examples/s]7102 examples [00:06, 1041.77 examples/s]7207 examples [00:06, 1040.31 examples/s]7312 examples [00:07, 1038.75 examples/s]7416 examples [00:07, 1039.01 examples/s]7520 examples [00:07, 1030.07 examples/s]7624 examples [00:07, 1031.39 examples/s]7728 examples [00:07, 1029.40 examples/s]7831 examples [00:07, 1018.42 examples/s]7936 examples [00:07, 1027.49 examples/s]8039 examples [00:07, 1009.54 examples/s]8145 examples [00:07, 1023.78 examples/s]8249 examples [00:07, 1026.70 examples/s]8352 examples [00:08, 1025.63 examples/s]8456 examples [00:08, 1027.56 examples/s]8560 examples [00:08, 1028.36 examples/s]8663 examples [00:08, 1024.71 examples/s]8766 examples [00:08, 1023.77 examples/s]8869 examples [00:08, 1025.39 examples/s]8972 examples [00:08, 1026.14 examples/s]9075 examples [00:08, 1014.86 examples/s]9178 examples [00:08, 1016.66 examples/s]9281 examples [00:09, 1018.75 examples/s]9383 examples [00:09, 1008.53 examples/s]9484 examples [00:09, 989.23 examples/s] 9586 examples [00:09, 997.86 examples/s]9689 examples [00:09, 1006.05 examples/s]9794 examples [00:09, 1016.81 examples/s]9899 examples [00:09, 1025.61 examples/s]10002 examples [00:09, 956.85 examples/s]10104 examples [00:09, 973.85 examples/s]10206 examples [00:09, 986.86 examples/s]10309 examples [00:10, 998.01 examples/s]10413 examples [00:10, 1007.88 examples/s]10517 examples [00:10, 1015.37 examples/s]10621 examples [00:10, 1019.83 examples/s]10726 examples [00:10, 1026.62 examples/s]10832 examples [00:10, 1034.69 examples/s]10938 examples [00:10, 1039.06 examples/s]11042 examples [00:10, 1029.95 examples/s]11146 examples [00:10, 1029.19 examples/s]11249 examples [00:10, 1013.93 examples/s]11354 examples [00:11, 1022.09 examples/s]11459 examples [00:11, 1028.99 examples/s]11562 examples [00:11, 1020.84 examples/s]11666 examples [00:11, 1025.93 examples/s]11769 examples [00:11, 1019.27 examples/s]11871 examples [00:11, 1015.39 examples/s]11976 examples [00:11, 1022.31 examples/s]12082 examples [00:11, 1030.64 examples/s]12186 examples [00:11, 1032.05 examples/s]12290 examples [00:11, 1031.15 examples/s]12396 examples [00:12, 1036.36 examples/s]12500 examples [00:12, 1031.58 examples/s]12606 examples [00:12, 1039.31 examples/s]12712 examples [00:12, 1042.86 examples/s]12817 examples [00:12, 1040.64 examples/s]12922 examples [00:12, 1040.42 examples/s]13028 examples [00:12, 1046.18 examples/s]13133 examples [00:12, 1039.97 examples/s]13238 examples [00:12, 1038.00 examples/s]13342 examples [00:12, 996.08 examples/s] 13442 examples [00:13, 980.40 examples/s]13543 examples [00:13, 988.36 examples/s]13647 examples [00:13, 1002.07 examples/s]13752 examples [00:13, 1014.89 examples/s]13854 examples [00:13, 1014.78 examples/s]13956 examples [00:13, 1007.45 examples/s]14062 examples [00:13, 1022.30 examples/s]14167 examples [00:13, 1027.22 examples/s]14272 examples [00:13, 1033.43 examples/s]14377 examples [00:14, 1036.40 examples/s]14483 examples [00:14, 1040.78 examples/s]14589 examples [00:14, 1044.89 examples/s]14694 examples [00:14, 1044.36 examples/s]14800 examples [00:14, 1047.34 examples/s]14905 examples [00:14, 1044.27 examples/s]15012 examples [00:14, 1049.47 examples/s]15117 examples [00:14, 1036.30 examples/s]15221 examples [00:14, 1037.38 examples/s]15325 examples [00:14, 1037.19 examples/s]15429 examples [00:15, 1022.41 examples/s]15535 examples [00:15, 1031.09 examples/s]15639 examples [00:15, 1032.81 examples/s]15743 examples [00:15, 1032.58 examples/s]15847 examples [00:15, 1033.12 examples/s]15951 examples [00:15, 1029.32 examples/s]16056 examples [00:15, 1033.81 examples/s]16161 examples [00:15, 1038.60 examples/s]16265 examples [00:15, 1033.11 examples/s]16369 examples [00:15, 1029.13 examples/s]16472 examples [00:16, 1028.41 examples/s]16577 examples [00:16, 1032.26 examples/s]16681 examples [00:16, 1032.73 examples/s]16785 examples [00:16, 1033.91 examples/s]16891 examples [00:16, 1040.25 examples/s]16996 examples [00:16, 1041.70 examples/s]17101 examples [00:16, 1044.16 examples/s]17207 examples [00:16, 1047.75 examples/s]17313 examples [00:16, 1050.67 examples/s]17419 examples [00:16, 1051.47 examples/s]17525 examples [00:17, 1049.26 examples/s]17631 examples [00:17, 1050.71 examples/s]17738 examples [00:17, 1054.77 examples/s]17844 examples [00:17, 1054.19 examples/s]17950 examples [00:17, 1052.91 examples/s]18056 examples [00:17, 1049.96 examples/s]18162 examples [00:17, 1032.83 examples/s]18266 examples [00:17, 1032.73 examples/s]18370 examples [00:17, 1031.80 examples/s]18474 examples [00:17, 1034.15 examples/s]18578 examples [00:18, 1027.06 examples/s]18681 examples [00:18, 1023.53 examples/s]18784 examples [00:18, 985.32 examples/s] 18883 examples [00:18, 969.21 examples/s]18988 examples [00:18, 991.82 examples/s]19091 examples [00:18, 1001.45 examples/s]19192 examples [00:18, 1001.81 examples/s]19296 examples [00:18, 1010.26 examples/s]19399 examples [00:18, 1014.53 examples/s]19501 examples [00:18, 991.59 examples/s] 19603 examples [00:19, 997.66 examples/s]19709 examples [00:19, 1015.32 examples/s]19811 examples [00:19, 989.91 examples/s] 19917 examples [00:19, 1006.76 examples/s]20018 examples [00:19, 956.37 examples/s] 20118 examples [00:19, 968.03 examples/s]20225 examples [00:19, 995.47 examples/s]20331 examples [00:19, 1012.25 examples/s]20438 examples [00:19, 1027.99 examples/s]20542 examples [00:20, 1031.04 examples/s]20649 examples [00:20, 1040.39 examples/s]20755 examples [00:20, 1044.71 examples/s]20862 examples [00:20, 1051.33 examples/s]20969 examples [00:20, 1053.38 examples/s]21075 examples [00:20, 1046.90 examples/s]21180 examples [00:20, 1047.31 examples/s]21286 examples [00:20, 1050.66 examples/s]21392 examples [00:20, 1053.20 examples/s]21499 examples [00:20, 1055.68 examples/s]21606 examples [00:21, 1057.86 examples/s]21712 examples [00:21, 1055.89 examples/s]21818 examples [00:21, 1048.40 examples/s]21925 examples [00:21, 1054.55 examples/s]22032 examples [00:21, 1057.53 examples/s]22138 examples [00:21, 1054.20 examples/s]22244 examples [00:21, 1052.81 examples/s]22351 examples [00:21, 1056.63 examples/s]22457 examples [00:21, 1045.19 examples/s]22564 examples [00:21, 1050.52 examples/s]22670 examples [00:22, 1046.68 examples/s]22775 examples [00:22, 1036.48 examples/s]22881 examples [00:22, 1042.98 examples/s]22988 examples [00:22, 1048.77 examples/s]23094 examples [00:22, 1051.41 examples/s]23200 examples [00:22, 1051.92 examples/s]23306 examples [00:22, 1041.11 examples/s]23411 examples [00:22, 1042.27 examples/s]23516 examples [00:22, 1039.92 examples/s]23621 examples [00:22, 1033.19 examples/s]23725 examples [00:23, 1034.55 examples/s]23829 examples [00:23, 1027.98 examples/s]23934 examples [00:23, 1031.88 examples/s]24038 examples [00:23, 1026.12 examples/s]24143 examples [00:23, 1032.29 examples/s]24249 examples [00:23, 1037.82 examples/s]24354 examples [00:23, 1039.22 examples/s]24460 examples [00:23, 1042.63 examples/s]24565 examples [00:23, 1040.87 examples/s]24671 examples [00:23, 1043.63 examples/s]24776 examples [00:24, 1039.39 examples/s]24880 examples [00:24, 1035.90 examples/s]24985 examples [00:24, 1037.63 examples/s]25090 examples [00:24, 1038.94 examples/s]25194 examples [00:24, 1038.50 examples/s]25298 examples [00:24, 1011.98 examples/s]25400 examples [00:24, 997.27 examples/s] 25504 examples [00:24, 1009.49 examples/s]25607 examples [00:24, 1014.10 examples/s]25709 examples [00:24, 995.67 examples/s] 25813 examples [00:25, 1007.55 examples/s]25915 examples [00:25, 1010.55 examples/s]26020 examples [00:25, 1019.38 examples/s]26125 examples [00:25, 1026.50 examples/s]26230 examples [00:25, 1032.21 examples/s]26334 examples [00:25, 1030.10 examples/s]26438 examples [00:25, 1032.91 examples/s]26544 examples [00:25, 1038.69 examples/s]26648 examples [00:25, 1037.93 examples/s]26752 examples [00:25, 1034.43 examples/s]26857 examples [00:26, 1037.73 examples/s]26961 examples [00:26, 1036.43 examples/s]27066 examples [00:26, 1039.26 examples/s]27172 examples [00:26, 1044.09 examples/s]27277 examples [00:26, 1041.22 examples/s]27382 examples [00:26, 1041.42 examples/s]27487 examples [00:26, 1039.23 examples/s]27593 examples [00:26, 1018.45 examples/s]27695 examples [00:26, 1009.78 examples/s]27797 examples [00:27, 1011.46 examples/s]27901 examples [00:27, 1017.38 examples/s]28004 examples [00:27, 1019.76 examples/s]28107 examples [00:27, 1016.90 examples/s]28212 examples [00:27, 1026.55 examples/s]28315 examples [00:27, 1024.92 examples/s]28421 examples [00:27, 1033.13 examples/s]28526 examples [00:27, 1036.90 examples/s]28632 examples [00:27, 1041.33 examples/s]28738 examples [00:27, 1045.23 examples/s]28846 examples [00:28, 1052.94 examples/s]28952 examples [00:28, 1053.44 examples/s]29058 examples [00:28, 1046.17 examples/s]29164 examples [00:28, 1048.79 examples/s]29270 examples [00:28, 1051.99 examples/s]29376 examples [00:28, 1035.74 examples/s]29483 examples [00:28, 1044.36 examples/s]29588 examples [00:28, 1042.77 examples/s]29694 examples [00:28, 1047.41 examples/s]29799 examples [00:28, 1047.47 examples/s]29906 examples [00:29, 1052.16 examples/s]30012 examples [00:29, 995.22 examples/s] 30113 examples [00:29, 985.53 examples/s]30218 examples [00:29, 1003.57 examples/s]30323 examples [00:29, 1014.46 examples/s]30427 examples [00:29, 1020.45 examples/s]30532 examples [00:29, 1028.50 examples/s]30636 examples [00:29, 1008.78 examples/s]30742 examples [00:29, 1022.54 examples/s]30846 examples [00:29, 1025.12 examples/s]30951 examples [00:30, 1030.99 examples/s]31056 examples [00:30, 1034.59 examples/s]31160 examples [00:30, 1031.22 examples/s]31264 examples [00:30, 1015.12 examples/s]31369 examples [00:30, 1024.12 examples/s]31472 examples [00:30, 1019.12 examples/s]31575 examples [00:30, 1019.79 examples/s]31678 examples [00:30, 991.06 examples/s] 31780 examples [00:30, 997.38 examples/s]31884 examples [00:30, 1003.87 examples/s]31985 examples [00:31, 1002.21 examples/s]32089 examples [00:31, 1013.10 examples/s]32191 examples [00:31, 1007.43 examples/s]32294 examples [00:31, 1012.71 examples/s]32399 examples [00:31, 1022.26 examples/s]32503 examples [00:31, 1027.23 examples/s]32606 examples [00:31, 1025.90 examples/s]32709 examples [00:31, 997.12 examples/s] 32814 examples [00:31, 1011.61 examples/s]32917 examples [00:32, 1015.68 examples/s]33021 examples [00:32, 1022.67 examples/s]33126 examples [00:32, 1028.49 examples/s]33230 examples [00:32, 1030.99 examples/s]33336 examples [00:32, 1037.34 examples/s]33443 examples [00:32, 1044.23 examples/s]33548 examples [00:32, 1020.35 examples/s]33653 examples [00:32, 1027.55 examples/s]33756 examples [00:32, 1025.25 examples/s]33861 examples [00:32, 1032.43 examples/s]33966 examples [00:33, 1036.39 examples/s]34073 examples [00:33, 1043.66 examples/s]34179 examples [00:33, 1046.11 examples/s]34284 examples [00:33, 1041.67 examples/s]34389 examples [00:33, 1043.36 examples/s]34496 examples [00:33, 1049.20 examples/s]34601 examples [00:33, 1049.00 examples/s]34706 examples [00:33, 1047.52 examples/s]34811 examples [00:33, 1028.81 examples/s]34917 examples [00:33, 1036.75 examples/s]35022 examples [00:34, 1039.49 examples/s]35127 examples [00:34, 1040.83 examples/s]35233 examples [00:34, 1045.17 examples/s]35338 examples [00:34, 1035.47 examples/s]35442 examples [00:34, 1035.51 examples/s]35548 examples [00:34, 1041.33 examples/s]35653 examples [00:34, 1042.52 examples/s]35758 examples [00:34, 1040.32 examples/s]35863 examples [00:34, 1037.39 examples/s]35968 examples [00:34, 1040.24 examples/s]36073 examples [00:35, 1035.00 examples/s]36178 examples [00:35, 1036.87 examples/s]36284 examples [00:35, 1043.17 examples/s]36389 examples [00:35, 1036.31 examples/s]36493 examples [00:35, 1036.35 examples/s]36597 examples [00:35, 1023.37 examples/s]36703 examples [00:35, 1031.12 examples/s]36807 examples [00:35, 1027.48 examples/s]36911 examples [00:35, 1029.71 examples/s]37017 examples [00:35, 1038.13 examples/s]37121 examples [00:36, 1037.37 examples/s]37227 examples [00:36, 1043.21 examples/s]37332 examples [00:36, 1044.57 examples/s]37437 examples [00:36, 1043.83 examples/s]37542 examples [00:36, 1041.42 examples/s]37648 examples [00:36, 1045.22 examples/s]37754 examples [00:36, 1049.08 examples/s]37859 examples [00:36, 1046.85 examples/s]37964 examples [00:36, 1046.12 examples/s]38069 examples [00:36, 1045.00 examples/s]38174 examples [00:37, 1025.13 examples/s]38280 examples [00:37, 1033.29 examples/s]38385 examples [00:37, 1037.72 examples/s]38490 examples [00:37, 1039.29 examples/s]38594 examples [00:37, 1039.21 examples/s]38698 examples [00:37, 1036.76 examples/s]38803 examples [00:37, 1040.52 examples/s]38909 examples [00:37, 1045.48 examples/s]39014 examples [00:37, 1044.99 examples/s]39120 examples [00:37, 1047.64 examples/s]39226 examples [00:38, 1049.36 examples/s]39332 examples [00:38, 1051.35 examples/s]39438 examples [00:38, 1050.02 examples/s]39544 examples [00:38, 1043.18 examples/s]39649 examples [00:38, 1043.55 examples/s]39755 examples [00:38, 1046.32 examples/s]39860 examples [00:38, 1035.39 examples/s]39967 examples [00:38, 1043.37 examples/s]40072 examples [00:38, 983.59 examples/s] 40178 examples [00:39, 1004.20 examples/s]40284 examples [00:39, 1018.90 examples/s]40390 examples [00:39, 1030.55 examples/s]40496 examples [00:39, 1038.45 examples/s]40601 examples [00:39, 1033.03 examples/s]40705 examples [00:39, 1034.14 examples/s]40810 examples [00:39, 1036.52 examples/s]40915 examples [00:39, 1040.01 examples/s]41020 examples [00:39, 1041.32 examples/s]41125 examples [00:39, 1039.73 examples/s]41230 examples [00:40, 1041.43 examples/s]41336 examples [00:40, 1043.91 examples/s]41441 examples [00:40, 1044.94 examples/s]41546 examples [00:40, 1045.97 examples/s]41651 examples [00:40, 1043.99 examples/s]41757 examples [00:40, 1047.32 examples/s]41863 examples [00:40, 1050.05 examples/s]41969 examples [00:40, 1044.14 examples/s]42074 examples [00:40, 1043.87 examples/s]42179 examples [00:40, 1041.30 examples/s]42284 examples [00:41, 1042.88 examples/s]42389 examples [00:41, 1044.34 examples/s]42494 examples [00:41, 1044.25 examples/s]42600 examples [00:41, 1048.11 examples/s]42705 examples [00:41, 1047.14 examples/s]42811 examples [00:41, 1049.57 examples/s]42917 examples [00:41, 1051.47 examples/s]43023 examples [00:41, 1037.45 examples/s]43127 examples [00:41, 1037.23 examples/s]43231 examples [00:41, 1035.95 examples/s]43335 examples [00:42, 1028.37 examples/s]43441 examples [00:42, 1036.14 examples/s]43545 examples [00:42, 1032.75 examples/s]43651 examples [00:42, 1040.31 examples/s]43756 examples [00:42, 1042.95 examples/s]43861 examples [00:42, 1042.01 examples/s]43966 examples [00:42, 1034.17 examples/s]44070 examples [00:42, 1035.88 examples/s]44174 examples [00:42, 1025.91 examples/s]44278 examples [00:42, 1029.48 examples/s]44383 examples [00:43, 1034.04 examples/s]44487 examples [00:43, 1021.10 examples/s]44593 examples [00:43, 1031.36 examples/s]44698 examples [00:43, 1036.72 examples/s]44802 examples [00:43, 1036.07 examples/s]44908 examples [00:43, 1040.26 examples/s]45014 examples [00:43, 1045.90 examples/s]45119 examples [00:43, 1046.78 examples/s]45225 examples [00:43, 1048.98 examples/s]45330 examples [00:43, 1046.21 examples/s]45435 examples [00:44, 1042.38 examples/s]45541 examples [00:44, 1046.05 examples/s]45647 examples [00:44, 1049.19 examples/s]45754 examples [00:44, 1054.07 examples/s]45861 examples [00:44, 1056.78 examples/s]45967 examples [00:44, 1052.08 examples/s]46073 examples [00:44, 1053.69 examples/s]46179 examples [00:44, 1054.25 examples/s]46286 examples [00:44, 1058.38 examples/s]46392 examples [00:44, 1057.45 examples/s]46498 examples [00:45, 1049.13 examples/s]46603 examples [00:45, 1048.50 examples/s]46708 examples [00:45, 1045.32 examples/s]46813 examples [00:45, 1026.77 examples/s]46918 examples [00:45, 1031.99 examples/s]47022 examples [00:45, 1031.06 examples/s]47127 examples [00:45, 1036.62 examples/s]47231 examples [00:45, 1032.61 examples/s]47335 examples [00:45, 966.29 examples/s] 47437 examples [00:46, 980.83 examples/s]47540 examples [00:46, 992.48 examples/s]47644 examples [00:46, 1003.45 examples/s]47748 examples [00:46, 1013.61 examples/s]47853 examples [00:46, 1021.88 examples/s]47956 examples [00:46, 1023.73 examples/s]48061 examples [00:46, 1031.18 examples/s]48165 examples [00:46, 1024.36 examples/s]48268 examples [00:46, 1024.14 examples/s]48373 examples [00:46, 1031.27 examples/s]48477 examples [00:47, 1033.86 examples/s]48581 examples [00:47, 1033.56 examples/s]48686 examples [00:47, 1036.73 examples/s]48790 examples [00:47, 1026.99 examples/s]48895 examples [00:47, 1031.31 examples/s]49000 examples [00:47, 1036.78 examples/s]49104 examples [00:47, 1028.27 examples/s]49207 examples [00:47, 1025.47 examples/s]49312 examples [00:47, 1031.18 examples/s]49416 examples [00:47, 1030.88 examples/s]49520 examples [00:48, 1029.80 examples/s]49623 examples [00:48, 1017.84 examples/s]49726 examples [00:48, 1019.27 examples/s]49831 examples [00:48, 1026.22 examples/s]49935 examples [00:48, 1027.98 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6310/50000 [00:00<00:00, 63099.01 examples/s] 38%|███▊      | 19145/50000 [00:00<00:00, 74454.06 examples/s] 64%|██████▍   | 31993/50000 [00:00<00:00, 85201.97 examples/s] 90%|████████▉ | 44816/50000 [00:00<00:00, 94738.74 examples/s]                                                               0 examples [00:00, ? examples/s]81 examples [00:00, 809.97 examples/s]180 examples [00:00, 855.37 examples/s]286 examples [00:00, 905.98 examples/s]392 examples [00:00, 945.79 examples/s]492 examples [00:00, 960.05 examples/s]597 examples [00:00, 984.31 examples/s]704 examples [00:00, 1007.19 examples/s]810 examples [00:00, 1020.63 examples/s]917 examples [00:00, 1032.91 examples/s]1023 examples [00:01, 1038.64 examples/s]1128 examples [00:01, 1041.78 examples/s]1234 examples [00:01, 1044.78 examples/s]1340 examples [00:01, 1047.42 examples/s]1447 examples [00:01, 1052.70 examples/s]1553 examples [00:01, 1054.24 examples/s]1659 examples [00:01, 1054.27 examples/s]1765 examples [00:01, 1052.77 examples/s]1871 examples [00:01, 1054.92 examples/s]1977 examples [00:01, 1056.00 examples/s]2084 examples [00:02, 1059.21 examples/s]2190 examples [00:02, 1056.07 examples/s]2296 examples [00:02, 1051.10 examples/s]2402 examples [00:02, 1050.77 examples/s]2508 examples [00:02, 1052.22 examples/s]2615 examples [00:02, 1054.97 examples/s]2721 examples [00:02, 1052.19 examples/s]2828 examples [00:02, 1055.72 examples/s]2934 examples [00:02, 1053.95 examples/s]3040 examples [00:02, 1054.88 examples/s]3146 examples [00:03, 1046.43 examples/s]3252 examples [00:03, 1050.33 examples/s]3358 examples [00:03, 1032.77 examples/s]3464 examples [00:03, 1038.25 examples/s]3568 examples [00:03, 1038.48 examples/s]3673 examples [00:03, 1040.40 examples/s]3778 examples [00:03, 1040.54 examples/s]3883 examples [00:03, 1032.12 examples/s]3989 examples [00:03, 1036.77 examples/s]4095 examples [00:03, 1042.00 examples/s]4201 examples [00:04, 1046.69 examples/s]4306 examples [00:04, 1042.73 examples/s]4411 examples [00:04, 1032.19 examples/s]4517 examples [00:04, 1040.14 examples/s]4622 examples [00:04, 1041.15 examples/s]4727 examples [00:04, 1038.71 examples/s]4832 examples [00:04, 1041.70 examples/s]4937 examples [00:04, 1039.11 examples/s]5043 examples [00:04, 1044.23 examples/s]5149 examples [00:04, 1048.12 examples/s]5256 examples [00:05, 1051.60 examples/s]5362 examples [00:05, 1046.25 examples/s]5467 examples [00:05, 1044.73 examples/s]5573 examples [00:05, 1048.41 examples/s]5679 examples [00:05, 1049.60 examples/s]5785 examples [00:05, 1051.65 examples/s]5891 examples [00:05, 1049.98 examples/s]5997 examples [00:05, 1046.46 examples/s]6102 examples [00:05, 1046.55 examples/s]6208 examples [00:05, 1047.91 examples/s]6313 examples [00:06, 1048.30 examples/s]6418 examples [00:06, 1044.99 examples/s]6523 examples [00:06, 1014.38 examples/s]6628 examples [00:06, 1022.67 examples/s]6735 examples [00:06, 1034.65 examples/s]6841 examples [00:06, 1040.57 examples/s]6947 examples [00:06, 1045.22 examples/s]7052 examples [00:06, 1040.61 examples/s]7157 examples [00:06, 1025.52 examples/s]7262 examples [00:06, 1032.56 examples/s]7368 examples [00:07, 1039.94 examples/s]7475 examples [00:07, 1046.15 examples/s]7580 examples [00:07, 1043.15 examples/s]7686 examples [00:07, 1046.75 examples/s]7793 examples [00:07, 1051.07 examples/s]7900 examples [00:07, 1053.72 examples/s]8006 examples [00:07, 1052.37 examples/s]8112 examples [00:07, 1042.38 examples/s]8217 examples [00:07, 1038.02 examples/s]8323 examples [00:07, 1042.28 examples/s]8428 examples [00:08, 991.95 examples/s] 8528 examples [00:08, 946.36 examples/s]8631 examples [00:08, 969.23 examples/s]8735 examples [00:08, 989.40 examples/s]8840 examples [00:08, 1005.69 examples/s]8942 examples [00:08, 996.90 examples/s] 9045 examples [00:08, 1006.42 examples/s]9152 examples [00:08, 1023.12 examples/s]9256 examples [00:08, 1027.69 examples/s]9359 examples [00:09, 1025.67 examples/s]9465 examples [00:09, 1033.34 examples/s]9569 examples [00:09, 1027.72 examples/s]9673 examples [00:09, 1029.58 examples/s]9777 examples [00:09, 1018.12 examples/s]9879 examples [00:09, 1005.94 examples/s]9980 examples [00:09, 1004.46 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIKAV7W/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIKAV7W/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f840875da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f840875da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f840875da60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84540ca0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f838e9d6400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f840875da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f840875da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f840875da60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83f2403358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f83f0612048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f83870e12f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f83870e12f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f83870e12f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f83870e1400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f83870e1400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f83870e1400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=ec0343eb55dcaa4711418c13815fb16b8bfda87184fad9557fa0f68133f19586
  Stored in directory: /tmp/pip-ephem-wheel-cache-vyq26v5g/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:30:23, 12.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:11:16, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:17:09, 25.8kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:30:32, 36.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:32:43, 52.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.81M/862M [00:01<3:09:36, 74.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:12:21, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:32:13, 153kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<1:04:20, 218kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<44:52, 310kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:01<31:22, 442kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:01<21:55, 628kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<15:23, 891kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<10:48, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<07:37, 1.78MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<05:51, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<05:59, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<07:51, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:16, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<04:35, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<07:43, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<07:01, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:14, 2.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:25, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<07:27, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:08<05:50, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:15, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<09:26, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<07:57, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:10<05:51, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<07:11, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<06:23, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:12<04:44, 2.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<06:24, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<04:11, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:14<03:06, 4.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<51:18, 254kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<37:14, 349kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<26:20, 493kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<21:25, 604kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<16:19, 793kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<11:41, 1.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<11:10, 1.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<10:27, 1.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<07:52, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:20<05:41, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<09:21, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<07:53, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<05:50, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<07:03, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:39, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<06:00, 2.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<04:20, 2.92MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:24:11, 151kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:00:12, 210kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<42:23, 298kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<32:32, 387kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<25:20, 497kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<18:22, 685kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<14:50, 845kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:39, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<08:28, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:49, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:27, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:28, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:38, 2.67MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<9:23:52, 22.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<6:34:49, 31.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<4:35:51, 44.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<3:17:14, 62.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:20:33, 87.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:38:54, 124kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:10:56, 173kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<50:57, 241kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<35:52, 341kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<27:49, 438kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<21:58, 554kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<15:53, 766kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<11:15, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<13:45, 881kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:51, 1.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:51, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:19, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:02, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:10, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:28, 1.85MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:58, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:23, 2.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:57, 3.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:39, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:55, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:12, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:45, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:14, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:48, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<09:38, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:56, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:51, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:50, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:11, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:33, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:02, 2.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:12, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:57, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:09, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:17, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:33, 2.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:10, 2.76MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:38, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:07, 2.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:52, 2.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:24, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:57, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:45, 3.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:52, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:39, 3.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:14, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:50, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:57, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:43, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:06, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:45, 2.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:34, 3.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:05, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:49, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:33, 2.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<03:19, 3.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:09, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:50, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:13<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:04, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:19, 2.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:50, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:39, 2.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:05, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:41, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:33, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:26, 3.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:57, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:33, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:27, 3.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:00, 3.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<8:21:59, 21.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<5:51:29, 30.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<4:04:57, 43.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<3:18:57, 53.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<2:21:24, 74.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:39:23, 106kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<1:10:56, 148kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<50:45, 206kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<35:43, 292kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<27:18, 381kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<21:19, 488kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<15:27, 672kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<10:54, 949kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:21:25, 127kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<58:02, 178kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<40:48, 253kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<30:52, 333kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<23:42, 433kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<17:05, 600kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<13:34, 752kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:32, 968kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<07:37, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:41, 1.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<06:23, 1.58MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:43, 2.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:40, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:01, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:43, 2.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:55, 2.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:29, 2.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:20, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:40, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:17, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:14, 3.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:36, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:05, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<02:58, 3.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<07:15, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<06:06, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:28, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:47, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:28, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:19, 2.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:34, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:24, 2.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:38, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:03, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:57, 3.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:34, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:16, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:37, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:26, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:46, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:34, 2.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:42, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:15, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:12, 2.91MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:02, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:03, 3.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:58, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:00, 3.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:16, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:56, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:56, 3.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:54, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:51, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:55, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:09, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:49, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:54, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:43, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:41, 2.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:42, 3.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:45, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:41, 2.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:37, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:01, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:58, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<02:52, 3.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<1:04:04, 136kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<45:42, 190kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<32:07, 270kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<24:25, 353kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<18:52, 457kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<13:33, 635kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<09:34, 895kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<10:23, 824kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<08:09, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<05:54, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:06, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:08, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:48, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:38, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:59, 1.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:51, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<02:47, 2.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<07:11, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:53, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:19, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:57, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:19, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:14, 2.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:10, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:40, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:42, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:40, 3.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<59:55, 136kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<42:46, 190kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<30:03, 270kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<22:50, 354kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<17:38, 458kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<12:41, 635kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<08:57, 896kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<09:35, 835kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<07:32, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<05:28, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:39, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:38, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:23, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<02:28, 3.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<33:24, 236kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<24:59, 315kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<17:48, 441kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<12:33, 623kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<11:16, 692kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<08:40, 898kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<06:15, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:10, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:58, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:39, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:38, 2.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<30:51, 248kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<23:09, 331kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<16:34, 461kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<12:45, 596kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<09:43, 781kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<06:56, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:36, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<05:23, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:55, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:29, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:54, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:54, 2.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:46, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:24, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:32, 2.90MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:30, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:11, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:24, 3.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:23, 2.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:59, 2.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:14, 3.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<01:39, 4.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<28:19, 254kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<21:17, 337kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<15:14, 470kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<11:44, 606kB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<08:56, 795kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<06:23, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<06:06, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:44, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:18, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<03:09, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:07, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:36, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:40, 2.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:08, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:20, 2.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:59, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:14, 3.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:10, 2.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:39, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:51, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:04, 3.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:55, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:08, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:03, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:09, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:13, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:21, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:37, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:17, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:28, 2.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:06, 2.09MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:56, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:12, 2.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:54, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:28, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:59, 3.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:06, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:28, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:36, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:09, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:41, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:56, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:07, 2.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:36, 818kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<06:01, 1.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:21, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:20, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:28, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:25, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:28, 2.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:54, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:25, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:32, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:03, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:06, 2.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:19, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:39, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:55, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<07:07, 824kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:39, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<04:06, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:05, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:09, 1.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:11, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:18, 2.50MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:56, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:25, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:33, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:59, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:21, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:40, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<01:56, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<06:51, 816kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:25, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:56, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:55, 1.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:03, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:06, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:14, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:38, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:09, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:20, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:46, 1.94MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:33, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:56, 2.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:28, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:58, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:20, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:43, 3.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:49, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:34, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:55, 2.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:26, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:55, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:17, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:40, 3.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:50, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:34, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:55, 2.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:24, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:48, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:37, 3.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:46, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:30, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:51, 2.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:20, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:46, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:10, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:35, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:53, 1.67MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:34, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:55, 2.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:21, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:12, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:39, 2.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:09, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:02, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:32, 3.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:03, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:59, 2.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:26, 3.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:14, 867kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<04:11, 1.09MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<03:01, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:03, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:08, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:24, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:43, 2.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:07, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:40, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:58, 2.22MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:18, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:37, 1.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:02, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:29, 2.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:30, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:14, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:03, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:25, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:54, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:22, 3.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:16, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:41, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:02, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:20, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:50, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:20, 3.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:21, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:05, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:33, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:55, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:15, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:45, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:17, 3.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:10, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:57, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:28, 2.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:49, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:10, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:41, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:14, 3.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:56, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:45, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:18, 2.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:43, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:01, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:35, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:09, 3.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:01, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:49, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:21, 2.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:41, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:00, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:35, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<01:08, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:00, 860kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:10, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:50<02:17, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:23, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:51, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:19, 2.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:39, 903kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:55, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:07, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:09, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:48, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:23, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<01:00, 3.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:09, 1.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:57, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:12, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:35, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:07, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:49, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:21, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:35, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:46, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:23, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:01, 2.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:36, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:27, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:05, 2.67MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:22, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:56, 3.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:33, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:24, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:03, 2.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:19, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:14, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:56, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:13, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:26, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:09, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:50, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:11, 816kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:31, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:48, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:47, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:32, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:07, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:18, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:29, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:09, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:49, 2.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:46, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:31, 1.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:06, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:53, 2.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<1:50:35, 21.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<1:17:07, 30.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<52:52, 42.9kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<39:40, 57.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<28:16, 80.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<19:47, 114kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<13:40, 162kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<10:24, 211kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<07:30, 292kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<05:14, 413kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:04, 520kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:19, 638kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:25, 872kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<01:42, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:52, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:31, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:06, 1.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:12, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:01, 1.92MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:45, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:33, 3.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:03, 928kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:52, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:59, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:21, 1.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:09, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:50, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:57, 1.84MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:50, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:35, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:01, 840kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:35, 1.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:08, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:09, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:09, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:53, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:37, 2.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:09, 304kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:46, 413kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:38, 582kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:07, 705kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:48, 829kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:19, 1.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:55, 1.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:38, 871kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:18, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:56, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:55, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:47, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:34, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:40, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:46, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:35, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:25, 2.92MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:47, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:30, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:30, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:21, 3.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<02:34, 422kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:54, 564kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:20, 791kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:07, 909kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:59, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:44, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:42, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:36, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:29, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:17, 2.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:19, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:14, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:15, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:19, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:15, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:10, 3.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:12, 437kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:53, 585kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:36, 819kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:29, 927kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:26, 1.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:19, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:12, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:31, 736kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:24, 941kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:16, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:14, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:14, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:10, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 827kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 1.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.43MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 799/400000 [00:00<00:49, 7985.19it/s]  0%|          | 1581/400000 [00:00<00:50, 7933.62it/s]  1%|          | 2403/400000 [00:00<00:49, 8015.05it/s]  1%|          | 3221/400000 [00:00<00:49, 8061.37it/s]  1%|          | 3964/400000 [00:00<00:50, 7858.57it/s]  1%|          | 4752/400000 [00:00<00:50, 7863.46it/s]  1%|▏         | 5569/400000 [00:00<00:49, 7952.53it/s]  2%|▏         | 6387/400000 [00:00<00:49, 8016.97it/s]  2%|▏         | 7198/400000 [00:00<00:48, 8042.70it/s]  2%|▏         | 8008/400000 [00:01<00:48, 8059.73it/s]  2%|▏         | 8827/400000 [00:01<00:48, 8097.06it/s]  2%|▏         | 9647/400000 [00:01<00:48, 8126.96it/s]  3%|▎         | 10470/400000 [00:01<00:47, 8154.76it/s]  3%|▎         | 11278/400000 [00:01<00:48, 7996.97it/s]  3%|▎         | 12074/400000 [00:01<00:49, 7877.03it/s]  3%|▎         | 12864/400000 [00:01<00:49, 7883.39it/s]  3%|▎         | 13676/400000 [00:01<00:48, 7952.64it/s]  4%|▎         | 14471/400000 [00:01<00:49, 7861.58it/s]  4%|▍         | 15257/400000 [00:01<00:49, 7824.57it/s]  4%|▍         | 16040/400000 [00:02<00:49, 7796.90it/s]  4%|▍         | 16842/400000 [00:02<00:48, 7860.38it/s]  4%|▍         | 17642/400000 [00:02<00:48, 7900.07it/s]  5%|▍         | 18433/400000 [00:02<00:48, 7901.54it/s]  5%|▍         | 19252/400000 [00:02<00:47, 7985.64it/s]  5%|▌         | 20061/400000 [00:02<00:47, 8014.38it/s]  5%|▌         | 20882/400000 [00:02<00:46, 8069.97it/s]  5%|▌         | 21705/400000 [00:02<00:46, 8114.87it/s]  6%|▌         | 22527/400000 [00:02<00:46, 8145.30it/s]  6%|▌         | 23350/400000 [00:02<00:46, 8170.37it/s]  6%|▌         | 24168/400000 [00:03<00:46, 8152.98it/s]  6%|▌         | 24990/400000 [00:03<00:45, 8171.73it/s]  6%|▋         | 25808/400000 [00:03<00:45, 8159.85it/s]  7%|▋         | 26629/400000 [00:03<00:45, 8173.90it/s]  7%|▋         | 27453/400000 [00:03<00:45, 8192.15it/s]  7%|▋         | 28273/400000 [00:03<00:45, 8147.09it/s]  7%|▋         | 29088/400000 [00:03<00:46, 8026.10it/s]  7%|▋         | 29892/400000 [00:03<00:46, 8021.82it/s]  8%|▊         | 30705/400000 [00:03<00:45, 8053.51it/s]  8%|▊         | 31511/400000 [00:03<00:45, 8024.63it/s]  8%|▊         | 32314/400000 [00:04<00:46, 7893.97it/s]  8%|▊         | 33133/400000 [00:04<00:45, 7980.13it/s]  8%|▊         | 33947/400000 [00:04<00:45, 8024.69it/s]  9%|▊         | 34768/400000 [00:04<00:45, 8076.47it/s]  9%|▉         | 35577/400000 [00:04<00:45, 7975.82it/s]  9%|▉         | 36387/400000 [00:04<00:45, 8011.37it/s]  9%|▉         | 37193/400000 [00:04<00:45, 8024.86it/s] 10%|▉         | 38012/400000 [00:04<00:44, 8071.25it/s] 10%|▉         | 38834/400000 [00:04<00:44, 8113.89it/s] 10%|▉         | 39646/400000 [00:04<00:44, 8017.29it/s] 10%|█         | 40460/400000 [00:05<00:44, 8051.67it/s] 10%|█         | 41278/400000 [00:05<00:44, 8088.43it/s] 11%|█         | 42099/400000 [00:05<00:44, 8122.18it/s] 11%|█         | 42918/400000 [00:05<00:43, 8141.72it/s] 11%|█         | 43740/400000 [00:05<00:43, 8164.86it/s] 11%|█         | 44557/400000 [00:05<00:43, 8151.77it/s] 11%|█▏        | 45377/400000 [00:05<00:43, 8163.89it/s] 12%|█▏        | 46198/400000 [00:05<00:43, 8177.48it/s] 12%|█▏        | 47021/400000 [00:05<00:43, 8190.24it/s] 12%|█▏        | 47845/400000 [00:05<00:42, 8205.03it/s] 12%|█▏        | 48666/400000 [00:06<00:42, 8179.12it/s] 12%|█▏        | 49488/400000 [00:06<00:42, 8188.87it/s] 13%|█▎        | 50311/400000 [00:06<00:42, 8198.67it/s] 13%|█▎        | 51131/400000 [00:06<00:43, 7972.06it/s] 13%|█▎        | 51930/400000 [00:06<00:43, 7975.40it/s] 13%|█▎        | 52741/400000 [00:06<00:43, 8013.14it/s] 13%|█▎        | 53544/400000 [00:06<00:43, 8003.78it/s] 14%|█▎        | 54345/400000 [00:06<00:43, 7902.63it/s] 14%|█▍        | 55160/400000 [00:06<00:43, 7973.22it/s] 14%|█▍        | 55981/400000 [00:06<00:42, 8041.72it/s] 14%|█▍        | 56795/400000 [00:07<00:42, 8070.06it/s] 14%|█▍        | 57608/400000 [00:07<00:42, 8087.50it/s] 15%|█▍        | 58421/400000 [00:07<00:42, 8100.19it/s] 15%|█▍        | 59241/400000 [00:07<00:41, 8128.67it/s] 15%|█▌        | 60061/400000 [00:07<00:41, 8148.46it/s] 15%|█▌        | 60876/400000 [00:07<00:42, 7990.49it/s] 15%|█▌        | 61676/400000 [00:07<00:43, 7806.64it/s] 16%|█▌        | 62487/400000 [00:07<00:42, 7894.10it/s] 16%|█▌        | 63305/400000 [00:07<00:42, 7977.59it/s] 16%|█▌        | 64125/400000 [00:07<00:41, 8041.95it/s] 16%|█▌        | 64933/400000 [00:08<00:41, 8050.85it/s] 16%|█▋        | 65744/400000 [00:08<00:41, 8068.37it/s] 17%|█▋        | 66560/400000 [00:08<00:41, 8092.89it/s] 17%|█▋        | 67380/400000 [00:08<00:40, 8123.12it/s] 17%|█▋        | 68198/400000 [00:08<00:40, 8138.98it/s] 17%|█▋        | 69013/400000 [00:08<00:41, 8063.90it/s] 17%|█▋        | 69830/400000 [00:08<00:40, 8094.23it/s] 18%|█▊        | 70647/400000 [00:08<00:40, 8115.41it/s] 18%|█▊        | 71460/400000 [00:08<00:40, 8119.25it/s] 18%|█▊        | 72278/400000 [00:08<00:40, 8134.69it/s] 18%|█▊        | 73092/400000 [00:09<00:40, 8021.10it/s] 18%|█▊        | 73908/400000 [00:09<00:40, 8060.13it/s] 19%|█▊        | 74722/400000 [00:09<00:40, 8081.64it/s] 19%|█▉        | 75543/400000 [00:09<00:39, 8118.97it/s] 19%|█▉        | 76365/400000 [00:09<00:39, 8147.12it/s] 19%|█▉        | 77180/400000 [00:09<00:39, 8138.01it/s] 19%|█▉        | 77996/400000 [00:09<00:39, 8142.36it/s] 20%|█▉        | 78812/400000 [00:09<00:39, 8146.85it/s] 20%|█▉        | 79631/400000 [00:09<00:39, 8158.12it/s] 20%|██        | 80453/400000 [00:09<00:39, 8175.03it/s] 20%|██        | 81271/400000 [00:10<00:39, 8041.66it/s] 21%|██        | 82086/400000 [00:10<00:39, 8073.83it/s] 21%|██        | 82906/400000 [00:10<00:39, 8110.10it/s] 21%|██        | 83718/400000 [00:10<00:39, 8101.40it/s] 21%|██        | 84536/400000 [00:10<00:38, 8123.18it/s] 21%|██▏       | 85349/400000 [00:10<00:38, 8115.08it/s] 22%|██▏       | 86161/400000 [00:10<00:39, 7977.20it/s] 22%|██▏       | 86970/400000 [00:10<00:39, 8008.81it/s] 22%|██▏       | 87790/400000 [00:10<00:38, 8062.34it/s] 22%|██▏       | 88611/400000 [00:10<00:38, 8105.40it/s] 22%|██▏       | 89430/400000 [00:11<00:38, 8128.14it/s] 23%|██▎       | 90246/400000 [00:11<00:38, 8137.65it/s] 23%|██▎       | 91068/400000 [00:11<00:37, 8160.83it/s] 23%|██▎       | 91885/400000 [00:11<00:37, 8123.94it/s] 23%|██▎       | 92701/400000 [00:11<00:37, 8132.18it/s] 23%|██▎       | 93523/400000 [00:11<00:37, 8156.82it/s] 24%|██▎       | 94339/400000 [00:11<00:37, 8146.57it/s] 24%|██▍       | 95154/400000 [00:11<00:37, 8120.73it/s] 24%|██▍       | 95974/400000 [00:11<00:37, 8141.73it/s] 24%|██▍       | 96793/400000 [00:12<00:37, 8153.61it/s] 24%|██▍       | 97615/400000 [00:12<00:37, 8171.21it/s] 25%|██▍       | 98433/400000 [00:12<00:36, 8157.88it/s] 25%|██▍       | 99249/400000 [00:12<00:36, 8155.00it/s] 25%|██▌       | 100069/400000 [00:12<00:36, 8167.12it/s] 25%|██▌       | 100886/400000 [00:12<00:36, 8164.09it/s] 25%|██▌       | 101707/400000 [00:12<00:36, 8177.75it/s] 26%|██▌       | 102525/400000 [00:12<00:36, 8147.26it/s] 26%|██▌       | 103340/400000 [00:12<00:36, 8115.77it/s] 26%|██▌       | 104163/400000 [00:12<00:36, 8147.20it/s] 26%|██▌       | 104981/400000 [00:13<00:36, 8155.37it/s] 26%|██▋       | 105803/400000 [00:13<00:35, 8173.44it/s] 27%|██▋       | 106621/400000 [00:13<00:35, 8155.73it/s] 27%|██▋       | 107442/400000 [00:13<00:35, 8171.93it/s] 27%|██▋       | 108263/400000 [00:13<00:35, 8181.09it/s] 27%|██▋       | 109082/400000 [00:13<00:35, 8176.43it/s] 27%|██▋       | 109904/400000 [00:13<00:35, 8186.56it/s] 28%|██▊       | 110723/400000 [00:13<00:35, 8166.22it/s] 28%|██▊       | 111540/400000 [00:13<00:35, 8129.64it/s] 28%|██▊       | 112359/400000 [00:13<00:35, 8145.71it/s] 28%|██▊       | 113181/400000 [00:14<00:35, 8165.57it/s] 28%|██▊       | 113998/400000 [00:14<00:35, 8166.81it/s] 29%|██▊       | 114815/400000 [00:14<00:35, 8136.64it/s] 29%|██▉       | 115632/400000 [00:14<00:34, 8145.30it/s] 29%|██▉       | 116447/400000 [00:14<00:34, 8112.37it/s] 29%|██▉       | 117263/400000 [00:14<00:34, 8125.86it/s] 30%|██▉       | 118084/400000 [00:14<00:34, 8149.82it/s] 30%|██▉       | 118900/400000 [00:14<00:34, 8150.74it/s] 30%|██▉       | 119716/400000 [00:14<00:35, 7967.27it/s] 30%|███       | 120514/400000 [00:14<00:35, 7903.16it/s] 30%|███       | 121336/400000 [00:15<00:34, 7994.33it/s] 31%|███       | 122156/400000 [00:15<00:34, 8053.56it/s] 31%|███       | 122977/400000 [00:15<00:34, 8097.12it/s] 31%|███       | 123788/400000 [00:15<00:34, 7961.05it/s] 31%|███       | 124611/400000 [00:15<00:34, 8038.14it/s] 31%|███▏      | 125431/400000 [00:15<00:33, 8084.09it/s] 32%|███▏      | 126241/400000 [00:15<00:33, 8083.59it/s] 32%|███▏      | 127050/400000 [00:15<00:33, 8083.44it/s] 32%|███▏      | 127864/400000 [00:15<00:33, 8099.39it/s] 32%|███▏      | 128687/400000 [00:15<00:33, 8135.40it/s] 32%|███▏      | 129508/400000 [00:16<00:33, 8157.22it/s] 33%|███▎      | 130329/400000 [00:16<00:33, 8170.95it/s] 33%|███▎      | 131147/400000 [00:16<00:32, 8150.11it/s] 33%|███▎      | 131963/400000 [00:16<00:32, 8145.01it/s] 33%|███▎      | 132785/400000 [00:16<00:32, 8165.11it/s] 33%|███▎      | 133607/400000 [00:16<00:32, 8178.34it/s] 34%|███▎      | 134428/400000 [00:16<00:32, 8186.08it/s] 34%|███▍      | 135247/400000 [00:16<00:32, 8172.09it/s] 34%|███▍      | 136065/400000 [00:16<00:32, 8119.25it/s] 34%|███▍      | 136886/400000 [00:16<00:32, 8144.29it/s] 34%|███▍      | 137707/400000 [00:17<00:32, 8163.38it/s] 35%|███▍      | 138524/400000 [00:17<00:32, 8062.56it/s] 35%|███▍      | 139331/400000 [00:17<00:32, 8042.07it/s] 35%|███▌      | 140138/400000 [00:17<00:32, 8048.46it/s] 35%|███▌      | 140958/400000 [00:17<00:32, 8090.66it/s] 35%|███▌      | 141779/400000 [00:17<00:31, 8123.18it/s] 36%|███▌      | 142592/400000 [00:17<00:31, 8110.59it/s] 36%|███▌      | 143409/400000 [00:17<00:31, 8126.83it/s] 36%|███▌      | 144223/400000 [00:17<00:31, 8130.70it/s] 36%|███▋      | 145044/400000 [00:17<00:31, 8153.67it/s] 36%|███▋      | 145865/400000 [00:18<00:31, 8167.73it/s] 37%|███▋      | 146684/400000 [00:18<00:30, 8174.22it/s] 37%|███▋      | 147503/400000 [00:18<00:30, 8177.44it/s] 37%|███▋      | 148321/400000 [00:18<00:30, 8137.40it/s] 37%|███▋      | 149141/400000 [00:18<00:30, 8155.92it/s] 37%|███▋      | 149958/400000 [00:18<00:30, 8160.00it/s] 38%|███▊      | 150780/400000 [00:18<00:30, 8177.37it/s] 38%|███▊      | 151598/400000 [00:18<00:30, 8173.53it/s] 38%|███▊      | 152416/400000 [00:18<00:30, 8124.01it/s] 38%|███▊      | 153238/400000 [00:18<00:30, 8150.30it/s] 39%|███▊      | 154062/400000 [00:19<00:30, 8176.42it/s] 39%|███▊      | 154880/400000 [00:19<00:30, 8166.05it/s] 39%|███▉      | 155701/400000 [00:19<00:29, 8177.79it/s] 39%|███▉      | 156519/400000 [00:19<00:30, 8058.23it/s] 39%|███▉      | 157332/400000 [00:19<00:30, 8077.19it/s] 40%|███▉      | 158153/400000 [00:19<00:29, 8115.88it/s] 40%|███▉      | 158976/400000 [00:19<00:29, 8147.40it/s] 40%|███▉      | 159797/400000 [00:19<00:29, 8165.34it/s] 40%|████      | 160614/400000 [00:19<00:29, 8136.46it/s] 40%|████      | 161430/400000 [00:19<00:29, 8141.79it/s] 41%|████      | 162247/400000 [00:20<00:29, 8147.63it/s] 41%|████      | 163069/400000 [00:20<00:29, 8167.65it/s] 41%|████      | 163891/400000 [00:20<00:28, 8182.99it/s] 41%|████      | 164710/400000 [00:20<00:28, 8150.28it/s] 41%|████▏     | 165526/400000 [00:20<00:28, 8126.35it/s] 42%|████▏     | 166349/400000 [00:20<00:28, 8154.45it/s] 42%|████▏     | 167169/400000 [00:20<00:28, 8166.04it/s] 42%|████▏     | 167989/400000 [00:20<00:28, 8174.74it/s] 42%|████▏     | 168807/400000 [00:20<00:28, 8109.88it/s] 42%|████▏     | 169627/400000 [00:20<00:28, 8136.12it/s] 43%|████▎     | 170448/400000 [00:21<00:28, 8155.68it/s] 43%|████▎     | 171270/400000 [00:21<00:27, 8171.98it/s] 43%|████▎     | 172091/400000 [00:21<00:27, 8182.80it/s] 43%|████▎     | 172910/400000 [00:21<00:27, 8149.21it/s] 43%|████▎     | 173725/400000 [00:21<00:27, 8144.03it/s] 44%|████▎     | 174546/400000 [00:21<00:27, 8163.09it/s] 44%|████▍     | 175363/400000 [00:21<00:27, 8150.36it/s] 44%|████▍     | 176187/400000 [00:21<00:27, 8175.74it/s] 44%|████▍     | 177005/400000 [00:21<00:27, 8111.84it/s] 44%|████▍     | 177823/400000 [00:21<00:27, 8131.69it/s] 45%|████▍     | 178644/400000 [00:22<00:27, 8152.16it/s] 45%|████▍     | 179460/400000 [00:22<00:27, 8059.17it/s] 45%|████▌     | 180267/400000 [00:22<00:27, 8060.05it/s] 45%|████▌     | 181075/400000 [00:22<00:27, 8065.29it/s] 45%|████▌     | 181893/400000 [00:22<00:26, 8099.01it/s] 46%|████▌     | 182710/400000 [00:22<00:26, 8118.46it/s] 46%|████▌     | 183532/400000 [00:22<00:26, 8147.05it/s] 46%|████▌     | 184349/400000 [00:22<00:26, 8152.89it/s] 46%|████▋     | 185165/400000 [00:22<00:26, 8130.88it/s] 46%|████▋     | 185986/400000 [00:22<00:26, 8153.95it/s] 47%|████▋     | 186805/400000 [00:23<00:26, 8163.52it/s] 47%|████▋     | 187624/400000 [00:23<00:25, 8170.76it/s] 47%|████▋     | 188446/400000 [00:23<00:25, 8184.33it/s] 47%|████▋     | 189265/400000 [00:23<00:25, 8165.73it/s] 48%|████▊     | 190082/400000 [00:23<00:25, 8166.96it/s] 48%|████▊     | 190899/400000 [00:23<00:25, 8144.78it/s] 48%|████▊     | 191722/400000 [00:23<00:25, 8168.54it/s] 48%|████▊     | 192539/400000 [00:23<00:25, 8163.80it/s] 48%|████▊     | 193356/400000 [00:23<00:25, 8114.44it/s] 49%|████▊     | 194176/400000 [00:23<00:25, 8137.98it/s] 49%|████▊     | 194990/400000 [00:24<00:25, 8015.53it/s] 49%|████▉     | 195808/400000 [00:24<00:25, 8062.42it/s] 49%|████▉     | 196627/400000 [00:24<00:25, 8098.88it/s] 49%|████▉     | 197438/400000 [00:24<00:25, 8024.57it/s] 50%|████▉     | 198249/400000 [00:24<00:25, 8048.90it/s] 50%|████▉     | 199063/400000 [00:24<00:24, 8075.82it/s] 50%|████▉     | 199882/400000 [00:24<00:24, 8109.52it/s] 50%|█████     | 200703/400000 [00:24<00:24, 8138.86it/s] 50%|█████     | 201518/400000 [00:24<00:24, 8076.38it/s] 51%|█████     | 202332/400000 [00:24<00:24, 8094.23it/s] 51%|█████     | 203153/400000 [00:25<00:24, 8127.73it/s] 51%|█████     | 203966/400000 [00:25<00:24, 8124.44it/s] 51%|█████     | 204787/400000 [00:25<00:23, 8149.11it/s] 51%|█████▏    | 205603/400000 [00:25<00:23, 8137.88it/s] 52%|█████▏    | 206417/400000 [00:25<00:23, 8099.55it/s] 52%|█████▏    | 207236/400000 [00:25<00:23, 8125.66it/s] 52%|█████▏    | 208049/400000 [00:25<00:23, 8102.81it/s] 52%|█████▏    | 208871/400000 [00:25<00:23, 8136.40it/s] 52%|█████▏    | 209685/400000 [00:25<00:23, 8115.72it/s] 53%|█████▎    | 210498/400000 [00:25<00:23, 8118.68it/s] 53%|█████▎    | 211320/400000 [00:26<00:23, 8147.54it/s] 53%|█████▎    | 212139/400000 [00:26<00:23, 8158.71it/s] 53%|█████▎    | 212958/400000 [00:26<00:22, 8167.51it/s] 53%|█████▎    | 213775/400000 [00:26<00:22, 8113.25it/s] 54%|█████▎    | 214591/400000 [00:26<00:22, 8125.71it/s] 54%|█████▍    | 215414/400000 [00:26<00:22, 8155.23it/s] 54%|█████▍    | 216232/400000 [00:26<00:22, 8160.24it/s] 54%|█████▍    | 217051/400000 [00:26<00:22, 8167.97it/s] 54%|█████▍    | 217868/400000 [00:26<00:22, 8140.39it/s] 55%|█████▍    | 218683/400000 [00:26<00:22, 8132.14it/s] 55%|█████▍    | 219504/400000 [00:27<00:22, 8155.25it/s] 55%|█████▌    | 220322/400000 [00:27<00:22, 8159.81it/s] 55%|█████▌    | 221139/400000 [00:27<00:21, 8156.27it/s] 55%|█████▌    | 221959/400000 [00:27<00:21, 8168.29it/s] 56%|█████▌    | 222776/400000 [00:27<00:21, 8157.23it/s] 56%|█████▌    | 223598/400000 [00:27<00:21, 8173.37it/s] 56%|█████▌    | 224416/400000 [00:27<00:21, 8148.77it/s] 56%|█████▋    | 225231/400000 [00:27<00:21, 8144.06it/s] 57%|█████▋    | 226046/400000 [00:27<00:21, 8133.98it/s] 57%|█████▋    | 226860/400000 [00:27<00:21, 8133.27it/s] 57%|█████▋    | 227681/400000 [00:28<00:21, 8153.47it/s] 57%|█████▋    | 228504/400000 [00:28<00:20, 8173.85it/s] 57%|█████▋    | 229322/400000 [00:28<00:20, 8168.49it/s] 58%|█████▊    | 230143/400000 [00:28<00:20, 8178.20it/s] 58%|█████▊    | 230961/400000 [00:28<00:20, 8120.46it/s] 58%|█████▊    | 231778/400000 [00:28<00:20, 8132.57it/s] 58%|█████▊    | 232592/400000 [00:28<00:20, 8130.63it/s] 58%|█████▊    | 233410/400000 [00:28<00:20, 8142.56it/s] 59%|█████▊    | 234225/400000 [00:28<00:20, 8132.65it/s] 59%|█████▉    | 235039/400000 [00:28<00:20, 8128.66it/s] 59%|█████▉    | 235859/400000 [00:29<00:20, 8149.10it/s] 59%|█████▉    | 236674/400000 [00:29<00:20, 8065.59it/s] 59%|█████▉    | 237489/400000 [00:29<00:20, 8089.25it/s] 60%|█████▉    | 238304/400000 [00:29<00:19, 8107.07it/s] 60%|█████▉    | 239115/400000 [00:29<00:19, 8052.39it/s] 60%|█████▉    | 239924/400000 [00:29<00:19, 8061.61it/s] 60%|██████    | 240740/400000 [00:29<00:19, 8090.12it/s] 60%|██████    | 241550/400000 [00:29<00:19, 8015.50it/s] 61%|██████    | 242352/400000 [00:29<00:19, 7904.43it/s] 61%|██████    | 243153/400000 [00:30<00:19, 7933.54it/s] 61%|██████    | 243947/400000 [00:30<00:19, 7842.03it/s] 61%|██████    | 244768/400000 [00:30<00:19, 7947.36it/s] 61%|██████▏   | 245587/400000 [00:30<00:19, 8016.07it/s] 62%|██████▏   | 246402/400000 [00:30<00:19, 8055.59it/s] 62%|██████▏   | 247209/400000 [00:30<00:18, 8054.70it/s] 62%|██████▏   | 248030/400000 [00:30<00:18, 8099.11it/s] 62%|██████▏   | 248851/400000 [00:30<00:18, 8130.24it/s] 62%|██████▏   | 249672/400000 [00:30<00:18, 8152.01it/s] 63%|██████▎   | 250488/400000 [00:30<00:18, 8148.29it/s] 63%|██████▎   | 251303/400000 [00:31<00:18, 7987.23it/s] 63%|██████▎   | 252103/400000 [00:31<00:18, 7945.50it/s] 63%|██████▎   | 252924/400000 [00:31<00:18, 8021.85it/s] 63%|██████▎   | 253746/400000 [00:31<00:18, 8079.72it/s] 64%|██████▎   | 254560/400000 [00:31<00:17, 8095.42it/s] 64%|██████▍   | 255370/400000 [00:31<00:18, 8025.25it/s] 64%|██████▍   | 256173/400000 [00:31<00:17, 7991.71it/s] 64%|██████▍   | 256993/400000 [00:31<00:17, 8052.24it/s] 64%|██████▍   | 257809/400000 [00:31<00:17, 8083.20it/s] 65%|██████▍   | 258618/400000 [00:31<00:17, 8054.73it/s] 65%|██████▍   | 259424/400000 [00:32<00:17, 8028.33it/s] 65%|██████▌   | 260245/400000 [00:32<00:17, 8079.14it/s] 65%|██████▌   | 261062/400000 [00:32<00:17, 8104.54it/s] 65%|██████▌   | 261881/400000 [00:32<00:16, 8127.43it/s] 66%|██████▌   | 262699/400000 [00:32<00:16, 8140.99it/s] 66%|██████▌   | 263514/400000 [00:32<00:16, 8116.42it/s] 66%|██████▌   | 264334/400000 [00:32<00:16, 8139.22it/s] 66%|██████▋   | 265148/400000 [00:32<00:16, 8122.48it/s] 66%|██████▋   | 265966/400000 [00:32<00:16, 8138.87it/s] 67%|██████▋   | 266780/400000 [00:32<00:16, 8106.64it/s] 67%|██████▋   | 267591/400000 [00:33<00:16, 8094.33it/s] 67%|██████▋   | 268408/400000 [00:33<00:16, 8116.15it/s] 67%|██████▋   | 269227/400000 [00:33<00:16, 8135.94it/s] 68%|██████▊   | 270045/400000 [00:33<00:15, 8147.84it/s] 68%|██████▊   | 270864/400000 [00:33<00:15, 8158.03it/s] 68%|██████▊   | 271680/400000 [00:33<00:15, 8140.46it/s] 68%|██████▊   | 272495/400000 [00:33<00:15, 8142.00it/s] 68%|██████▊   | 273312/400000 [00:33<00:15, 8148.80it/s] 69%|██████▊   | 274134/400000 [00:33<00:15, 8167.66it/s] 69%|██████▊   | 274951/400000 [00:33<00:15, 8138.76it/s] 69%|██████▉   | 275765/400000 [00:34<00:15, 8113.58it/s] 69%|██████▉   | 276577/400000 [00:34<00:15, 8112.43it/s] 69%|██████▉   | 277389/400000 [00:34<00:15, 8113.95it/s] 70%|██████▉   | 278201/400000 [00:34<00:15, 7968.77it/s] 70%|██████▉   | 278999/400000 [00:34<00:15, 7969.78it/s] 70%|██████▉   | 279797/400000 [00:34<00:15, 7954.27it/s] 70%|███████   | 280611/400000 [00:34<00:14, 8007.91it/s] 70%|███████   | 281413/400000 [00:34<00:15, 7810.98it/s] 71%|███████   | 282229/400000 [00:34<00:14, 7912.27it/s] 71%|███████   | 283041/400000 [00:34<00:14, 7973.19it/s] 71%|███████   | 283849/400000 [00:35<00:14, 8002.13it/s] 71%|███████   | 284650/400000 [00:35<00:14, 7985.94it/s] 71%|███████▏  | 285465/400000 [00:35<00:14, 8033.80it/s] 72%|███████▏  | 286283/400000 [00:35<00:14, 8074.77it/s] 72%|███████▏  | 287104/400000 [00:35<00:13, 8114.22it/s] 72%|███████▏  | 287916/400000 [00:35<00:13, 8101.51it/s] 72%|███████▏  | 288727/400000 [00:35<00:13, 8095.93it/s] 72%|███████▏  | 289538/400000 [00:35<00:13, 8098.07it/s] 73%|███████▎  | 290358/400000 [00:35<00:13, 8124.07it/s] 73%|███████▎  | 291180/400000 [00:35<00:13, 8149.87it/s] 73%|███████▎  | 291996/400000 [00:36<00:13, 8148.73it/s] 73%|███████▎  | 292811/400000 [00:36<00:13, 8142.25it/s] 73%|███████▎  | 293632/400000 [00:36<00:13, 8161.81it/s] 74%|███████▎  | 294450/400000 [00:36<00:12, 8164.34it/s] 74%|███████▍  | 295269/400000 [00:36<00:12, 8171.51it/s] 74%|███████▍  | 296087/400000 [00:36<00:12, 8169.95it/s] 74%|███████▍  | 296905/400000 [00:36<00:12, 8142.91it/s] 74%|███████▍  | 297721/400000 [00:36<00:12, 8145.87it/s] 75%|███████▍  | 298541/400000 [00:36<00:12, 8159.24it/s] 75%|███████▍  | 299362/400000 [00:36<00:12, 8171.77it/s] 75%|███████▌  | 300182/400000 [00:37<00:12, 8178.13it/s] 75%|███████▌  | 301000/400000 [00:37<00:12, 8148.38it/s] 75%|███████▌  | 301816/400000 [00:37<00:12, 8148.99it/s] 76%|███████▌  | 302637/400000 [00:37<00:11, 8165.02it/s] 76%|███████▌  | 303454/400000 [00:37<00:11, 8159.11it/s] 76%|███████▌  | 304274/400000 [00:37<00:11, 8169.11it/s] 76%|███████▋  | 305091/400000 [00:37<00:11, 8151.40it/s] 76%|███████▋  | 305907/400000 [00:37<00:11, 8145.12it/s] 77%|███████▋  | 306725/400000 [00:37<00:11, 8152.83it/s] 77%|███████▋  | 307545/400000 [00:37<00:11, 8165.38it/s] 77%|███████▋  | 308362/400000 [00:38<00:11, 8161.12it/s] 77%|███████▋  | 309179/400000 [00:38<00:11, 8148.38it/s] 77%|███████▋  | 309998/400000 [00:38<00:11, 8158.88it/s] 78%|███████▊  | 310815/400000 [00:38<00:10, 8159.48it/s] 78%|███████▊  | 311633/400000 [00:38<00:10, 8165.56it/s] 78%|███████▊  | 312450/400000 [00:38<00:10, 8160.86it/s] 78%|███████▊  | 313267/400000 [00:38<00:10, 8128.09it/s] 79%|███████▊  | 314081/400000 [00:38<00:10, 8129.18it/s] 79%|███████▊  | 314894/400000 [00:38<00:10, 8122.91it/s] 79%|███████▉  | 315711/400000 [00:38<00:10, 8134.52it/s] 79%|███████▉  | 316526/400000 [00:39<00:10, 8138.73it/s] 79%|███████▉  | 317340/400000 [00:39<00:10, 8102.41it/s] 80%|███████▉  | 318159/400000 [00:39<00:10, 8125.82it/s] 80%|███████▉  | 318972/400000 [00:39<00:09, 8116.09it/s] 80%|███████▉  | 319793/400000 [00:39<00:09, 8142.39it/s] 80%|████████  | 320608/400000 [00:39<00:09, 8131.84it/s] 80%|████████  | 321422/400000 [00:39<00:09, 8116.11it/s] 81%|████████  | 322237/400000 [00:39<00:09, 8124.30it/s] 81%|████████  | 323054/400000 [00:39<00:09, 8136.89it/s] 81%|████████  | 323868/400000 [00:39<00:09, 8104.64it/s] 81%|████████  | 324679/400000 [00:40<00:09, 8094.55it/s] 81%|████████▏ | 325489/400000 [00:40<00:09, 8071.57it/s] 82%|████████▏ | 326302/400000 [00:40<00:09, 8087.07it/s] 82%|████████▏ | 327118/400000 [00:40<00:08, 8106.10it/s] 82%|████████▏ | 327929/400000 [00:40<00:08, 8087.41it/s] 82%|████████▏ | 328746/400000 [00:40<00:08, 8110.84it/s] 82%|████████▏ | 329558/400000 [00:40<00:08, 8043.79it/s] 83%|████████▎ | 330372/400000 [00:40<00:08, 8072.37it/s] 83%|████████▎ | 331193/400000 [00:40<00:08, 8111.09it/s] 83%|████████▎ | 332005/400000 [00:40<00:08, 8107.72it/s] 83%|████████▎ | 332826/400000 [00:41<00:08, 8137.35it/s] 83%|████████▎ | 333640/400000 [00:41<00:08, 8125.83it/s] 84%|████████▎ | 334463/400000 [00:41<00:08, 8155.70it/s] 84%|████████▍ | 335279/400000 [00:41<00:07, 8149.97it/s] 84%|████████▍ | 336095/400000 [00:41<00:07, 8133.54it/s] 84%|████████▍ | 336914/400000 [00:41<00:07, 8147.69it/s] 84%|████████▍ | 337729/400000 [00:41<00:07, 8125.39it/s] 85%|████████▍ | 338548/400000 [00:41<00:07, 8142.81it/s] 85%|████████▍ | 339367/400000 [00:41<00:07, 8154.46it/s] 85%|████████▌ | 340183/400000 [00:41<00:07, 8128.15it/s] 85%|████████▌ | 341002/400000 [00:42<00:07, 8145.31it/s] 85%|████████▌ | 341817/400000 [00:42<00:07, 8132.65it/s] 86%|████████▌ | 342631/400000 [00:42<00:07, 8129.25it/s] 86%|████████▌ | 343449/400000 [00:42<00:06, 8141.50it/s] 86%|████████▌ | 344269/400000 [00:42<00:06, 8158.20it/s] 86%|████████▋ | 345088/400000 [00:42<00:06, 8167.10it/s] 86%|████████▋ | 345905/400000 [00:42<00:06, 8125.40it/s] 87%|████████▋ | 346721/400000 [00:42<00:06, 8133.83it/s] 87%|████████▋ | 347542/400000 [00:42<00:06, 8155.38it/s] 87%|████████▋ | 348358/400000 [00:42<00:06, 8145.59it/s] 87%|████████▋ | 349177/400000 [00:43<00:06, 8158.66it/s] 87%|████████▋ | 349993/400000 [00:43<00:06, 8109.22it/s] 88%|████████▊ | 350809/400000 [00:43<00:06, 8121.55it/s] 88%|████████▊ | 351625/400000 [00:43<00:05, 8132.99it/s] 88%|████████▊ | 352446/400000 [00:43<00:05, 8153.65it/s] 88%|████████▊ | 353262/400000 [00:43<00:05, 8148.78it/s] 89%|████████▊ | 354077/400000 [00:43<00:05, 8141.59it/s] 89%|████████▊ | 354897/400000 [00:43<00:05, 8157.98it/s] 89%|████████▉ | 355714/400000 [00:43<00:05, 8161.08it/s] 89%|████████▉ | 356536/400000 [00:43<00:05, 8176.99it/s] 89%|████████▉ | 357354/400000 [00:44<00:05, 8136.64it/s] 90%|████████▉ | 358168/400000 [00:44<00:05, 7927.96it/s] 90%|████████▉ | 358984/400000 [00:44<00:05, 7993.87it/s] 90%|████████▉ | 359797/400000 [00:44<00:05, 8032.33it/s] 90%|█████████ | 360618/400000 [00:44<00:04, 8082.75it/s] 90%|█████████ | 361431/400000 [00:44<00:04, 8094.16it/s] 91%|█████████ | 362241/400000 [00:44<00:04, 8070.15it/s] 91%|█████████ | 363053/400000 [00:44<00:04, 8082.72it/s] 91%|█████████ | 363873/400000 [00:44<00:04, 8116.22it/s] 91%|█████████ | 364685/400000 [00:45<00:04, 8076.10it/s] 91%|█████████▏| 365498/400000 [00:45<00:04, 8090.69it/s] 92%|█████████▏| 366308/400000 [00:45<00:04, 8091.57it/s] 92%|█████████▏| 367125/400000 [00:45<00:04, 8113.74it/s] 92%|█████████▏| 367945/400000 [00:45<00:03, 8139.16it/s] 92%|█████████▏| 368761/400000 [00:45<00:03, 8143.96it/s] 92%|█████████▏| 369582/400000 [00:45<00:03, 8160.94it/s] 93%|█████████▎| 370399/400000 [00:45<00:03, 8148.71it/s] 93%|█████████▎| 371217/400000 [00:45<00:03, 8156.10it/s] 93%|█████████▎| 372033/400000 [00:45<00:03, 8099.46it/s] 93%|█████████▎| 372844/400000 [00:46<00:03, 8094.70it/s] 93%|█████████▎| 373666/400000 [00:46<00:03, 8129.17it/s] 94%|█████████▎| 374480/400000 [00:46<00:03, 8123.25it/s] 94%|█████████▍| 375297/400000 [00:46<00:03, 8135.05it/s] 94%|█████████▍| 376111/400000 [00:46<00:02, 8131.89it/s] 94%|█████████▍| 376931/400000 [00:46<00:02, 8150.13it/s] 94%|█████████▍| 377753/400000 [00:46<00:02, 8168.51it/s] 95%|█████████▍| 378570/400000 [00:46<00:02, 8164.88it/s] 95%|█████████▍| 379387/400000 [00:46<00:02, 8147.45it/s] 95%|█████████▌| 380202/400000 [00:46<00:02, 8121.37it/s] 95%|█████████▌| 381022/400000 [00:47<00:02, 8143.87it/s] 95%|█████████▌| 381841/400000 [00:47<00:02, 8156.93it/s] 96%|█████████▌| 382657/400000 [00:47<00:02, 8147.53it/s] 96%|█████████▌| 383472/400000 [00:47<00:02, 8077.37it/s] 96%|█████████▌| 384291/400000 [00:47<00:01, 8110.14it/s] 96%|█████████▋| 385109/400000 [00:47<00:01, 8129.56it/s] 96%|█████████▋| 385929/400000 [00:47<00:01, 8150.04it/s] 97%|█████████▋| 386747/400000 [00:47<00:01, 8157.39it/s] 97%|█████████▋| 387563/400000 [00:47<00:01, 8125.45it/s] 97%|█████████▋| 388382/400000 [00:47<00:01, 8142.03it/s] 97%|█████████▋| 389200/400000 [00:48<00:01, 8152.88it/s] 98%|█████████▊| 390018/400000 [00:48<00:01, 8159.64it/s] 98%|█████████▊| 390837/400000 [00:48<00:01, 8166.49it/s] 98%|█████████▊| 391654/400000 [00:48<00:01, 8116.52it/s] 98%|█████████▊| 392473/400000 [00:48<00:00, 8138.04it/s] 98%|█████████▊| 393287/400000 [00:48<00:00, 8043.73it/s] 99%|█████████▊| 394092/400000 [00:48<00:00, 8030.39it/s] 99%|█████████▊| 394913/400000 [00:48<00:00, 8082.62it/s] 99%|█████████▉| 395722/400000 [00:48<00:00, 8061.52it/s] 99%|█████████▉| 396530/400000 [00:48<00:00, 8066.51it/s] 99%|█████████▉| 397339/400000 [00:49<00:00, 8071.56it/s]100%|█████████▉| 398158/400000 [00:49<00:00, 8106.40it/s]100%|█████████▉| 398977/400000 [00:49<00:00, 8129.39it/s]100%|█████████▉| 399791/400000 [00:49<00:00, 8098.65it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8105.15it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f83870ffbe0>, <torchtext.data.dataset.TabularDataset object at 0x7f83870ffd30>, <torchtext.vocab.Vocab object at 0x7f83870ffc50>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.49 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.49 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.37 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.86 file/s]2020-07-27 18:21:34.590430: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-27 18:21:34.594786: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-27 18:21:34.594972: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56186849ec70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-27 18:21:34.594988: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147519.55it/s] 87%|████████▋ | 8667136/9912422 [00:00<00:05, 210586.99it/s]9920512it [00:00, 44608685.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 377064.70it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156122.44it/s]1654784it [00:00, 11253925.36it/s]                         
0it [00:00, ?it/s]8192it [00:00, 249818.15it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
