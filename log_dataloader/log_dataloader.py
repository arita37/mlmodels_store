
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fbb351c1ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fbb351c1ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fbb9fe31510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fbb9fe31510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fbbbf05eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fbbbf05eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbb4d1d9a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbb4d1d9a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbb4d1d9a60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████▏     | 4104192/9912422 [00:00<00:00, 40971821.65it/s]9920512it [00:00, 32802861.99it/s]                             
0it [00:00, ?it/s]32768it [00:00, 533608.81it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162727.45it/s]1654784it [00:00, 11360171.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200704.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbb4d1644a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbb3508b9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbb4d1d96a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbb4d1d96a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fbb4d1d96a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:59,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:59,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:58,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:58,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:58,  2.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 34.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 72.87 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.82s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.58 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.83s/ url]
0 examples [00:00, ? examples/s]2020-08-01 18:09:18.497146: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-01 18:09:18.512109: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-01 18:09:18.512316: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5605dce11e00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-01 18:09:18.512335: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
13 examples [00:00, 128.88 examples/s]106 examples [00:00, 173.77 examples/s]205 examples [00:00, 230.72 examples/s]308 examples [00:00, 300.52 examples/s]402 examples [00:00, 377.24 examples/s]490 examples [00:00, 454.81 examples/s]583 examples [00:00, 536.93 examples/s]681 examples [00:00, 620.47 examples/s]783 examples [00:00, 702.96 examples/s]877 examples [00:01, 759.47 examples/s]973 examples [00:01, 808.36 examples/s]1071 examples [00:01, 851.81 examples/s]1168 examples [00:01, 884.12 examples/s]1264 examples [00:01, 878.62 examples/s]1357 examples [00:01, 870.80 examples/s]1448 examples [00:01, 880.71 examples/s]1543 examples [00:01, 898.34 examples/s]1635 examples [00:01, 900.65 examples/s]1730 examples [00:01, 912.35 examples/s]1823 examples [00:02, 916.24 examples/s]1920 examples [00:02, 931.59 examples/s]2017 examples [00:02, 941.95 examples/s]2115 examples [00:02, 951.49 examples/s]2216 examples [00:02, 966.83 examples/s]2313 examples [00:02, 943.00 examples/s]2408 examples [00:02, 944.42 examples/s]2505 examples [00:02, 949.76 examples/s]2604 examples [00:02, 959.58 examples/s]2701 examples [00:02, 961.03 examples/s]2798 examples [00:03, 953.90 examples/s]2895 examples [00:03, 958.08 examples/s]2994 examples [00:03, 967.17 examples/s]3095 examples [00:03, 978.67 examples/s]3193 examples [00:03, 968.70 examples/s]3290 examples [00:03, 959.63 examples/s]3387 examples [00:03, 958.71 examples/s]3483 examples [00:03, 957.71 examples/s]3579 examples [00:03, 957.56 examples/s]3675 examples [00:03, 944.88 examples/s]3770 examples [00:04, 943.34 examples/s]3871 examples [00:04, 960.27 examples/s]3968 examples [00:04, 952.94 examples/s]4064 examples [00:04, 933.34 examples/s]4158 examples [00:04, 929.24 examples/s]4252 examples [00:04, 930.10 examples/s]4350 examples [00:04, 941.78 examples/s]4446 examples [00:04, 946.13 examples/s]4541 examples [00:04, 939.63 examples/s]4638 examples [00:04, 947.15 examples/s]4737 examples [00:05, 957.97 examples/s]4836 examples [00:05, 967.09 examples/s]4938 examples [00:05, 980.29 examples/s]5039 examples [00:05, 988.47 examples/s]5140 examples [00:05, 993.78 examples/s]5240 examples [00:05, 988.73 examples/s]5341 examples [00:05, 993.76 examples/s]5441 examples [00:05, 987.69 examples/s]5541 examples [00:05, 989.49 examples/s]5640 examples [00:05, 973.06 examples/s]5739 examples [00:06, 975.93 examples/s]5838 examples [00:06, 979.97 examples/s]5937 examples [00:06, 925.74 examples/s]6031 examples [00:06, 915.00 examples/s]6129 examples [00:06, 932.50 examples/s]6223 examples [00:06, 931.73 examples/s]6317 examples [00:06, 922.95 examples/s]6418 examples [00:06, 946.63 examples/s]6519 examples [00:06, 961.25 examples/s]6616 examples [00:07, 948.50 examples/s]6712 examples [00:07, 941.99 examples/s]6807 examples [00:07, 934.68 examples/s]6908 examples [00:07, 953.53 examples/s]7007 examples [00:07, 963.89 examples/s]7104 examples [00:07, 951.72 examples/s]7200 examples [00:07, 949.76 examples/s]7298 examples [00:07, 957.24 examples/s]7398 examples [00:07, 966.64 examples/s]7495 examples [00:07, 914.91 examples/s]7597 examples [00:08, 942.39 examples/s]7692 examples [00:08, 926.78 examples/s]7786 examples [00:08, 919.91 examples/s]7884 examples [00:08, 935.61 examples/s]7981 examples [00:08, 944.80 examples/s]8076 examples [00:08, 942.32 examples/s]8171 examples [00:08, 921.44 examples/s]8270 examples [00:08, 939.53 examples/s]8366 examples [00:08, 945.25 examples/s]8464 examples [00:08, 951.21 examples/s]8565 examples [00:09, 965.72 examples/s]8662 examples [00:09, 959.71 examples/s]8761 examples [00:09, 966.32 examples/s]8860 examples [00:09, 972.41 examples/s]8958 examples [00:09, 973.29 examples/s]9056 examples [00:09, 973.35 examples/s]9156 examples [00:09, 980.55 examples/s]9258 examples [00:09, 992.06 examples/s]9358 examples [00:09, 992.27 examples/s]9458 examples [00:10, 957.52 examples/s]9555 examples [00:10, 943.33 examples/s]9650 examples [00:10, 941.55 examples/s]9749 examples [00:10, 954.93 examples/s]9848 examples [00:10, 963.29 examples/s]9945 examples [00:10, 948.92 examples/s]10041 examples [00:10, 906.92 examples/s]10138 examples [00:10, 924.71 examples/s]10241 examples [00:10, 951.93 examples/s]10342 examples [00:10, 968.54 examples/s]10444 examples [00:11, 981.72 examples/s]10543 examples [00:11, 983.95 examples/s]10642 examples [00:11, 984.84 examples/s]10744 examples [00:11, 993.54 examples/s]10845 examples [00:11, 997.41 examples/s]10945 examples [00:11, 963.22 examples/s]11043 examples [00:11, 966.23 examples/s]11140 examples [00:11, 944.71 examples/s]11235 examples [00:11, 942.15 examples/s]11330 examples [00:11, 933.63 examples/s]11428 examples [00:12, 945.71 examples/s]11523 examples [00:12, 944.94 examples/s]11618 examples [00:12, 940.35 examples/s]11717 examples [00:12, 954.69 examples/s]11818 examples [00:12, 967.81 examples/s]11915 examples [00:12, 961.73 examples/s]12012 examples [00:12, 948.20 examples/s]12107 examples [00:12, 939.77 examples/s]12204 examples [00:12, 948.06 examples/s]12299 examples [00:12, 941.48 examples/s]12399 examples [00:13, 955.75 examples/s]12497 examples [00:13, 962.32 examples/s]12594 examples [00:13, 945.92 examples/s]12691 examples [00:13, 951.91 examples/s]12790 examples [00:13, 961.38 examples/s]12887 examples [00:13, 958.17 examples/s]12983 examples [00:13, 957.60 examples/s]13079 examples [00:13, 942.62 examples/s]13176 examples [00:13, 949.83 examples/s]13273 examples [00:14, 955.58 examples/s]13371 examples [00:14, 961.47 examples/s]13468 examples [00:14, 962.69 examples/s]13565 examples [00:14, 925.58 examples/s]13666 examples [00:14, 947.53 examples/s]13762 examples [00:14, 949.66 examples/s]13858 examples [00:14, 949.88 examples/s]13954 examples [00:14, 947.31 examples/s]14050 examples [00:14, 949.60 examples/s]14151 examples [00:14, 964.75 examples/s]14251 examples [00:15, 973.37 examples/s]14349 examples [00:15, 970.31 examples/s]14447 examples [00:15, 939.25 examples/s]14542 examples [00:15, 920.21 examples/s]14635 examples [00:15, 915.24 examples/s]14728 examples [00:15, 917.04 examples/s]14820 examples [00:15, 913.74 examples/s]14915 examples [00:15, 922.67 examples/s]15008 examples [00:15, 923.57 examples/s]15104 examples [00:15, 933.81 examples/s]15200 examples [00:16, 938.65 examples/s]15296 examples [00:16, 943.49 examples/s]15391 examples [00:16, 940.99 examples/s]15490 examples [00:16, 953.58 examples/s]15587 examples [00:16, 955.93 examples/s]15689 examples [00:16, 971.16 examples/s]15787 examples [00:16, 953.00 examples/s]15884 examples [00:16, 954.85 examples/s]15980 examples [00:16, 933.38 examples/s]16074 examples [00:16, 933.53 examples/s]16168 examples [00:17, 920.28 examples/s]16261 examples [00:17, 913.77 examples/s]16353 examples [00:17, 914.76 examples/s]16445 examples [00:17, 912.54 examples/s]16537 examples [00:17, 912.57 examples/s]16632 examples [00:17, 922.71 examples/s]16726 examples [00:17, 926.64 examples/s]16820 examples [00:17, 928.67 examples/s]16916 examples [00:17, 937.71 examples/s]17010 examples [00:18, 927.74 examples/s]17108 examples [00:18, 942.77 examples/s]17206 examples [00:18, 953.53 examples/s]17302 examples [00:18, 952.31 examples/s]17398 examples [00:18, 942.97 examples/s]17493 examples [00:18, 940.65 examples/s]17590 examples [00:18, 947.71 examples/s]17686 examples [00:18, 949.51 examples/s]17781 examples [00:18, 941.04 examples/s]17876 examples [00:18, 932.77 examples/s]17970 examples [00:19, 919.78 examples/s]18063 examples [00:19, 899.97 examples/s]18160 examples [00:19, 919.58 examples/s]18257 examples [00:19, 933.62 examples/s]18351 examples [00:19, 932.43 examples/s]18445 examples [00:19, 924.20 examples/s]18539 examples [00:19, 927.44 examples/s]18636 examples [00:19, 938.83 examples/s]18730 examples [00:19, 922.29 examples/s]18826 examples [00:19, 931.07 examples/s]18920 examples [00:20, 924.13 examples/s]19016 examples [00:20, 933.24 examples/s]19112 examples [00:20, 938.83 examples/s]19206 examples [00:20, 938.98 examples/s]19300 examples [00:20, 925.98 examples/s]19394 examples [00:20, 929.32 examples/s]19489 examples [00:20, 935.41 examples/s]19585 examples [00:20, 941.36 examples/s]19680 examples [00:20, 920.66 examples/s]19782 examples [00:20, 946.45 examples/s]19884 examples [00:21, 966.88 examples/s]19986 examples [00:21, 981.04 examples/s]20085 examples [00:21, 924.02 examples/s]20179 examples [00:21, 916.23 examples/s]20272 examples [00:21, 914.28 examples/s]20364 examples [00:21, 912.25 examples/s]20456 examples [00:21, 905.58 examples/s]20550 examples [00:21, 915.27 examples/s]20643 examples [00:21, 918.39 examples/s]20735 examples [00:22, 902.59 examples/s]20826 examples [00:22, 904.54 examples/s]20920 examples [00:22, 913.70 examples/s]21012 examples [00:22, 906.10 examples/s]21103 examples [00:22, 902.02 examples/s]21196 examples [00:22, 907.96 examples/s]21291 examples [00:22, 918.27 examples/s]21385 examples [00:22, 923.12 examples/s]21478 examples [00:22, 923.45 examples/s]21573 examples [00:22, 929.33 examples/s]21666 examples [00:23, 908.55 examples/s]21763 examples [00:23, 923.65 examples/s]21857 examples [00:23, 925.92 examples/s]21951 examples [00:23, 928.90 examples/s]22045 examples [00:23, 931.83 examples/s]22145 examples [00:23, 948.73 examples/s]22242 examples [00:23, 953.50 examples/s]22338 examples [00:23, 954.53 examples/s]22434 examples [00:23, 948.53 examples/s]22529 examples [00:23, 930.57 examples/s]22625 examples [00:24, 938.50 examples/s]22719 examples [00:24, 905.23 examples/s]22816 examples [00:24, 921.19 examples/s]22909 examples [00:24, 923.79 examples/s]23002 examples [00:24, 919.81 examples/s]23098 examples [00:24, 929.37 examples/s]23192 examples [00:24, 929.81 examples/s]23289 examples [00:24, 939.77 examples/s]23384 examples [00:24, 939.46 examples/s]23479 examples [00:24, 927.55 examples/s]23573 examples [00:25, 928.50 examples/s]23666 examples [00:25, 919.79 examples/s]23762 examples [00:25, 930.86 examples/s]23856 examples [00:25, 931.90 examples/s]23950 examples [00:25, 925.81 examples/s]24043 examples [00:25, 893.02 examples/s]24133 examples [00:25, 841.90 examples/s]24218 examples [00:25, 820.24 examples/s]24307 examples [00:25, 837.74 examples/s]24398 examples [00:26, 856.62 examples/s]24496 examples [00:26, 889.82 examples/s]24594 examples [00:26, 913.19 examples/s]24687 examples [00:26, 917.49 examples/s]24780 examples [00:26, 915.28 examples/s]24872 examples [00:26, 895.59 examples/s]24963 examples [00:26, 898.09 examples/s]25054 examples [00:26, 886.73 examples/s]25147 examples [00:26, 899.25 examples/s]25245 examples [00:26, 921.38 examples/s]25338 examples [00:27, 912.46 examples/s]25430 examples [00:27, 896.60 examples/s]25527 examples [00:27, 915.25 examples/s]25623 examples [00:27, 926.52 examples/s]25717 examples [00:27, 929.31 examples/s]25811 examples [00:27, 916.60 examples/s]25904 examples [00:27, 917.66 examples/s]25996 examples [00:27, 883.68 examples/s]26085 examples [00:27, 875.31 examples/s]26173 examples [00:27, 863.83 examples/s]26264 examples [00:28, 874.20 examples/s]26355 examples [00:28, 883.07 examples/s]26446 examples [00:28, 888.54 examples/s]26541 examples [00:28, 903.72 examples/s]26633 examples [00:28, 907.17 examples/s]26724 examples [00:28, 898.94 examples/s]26819 examples [00:28, 912.74 examples/s]26911 examples [00:28, 908.65 examples/s]27004 examples [00:28, 913.15 examples/s]27100 examples [00:28, 924.52 examples/s]27194 examples [00:29, 929.10 examples/s]27295 examples [00:29, 949.97 examples/s]27391 examples [00:29, 950.75 examples/s]27487 examples [00:29, 929.45 examples/s]27581 examples [00:29, 917.96 examples/s]27673 examples [00:29, 904.05 examples/s]27769 examples [00:29, 919.92 examples/s]27864 examples [00:29, 927.67 examples/s]27967 examples [00:29, 955.61 examples/s]28069 examples [00:30, 972.10 examples/s]28167 examples [00:30, 956.76 examples/s]28270 examples [00:30, 974.58 examples/s]28368 examples [00:30, 908.82 examples/s]28462 examples [00:30, 915.81 examples/s]28559 examples [00:30, 928.08 examples/s]28656 examples [00:30, 938.82 examples/s]28751 examples [00:30, 915.87 examples/s]28847 examples [00:30, 926.29 examples/s]28950 examples [00:30, 954.21 examples/s]29048 examples [00:31, 959.59 examples/s]29150 examples [00:31, 973.83 examples/s]29252 examples [00:31, 987.07 examples/s]29356 examples [00:31, 1001.42 examples/s]29461 examples [00:31, 1013.13 examples/s]29563 examples [00:31, 992.41 examples/s] 29664 examples [00:31, 996.36 examples/s]29769 examples [00:31, 1008.92 examples/s]29871 examples [00:31, 984.36 examples/s] 29970 examples [00:31, 971.25 examples/s]30068 examples [00:32, 910.85 examples/s]30161 examples [00:32, 914.54 examples/s]30257 examples [00:32, 926.69 examples/s]30354 examples [00:32, 936.83 examples/s]30451 examples [00:32, 944.90 examples/s]30546 examples [00:32, 939.41 examples/s]30643 examples [00:32, 946.80 examples/s]30738 examples [00:32, 944.97 examples/s]30833 examples [00:32, 940.47 examples/s]30928 examples [00:33, 936.63 examples/s]31022 examples [00:33, 928.12 examples/s]31115 examples [00:33, 923.66 examples/s]31208 examples [00:33, 923.33 examples/s]31307 examples [00:33, 940.49 examples/s]31402 examples [00:33, 930.23 examples/s]31496 examples [00:33, 916.18 examples/s]31590 examples [00:33, 920.61 examples/s]31683 examples [00:33, 895.40 examples/s]31773 examples [00:33, 889.32 examples/s]31863 examples [00:34, 890.71 examples/s]31953 examples [00:34, 889.18 examples/s]32043 examples [00:34, 892.21 examples/s]32133 examples [00:34, 892.65 examples/s]32229 examples [00:34, 910.50 examples/s]32322 examples [00:34, 914.25 examples/s]32415 examples [00:34, 915.69 examples/s]32513 examples [00:34, 931.64 examples/s]32607 examples [00:34, 924.70 examples/s]32701 examples [00:34, 928.62 examples/s]32799 examples [00:35, 940.77 examples/s]32894 examples [00:35, 924.11 examples/s]32987 examples [00:35, 917.93 examples/s]33079 examples [00:35, 914.03 examples/s]33171 examples [00:35, 909.87 examples/s]33264 examples [00:35, 914.50 examples/s]33356 examples [00:35, 913.05 examples/s]33450 examples [00:35, 917.95 examples/s]33542 examples [00:35, 912.84 examples/s]33638 examples [00:35, 925.45 examples/s]33733 examples [00:36, 932.20 examples/s]33827 examples [00:36, 918.93 examples/s]33919 examples [00:36, 904.26 examples/s]34010 examples [00:36, 895.14 examples/s]34105 examples [00:36, 910.26 examples/s]34197 examples [00:36, 903.91 examples/s]34291 examples [00:36, 913.93 examples/s]34389 examples [00:36, 932.13 examples/s]34489 examples [00:36, 950.56 examples/s]34589 examples [00:36, 963.38 examples/s]34686 examples [00:37, 933.42 examples/s]34785 examples [00:37, 949.09 examples/s]34882 examples [00:37, 954.05 examples/s]34981 examples [00:37, 962.44 examples/s]35078 examples [00:37, 960.54 examples/s]35175 examples [00:37, 959.16 examples/s]35271 examples [00:37, 949.26 examples/s]35370 examples [00:37, 960.61 examples/s]35469 examples [00:37, 969.14 examples/s]35570 examples [00:38, 978.72 examples/s]35668 examples [00:38, 975.19 examples/s]35766 examples [00:38, 942.81 examples/s]35861 examples [00:38, 939.90 examples/s]35956 examples [00:38, 901.49 examples/s]36054 examples [00:38, 923.34 examples/s]36151 examples [00:38, 935.26 examples/s]36246 examples [00:38, 938.89 examples/s]36341 examples [00:38, 937.04 examples/s]36435 examples [00:38, 927.84 examples/s]36529 examples [00:39, 930.78 examples/s]36627 examples [00:39, 943.63 examples/s]36723 examples [00:39, 948.12 examples/s]36818 examples [00:39, 932.98 examples/s]36912 examples [00:39, 924.31 examples/s]37006 examples [00:39, 926.71 examples/s]37103 examples [00:39, 939.04 examples/s]37199 examples [00:39, 942.32 examples/s]37295 examples [00:39, 945.72 examples/s]37390 examples [00:39, 939.34 examples/s]37487 examples [00:40, 947.86 examples/s]37582 examples [00:40, 922.30 examples/s]37675 examples [00:40, 919.44 examples/s]37772 examples [00:40, 931.74 examples/s]37866 examples [00:40, 928.78 examples/s]37959 examples [00:40, 924.15 examples/s]38054 examples [00:40, 929.30 examples/s]38147 examples [00:40, 914.94 examples/s]38242 examples [00:40, 923.97 examples/s]38335 examples [00:40, 924.71 examples/s]38428 examples [00:41, 914.13 examples/s]38520 examples [00:41, 915.18 examples/s]38612 examples [00:41, 911.95 examples/s]38706 examples [00:41, 917.86 examples/s]38798 examples [00:41, 907.19 examples/s]38889 examples [00:41, 879.65 examples/s]38982 examples [00:41, 891.58 examples/s]39072 examples [00:41, 892.41 examples/s]39166 examples [00:41, 904.54 examples/s]39259 examples [00:42, 910.91 examples/s]39351 examples [00:42, 900.17 examples/s]39442 examples [00:42, 873.20 examples/s]39531 examples [00:42, 876.08 examples/s]39619 examples [00:42, 872.48 examples/s]39710 examples [00:42, 883.17 examples/s]39799 examples [00:42, 884.60 examples/s]39890 examples [00:42, 890.25 examples/s]39980 examples [00:42, 880.89 examples/s]40069 examples [00:42, 821.88 examples/s]40159 examples [00:43, 841.85 examples/s]40248 examples [00:43, 855.19 examples/s]40337 examples [00:43, 862.97 examples/s]40424 examples [00:43, 863.73 examples/s]40511 examples [00:43, 856.41 examples/s]40605 examples [00:43, 877.24 examples/s]40696 examples [00:43, 884.57 examples/s]40785 examples [00:43, 861.08 examples/s]40877 examples [00:43, 876.37 examples/s]40976 examples [00:43, 906.76 examples/s]41074 examples [00:44, 926.56 examples/s]41172 examples [00:44, 941.36 examples/s]41267 examples [00:44, 940.23 examples/s]41363 examples [00:44, 944.10 examples/s]41459 examples [00:44, 948.20 examples/s]41560 examples [00:44, 965.73 examples/s]41660 examples [00:44, 975.61 examples/s]41758 examples [00:44, 964.31 examples/s]41855 examples [00:44, 965.82 examples/s]41952 examples [00:44, 967.05 examples/s]42054 examples [00:45, 979.55 examples/s]42153 examples [00:45, 981.58 examples/s]42252 examples [00:45, 979.86 examples/s]42351 examples [00:45, 975.53 examples/s]42449 examples [00:45, 971.59 examples/s]42554 examples [00:45, 992.77 examples/s]42661 examples [00:45, 1011.99 examples/s]42763 examples [00:45, 1002.54 examples/s]42864 examples [00:45, 974.53 examples/s] 42962 examples [00:46, 973.42 examples/s]43060 examples [00:46, 967.07 examples/s]43157 examples [00:46, 958.41 examples/s]43254 examples [00:46, 958.87 examples/s]43350 examples [00:46, 947.11 examples/s]43445 examples [00:46, 947.58 examples/s]43540 examples [00:46, 947.53 examples/s]43640 examples [00:46, 960.58 examples/s]43737 examples [00:46, 952.71 examples/s]43833 examples [00:46, 948.78 examples/s]43931 examples [00:47, 955.48 examples/s]44027 examples [00:47, 930.24 examples/s]44122 examples [00:47, 934.07 examples/s]44221 examples [00:47, 949.85 examples/s]44319 examples [00:47, 957.22 examples/s]44420 examples [00:47, 970.32 examples/s]44518 examples [00:47, 966.02 examples/s]44616 examples [00:47, 967.97 examples/s]44713 examples [00:47, 951.49 examples/s]44809 examples [00:47, 935.98 examples/s]44905 examples [00:48, 941.28 examples/s]45001 examples [00:48, 943.90 examples/s]45096 examples [00:48, 940.26 examples/s]45197 examples [00:48, 959.41 examples/s]45294 examples [00:48, 949.40 examples/s]45390 examples [00:48, 949.44 examples/s]45489 examples [00:48, 961.21 examples/s]45591 examples [00:48, 975.80 examples/s]45693 examples [00:48, 986.55 examples/s]45792 examples [00:48, 970.45 examples/s]45890 examples [00:49, 970.24 examples/s]45989 examples [00:49, 974.44 examples/s]46087 examples [00:49, 964.43 examples/s]46184 examples [00:49, 943.83 examples/s]46279 examples [00:49, 916.45 examples/s]46373 examples [00:49, 922.61 examples/s]46471 examples [00:49, 937.96 examples/s]46566 examples [00:49, 936.59 examples/s]46660 examples [00:49, 934.89 examples/s]46754 examples [00:50, 935.87 examples/s]46849 examples [00:50, 939.57 examples/s]46949 examples [00:50, 956.30 examples/s]47045 examples [00:50, 942.87 examples/s]47140 examples [00:50, 930.29 examples/s]47234 examples [00:50, 931.38 examples/s]47332 examples [00:50, 942.11 examples/s]47428 examples [00:50, 944.92 examples/s]47523 examples [00:50, 932.88 examples/s]47619 examples [00:50, 939.73 examples/s]47714 examples [00:51, 933.37 examples/s]47808 examples [00:51, 934.07 examples/s]47904 examples [00:51, 939.47 examples/s]47998 examples [00:51, 935.22 examples/s]48092 examples [00:51, 929.29 examples/s]48185 examples [00:51, 927.11 examples/s]48278 examples [00:51, 927.89 examples/s]48372 examples [00:51, 929.08 examples/s]48465 examples [00:51, 921.07 examples/s]48559 examples [00:51, 925.33 examples/s]48652 examples [00:52, 923.56 examples/s]48746 examples [00:52, 925.20 examples/s]48839 examples [00:52, 916.06 examples/s]48934 examples [00:52, 923.44 examples/s]49030 examples [00:52, 932.45 examples/s]49125 examples [00:52, 935.02 examples/s]49223 examples [00:52, 946.82 examples/s]49318 examples [00:52, 942.94 examples/s]49414 examples [00:52, 945.43 examples/s]49509 examples [00:52, 942.88 examples/s]49604 examples [00:53, 939.82 examples/s]49702 examples [00:53, 948.77 examples/s]49797 examples [00:53, 944.11 examples/s]49893 examples [00:53, 946.57 examples/s]49990 examples [00:53, 953.40 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6709/50000 [00:00<00:00, 67089.42 examples/s] 38%|███▊      | 18854/50000 [00:00<00:00, 77494.98 examples/s] 63%|██████▎   | 31266/50000 [00:00<00:00, 87335.93 examples/s] 87%|████████▋ | 43584/50000 [00:00<00:00, 95687.83 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 752.28 examples/s]177 examples [00:00, 814.22 examples/s]271 examples [00:00, 846.12 examples/s]364 examples [00:00, 867.02 examples/s]462 examples [00:00, 897.20 examples/s]555 examples [00:00, 905.45 examples/s]653 examples [00:00, 925.75 examples/s]751 examples [00:00, 940.84 examples/s]843 examples [00:00, 934.29 examples/s]938 examples [00:01, 938.18 examples/s]1036 examples [00:01, 949.21 examples/s]1140 examples [00:01, 974.00 examples/s]1240 examples [00:01, 979.47 examples/s]1338 examples [00:01, 979.23 examples/s]1436 examples [00:01, 978.16 examples/s]1534 examples [00:01, 969.22 examples/s]1635 examples [00:01, 978.52 examples/s]1735 examples [00:01, 983.19 examples/s]1836 examples [00:01, 990.26 examples/s]1936 examples [00:02, 991.09 examples/s]2036 examples [00:02, 986.20 examples/s]2136 examples [00:02, 987.51 examples/s]2238 examples [00:02, 994.99 examples/s]2338 examples [00:02, 993.47 examples/s]2438 examples [00:02, 978.02 examples/s]2536 examples [00:02, 945.84 examples/s]2633 examples [00:02, 952.02 examples/s]2729 examples [00:02, 945.33 examples/s]2824 examples [00:02, 946.25 examples/s]2919 examples [00:03, 947.37 examples/s]3014 examples [00:03, 937.88 examples/s]3108 examples [00:03, 893.73 examples/s]3208 examples [00:03, 921.13 examples/s]3301 examples [00:03, 913.68 examples/s]3398 examples [00:03, 928.69 examples/s]3492 examples [00:03, 895.30 examples/s]3592 examples [00:03, 923.61 examples/s]3685 examples [00:03, 919.82 examples/s]3778 examples [00:03, 917.23 examples/s]3871 examples [00:04, 897.44 examples/s]3962 examples [00:04, 882.77 examples/s]4057 examples [00:04, 900.84 examples/s]4152 examples [00:04, 914.49 examples/s]4248 examples [00:04, 926.54 examples/s]4342 examples [00:04, 928.34 examples/s]4435 examples [00:04, 918.64 examples/s]4532 examples [00:04, 932.23 examples/s]4626 examples [00:04, 929.03 examples/s]4720 examples [00:05, 930.44 examples/s]4814 examples [00:05, 932.12 examples/s]4908 examples [00:05, 919.87 examples/s]5002 examples [00:05, 923.52 examples/s]5095 examples [00:05, 922.14 examples/s]5193 examples [00:05, 938.27 examples/s]5293 examples [00:05, 953.54 examples/s]5389 examples [00:05, 953.66 examples/s]5492 examples [00:05, 974.97 examples/s]5590 examples [00:05, 970.92 examples/s]5692 examples [00:06, 982.83 examples/s]5791 examples [00:06, 983.16 examples/s]5890 examples [00:06, 978.47 examples/s]5991 examples [00:06, 987.63 examples/s]6090 examples [00:06, 956.07 examples/s]6186 examples [00:06, 956.70 examples/s]6282 examples [00:06, 957.66 examples/s]6378 examples [00:06, 944.07 examples/s]6474 examples [00:06, 946.68 examples/s]6571 examples [00:06, 952.16 examples/s]6673 examples [00:07, 970.02 examples/s]6774 examples [00:07, 980.18 examples/s]6873 examples [00:07, 973.30 examples/s]6973 examples [00:07, 981.11 examples/s]7074 examples [00:07, 987.63 examples/s]7175 examples [00:07, 991.82 examples/s]7275 examples [00:07, 977.69 examples/s]7373 examples [00:07, 958.96 examples/s]7471 examples [00:07, 964.21 examples/s]7569 examples [00:07, 966.46 examples/s]7670 examples [00:08, 977.53 examples/s]7769 examples [00:08, 980.35 examples/s]7868 examples [00:08, 964.13 examples/s]7970 examples [00:08, 980.07 examples/s]8069 examples [00:08, 965.29 examples/s]8167 examples [00:08, 968.00 examples/s]8264 examples [00:08, 954.12 examples/s]8360 examples [00:08, 935.33 examples/s]8455 examples [00:08, 937.50 examples/s]8552 examples [00:08, 946.65 examples/s]8649 examples [00:09, 952.35 examples/s]8749 examples [00:09, 964.22 examples/s]8846 examples [00:09, 956.28 examples/s]8942 examples [00:09, 954.73 examples/s]9043 examples [00:09, 968.70 examples/s]9140 examples [00:09, 964.50 examples/s]9237 examples [00:09, 954.39 examples/s]9333 examples [00:09, 942.43 examples/s]9428 examples [00:09, 940.57 examples/s]9523 examples [00:10, 918.90 examples/s]9618 examples [00:10, 927.53 examples/s]9713 examples [00:10, 933.95 examples/s]9807 examples [00:10, 915.79 examples/s]9899 examples [00:10, 914.70 examples/s]9997 examples [00:10, 932.76 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP41B31/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP41B31/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbb4d1d9a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbb4d1d9a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbb4d1d9a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbb98b460b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbae02434a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbb4d1d9a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbb4d1d9a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbb4d1d9a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbb3508b5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbb3508b0f0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.80 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.80 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.80 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.85 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.35 file/s]2020-08-01 18:10:31.144523: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-01 18:10:31.149071: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-01 18:10:31.149260: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56181cdf7d30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-01 18:10:31.149279: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 152296.38it/s] 75%|███████▍  | 7413760/9912422 [00:00<00:11, 217374.46it/s]9920512it [00:00, 43479750.72it/s]                           
0it [00:00, ?it/s]32768it [00:00, 570305.05it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 131319.22it/s]1654784it [00:00, 9811184.17it/s]                          
0it [00:00, ?it/s]8192it [00:00, 175727.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
