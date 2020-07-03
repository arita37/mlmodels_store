
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f59cdb5b0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f59cdb5b0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5a38ea71e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5a38ea71e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5a571eee18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5a571eee18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f59e61d6620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f59e61d6620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f59e61d6620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3522560/9912422 [00:00<00:00, 35221518.83it/s]9920512it [00:00, 34334561.58it/s]                             
0it [00:00, ?it/s]32768it [00:00, 916455.20it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161227.04it/s]1654784it [00:00, 11451772.13it/s]                         
0it [00:00, ?it/s]8192it [00:00, 159977.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59e34e2978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59e3721be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f59e61d6268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f59e61d6268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f59e61d6268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:04,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:03,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:02,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:01,  2.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:43,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:42,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:41,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:41,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:41,  3.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:29,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:28,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:27,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:27,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:27,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:27,  4.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:19,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:18,  6.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:11,  9.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 13.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 37.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 74.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 74.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.36 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.74s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 83.36 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.74s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.74s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 43.36 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.74s/ url]
0 examples [00:00, ? examples/s]2020-07-03 18:07:26.767311: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-03 18:07:26.771175: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-07-03 18:07:26.771276: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad133ba920 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-03 18:07:26.771288: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
108 examples [00:00, 1071.95 examples/s]253 examples [00:00, 1160.92 examples/s]396 examples [00:00, 1229.16 examples/s]517 examples [00:00, 1223.02 examples/s]660 examples [00:00, 1277.16 examples/s]800 examples [00:00, 1311.67 examples/s]942 examples [00:00, 1341.50 examples/s]1086 examples [00:00, 1368.73 examples/s]1227 examples [00:00, 1379.10 examples/s]1370 examples [00:01, 1391.36 examples/s]1515 examples [00:01, 1408.30 examples/s]1660 examples [00:01, 1420.21 examples/s]1803 examples [00:01, 1421.89 examples/s]1945 examples [00:01, 1384.86 examples/s]2091 examples [00:01, 1404.98 examples/s]2234 examples [00:01, 1410.74 examples/s]2375 examples [00:01, 1408.70 examples/s]2516 examples [00:01, 1375.00 examples/s]2654 examples [00:01, 1375.02 examples/s]2799 examples [00:02, 1394.61 examples/s]2942 examples [00:02, 1402.97 examples/s]3083 examples [00:02, 1395.30 examples/s]3223 examples [00:02, 1372.13 examples/s]3362 examples [00:02, 1375.41 examples/s]3500 examples [00:02, 1374.81 examples/s]3647 examples [00:02, 1401.50 examples/s]3794 examples [00:02, 1418.82 examples/s]3937 examples [00:02, 1421.24 examples/s]4080 examples [00:02, 1410.00 examples/s]4225 examples [00:03, 1418.51 examples/s]4367 examples [00:03, 1344.31 examples/s]4510 examples [00:03, 1366.48 examples/s]4648 examples [00:03, 1364.06 examples/s]4785 examples [00:03, 1363.94 examples/s]4930 examples [00:03, 1387.23 examples/s]5074 examples [00:03, 1401.56 examples/s]5219 examples [00:03, 1415.21 examples/s]5361 examples [00:03, 1397.33 examples/s]5501 examples [00:03, 1396.31 examples/s]5643 examples [00:04, 1400.31 examples/s]5784 examples [00:04, 1369.94 examples/s]5926 examples [00:04, 1382.99 examples/s]6065 examples [00:04, 1353.57 examples/s]6204 examples [00:04, 1363.02 examples/s]6346 examples [00:04, 1377.56 examples/s]6484 examples [00:04, 1358.93 examples/s]6621 examples [00:04, 1357.13 examples/s]6757 examples [00:04, 1336.82 examples/s]6894 examples [00:04, 1346.59 examples/s]7041 examples [00:05, 1378.94 examples/s]7186 examples [00:05, 1398.88 examples/s]7327 examples [00:05, 1401.55 examples/s]7468 examples [00:05, 1389.67 examples/s]7608 examples [00:05, 1375.03 examples/s]7746 examples [00:05, 1333.97 examples/s]7890 examples [00:05, 1362.86 examples/s]8035 examples [00:05, 1385.35 examples/s]8176 examples [00:05, 1392.53 examples/s]8317 examples [00:06, 1396.78 examples/s]8462 examples [00:06, 1410.51 examples/s]8606 examples [00:06, 1418.07 examples/s]8749 examples [00:06, 1420.81 examples/s]8892 examples [00:06, 1366.71 examples/s]9030 examples [00:06, 1366.27 examples/s]9172 examples [00:06, 1375.55 examples/s]9310 examples [00:06, 1354.96 examples/s]9452 examples [00:06, 1372.64 examples/s]9590 examples [00:06, 1370.55 examples/s]9730 examples [00:07, 1378.04 examples/s]9870 examples [00:07, 1381.85 examples/s]10009 examples [00:07, 1320.65 examples/s]10149 examples [00:07, 1341.69 examples/s]10289 examples [00:07, 1356.53 examples/s]10426 examples [00:07, 1358.39 examples/s]10571 examples [00:07, 1383.15 examples/s]10714 examples [00:07, 1394.73 examples/s]10854 examples [00:07, 1375.14 examples/s]10992 examples [00:07, 1373.80 examples/s]11135 examples [00:08, 1389.27 examples/s]11281 examples [00:08, 1408.09 examples/s]11422 examples [00:08, 1391.33 examples/s]11562 examples [00:08, 1371.67 examples/s]11700 examples [00:08, 1365.55 examples/s]11839 examples [00:08, 1371.24 examples/s]11982 examples [00:08, 1387.61 examples/s]12125 examples [00:08, 1399.92 examples/s]12269 examples [00:08, 1411.44 examples/s]12411 examples [00:08, 1412.31 examples/s]12553 examples [00:09, 1411.28 examples/s]12699 examples [00:09, 1424.42 examples/s]12842 examples [00:09, 1411.56 examples/s]12984 examples [00:09, 1393.98 examples/s]13124 examples [00:09, 1374.16 examples/s]13263 examples [00:09, 1375.86 examples/s]13409 examples [00:09, 1399.66 examples/s]13551 examples [00:09, 1405.06 examples/s]13696 examples [00:09, 1416.88 examples/s]13838 examples [00:10, 1389.97 examples/s]13982 examples [00:10, 1403.31 examples/s]14126 examples [00:10, 1413.27 examples/s]14269 examples [00:10, 1417.87 examples/s]14411 examples [00:10, 1406.12 examples/s]14552 examples [00:10, 1366.14 examples/s]14699 examples [00:10, 1393.80 examples/s]14846 examples [00:10, 1414.29 examples/s]14990 examples [00:10, 1420.31 examples/s]15133 examples [00:10, 1376.70 examples/s]15272 examples [00:11, 1361.82 examples/s]15413 examples [00:11, 1375.17 examples/s]15559 examples [00:11, 1397.87 examples/s]15700 examples [00:11, 1397.90 examples/s]15844 examples [00:11, 1409.62 examples/s]15986 examples [00:11, 1391.43 examples/s]16132 examples [00:11, 1410.97 examples/s]16274 examples [00:11, 1397.46 examples/s]16414 examples [00:11, 1384.69 examples/s]16558 examples [00:11, 1400.51 examples/s]16699 examples [00:12, 1384.44 examples/s]16844 examples [00:12, 1403.34 examples/s]16991 examples [00:12, 1420.61 examples/s]17135 examples [00:12, 1424.89 examples/s]17279 examples [00:12, 1428.86 examples/s]17422 examples [00:12, 1400.01 examples/s]17563 examples [00:12, 1394.28 examples/s]17709 examples [00:12, 1410.76 examples/s]17854 examples [00:12, 1420.10 examples/s]18001 examples [00:12, 1433.41 examples/s]18145 examples [00:13, 1412.40 examples/s]18290 examples [00:13, 1423.32 examples/s]18437 examples [00:13, 1434.72 examples/s]18581 examples [00:13, 1426.56 examples/s]18725 examples [00:13, 1430.51 examples/s]18869 examples [00:13, 1405.63 examples/s]19012 examples [00:13, 1411.86 examples/s]19157 examples [00:13, 1421.64 examples/s]19300 examples [00:13, 1420.03 examples/s]19443 examples [00:13, 1380.71 examples/s]19582 examples [00:14, 1380.05 examples/s]19726 examples [00:14, 1397.30 examples/s]19866 examples [00:14, 1387.41 examples/s]20005 examples [00:14, 1305.84 examples/s]20138 examples [00:14, 1310.22 examples/s]20270 examples [00:14, 1309.08 examples/s]20416 examples [00:14, 1348.51 examples/s]20561 examples [00:14, 1375.32 examples/s]20705 examples [00:14, 1392.77 examples/s]20848 examples [00:15, 1403.59 examples/s]20989 examples [00:15, 1370.88 examples/s]21129 examples [00:15, 1377.19 examples/s]21269 examples [00:15, 1381.91 examples/s]21411 examples [00:15, 1392.34 examples/s]21555 examples [00:15, 1404.10 examples/s]21696 examples [00:15, 1402.65 examples/s]21842 examples [00:15, 1418.47 examples/s]21984 examples [00:15, 1400.15 examples/s]22125 examples [00:15, 1391.05 examples/s]22271 examples [00:16, 1410.06 examples/s]22413 examples [00:16, 1404.39 examples/s]22554 examples [00:16, 1395.94 examples/s]22694 examples [00:16, 1350.02 examples/s]22834 examples [00:16, 1363.80 examples/s]22978 examples [00:16, 1383.94 examples/s]23117 examples [00:16, 1372.26 examples/s]23257 examples [00:16, 1380.43 examples/s]23396 examples [00:16, 1380.59 examples/s]23536 examples [00:16, 1384.42 examples/s]23675 examples [00:17, 1344.48 examples/s]23812 examples [00:17, 1349.61 examples/s]23955 examples [00:17, 1372.12 examples/s]24099 examples [00:17, 1391.00 examples/s]24239 examples [00:17, 1388.66 examples/s]24379 examples [00:17, 1367.46 examples/s]24516 examples [00:17, 1364.64 examples/s]24657 examples [00:17, 1375.41 examples/s]24800 examples [00:17, 1391.00 examples/s]24945 examples [00:17, 1406.06 examples/s]25089 examples [00:18, 1414.14 examples/s]25231 examples [00:18, 1404.30 examples/s]25376 examples [00:18, 1415.32 examples/s]25518 examples [00:18, 1415.23 examples/s]25660 examples [00:18, 1374.45 examples/s]25798 examples [00:18, 1366.10 examples/s]25935 examples [00:18, 1364.14 examples/s]26078 examples [00:18, 1383.21 examples/s]26222 examples [00:18, 1398.37 examples/s]26365 examples [00:18, 1407.40 examples/s]26509 examples [00:19, 1416.28 examples/s]26651 examples [00:19, 1404.41 examples/s]26792 examples [00:19, 1404.75 examples/s]26933 examples [00:19, 1391.50 examples/s]27073 examples [00:19, 1389.68 examples/s]27213 examples [00:19, 1380.15 examples/s]27352 examples [00:19, 1356.28 examples/s]27488 examples [00:19, 1353.14 examples/s]27632 examples [00:19, 1375.53 examples/s]27772 examples [00:20, 1381.50 examples/s]27911 examples [00:20, 1349.05 examples/s]28047 examples [00:20, 1337.63 examples/s]28190 examples [00:20, 1363.13 examples/s]28331 examples [00:20, 1375.10 examples/s]28472 examples [00:20, 1385.17 examples/s]28616 examples [00:20, 1399.48 examples/s]28757 examples [00:20, 1396.81 examples/s]28900 examples [00:20, 1406.21 examples/s]29045 examples [00:20, 1417.30 examples/s]29187 examples [00:21, 1415.63 examples/s]29329 examples [00:21, 1402.58 examples/s]29470 examples [00:21, 1370.78 examples/s]29612 examples [00:21, 1384.52 examples/s]29752 examples [00:21, 1389.01 examples/s]29892 examples [00:21, 1337.20 examples/s]30027 examples [00:21, 1293.40 examples/s]30163 examples [00:21, 1310.96 examples/s]30302 examples [00:21, 1331.82 examples/s]30447 examples [00:21, 1363.50 examples/s]30590 examples [00:22, 1382.76 examples/s]30736 examples [00:22, 1403.59 examples/s]30877 examples [00:22, 1396.18 examples/s]31021 examples [00:22, 1405.89 examples/s]31163 examples [00:22, 1409.30 examples/s]31305 examples [00:22, 1394.84 examples/s]31449 examples [00:22, 1406.45 examples/s]31590 examples [00:22, 1385.13 examples/s]31732 examples [00:22, 1394.30 examples/s]31875 examples [00:22, 1402.23 examples/s]32019 examples [00:23, 1411.19 examples/s]32161 examples [00:23, 1365.41 examples/s]32298 examples [00:23, 1343.89 examples/s]32433 examples [00:23, 1344.06 examples/s]32578 examples [00:23, 1373.92 examples/s]32721 examples [00:23, 1387.95 examples/s]32862 examples [00:23, 1391.66 examples/s]33003 examples [00:23, 1395.97 examples/s]33146 examples [00:23, 1405.31 examples/s]33289 examples [00:24, 1410.62 examples/s]33433 examples [00:24, 1417.20 examples/s]33576 examples [00:24, 1418.15 examples/s]33718 examples [00:24, 1381.81 examples/s]33857 examples [00:24, 1367.78 examples/s]33997 examples [00:24, 1376.54 examples/s]34140 examples [00:24, 1391.28 examples/s]34280 examples [00:24, 1389.46 examples/s]34420 examples [00:24, 1386.48 examples/s]34559 examples [00:24, 1371.34 examples/s]34700 examples [00:25, 1379.89 examples/s]34839 examples [00:25, 1378.70 examples/s]34977 examples [00:25, 1378.96 examples/s]35116 examples [00:25, 1380.27 examples/s]35260 examples [00:25, 1396.99 examples/s]35403 examples [00:25, 1403.98 examples/s]35544 examples [00:25, 1358.19 examples/s]35682 examples [00:25, 1363.57 examples/s]35824 examples [00:25, 1379.79 examples/s]35964 examples [00:25, 1384.07 examples/s]36103 examples [00:26, 1366.62 examples/s]36246 examples [00:26, 1379.78 examples/s]36385 examples [00:26, 1332.25 examples/s]36521 examples [00:26, 1339.05 examples/s]36662 examples [00:26, 1357.88 examples/s]36805 examples [00:26, 1376.02 examples/s]36949 examples [00:26, 1394.24 examples/s]37089 examples [00:26, 1386.85 examples/s]37231 examples [00:26, 1393.90 examples/s]37375 examples [00:26, 1406.52 examples/s]37518 examples [00:27, 1410.85 examples/s]37660 examples [00:27, 1410.59 examples/s]37802 examples [00:27, 1388.95 examples/s]37942 examples [00:27, 1371.62 examples/s]38081 examples [00:27, 1375.42 examples/s]38219 examples [00:27, 1357.76 examples/s]38358 examples [00:27, 1366.40 examples/s]38495 examples [00:27, 1363.48 examples/s]38636 examples [00:27, 1376.46 examples/s]38780 examples [00:27, 1393.41 examples/s]38920 examples [00:28, 1394.20 examples/s]39060 examples [00:28, 1390.66 examples/s]39200 examples [00:28, 1352.00 examples/s]39336 examples [00:28, 1335.76 examples/s]39470 examples [00:28, 1322.60 examples/s]39606 examples [00:28, 1331.92 examples/s]39740 examples [00:28, 1333.90 examples/s]39874 examples [00:28, 1332.28 examples/s]40008 examples [00:28, 1277.30 examples/s]40150 examples [00:29, 1316.67 examples/s]40289 examples [00:29, 1336.63 examples/s]40424 examples [00:29, 1307.98 examples/s]40556 examples [00:29, 1283.18 examples/s]40698 examples [00:29, 1319.21 examples/s]40831 examples [00:29, 1320.73 examples/s]40964 examples [00:29, 1260.90 examples/s]41091 examples [00:29, 1200.52 examples/s]41219 examples [00:29, 1220.89 examples/s]41348 examples [00:29, 1238.44 examples/s]41473 examples [00:30, 1231.73 examples/s]41609 examples [00:30, 1267.47 examples/s]41753 examples [00:30, 1312.89 examples/s]41886 examples [00:30, 1314.27 examples/s]42025 examples [00:30, 1335.43 examples/s]42166 examples [00:30, 1354.44 examples/s]42307 examples [00:30, 1370.64 examples/s]42446 examples [00:30, 1375.15 examples/s]42584 examples [00:30, 1344.74 examples/s]42727 examples [00:30, 1368.90 examples/s]42865 examples [00:31, 1362.14 examples/s]43005 examples [00:31, 1371.99 examples/s]43143 examples [00:31, 1371.95 examples/s]43281 examples [00:31, 1356.08 examples/s]43421 examples [00:31, 1366.69 examples/s]43561 examples [00:31, 1376.26 examples/s]43699 examples [00:31, 1366.23 examples/s]43837 examples [00:31, 1368.18 examples/s]43974 examples [00:31, 1335.69 examples/s]44118 examples [00:32, 1363.95 examples/s]44256 examples [00:32, 1365.89 examples/s]44398 examples [00:32, 1380.05 examples/s]44543 examples [00:32, 1399.73 examples/s]44684 examples [00:32, 1370.21 examples/s]44822 examples [00:32, 1352.59 examples/s]44958 examples [00:32, 1329.10 examples/s]45100 examples [00:32, 1353.92 examples/s]45236 examples [00:32, 1322.61 examples/s]45369 examples [00:32, 1321.94 examples/s]45503 examples [00:33, 1327.30 examples/s]45636 examples [00:33, 1264.76 examples/s]45775 examples [00:33, 1298.74 examples/s]45908 examples [00:33, 1306.84 examples/s]46049 examples [00:33, 1335.25 examples/s]46184 examples [00:33, 1334.73 examples/s]46325 examples [00:33, 1355.11 examples/s]46461 examples [00:33, 1348.31 examples/s]46599 examples [00:33, 1357.10 examples/s]46739 examples [00:33, 1369.49 examples/s]46878 examples [00:34, 1373.89 examples/s]47019 examples [00:34, 1382.82 examples/s]47162 examples [00:34, 1394.85 examples/s]47302 examples [00:34, 1364.49 examples/s]47440 examples [00:34, 1365.49 examples/s]47583 examples [00:34, 1382.13 examples/s]47727 examples [00:34, 1398.62 examples/s]47868 examples [00:34, 1387.98 examples/s]48007 examples [00:34, 1367.65 examples/s]48148 examples [00:34, 1377.53 examples/s]48291 examples [00:35, 1390.62 examples/s]48431 examples [00:35, 1377.15 examples/s]48569 examples [00:35, 1361.03 examples/s]48708 examples [00:35, 1369.12 examples/s]48850 examples [00:35, 1382.70 examples/s]48989 examples [00:35, 1377.02 examples/s]49131 examples [00:35, 1388.62 examples/s]49274 examples [00:35, 1400.15 examples/s]49418 examples [00:35, 1409.71 examples/s]49560 examples [00:35, 1383.89 examples/s]49703 examples [00:36, 1395.02 examples/s]49848 examples [00:36, 1410.24 examples/s]49990 examples [00:36, 1412.99 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 23%|██▎       | 11303/50000 [00:00<00:00, 113024.18 examples/s] 56%|█████▌    | 27796/50000 [00:00<00:00, 124807.44 examples/s] 90%|████████▉ | 44935/50000 [00:00<00:00, 135885.59 examples/s]                                                                0 examples [00:00, ? examples/s]122 examples [00:00, 1216.33 examples/s]263 examples [00:00, 1267.54 examples/s]402 examples [00:00, 1301.04 examples/s]546 examples [00:00, 1338.75 examples/s]692 examples [00:00, 1372.10 examples/s]838 examples [00:00, 1394.78 examples/s]984 examples [00:00, 1413.46 examples/s]1127 examples [00:00, 1415.91 examples/s]1263 examples [00:00, 1396.80 examples/s]1402 examples [00:01, 1392.84 examples/s]1549 examples [00:01, 1414.70 examples/s]1695 examples [00:01, 1425.30 examples/s]1837 examples [00:01, 1423.42 examples/s]1981 examples [00:01, 1425.40 examples/s]2123 examples [00:01, 1413.60 examples/s]2264 examples [00:01, 1399.23 examples/s]2404 examples [00:01, 1381.46 examples/s]2542 examples [00:01, 1365.92 examples/s]2682 examples [00:01, 1375.28 examples/s]2826 examples [00:02, 1392.30 examples/s]2966 examples [00:02, 1389.99 examples/s]3111 examples [00:02, 1405.12 examples/s]3252 examples [00:02, 1382.41 examples/s]3391 examples [00:02, 1372.92 examples/s]3535 examples [00:02, 1391.77 examples/s]3675 examples [00:02, 1384.57 examples/s]3814 examples [00:02, 1385.31 examples/s]3953 examples [00:02, 1378.37 examples/s]4100 examples [00:02, 1404.30 examples/s]4241 examples [00:03, 1365.64 examples/s]4386 examples [00:03, 1388.14 examples/s]4534 examples [00:03, 1414.28 examples/s]4676 examples [00:03, 1396.47 examples/s]4817 examples [00:03, 1400.21 examples/s]4958 examples [00:03, 1368.15 examples/s]5096 examples [00:03, 1338.40 examples/s]5231 examples [00:03, 1338.99 examples/s]5366 examples [00:03, 1326.05 examples/s]5513 examples [00:03, 1364.00 examples/s]5659 examples [00:04, 1390.28 examples/s]5804 examples [00:04, 1405.49 examples/s]5945 examples [00:04, 1399.94 examples/s]6086 examples [00:04, 1387.10 examples/s]6230 examples [00:04, 1402.33 examples/s]6374 examples [00:04, 1411.17 examples/s]6516 examples [00:04, 1403.78 examples/s]6657 examples [00:04, 1396.71 examples/s]6797 examples [00:04, 1392.83 examples/s]6943 examples [00:04, 1412.24 examples/s]7087 examples [00:05, 1420.07 examples/s]7230 examples [00:05, 1417.77 examples/s]7372 examples [00:05, 1399.59 examples/s]7513 examples [00:05, 1391.18 examples/s]7653 examples [00:05, 1378.19 examples/s]7799 examples [00:05, 1401.20 examples/s]7943 examples [00:05, 1408.81 examples/s]8087 examples [00:05, 1415.31 examples/s]8229 examples [00:05, 1376.26 examples/s]8368 examples [00:06, 1379.04 examples/s]8507 examples [00:06, 1348.48 examples/s]8645 examples [00:06, 1356.87 examples/s]8791 examples [00:06, 1385.21 examples/s]8930 examples [00:06, 1375.89 examples/s]9072 examples [00:06, 1388.59 examples/s]9219 examples [00:06, 1411.08 examples/s]9361 examples [00:06, 1411.42 examples/s]9505 examples [00:06, 1419.40 examples/s]9648 examples [00:06, 1364.80 examples/s]9786 examples [00:07, 1366.55 examples/s]9930 examples [00:07, 1385.88 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteMMZ158/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteMMZ158/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f59e61d6620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f59e61d6620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f59e61d6620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59803d7278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f598040cda0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f59e61d6620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f59e61d6620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f59e61d6620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59e3721470>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f59e3721518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f595dc66158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f595dc66158> 

  function with postional parmater data_info <function split_train_valid at 0x7f595dc66158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f595dc66268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f595dc66268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f595dc66268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=b1b69872c1cd76871e0b9ba343be3d96058d84e5345a212638d3769f80ca8d18
  Stored in directory: /tmp/pip-ephem-wheel-cache-0pagsq7j/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:57:54, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:36:24, 15.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:58:51, 21.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:41:29, 31.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.91M/862M [00:01<5:23:04, 44.4kB/s].vector_cache/glove.6B.zip:   1%|          | 5.87M/862M [00:01<3:45:13, 63.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.40M/862M [00:01<2:37:07, 90.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.9M/862M [00:01<1:49:30, 129kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<1:16:19, 184kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.3M/862M [00:01<53:18, 263kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:01<37:11, 374kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:01<26:01, 532kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:02<18:12, 757kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<12:47, 1.07MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.7M/862M [00:02<08:59, 1.52MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.4M/862M [00:02<06:22, 2.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<04:43, 2.86MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:12, 2.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<05:37, 2.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:04<04:23, 3.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:04<03:14, 4.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<12:59, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<10:57, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:06<08:07, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<08:10, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<07:31, 1.77MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<05:43, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:08<04:16, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<03:22, 3.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<32:44, 404kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<25:17, 523kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<18:17, 722kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<12:53, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<1:04:03, 206kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<46:18, 284kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<32:41, 402kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<25:35, 512kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<20:49, 628kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<15:11, 861kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:57, 1.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<10:38, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<08:46, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<06:27, 2.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<07:35, 1.71MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:54, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<05:11, 2.49MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:18<03:48, 3.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<13:58, 922kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<11:27, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<08:24, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<08:24, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:19, 1.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<05:28, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:22<03:59, 3.20MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<1:48:53, 117kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<1:18:00, 163kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<55:00, 231kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<38:33, 329kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<40:30, 313kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<29:53, 424kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<21:16, 595kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<15:26, 817kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:52:44, 19.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<7:38:14, 27.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<5:20:33, 39.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<3:46:13, 55.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<2:39:34, 78.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<1:51:46, 112kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:20:51, 154kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<57:39, 216kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<40:37, 306kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<28:29, 435kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:09:53, 177kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<50:08, 247kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<35:20, 350kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<27:35, 447kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<21:48, 565kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<15:47, 780kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<11:16, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:50, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<10:54, 1.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:16, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<05:54, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<47:56, 254kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<34:46, 350kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<24:37, 494kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<20:01, 606kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:31, 734kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:11, 993kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<08:37, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<21:17, 566kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<16:07, 747kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<11:30, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<08:12, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<53:20, 225kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<39:46, 301kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<28:19, 423kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<20:07, 594kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<16:49, 709kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<13:00, 916kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<09:24, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:20, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:04, 1.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:58, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<05:00, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<1:27:35, 135kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<1:02:30, 188kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<43:54, 268kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<33:24, 351kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<24:34, 477kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<17:28, 669kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<12:20, 944kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:35:12, 122kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:09:00, 169kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<48:42, 239kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<34:22, 338kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<26:42, 434kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<19:53, 582kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:08, 816kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<12:34, 915kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:07, 1.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:16, 1.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:30, 1.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<07:06, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:12, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:48, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<47:41, 238kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<34:31, 329kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<24:24, 465kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<19:42, 574kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<14:55, 757kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:42, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<10:08, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:14, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<06:00, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:51, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:05, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:31, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:01, 2.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:21, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:17, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:41, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:51, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:21, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:56, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:40, 3.00MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:57, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:49, 2.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<02:58, 3.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:15, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:30, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:05, 2.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:21, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:56, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:39, 2.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:27, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:37, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:01, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:47, 2.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:09, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:59, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:50, 2.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<02:49, 3.77MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<15:22, 691kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<11:57, 887kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<08:39, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:17, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:59, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:09, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:44, 2.81MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<10:27:41, 16.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<7:21:22, 23.7kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<5:09:07, 33.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<3:35:26, 48.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<3:46:06, 46.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<2:39:15, 65.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<1:51:28, 93.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:20:10, 129kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<57:07, 181kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<40:09, 257kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<30:32, 336kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<21:44, 472kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<15:27, 662kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<13:10, 774kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:15, 994kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<07:23, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<05:18, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<1:10:31, 144kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<50:22, 201kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<35:23, 285kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<27:05, 372kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<19:58, 504kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<14:12, 706kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<12:16, 814kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<10:36, 942kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:55, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<07:08, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:59, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:26, 2.22MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:24, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:35, 2.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:49, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:20, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:11, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:06, 3.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:02, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:22, 2.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:16, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<02:33, 3.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:24, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:46, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:34, 2.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:44, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:18, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:15, 2.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:28, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:05, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:06, 3.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:24, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:02, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:02, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:19, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:58, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:00, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:57, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:58, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:55, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:58, 3.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:14, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:54, 3.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:12, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<02:56, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:50, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:54, 3.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:09, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:50, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:54, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:47, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:50, 3.13MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:05, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:45, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<02:50, 3.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:43, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:47, 3.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:41, 2.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:46, 3.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:59, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:39, 2.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:46, 3.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:57, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:38, 2.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:45, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:55, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:33, 2.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<02:35, 3.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:32, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:55, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:40, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:57, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:50, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:52, 2.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:08, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:45, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:49, 2.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:53, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:33, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:39, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:47, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:36, 3.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:25, 2.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:39, 3.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<01:55, 4.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:38<03:40, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<8:29:51, 15.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<5:58:07, 22.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<4:10:30, 31.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<2:54:49, 45.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<2:05:22, 63.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<1:28:30, 89.5kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<1:01:58, 127kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<43:22, 181kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<35:34, 221kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<26:37, 295kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<18:58, 413kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<13:19, 586kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<13:25, 580kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<10:11, 763kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<07:19, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:53, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<06:24, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:52, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<03:28, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<57:32, 133kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<41:01, 186kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<28:48, 264kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<21:50, 347kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<16:49, 450kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<12:08, 623kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<09:39, 777kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:32, 996kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:32, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:23, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:08, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:04, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:41, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:35, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:00, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:16, 3.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:40, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:01, 1.80MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:57, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:44, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:04, 1.76MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:23, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:06, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:19, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:43, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:57, 2.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:56, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:12, 3.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:08, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:35, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:51, 2.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:12<02:04, 3.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<57:49, 118kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<41:07, 166kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<28:51, 235kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<21:40, 312kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<16:34, 408kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<11:55, 565kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<08:21, 800kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<1:23:41, 79.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<59:12, 113kB/s]   .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<41:27, 161kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<30:25, 218kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<21:57, 301kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<15:28, 426kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<12:19, 531kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<09:57, 658kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<07:14, 903kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:06, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:37, 851kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<06:00, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<04:21, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:32, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:29, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:27, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:25, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:02, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:15, 2.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:03, 2.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:25, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:42, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:53, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:39, 2.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:00, 3.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<01:28, 4.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<46:39, 132kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<33:53, 181kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<23:59, 255kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<16:43, 363kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<6:01:18, 16.8kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<4:13:16, 23.9kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<2:56:41, 34.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<2:04:21, 48.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:28:14, 68.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<1:01:56, 96.6kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<43:56, 135kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<31:20, 189kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<21:59, 268kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<16:40, 352kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<12:51, 456kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<09:16, 630kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<06:30, 890kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<27:03, 214kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<19:30, 297kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<13:43, 420kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<11:00, 520kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<08:00, 714kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<05:42, 997kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:31, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:27, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<03:14, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:33, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:38, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:46, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:00, 2.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:34, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:46, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:46, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:12, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:48, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:05, 2.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:00, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:22, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:42, 3.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<5:07:26, 17.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<3:35:29, 24.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<2:30:14, 35.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<1:45:40, 49.7kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<1:14:59, 69.9kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<52:37, 99.4kB/s]  .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<37:18, 139kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<26:38, 194kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<18:41, 275kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<14:10, 361kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<10:57, 466kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<07:54, 644kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<06:17, 801kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:55, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:33, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:38, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:02, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:14, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:53, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:37, 2.98MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:42, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:13, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:40, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:21, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:45, 2.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:19, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:36, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:03, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:28, 3.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<07:58, 580kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<09:27, 489kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<07:04, 645kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:28, 833kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:54, 1.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<02:47, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<07:34, 593kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<06:21, 706kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<04:41, 954kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<03:19, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:08, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:17, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<02:22, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<01:42, 2.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<18:45, 232kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<13:33, 321kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<09:32, 453kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<07:38, 561kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:42, 749kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:10, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<05:51, 719kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:31, 930kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<03:15, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:14, 1.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:41, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:57, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:19, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:02, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:31, 2.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:00, 2.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:48, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:20, 2.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:52, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:42, 2.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:16, 3.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:48, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:39, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:15, 3.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:46, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:37, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:13, 3.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:43, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:57, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:32, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:06, 3.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:59, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:28, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:48, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:06, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<01:50, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:22, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:46, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:35, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:11, 2.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:38, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:30, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:07, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:34, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:26, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:32, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:45, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:23, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:29, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:22, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:01, 3.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:27, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:20, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:00, 3.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:26, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:39, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:18, 2.38MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<00:56, 3.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:51, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:36, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:11, 2.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:31, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:19, 2.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:56, 3.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<2:48:27, 17.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<1:57:58, 24.6kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<1:21:54, 35.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<57:16, 49.7kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<40:18, 70.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<28:01, 100kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<20:00, 139kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<14:33, 190kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<10:17, 268kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<07:30, 361kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<05:30, 490kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<03:53, 688kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<03:18, 798kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:34, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:50, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:52, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:34, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:09, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:54, 2.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:12, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:48, 2.94MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:07, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:16, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:59, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<00:42, 3.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:53, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:33, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:08, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<01:18, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:21, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:03, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:04, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:58, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:43, 2.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:59, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:07, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:52, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:55, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:51, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:38, 3.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:53, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:49, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:37, 3.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:47, 2.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:49, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:34, 3.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:48, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:56, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:31, 3.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:50, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:37, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:53, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:41, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:29, 3.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<1:29:15, 17.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<1:02:22, 24.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<42:58, 35.1kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<29:41, 49.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<21:02, 69.9kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<14:39, 99.4kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<10:08, 142kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<07:19, 192kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<05:15, 266kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<03:38, 377kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:47, 478kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:13, 599kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:36, 824kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:08, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:05, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:53, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:38, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:42, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:56, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:45, 1.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:38, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:39, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:30, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:33, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:25, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:27, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:29, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:23, 2.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:24, 2.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:27, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:21, 2.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:22, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:25, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:19, 2.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:13, 3.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:38, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:35, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:27, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:24, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:25, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:18, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:08<00:13, 3.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:36, 1.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:32, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:24, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:21, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:14<00:08, 3.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<08:45, 50.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<06:09, 71.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<04:11, 101kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<02:42, 145kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:59, 189kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:26, 257kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:59, 361kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:40, 509kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:29, 620kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:23, 771kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:17, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:11, 1.40MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:07, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:17, 820kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:15, 933kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:12, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:08, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:05, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:22<00:03, 2.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:10, 988kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:11, 849kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<02:29, 37.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<01:19, 54.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:21, 77.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:19, 76.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:13, 105kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:05, 149kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 794/400000 [00:00<00:50, 7936.30it/s]  0%|          | 1948/400000 [00:00<00:45, 8756.63it/s]  1%|          | 3142/400000 [00:00<00:41, 9516.46it/s]  1%|          | 4336/400000 [00:00<00:39, 10132.74it/s]  1%|▏         | 5507/400000 [00:00<00:37, 10558.45it/s]  2%|▏         | 6665/400000 [00:00<00:36, 10844.63it/s]  2%|▏         | 7787/400000 [00:00<00:35, 10954.06it/s]  2%|▏         | 8983/400000 [00:00<00:34, 11236.09it/s]  3%|▎         | 10127/400000 [00:00<00:34, 11294.61it/s]  3%|▎         | 11318/400000 [00:01<00:33, 11471.63it/s]  3%|▎         | 12468/400000 [00:01<00:33, 11478.98it/s]  3%|▎         | 13672/400000 [00:01<00:33, 11639.59it/s]  4%|▎         | 14853/400000 [00:01<00:32, 11688.62it/s]  4%|▍         | 16018/400000 [00:01<00:32, 11643.92it/s]  4%|▍         | 17230/400000 [00:01<00:32, 11780.56it/s]  5%|▍         | 18443/400000 [00:01<00:32, 11881.97it/s]  5%|▍         | 19631/400000 [00:01<00:33, 11426.01it/s]  5%|▌         | 20868/400000 [00:01<00:32, 11693.11it/s]  6%|▌         | 22082/400000 [00:01<00:31, 11821.15it/s]  6%|▌         | 23289/400000 [00:02<00:31, 11892.11it/s]  6%|▌         | 24495/400000 [00:02<00:31, 11862.21it/s]  6%|▋         | 25683/400000 [00:02<00:31, 11749.60it/s]  7%|▋         | 26891/400000 [00:02<00:31, 11845.62it/s]  7%|▋         | 28077/400000 [00:02<00:31, 11823.76it/s]  7%|▋         | 29271/400000 [00:02<00:31, 11857.54it/s]  8%|▊         | 30458/400000 [00:02<00:32, 11447.67it/s]  8%|▊         | 31607/400000 [00:02<00:32, 11397.46it/s]  8%|▊         | 32788/400000 [00:02<00:31, 11516.74it/s]  8%|▊         | 33987/400000 [00:02<00:31, 11652.18it/s]  9%|▉         | 35200/400000 [00:03<00:30, 11790.58it/s]  9%|▉         | 36381/400000 [00:03<00:30, 11737.43it/s]  9%|▉         | 37556/400000 [00:03<00:30, 11717.36it/s] 10%|▉         | 38763/400000 [00:03<00:30, 11820.16it/s] 10%|▉         | 39993/400000 [00:03<00:30, 11958.00it/s] 10%|█         | 41190/400000 [00:03<00:30, 11900.50it/s] 11%|█         | 42381/400000 [00:03<00:30, 11840.05it/s] 11%|█         | 43566/400000 [00:03<00:30, 11700.13it/s] 11%|█         | 44751/400000 [00:03<00:30, 11744.51it/s] 11%|█▏        | 45950/400000 [00:03<00:29, 11816.19it/s] 12%|█▏        | 47133/400000 [00:04<00:29, 11768.08it/s] 12%|█▏        | 48334/400000 [00:04<00:29, 11838.33it/s] 12%|█▏        | 49519/400000 [00:04<00:30, 11682.31it/s] 13%|█▎        | 50714/400000 [00:04<00:29, 11759.37it/s] 13%|█▎        | 51906/400000 [00:04<00:29, 11806.28it/s] 13%|█▎        | 53134/400000 [00:04<00:29, 11943.65it/s] 14%|█▎        | 54330/400000 [00:04<00:29, 11663.05it/s] 14%|█▍        | 55499/400000 [00:04<00:29, 11511.85it/s] 14%|█▍        | 56689/400000 [00:04<00:29, 11621.89it/s] 14%|█▍        | 57853/400000 [00:04<00:29, 11621.84it/s] 15%|█▍        | 59083/400000 [00:05<00:28, 11814.98it/s] 15%|█▌        | 60281/400000 [00:05<00:28, 11862.79it/s] 15%|█▌        | 61469/400000 [00:05<00:28, 11733.85it/s] 16%|█▌        | 62644/400000 [00:05<00:28, 11712.22it/s] 16%|█▌        | 63839/400000 [00:05<00:28, 11780.88it/s] 16%|█▋        | 65020/400000 [00:05<00:28, 11788.37it/s] 17%|█▋        | 66200/400000 [00:05<00:29, 11367.82it/s] 17%|█▋        | 67341/400000 [00:05<00:29, 11239.14it/s] 17%|█▋        | 68517/400000 [00:05<00:29, 11388.32it/s] 17%|█▋        | 69709/400000 [00:05<00:28, 11540.42it/s] 18%|█▊        | 70915/400000 [00:06<00:28, 11689.68it/s] 18%|█▊        | 72086/400000 [00:06<00:28, 11634.61it/s] 18%|█▊        | 73258/400000 [00:06<00:28, 11659.54it/s] 19%|█▊        | 74483/400000 [00:06<00:27, 11828.33it/s] 19%|█▉        | 75701/400000 [00:06<00:27, 11930.96it/s] 19%|█▉        | 76906/400000 [00:06<00:27, 11964.72it/s] 20%|█▉        | 78104/400000 [00:06<00:27, 11885.57it/s] 20%|█▉        | 79294/400000 [00:06<00:27, 11636.61it/s] 20%|██        | 80482/400000 [00:06<00:27, 11708.17it/s] 20%|██        | 81662/400000 [00:06<00:27, 11734.84it/s] 21%|██        | 82837/400000 [00:07<00:27, 11609.67it/s] 21%|██        | 83999/400000 [00:07<00:27, 11587.73it/s] 21%|██▏       | 85159/400000 [00:07<00:27, 11499.49it/s] 22%|██▏       | 86377/400000 [00:07<00:26, 11695.04it/s] 22%|██▏       | 87548/400000 [00:07<00:26, 11696.66it/s] 22%|██▏       | 88719/400000 [00:07<00:26, 11688.78it/s] 22%|██▏       | 89889/400000 [00:07<00:26, 11558.98it/s] 23%|██▎       | 91046/400000 [00:07<00:27, 11426.01it/s] 23%|██▎       | 92190/400000 [00:07<00:26, 11429.24it/s] 23%|██▎       | 93412/400000 [00:08<00:26, 11654.25it/s] 24%|██▎       | 94620/400000 [00:08<00:25, 11776.97it/s] 24%|██▍       | 95825/400000 [00:08<00:25, 11856.16it/s] 24%|██▍       | 97012/400000 [00:08<00:26, 11597.53it/s] 25%|██▍       | 98223/400000 [00:08<00:25, 11745.85it/s] 25%|██▍       | 99445/400000 [00:08<00:25, 11883.69it/s] 25%|██▌       | 100636/400000 [00:08<00:25, 11775.95it/s] 25%|██▌       | 101816/400000 [00:08<00:25, 11702.09it/s] 26%|██▌       | 102988/400000 [00:08<00:25, 11556.38it/s] 26%|██▌       | 104158/400000 [00:08<00:25, 11597.93it/s] 26%|██▋       | 105372/400000 [00:09<00:25, 11753.62it/s] 27%|██▋       | 106549/400000 [00:09<00:25, 11643.96it/s] 27%|██▋       | 107766/400000 [00:09<00:24, 11794.75it/s] 27%|██▋       | 108947/400000 [00:09<00:24, 11643.63it/s] 28%|██▊       | 110155/400000 [00:09<00:24, 11769.35it/s] 28%|██▊       | 111367/400000 [00:09<00:24, 11871.84it/s] 28%|██▊       | 112556/400000 [00:09<00:25, 11254.83it/s] 28%|██▊       | 113689/400000 [00:09<00:27, 10524.14it/s] 29%|██▊       | 114757/400000 [00:09<00:28, 10069.90it/s] 29%|██▉       | 115779/400000 [00:10<00:28, 9935.03it/s]  29%|██▉       | 116891/400000 [00:10<00:27, 10262.46it/s] 30%|██▉       | 118105/400000 [00:10<00:26, 10760.30it/s] 30%|██▉       | 119232/400000 [00:10<00:25, 10905.72it/s] 30%|███       | 120411/400000 [00:10<00:25, 11155.27it/s] 30%|███       | 121538/400000 [00:10<00:24, 11187.78it/s] 31%|███       | 122727/400000 [00:10<00:24, 11387.23it/s] 31%|███       | 123871/400000 [00:10<00:24, 11141.30it/s] 31%|███       | 124990/400000 [00:10<00:25, 10812.17it/s] 32%|███▏      | 126158/400000 [00:10<00:24, 11058.62it/s] 32%|███▏      | 127340/400000 [00:11<00:24, 11274.49it/s] 32%|███▏      | 128473/400000 [00:11<00:24, 11287.83it/s] 32%|███▏      | 129702/400000 [00:11<00:23, 11568.34it/s] 33%|███▎      | 130881/400000 [00:11<00:23, 11632.93it/s] 33%|███▎      | 132048/400000 [00:11<00:23, 11591.11it/s] 33%|███▎      | 133273/400000 [00:11<00:22, 11781.07it/s] 34%|███▎      | 134454/400000 [00:11<00:22, 11769.42it/s] 34%|███▍      | 135633/400000 [00:11<00:22, 11621.35it/s] 34%|███▍      | 136809/400000 [00:11<00:22, 11660.37it/s] 35%|███▍      | 138008/400000 [00:11<00:22, 11756.23it/s] 35%|███▍      | 139240/400000 [00:12<00:21, 11917.71it/s] 35%|███▌      | 140433/400000 [00:12<00:21, 11892.48it/s] 35%|███▌      | 141624/400000 [00:12<00:21, 11846.90it/s] 36%|███▌      | 142810/400000 [00:12<00:21, 11787.28it/s] 36%|███▌      | 144014/400000 [00:12<00:21, 11861.75it/s] 36%|███▋      | 145214/400000 [00:12<00:21, 11900.81it/s] 37%|███▋      | 146405/400000 [00:12<00:21, 11742.32it/s] 37%|███▋      | 147612/400000 [00:12<00:21, 11836.50it/s] 37%|███▋      | 148797/400000 [00:12<00:21, 11750.20it/s] 37%|███▋      | 149973/400000 [00:12<00:21, 11650.70it/s] 38%|███▊      | 151155/400000 [00:13<00:21, 11699.38it/s] 38%|███▊      | 152384/400000 [00:13<00:20, 11870.09it/s] 38%|███▊      | 153587/400000 [00:13<00:20, 11917.51it/s] 39%|███▊      | 154780/400000 [00:13<00:20, 11921.25it/s] 39%|███▉      | 155973/400000 [00:13<00:20, 11881.72it/s] 39%|███▉      | 157174/400000 [00:13<00:20, 11917.63it/s] 40%|███▉      | 158367/400000 [00:13<00:20, 11700.06it/s] 40%|███▉      | 159558/400000 [00:13<00:20, 11761.38it/s] 40%|████      | 160735/400000 [00:13<00:20, 11742.76it/s] 40%|████      | 161937/400000 [00:13<00:20, 11823.12it/s] 41%|████      | 163156/400000 [00:14<00:19, 11930.21it/s] 41%|████      | 164384/400000 [00:14<00:19, 12030.64it/s] 41%|████▏     | 165588/400000 [00:14<00:19, 11897.45it/s] 42%|████▏     | 166795/400000 [00:14<00:19, 11945.39it/s] 42%|████▏     | 167991/400000 [00:14<00:19, 11871.57it/s] 42%|████▏     | 169220/400000 [00:14<00:19, 11991.13it/s] 43%|████▎     | 170420/400000 [00:14<00:20, 11222.01it/s] 43%|████▎     | 171608/400000 [00:14<00:20, 11410.13it/s] 43%|████▎     | 172770/400000 [00:14<00:19, 11469.64it/s] 43%|████▎     | 173974/400000 [00:14<00:19, 11633.15it/s] 44%|████▍     | 175188/400000 [00:15<00:19, 11779.43it/s] 44%|████▍     | 176406/400000 [00:15<00:18, 11895.38it/s] 44%|████▍     | 177599/400000 [00:15<00:18, 11877.81it/s] 45%|████▍     | 178789/400000 [00:15<00:19, 11394.61it/s] 45%|████▍     | 179935/400000 [00:15<00:19, 11347.08it/s] 45%|████▌     | 181139/400000 [00:15<00:18, 11545.70it/s] 46%|████▌     | 182298/400000 [00:15<00:19, 11399.34it/s] 46%|████▌     | 183488/400000 [00:15<00:18, 11543.47it/s] 46%|████▌     | 184650/400000 [00:15<00:18, 11563.73it/s] 46%|████▋     | 185896/400000 [00:16<00:18, 11817.80it/s] 47%|████▋     | 187122/400000 [00:16<00:17, 11946.08it/s] 47%|████▋     | 188370/400000 [00:16<00:17, 12099.39it/s] 47%|████▋     | 189582/400000 [00:16<00:17, 12079.44it/s] 48%|████▊     | 190792/400000 [00:16<00:17, 12008.85it/s] 48%|████▊     | 191994/400000 [00:16<00:17, 11962.94it/s] 48%|████▊     | 193192/400000 [00:16<00:17, 11960.24it/s] 49%|████▊     | 194389/400000 [00:16<00:17, 11784.74it/s] 49%|████▉     | 195591/400000 [00:16<00:17, 11852.65it/s] 49%|████▉     | 196778/400000 [00:16<00:17, 11666.38it/s] 49%|████▉     | 197982/400000 [00:17<00:17, 11775.91it/s] 50%|████▉     | 199178/400000 [00:17<00:16, 11830.39it/s] 50%|█████     | 200366/400000 [00:17<00:16, 11842.35it/s] 50%|█████     | 201589/400000 [00:17<00:16, 11954.71it/s] 51%|█████     | 202786/400000 [00:17<00:16, 11944.80it/s] 51%|█████     | 203981/400000 [00:17<00:16, 11603.64it/s] 51%|█████▏    | 205153/400000 [00:17<00:16, 11637.98it/s] 52%|█████▏    | 206375/400000 [00:17<00:16, 11806.33it/s] 52%|█████▏    | 207614/400000 [00:17<00:16, 11973.02it/s] 52%|█████▏    | 208814/400000 [00:17<00:16, 11763.58it/s] 53%|█████▎    | 210042/400000 [00:18<00:15, 11912.08it/s] 53%|█████▎    | 211236/400000 [00:18<00:15, 11905.63it/s] 53%|█████▎    | 212428/400000 [00:18<00:15, 11899.43it/s] 53%|█████▎    | 213619/400000 [00:18<00:15, 11839.37it/s] 54%|█████▎    | 214804/400000 [00:18<00:15, 11674.83it/s] 54%|█████▍    | 216018/400000 [00:18<00:15, 11807.96it/s] 54%|█████▍    | 217200/400000 [00:18<00:15, 11751.87it/s] 55%|█████▍    | 218386/400000 [00:18<00:15, 11781.82it/s] 55%|█████▍    | 219617/400000 [00:18<00:15, 11933.29it/s] 55%|█████▌    | 220812/400000 [00:18<00:15, 11827.90it/s] 55%|█████▌    | 221998/400000 [00:19<00:15, 11835.32it/s] 56%|█████▌    | 223192/400000 [00:19<00:14, 11865.50it/s] 56%|█████▌    | 224414/400000 [00:19<00:14, 11969.30it/s] 56%|█████▋    | 225629/400000 [00:19<00:14, 12021.05it/s] 57%|█████▋    | 226832/400000 [00:19<00:14, 11893.39it/s] 57%|█████▋    | 228060/400000 [00:19<00:14, 12002.93it/s] 57%|█████▋    | 229261/400000 [00:19<00:14, 11818.26it/s] 58%|█████▊    | 230444/400000 [00:19<00:14, 11812.41it/s] 58%|█████▊    | 231661/400000 [00:19<00:14, 11915.93it/s] 58%|█████▊    | 232854/400000 [00:19<00:14, 11863.17it/s] 59%|█████▊    | 234052/400000 [00:20<00:13, 11896.59it/s] 59%|█████▉    | 235243/400000 [00:20<00:13, 11799.45it/s] 59%|█████▉    | 236424/400000 [00:20<00:13, 11784.98it/s] 59%|█████▉    | 237652/400000 [00:20<00:13, 11927.67it/s] 60%|█████▉    | 238846/400000 [00:20<00:13, 11807.24it/s] 60%|██████    | 240028/400000 [00:20<00:13, 11777.47it/s] 60%|██████    | 241211/400000 [00:20<00:13, 11790.50it/s] 61%|██████    | 242423/400000 [00:20<00:13, 11887.20it/s] 61%|██████    | 243636/400000 [00:20<00:13, 11958.09it/s] 61%|██████    | 244833/400000 [00:20<00:13, 11824.25it/s] 62%|██████▏   | 246020/400000 [00:21<00:13, 11836.75it/s] 62%|██████▏   | 247228/400000 [00:21<00:12, 11906.60it/s] 62%|██████▏   | 248440/400000 [00:21<00:12, 11969.78it/s] 62%|██████▏   | 249638/400000 [00:21<00:12, 11857.53it/s] 63%|██████▎   | 250881/400000 [00:21<00:12, 12022.43it/s] 63%|██████▎   | 252085/400000 [00:21<00:12, 11583.82it/s] 63%|██████▎   | 253263/400000 [00:21<00:12, 11641.41it/s] 64%|██████▎   | 254484/400000 [00:21<00:12, 11802.92it/s] 64%|██████▍   | 255667/400000 [00:21<00:12, 11647.75it/s] 64%|██████▍   | 256886/400000 [00:21<00:12, 11803.77it/s] 65%|██████▍   | 258069/400000 [00:22<00:12, 11722.12it/s] 65%|██████▍   | 259283/400000 [00:22<00:11, 11844.08it/s] 65%|██████▌   | 260496/400000 [00:22<00:11, 11928.05it/s] 65%|██████▌   | 261690/400000 [00:22<00:11, 11930.58it/s] 66%|██████▌   | 262917/400000 [00:22<00:11, 12026.74it/s] 66%|██████▌   | 264121/400000 [00:22<00:11, 11847.42it/s] 66%|██████▋   | 265307/400000 [00:22<00:11, 11840.45it/s] 67%|██████▋   | 266506/400000 [00:22<00:11, 11883.97it/s] 67%|██████▋   | 267712/400000 [00:22<00:11, 11935.88it/s] 67%|██████▋   | 268919/400000 [00:23<00:10, 11974.81it/s] 68%|██████▊   | 270117/400000 [00:23<00:10, 11836.52it/s] 68%|██████▊   | 271350/400000 [00:23<00:10, 11978.12it/s] 68%|██████▊   | 272549/400000 [00:23<00:10, 11888.46it/s] 68%|██████▊   | 273741/400000 [00:23<00:10, 11895.60it/s] 69%|██████▊   | 274944/400000 [00:23<00:10, 11935.41it/s] 69%|██████▉   | 276138/400000 [00:23<00:10, 11765.23it/s] 69%|██████▉   | 277316/400000 [00:23<00:10, 11760.22it/s] 70%|██████▉   | 278541/400000 [00:23<00:10, 11900.19it/s] 70%|██████▉   | 279769/400000 [00:23<00:10, 12008.51it/s] 70%|███████   | 280999/400000 [00:24<00:09, 12094.22it/s] 71%|███████   | 282210/400000 [00:24<00:09, 11887.04it/s] 71%|███████   | 283440/400000 [00:24<00:09, 12007.79it/s] 71%|███████   | 284671/400000 [00:24<00:09, 12096.21it/s] 71%|███████▏  | 285882/400000 [00:24<00:09, 12019.39it/s] 72%|███████▏  | 287108/400000 [00:24<00:09, 12090.17it/s] 72%|███████▏  | 288318/400000 [00:24<00:09, 11885.02it/s] 72%|███████▏  | 289508/400000 [00:24<00:09, 11692.43it/s] 73%|███████▎  | 290731/400000 [00:24<00:09, 11846.36it/s] 73%|███████▎  | 291946/400000 [00:24<00:09, 11934.46it/s] 73%|███████▎  | 293151/400000 [00:25<00:08, 11966.10it/s] 74%|███████▎  | 294349/400000 [00:25<00:09, 11688.80it/s] 74%|███████▍  | 295533/400000 [00:25<00:08, 11733.35it/s] 74%|███████▍  | 296769/400000 [00:25<00:08, 11914.26it/s] 75%|███████▍  | 298001/400000 [00:25<00:08, 12032.19it/s] 75%|███████▍  | 299217/400000 [00:25<00:08, 12070.09it/s] 75%|███████▌  | 300426/400000 [00:25<00:08, 11789.30it/s] 75%|███████▌  | 301648/400000 [00:25<00:08, 11915.22it/s] 76%|███████▌  | 302842/400000 [00:25<00:08, 11921.85it/s] 76%|███████▌  | 304036/400000 [00:25<00:08, 11913.19it/s] 76%|███████▋  | 305263/400000 [00:26<00:07, 12015.32it/s] 77%|███████▋  | 306466/400000 [00:26<00:07, 11785.80it/s] 77%|███████▋  | 307671/400000 [00:26<00:07, 11861.61it/s] 77%|███████▋  | 308902/400000 [00:26<00:07, 11992.55it/s] 78%|███████▊  | 310118/400000 [00:26<00:07, 12039.94it/s] 78%|███████▊  | 311334/400000 [00:26<00:07, 12073.84it/s] 78%|███████▊  | 312543/400000 [00:26<00:07, 11775.30it/s] 78%|███████▊  | 313726/400000 [00:26<00:07, 11790.53it/s] 79%|███████▊  | 314922/400000 [00:26<00:07, 11839.95it/s] 79%|███████▉  | 316136/400000 [00:26<00:07, 11928.02it/s] 79%|███████▉  | 317356/400000 [00:27<00:06, 12006.14it/s] 80%|███████▉  | 318558/400000 [00:27<00:06, 11825.66it/s] 80%|███████▉  | 319776/400000 [00:27<00:06, 11927.19it/s] 80%|████████  | 321003/400000 [00:27<00:06, 12026.81it/s] 81%|████████  | 322207/400000 [00:27<00:06, 11962.78it/s] 81%|████████  | 323404/400000 [00:27<00:06, 11920.65it/s] 81%|████████  | 324597/400000 [00:27<00:06, 11410.77it/s] 81%|████████▏ | 325826/400000 [00:27<00:06, 11660.08it/s] 82%|████████▏ | 327052/400000 [00:27<00:06, 11832.88it/s] 82%|████████▏ | 328252/400000 [00:27<00:06, 11880.43it/s] 82%|████████▏ | 329471/400000 [00:28<00:05, 11968.83it/s] 83%|████████▎ | 330671/400000 [00:28<00:05, 11840.93it/s] 83%|████████▎ | 331891/400000 [00:28<00:05, 11946.40it/s] 83%|████████▎ | 333088/400000 [00:28<00:05, 11890.17it/s] 84%|████████▎ | 334287/400000 [00:28<00:05, 11918.51it/s] 84%|████████▍ | 335480/400000 [00:28<00:05, 11884.40it/s] 84%|████████▍ | 336670/400000 [00:28<00:05, 11517.72it/s] 84%|████████▍ | 337896/400000 [00:28<00:05, 11730.14it/s] 85%|████████▍ | 339123/400000 [00:28<00:05, 11885.81it/s] 85%|████████▌ | 340315/400000 [00:29<00:05, 11736.83it/s] 85%|████████▌ | 341508/400000 [00:29<00:04, 11793.22it/s] 86%|████████▌ | 342689/400000 [00:29<00:04, 11651.73it/s] 86%|████████▌ | 343928/400000 [00:29<00:04, 11863.67it/s] 86%|████████▋ | 345123/400000 [00:29<00:04, 11887.02it/s] 87%|████████▋ | 346354/400000 [00:29<00:04, 12010.23it/s] 87%|████████▋ | 347557/400000 [00:29<00:04, 11981.08it/s] 87%|████████▋ | 348757/400000 [00:29<00:04, 11761.87it/s] 87%|████████▋ | 349956/400000 [00:29<00:04, 11828.25it/s] 88%|████████▊ | 351177/400000 [00:29<00:04, 11939.03it/s] 88%|████████▊ | 352372/400000 [00:30<00:03, 11922.31it/s] 88%|████████▊ | 353584/400000 [00:30<00:03, 11979.95it/s] 89%|████████▊ | 354783/400000 [00:30<00:03, 11820.60it/s] 89%|████████▉ | 355966/400000 [00:30<00:03, 11789.93it/s] 89%|████████▉ | 357185/400000 [00:30<00:03, 11905.44it/s] 90%|████████▉ | 358377/400000 [00:30<00:03, 11837.80it/s] 90%|████████▉ | 359585/400000 [00:30<00:03, 11906.12it/s] 90%|█████████ | 360777/400000 [00:30<00:03, 11654.98it/s] 90%|█████████ | 361983/400000 [00:30<00:03, 11772.89it/s] 91%|█████████ | 363162/400000 [00:30<00:03, 11728.69it/s] 91%|█████████ | 364372/400000 [00:31<00:03, 11836.29it/s] 91%|█████████▏| 365582/400000 [00:31<00:02, 11911.92it/s] 92%|█████████▏| 366774/400000 [00:31<00:02, 11729.91it/s] 92%|█████████▏| 368007/400000 [00:31<00:02, 11902.40it/s] 92%|█████████▏| 369199/400000 [00:31<00:02, 11770.20it/s] 93%|█████████▎| 370409/400000 [00:31<00:02, 11865.29it/s] 93%|█████████▎| 371597/400000 [00:31<00:02, 11623.96it/s] 93%|█████████▎| 372762/400000 [00:31<00:02, 11567.53it/s] 93%|█████████▎| 373975/400000 [00:31<00:02, 11726.72it/s] 94%|█████████▍| 375172/400000 [00:31<00:02, 11798.60it/s] 94%|█████████▍| 376386/400000 [00:32<00:01, 11898.15it/s] 94%|█████████▍| 377605/400000 [00:32<00:01, 11983.62it/s] 95%|█████████▍| 378805/400000 [00:32<00:01, 11799.68it/s] 95%|█████████▌| 380006/400000 [00:32<00:01, 11859.77it/s] 95%|█████████▌| 381232/400000 [00:32<00:01, 11976.68it/s] 96%|█████████▌| 382447/400000 [00:32<00:01, 12027.20it/s] 96%|█████████▌| 383658/400000 [00:32<00:01, 12051.04it/s] 96%|█████████▌| 384864/400000 [00:32<00:01, 11928.42it/s] 97%|█████████▋| 386092/400000 [00:32<00:01, 12030.51it/s] 97%|█████████▋| 387322/400000 [00:32<00:01, 12107.44it/s] 97%|█████████▋| 388534/400000 [00:33<00:00, 12077.45it/s] 97%|█████████▋| 389743/400000 [00:33<00:00, 11998.75it/s] 98%|█████████▊| 390944/400000 [00:33<00:00, 11801.19it/s] 98%|█████████▊| 392154/400000 [00:33<00:00, 11888.37it/s] 98%|█████████▊| 393363/400000 [00:33<00:00, 11947.21it/s] 99%|█████████▊| 394559/400000 [00:33<00:00, 11875.71it/s] 99%|█████████▉| 395748/400000 [00:33<00:00, 11566.41it/s] 99%|█████████▉| 396907/400000 [00:33<00:00, 11444.52it/s]100%|█████████▉| 398054/400000 [00:33<00:00, 11251.64it/s]100%|█████████▉| 399284/400000 [00:33<00:00, 11544.54it/s]100%|█████████▉| 399999/400000 [00:34<00:00, 11745.79it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f595dc79b70>, <torchtext.data.dataset.TabularDataset object at 0x7f595dc79cc0>, <torchtext.vocab.Vocab object at 0x7f595dc79be0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.27 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.00 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.00 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.29 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.29 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.73 file/s]2020-07-03 18:15:55.539998: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-03 18:15:55.544133: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-07-03 18:15:55.544306: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555eb3be1200 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-03 18:15:55.544319: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3080192/9912422 [00:00<00:00, 30625467.59it/s]9920512it [00:00, 33793097.06it/s]                             
0it [00:00, ?it/s]32768it [00:00, 968951.26it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468949.02it/s]1654784it [00:00, 11685687.17it/s]                         
0it [00:00, ?it/s]8192it [00:00, 179158.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
