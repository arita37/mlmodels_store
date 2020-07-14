
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5ecab44048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5ecab44048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5f366ec1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5f366ec1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5f54a34ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5f54a34ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ee3a1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ee3a1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ee3a1a620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3203072/9912422 [00:00<00:00, 32025405.73it/s]9920512it [00:00, 31141236.31it/s]                             
0it [00:00, ?it/s]32768it [00:00, 503345.36it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160823.31it/s]1654784it [00:00, 11351661.86it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180003.24it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ee0c405f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ec15d0c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5ee3a1a268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5ee3a1a268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5ee3a1a268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:38,  1.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:38,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:37,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:37,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:36,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:36,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:35,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:43,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:43,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:43,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:42,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:20,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:20,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:20,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 34.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 42.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 44.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 44.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 46.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 46.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 48.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 48.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 50.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 50.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 51.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 51.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 52.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 52.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 53.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 52.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 52.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 52.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 52.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 52.59 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 51.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.40s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 51.92 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.44s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 51.92 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.44s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.44s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.79 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.44s/ url]
0 examples [00:00, ? examples/s]2020-07-14 00:08:49.595370: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-14 00:08:49.607827: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-14 00:08:49.608086: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bdf97227c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 00:08:49.608110: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
17 examples [00:00, 169.09 examples/s]116 examples [00:00, 224.95 examples/s]210 examples [00:00, 291.46 examples/s]310 examples [00:00, 369.75 examples/s]405 examples [00:00, 452.46 examples/s]506 examples [00:00, 541.57 examples/s]607 examples [00:00, 628.60 examples/s]703 examples [00:00, 700.00 examples/s]805 examples [00:00, 771.81 examples/s]905 examples [00:01, 828.47 examples/s]1001 examples [00:01, 841.69 examples/s]1101 examples [00:01, 881.77 examples/s]1197 examples [00:01, 903.69 examples/s]1294 examples [00:01, 922.02 examples/s]1392 examples [00:01, 938.46 examples/s]1493 examples [00:01, 956.85 examples/s]1591 examples [00:01, 960.09 examples/s]1689 examples [00:01, 952.09 examples/s]1790 examples [00:01, 966.60 examples/s]1891 examples [00:02, 976.91 examples/s]1990 examples [00:02, 976.11 examples/s]2088 examples [00:02, 969.32 examples/s]2186 examples [00:02, 964.53 examples/s]2289 examples [00:02, 982.00 examples/s]2389 examples [00:02, 985.55 examples/s]2488 examples [00:02, 984.42 examples/s]2587 examples [00:02, 976.98 examples/s]2685 examples [00:02, 964.55 examples/s]2786 examples [00:02, 975.23 examples/s]2884 examples [00:03, 972.87 examples/s]2982 examples [00:03, 974.99 examples/s]3086 examples [00:03, 991.86 examples/s]3186 examples [00:03, 982.07 examples/s]3287 examples [00:03, 987.51 examples/s]3389 examples [00:03, 996.01 examples/s]3489 examples [00:03, 997.17 examples/s]3595 examples [00:03, 1013.32 examples/s]3697 examples [00:03, 1004.75 examples/s]3801 examples [00:03, 1014.75 examples/s]3905 examples [00:04, 1020.30 examples/s]4008 examples [00:04, 987.05 examples/s] 4110 examples [00:04, 996.21 examples/s]4210 examples [00:04, 982.72 examples/s]4310 examples [00:04, 985.28 examples/s]4409 examples [00:04, 985.83 examples/s]4510 examples [00:04, 992.35 examples/s]4614 examples [00:04, 1005.17 examples/s]4715 examples [00:04, 998.17 examples/s] 4819 examples [00:04, 1010.10 examples/s]4924 examples [00:05, 1019.99 examples/s]5028 examples [00:05, 1024.57 examples/s]5133 examples [00:05, 1029.53 examples/s]5237 examples [00:05, 1006.59 examples/s]5341 examples [00:05, 1014.38 examples/s]5447 examples [00:05, 1025.63 examples/s]5550 examples [00:05, 1025.93 examples/s]5653 examples [00:05, 1018.13 examples/s]5755 examples [00:05, 1003.14 examples/s]5856 examples [00:05, 1002.25 examples/s]5961 examples [00:06, 1014.11 examples/s]6063 examples [00:06, 1013.31 examples/s]6166 examples [00:06, 1015.94 examples/s]6268 examples [00:06, 1010.40 examples/s]6372 examples [00:06, 1018.26 examples/s]6474 examples [00:06, 1015.95 examples/s]6578 examples [00:06, 1021.03 examples/s]6681 examples [00:06, 1011.57 examples/s]6785 examples [00:06, 1017.10 examples/s]6887 examples [00:07, 1008.56 examples/s]6989 examples [00:07, 1010.57 examples/s]7091 examples [00:07, 976.68 examples/s] 7191 examples [00:07, 982.73 examples/s]7291 examples [00:07, 984.96 examples/s]7395 examples [00:07, 999.10 examples/s]7499 examples [00:07, 1010.10 examples/s]7601 examples [00:07, 1010.55 examples/s]7703 examples [00:07, 1003.43 examples/s]7806 examples [00:07, 1009.62 examples/s]7908 examples [00:08, 999.40 examples/s] 8009 examples [00:08, 995.59 examples/s]8114 examples [00:08, 1009.69 examples/s]8216 examples [00:08, 1012.52 examples/s]8318 examples [00:08, 1002.75 examples/s]8419 examples [00:08, 998.66 examples/s] 8523 examples [00:08, 1008.67 examples/s]8624 examples [00:08, 993.81 examples/s] 8724 examples [00:08, 965.89 examples/s]8821 examples [00:08, 944.33 examples/s]8920 examples [00:09, 955.63 examples/s]9023 examples [00:09, 975.87 examples/s]9125 examples [00:09, 985.89 examples/s]9224 examples [00:09, 980.71 examples/s]9324 examples [00:09, 985.86 examples/s]9423 examples [00:09, 974.18 examples/s]9526 examples [00:09, 989.98 examples/s]9629 examples [00:09, 1000.43 examples/s]9730 examples [00:09, 1002.57 examples/s]9831 examples [00:09, 1002.77 examples/s]9934 examples [00:10, 1009.04 examples/s]10035 examples [00:10, 948.97 examples/s]10131 examples [00:10, 927.94 examples/s]10226 examples [00:10, 933.01 examples/s]10320 examples [00:10, 933.24 examples/s]10414 examples [00:10, 913.01 examples/s]10516 examples [00:10, 942.33 examples/s]10611 examples [00:10, 944.48 examples/s]10714 examples [00:10, 966.09 examples/s]10811 examples [00:11, 963.18 examples/s]10908 examples [00:11, 964.63 examples/s]11013 examples [00:11, 986.49 examples/s]11120 examples [00:11, 1008.32 examples/s]11222 examples [00:11, 994.03 examples/s] 11322 examples [00:11, 988.21 examples/s]11422 examples [00:11, 987.97 examples/s]11521 examples [00:11, 975.92 examples/s]11619 examples [00:11, 955.71 examples/s]11716 examples [00:11, 959.45 examples/s]11813 examples [00:12, 957.19 examples/s]11914 examples [00:12, 969.94 examples/s]12012 examples [00:12, 939.19 examples/s]12115 examples [00:12, 963.47 examples/s]12214 examples [00:12, 970.17 examples/s]12312 examples [00:12, 962.07 examples/s]12418 examples [00:12, 989.35 examples/s]12521 examples [00:12, 1000.42 examples/s]12623 examples [00:12, 1004.55 examples/s]12728 examples [00:12, 1015.52 examples/s]12830 examples [00:13, 1010.64 examples/s]12934 examples [00:13, 1017.73 examples/s]13037 examples [00:13, 1020.95 examples/s]13140 examples [00:13, 994.43 examples/s] 13243 examples [00:13, 1002.41 examples/s]13344 examples [00:13, 991.03 examples/s] 13444 examples [00:13, 990.98 examples/s]13545 examples [00:13, 967.28 examples/s]13645 examples [00:13, 975.44 examples/s]13747 examples [00:13, 986.59 examples/s]13846 examples [00:14, 973.51 examples/s]13946 examples [00:14, 979.42 examples/s]14045 examples [00:14, 979.05 examples/s]14143 examples [00:14, 971.22 examples/s]14244 examples [00:14, 980.35 examples/s]14343 examples [00:14, 977.70 examples/s]14447 examples [00:14, 994.63 examples/s]14548 examples [00:14, 996.59 examples/s]14654 examples [00:14, 1013.99 examples/s]14758 examples [00:15, 1020.35 examples/s]14861 examples [00:15, 1008.26 examples/s]14962 examples [00:15, 1003.16 examples/s]15066 examples [00:15, 1011.66 examples/s]15168 examples [00:15, 999.03 examples/s] 15269 examples [00:15, 1001.62 examples/s]15370 examples [00:15, 986.53 examples/s] 15469 examples [00:15, 981.30 examples/s]15568 examples [00:15, 974.52 examples/s]15667 examples [00:15, 977.33 examples/s]15765 examples [00:16, 973.59 examples/s]15863 examples [00:16, 945.28 examples/s]15960 examples [00:16, 952.10 examples/s]16056 examples [00:16, 905.99 examples/s]16148 examples [00:16, 902.20 examples/s]16242 examples [00:16, 912.54 examples/s]16339 examples [00:16, 926.92 examples/s]16443 examples [00:16, 957.20 examples/s]16547 examples [00:16, 979.28 examples/s]16652 examples [00:16, 997.13 examples/s]16755 examples [00:17, 1005.89 examples/s]16856 examples [00:17, 988.55 examples/s] 16956 examples [00:17, 956.57 examples/s]17053 examples [00:17, 938.28 examples/s]17148 examples [00:17, 937.44 examples/s]17245 examples [00:17, 944.08 examples/s]17341 examples [00:17, 947.75 examples/s]17436 examples [00:17, 948.13 examples/s]17538 examples [00:17, 968.59 examples/s]17639 examples [00:18, 978.51 examples/s]17739 examples [00:18, 982.94 examples/s]17843 examples [00:18, 996.98 examples/s]17943 examples [00:18, 995.32 examples/s]18043 examples [00:18, 991.77 examples/s]18143 examples [00:18, 985.49 examples/s]18242 examples [00:18, 958.92 examples/s]18342 examples [00:18, 970.75 examples/s]18440 examples [00:18, 971.34 examples/s]18541 examples [00:18, 982.32 examples/s]18640 examples [00:19, 976.59 examples/s]18738 examples [00:19, 962.50 examples/s]18839 examples [00:19, 974.85 examples/s]18938 examples [00:19, 977.41 examples/s]19036 examples [00:19, 968.91 examples/s]19136 examples [00:19, 977.89 examples/s]19234 examples [00:19, 941.23 examples/s]19329 examples [00:19, 934.73 examples/s]19428 examples [00:19, 948.26 examples/s]19525 examples [00:19, 952.16 examples/s]19626 examples [00:20, 966.89 examples/s]19723 examples [00:20, 967.30 examples/s]19825 examples [00:20, 980.11 examples/s]19924 examples [00:20, 974.34 examples/s]20022 examples [00:20, 910.00 examples/s]20116 examples [00:20, 916.92 examples/s]20212 examples [00:20, 928.67 examples/s]20313 examples [00:20, 949.42 examples/s]20409 examples [00:20, 948.34 examples/s]20511 examples [00:20, 966.99 examples/s]20612 examples [00:21, 977.49 examples/s]20710 examples [00:21, 972.01 examples/s]20808 examples [00:21, 962.68 examples/s]20905 examples [00:21, 951.23 examples/s]21003 examples [00:21, 957.83 examples/s]21099 examples [00:21, 943.03 examples/s]21195 examples [00:21, 945.76 examples/s]21293 examples [00:21, 955.28 examples/s]21389 examples [00:21, 919.46 examples/s]21482 examples [00:22, 918.40 examples/s]21575 examples [00:22, 905.54 examples/s]21666 examples [00:22, 880.62 examples/s]21756 examples [00:22, 884.82 examples/s]21845 examples [00:22, 875.01 examples/s]21942 examples [00:22, 900.89 examples/s]22033 examples [00:22, 860.50 examples/s]22120 examples [00:22, 855.13 examples/s]22218 examples [00:22, 886.82 examples/s]22321 examples [00:22, 924.07 examples/s]22423 examples [00:23, 948.32 examples/s]22519 examples [00:23, 945.02 examples/s]22619 examples [00:23, 958.23 examples/s]22716 examples [00:23, 957.76 examples/s]22823 examples [00:23, 985.68 examples/s]22922 examples [00:23, 983.65 examples/s]23021 examples [00:23, 982.68 examples/s]23120 examples [00:23, 959.01 examples/s]23222 examples [00:23, 973.18 examples/s]23323 examples [00:23, 981.49 examples/s]23428 examples [00:24, 1000.89 examples/s]23534 examples [00:24, 1017.44 examples/s]23641 examples [00:24, 1032.18 examples/s]23749 examples [00:24, 1045.94 examples/s]23858 examples [00:24, 1057.55 examples/s]23964 examples [00:24, 1046.57 examples/s]24069 examples [00:24, 1031.07 examples/s]24177 examples [00:24, 1044.23 examples/s]24285 examples [00:24, 1053.42 examples/s]24394 examples [00:24, 1064.03 examples/s]24501 examples [00:25, 1065.73 examples/s]24608 examples [00:25, 1054.43 examples/s]24714 examples [00:25, 1049.51 examples/s]24820 examples [00:25, 1024.11 examples/s]24926 examples [00:25, 1033.96 examples/s]25030 examples [00:25, 1018.61 examples/s]25133 examples [00:25, 979.85 examples/s] 25232 examples [00:25, 969.55 examples/s]25336 examples [00:25, 988.51 examples/s]25440 examples [00:26, 1001.58 examples/s]25546 examples [00:26, 1018.03 examples/s]25649 examples [00:26, 1019.72 examples/s]25757 examples [00:26, 1034.32 examples/s]25861 examples [00:26, 1026.96 examples/s]25967 examples [00:26, 1034.14 examples/s]26071 examples [00:26, 1031.71 examples/s]26175 examples [00:26, 1032.94 examples/s]26279 examples [00:26, 1034.18 examples/s]26383 examples [00:26, 1019.52 examples/s]26492 examples [00:27, 1038.52 examples/s]26599 examples [00:27, 1045.93 examples/s]26704 examples [00:27, 1030.90 examples/s]26808 examples [00:27, 1011.66 examples/s]26911 examples [00:27, 1015.04 examples/s]27018 examples [00:27, 1030.29 examples/s]27122 examples [00:27, 1022.98 examples/s]27225 examples [00:27, 1013.38 examples/s]27327 examples [00:27, 1014.25 examples/s]27433 examples [00:27, 1026.82 examples/s]27537 examples [00:28, 1029.82 examples/s]27641 examples [00:28, 1032.48 examples/s]27745 examples [00:28, 1017.25 examples/s]27849 examples [00:28, 1022.38 examples/s]27952 examples [00:28, 1023.58 examples/s]28055 examples [00:28, 1024.60 examples/s]28158 examples [00:28, 999.65 examples/s] 28259 examples [00:28, 986.67 examples/s]28364 examples [00:28, 1004.13 examples/s]28468 examples [00:28, 1014.41 examples/s]28574 examples [00:29, 1025.21 examples/s]28680 examples [00:29, 1034.01 examples/s]28784 examples [00:29, 1029.93 examples/s]28891 examples [00:29, 1040.63 examples/s]28997 examples [00:29, 1044.77 examples/s]29102 examples [00:29, 1041.80 examples/s]29208 examples [00:29, 1045.22 examples/s]29313 examples [00:29, 1031.13 examples/s]29419 examples [00:29, 1039.17 examples/s]29527 examples [00:30, 1049.78 examples/s]29633 examples [00:30, 1052.73 examples/s]29739 examples [00:30, 1039.13 examples/s]29843 examples [00:30, 1003.51 examples/s]29950 examples [00:30, 1020.04 examples/s]30053 examples [00:30, 972.59 examples/s] 30158 examples [00:30, 992.56 examples/s]30260 examples [00:30, 1000.31 examples/s]30361 examples [00:30, 988.92 examples/s] 30466 examples [00:30, 1006.15 examples/s]30574 examples [00:31, 1024.97 examples/s]30677 examples [00:31, 1026.04 examples/s]30782 examples [00:31, 1030.67 examples/s]30886 examples [00:31, 1018.25 examples/s]30990 examples [00:31, 1022.03 examples/s]31093 examples [00:31, 995.54 examples/s] 31198 examples [00:31, 1009.01 examples/s]31300 examples [00:31, 983.29 examples/s] 31399 examples [00:31, 980.57 examples/s]31505 examples [00:31, 1002.62 examples/s]31613 examples [00:32, 1022.91 examples/s]31716 examples [00:32, 1015.82 examples/s]31818 examples [00:32, 1003.64 examples/s]31919 examples [00:32, 992.66 examples/s] 32019 examples [00:32, 986.47 examples/s]32122 examples [00:32, 997.00 examples/s]32226 examples [00:32, 1008.58 examples/s]32332 examples [00:32, 1023.14 examples/s]32435 examples [00:32, 1012.19 examples/s]32541 examples [00:32, 1023.60 examples/s]32647 examples [00:33, 1030.79 examples/s]32751 examples [00:33, 1030.50 examples/s]32858 examples [00:33, 1039.59 examples/s]32963 examples [00:33, 1019.92 examples/s]33066 examples [00:33, 1012.84 examples/s]33172 examples [00:33, 1024.68 examples/s]33275 examples [00:33, 1025.62 examples/s]33378 examples [00:33, 998.76 examples/s] 33479 examples [00:33, 998.66 examples/s]33589 examples [00:34, 1026.77 examples/s]33697 examples [00:34, 1041.42 examples/s]33802 examples [00:34, 1039.53 examples/s]33907 examples [00:34, 1036.16 examples/s]34011 examples [00:34, 1033.82 examples/s]34116 examples [00:34, 1036.44 examples/s]34224 examples [00:34, 1046.37 examples/s]34329 examples [00:34, 1042.29 examples/s]34434 examples [00:34, 1000.28 examples/s]34545 examples [00:34, 1028.80 examples/s]34649 examples [00:35, 1022.19 examples/s]34752 examples [00:35, 998.96 examples/s] 34853 examples [00:35, 991.27 examples/s]34953 examples [00:35, 985.99 examples/s]35057 examples [00:35, 999.26 examples/s]35158 examples [00:35, 996.99 examples/s]35258 examples [00:35, 985.90 examples/s]35360 examples [00:35, 991.35 examples/s]35460 examples [00:35, 958.57 examples/s]35557 examples [00:35, 949.28 examples/s]35653 examples [00:36, 944.98 examples/s]35752 examples [00:36, 957.11 examples/s]35856 examples [00:36, 979.57 examples/s]35956 examples [00:36, 983.19 examples/s]36055 examples [00:36, 981.89 examples/s]36154 examples [00:36, 964.74 examples/s]36251 examples [00:36, 948.60 examples/s]36349 examples [00:36, 955.28 examples/s]36445 examples [00:36, 956.51 examples/s]36541 examples [00:37, 956.55 examples/s]36637 examples [00:37, 953.92 examples/s]36734 examples [00:37, 958.68 examples/s]36830 examples [00:37, 955.92 examples/s]36926 examples [00:37, 936.56 examples/s]37031 examples [00:37, 965.50 examples/s]37132 examples [00:37, 978.37 examples/s]37234 examples [00:37, 990.32 examples/s]37334 examples [00:37, 968.57 examples/s]37432 examples [00:37, 943.54 examples/s]37527 examples [00:38, 922.30 examples/s]37620 examples [00:38, 919.14 examples/s]37717 examples [00:38, 932.22 examples/s]37819 examples [00:38, 956.06 examples/s]37917 examples [00:38, 962.83 examples/s]38016 examples [00:38, 970.68 examples/s]38114 examples [00:38, 972.08 examples/s]38216 examples [00:38, 984.15 examples/s]38315 examples [00:38, 945.86 examples/s]38410 examples [00:38, 920.15 examples/s]38514 examples [00:39, 952.02 examples/s]38614 examples [00:39, 964.40 examples/s]38719 examples [00:39, 986.36 examples/s]38819 examples [00:39, 958.72 examples/s]38917 examples [00:39, 962.08 examples/s]39019 examples [00:39, 976.23 examples/s]39121 examples [00:39, 987.77 examples/s]39221 examples [00:39, 965.84 examples/s]39318 examples [00:39, 952.47 examples/s]39415 examples [00:40, 955.23 examples/s]39512 examples [00:40, 957.86 examples/s]39611 examples [00:40, 964.12 examples/s]39708 examples [00:40, 951.35 examples/s]39808 examples [00:40, 963.71 examples/s]39910 examples [00:40, 978.10 examples/s]40008 examples [00:40, 925.08 examples/s]40109 examples [00:40, 947.17 examples/s]40207 examples [00:40, 956.41 examples/s]40304 examples [00:40, 955.25 examples/s]40409 examples [00:41, 979.34 examples/s]40514 examples [00:41, 997.16 examples/s]40615 examples [00:41, 982.95 examples/s]40715 examples [00:41, 987.58 examples/s]40814 examples [00:41, 970.36 examples/s]40912 examples [00:41, 950.65 examples/s]41008 examples [00:41, 918.73 examples/s]41101 examples [00:41, 915.38 examples/s]41197 examples [00:41, 924.71 examples/s]41290 examples [00:41, 912.85 examples/s]41384 examples [00:42, 919.20 examples/s]41481 examples [00:42, 932.14 examples/s]41575 examples [00:42, 910.38 examples/s]41667 examples [00:42, 888.86 examples/s]41757 examples [00:42, 876.06 examples/s]41854 examples [00:42, 901.01 examples/s]41952 examples [00:42, 923.14 examples/s]42053 examples [00:42, 945.54 examples/s]42153 examples [00:42, 958.55 examples/s]42250 examples [00:43, 947.57 examples/s]42352 examples [00:43, 968.02 examples/s]42450 examples [00:43, 967.33 examples/s]42553 examples [00:43, 983.57 examples/s]42661 examples [00:43, 1008.18 examples/s]42763 examples [00:43, 1004.26 examples/s]42871 examples [00:43, 1024.93 examples/s]42979 examples [00:43, 1040.83 examples/s]43084 examples [00:43, 1025.53 examples/s]43187 examples [00:43, 995.59 examples/s] 43287 examples [00:44, 992.07 examples/s]43391 examples [00:44, 1005.52 examples/s]43492 examples [00:44, 1000.67 examples/s]43593 examples [00:44, 1001.98 examples/s]43699 examples [00:44, 1016.39 examples/s]43801 examples [00:44, 1008.75 examples/s]43905 examples [00:44, 1014.92 examples/s]44007 examples [00:44, 1016.42 examples/s]44109 examples [00:44, 1000.18 examples/s]44210 examples [00:44, 978.45 examples/s] 44309 examples [00:45, 958.98 examples/s]44406 examples [00:45, 955.27 examples/s]44502 examples [00:45, 956.36 examples/s]44601 examples [00:45, 963.85 examples/s]44701 examples [00:45, 972.86 examples/s]44799 examples [00:45, 952.89 examples/s]44895 examples [00:45, 930.84 examples/s]44998 examples [00:45, 955.31 examples/s]45100 examples [00:45, 972.67 examples/s]45206 examples [00:45, 994.50 examples/s]45308 examples [00:46, 1000.94 examples/s]45410 examples [00:46, 1003.59 examples/s]45511 examples [00:46, 983.64 examples/s] 45610 examples [00:46, 979.36 examples/s]45709 examples [00:46, 976.52 examples/s]45807 examples [00:46, 971.11 examples/s]45905 examples [00:46, 971.97 examples/s]46003 examples [00:46, 972.88 examples/s]46103 examples [00:46, 979.59 examples/s]46201 examples [00:47, 952.75 examples/s]46302 examples [00:47, 968.18 examples/s]46400 examples [00:47, 944.71 examples/s]46495 examples [00:47, 921.36 examples/s]46594 examples [00:47, 939.00 examples/s]46694 examples [00:47, 953.60 examples/s]46793 examples [00:47, 963.29 examples/s]46896 examples [00:47, 981.15 examples/s]46995 examples [00:47, 977.54 examples/s]47094 examples [00:47, 981.03 examples/s]47193 examples [00:48, 982.63 examples/s]47292 examples [00:48, 977.24 examples/s]47390 examples [00:48, 964.63 examples/s]47491 examples [00:48, 976.16 examples/s]47589 examples [00:48, 971.98 examples/s]47688 examples [00:48, 974.94 examples/s]47789 examples [00:48, 982.91 examples/s]47888 examples [00:48, 978.04 examples/s]47990 examples [00:48, 987.38 examples/s]48089 examples [00:48, 970.19 examples/s]48187 examples [00:49, 953.23 examples/s]48285 examples [00:49, 960.94 examples/s]48385 examples [00:49, 971.31 examples/s]48483 examples [00:49, 909.92 examples/s]48575 examples [00:49, 900.77 examples/s]48674 examples [00:49, 923.32 examples/s]48774 examples [00:49, 943.82 examples/s]48874 examples [00:49, 958.99 examples/s]48973 examples [00:49, 967.39 examples/s]49074 examples [00:49, 977.59 examples/s]49173 examples [00:50, 949.75 examples/s]49269 examples [00:50, 948.04 examples/s]49365 examples [00:50, 946.52 examples/s]49460 examples [00:50, 935.65 examples/s]49555 examples [00:50, 937.25 examples/s]49651 examples [00:50, 943.00 examples/s]49753 examples [00:50, 962.75 examples/s]49850 examples [00:50, 952.32 examples/s]49946 examples [00:50, 952.17 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7061/50000 [00:00<00:00, 70603.67 examples/s] 39%|███▉      | 19705/50000 [00:00<00:00, 81385.30 examples/s] 65%|██████▍   | 32267/50000 [00:00<00:00, 90997.61 examples/s] 91%|█████████ | 45321/50000 [00:00<00:00, 100090.81 examples/s]                                                                0 examples [00:00, ? examples/s]77 examples [00:00, 765.36 examples/s]167 examples [00:00, 800.06 examples/s]264 examples [00:00, 843.98 examples/s]361 examples [00:00, 875.95 examples/s]454 examples [00:00, 889.30 examples/s]551 examples [00:00, 911.03 examples/s]642 examples [00:00, 908.35 examples/s]740 examples [00:00, 928.68 examples/s]833 examples [00:00, 928.25 examples/s]924 examples [00:01, 920.26 examples/s]1022 examples [00:01, 936.44 examples/s]1115 examples [00:01, 925.42 examples/s]1207 examples [00:01, 920.31 examples/s]1310 examples [00:01, 948.78 examples/s]1409 examples [00:01, 960.18 examples/s]1512 examples [00:01, 977.91 examples/s]1610 examples [00:01, 958.00 examples/s]1710 examples [00:01, 969.41 examples/s]1808 examples [00:01, 951.61 examples/s]1904 examples [00:02, 930.52 examples/s]1998 examples [00:02, 905.54 examples/s]2090 examples [00:02, 909.11 examples/s]2182 examples [00:02, 902.68 examples/s]2273 examples [00:02, 896.96 examples/s]2367 examples [00:02, 906.68 examples/s]2458 examples [00:02, 896.43 examples/s]2553 examples [00:02, 910.60 examples/s]2655 examples [00:02, 939.03 examples/s]2759 examples [00:02, 965.12 examples/s]2863 examples [00:03, 985.36 examples/s]2972 examples [00:03, 1014.41 examples/s]3075 examples [00:03, 1017.50 examples/s]3178 examples [00:03, 1020.34 examples/s]3281 examples [00:03, 1020.29 examples/s]3384 examples [00:03, 1013.89 examples/s]3488 examples [00:03, 1018.79 examples/s]3590 examples [00:03, 1018.09 examples/s]3693 examples [00:03, 1020.33 examples/s]3796 examples [00:03, 1015.96 examples/s]3898 examples [00:04, 996.87 examples/s] 4000 examples [00:04, 1001.42 examples/s]4101 examples [00:04, 1002.22 examples/s]4203 examples [00:04, 1006.54 examples/s]4304 examples [00:04, 1004.23 examples/s]4413 examples [00:04, 1027.06 examples/s]4521 examples [00:04, 1041.94 examples/s]4626 examples [00:04, 1007.90 examples/s]4728 examples [00:04, 1001.26 examples/s]4829 examples [00:04, 999.14 examples/s] 4930 examples [00:05, 994.68 examples/s]5030 examples [00:05, 994.97 examples/s]5130 examples [00:05, 985.89 examples/s]5229 examples [00:05, 982.23 examples/s]5331 examples [00:05, 992.90 examples/s]5431 examples [00:05, 979.90 examples/s]5530 examples [00:05, 969.95 examples/s]5628 examples [00:05, 960.30 examples/s]5725 examples [00:05, 940.36 examples/s]5823 examples [00:06, 949.68 examples/s]5924 examples [00:06, 964.31 examples/s]6021 examples [00:06, 949.05 examples/s]6117 examples [00:06, 937.69 examples/s]6211 examples [00:06, 885.31 examples/s]6301 examples [00:06, 865.37 examples/s]6394 examples [00:06, 881.70 examples/s]6491 examples [00:06, 904.66 examples/s]6585 examples [00:06, 913.72 examples/s]6680 examples [00:06, 923.85 examples/s]6779 examples [00:07, 942.68 examples/s]6879 examples [00:07, 957.42 examples/s]6976 examples [00:07, 942.40 examples/s]7074 examples [00:07, 952.20 examples/s]7173 examples [00:07, 963.11 examples/s]7274 examples [00:07, 975.54 examples/s]7376 examples [00:07, 988.47 examples/s]7475 examples [00:07, 974.70 examples/s]7573 examples [00:07, 883.79 examples/s]7670 examples [00:08, 907.10 examples/s]7773 examples [00:08, 940.30 examples/s]7869 examples [00:08, 940.14 examples/s]7964 examples [00:08, 937.08 examples/s]8063 examples [00:08, 951.53 examples/s]8161 examples [00:08, 959.47 examples/s]8262 examples [00:08, 973.77 examples/s]8362 examples [00:08, 980.62 examples/s]8461 examples [00:08, 973.13 examples/s]8559 examples [00:08, 969.72 examples/s]8663 examples [00:09, 987.54 examples/s]8765 examples [00:09, 994.78 examples/s]8865 examples [00:09, 984.80 examples/s]8964 examples [00:09, 946.79 examples/s]9060 examples [00:09, 940.73 examples/s]9166 examples [00:09, 973.54 examples/s]9272 examples [00:09, 995.33 examples/s]9372 examples [00:09, 976.16 examples/s]9472 examples [00:09, 982.20 examples/s]9577 examples [00:09, 1000.10 examples/s]9679 examples [00:10, 1003.49 examples/s]9785 examples [00:10, 1019.23 examples/s]9890 examples [00:10, 1025.05 examples/s]9993 examples [00:10, 1001.19 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9L3DDS/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9L3DDS/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ee3a1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ee3a1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ee3a1a620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e6ceb4588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e6cef1d30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ee3a1a620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ee3a1a620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ee3a1a620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e6cf2cef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ec15d0358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5e5b4d52f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5e5b4d52f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f5e5b4d52f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5e5b4d5400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5e5b4d5400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5e5b4d5400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=30b70d7951dde8f0a748077b859edf1ca196818e80725c060df8a6f49bfa9b3d
  Stored in directory: /tmp/pip-ephem-wheel-cache-8zsnk4g3/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:11:13, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:57:43, 18.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:07:38, 26.2kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:23:53, 37.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:28:04, 53.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.81M/862M [00:01<3:06:36, 76.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:10:08, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:30:38, 155kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:01<1:03:19, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:01<44:09, 316kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<30:53, 449kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:01<21:36, 639kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:01<15:08, 907kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:02<10:38, 1.28MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<07:30, 1.81MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<05:19, 2.54MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<05:45, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:02<04:07, 3.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<36:46, 365kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<27:58, 480kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<20:07, 666kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:05<14:13, 940kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<20:58, 637kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<16:19, 818kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<11:46, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:07<08:23, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<58:51, 226kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<44:24, 299kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<31:45, 418kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<22:20, 592kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<21:04, 627kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<16:20, 809kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<11:48, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<10:58, 1.20MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<09:07, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<06:41, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:31, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<07:55, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<06:12, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:26, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<05:51, 2.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:16<04:25, 2.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<06:06, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<06:52, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<05:22, 2.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<03:53, 3.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<13:18, 967kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<10:38, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:20<07:43, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<08:20, 1.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<08:27, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:22<06:30, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:22<04:40, 2.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<16:38, 765kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<12:57, 983kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:24<09:22, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:31, 1.33MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:20, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:06, 1.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:14, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:06, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:16, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:39, 2.70MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:11, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:52, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:26, 2.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:49, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:07, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<03:57, 3.14MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<02:54, 4.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<52:01, 238kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<37:36, 329kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<26:35, 465kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:29, 574kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<17:31, 703kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<12:48, 961kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<09:04, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<17:18, 708kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<13:08, 932kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<09:40, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<06:52, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<53:56, 226kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<40:13, 303kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<28:44, 424kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<22:00, 551kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:38, 728kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<11:56, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:08, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:59, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:33, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:25, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:47, 2.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:09, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:06, 2.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:09, 2.29MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:52, 3.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:30, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:46, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:12, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:55, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:19, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:56, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:44, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:18, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:04, 1.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:44, 2.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:27, 3.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:33, 1.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:00, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:08, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:45, 3.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<27:19, 418kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<21:27, 533kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<15:29, 737kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<10:58, 1.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<12:48, 888kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<10:07, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:22, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:46, 1.45MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<07:44, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:58, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:59, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:22, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:59, 2.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:24, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:03, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:26, 3.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<21:35, 514kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<16:16, 681kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<11:39, 949kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:11<08:16, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<46:44, 236kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<33:48, 326kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<23:53, 460kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<19:15, 569kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<14:35, 750kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<10:32, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:15<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:37:50, 111kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:09:19, 157kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<48:50, 222kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<34:10, 317kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:04:48, 167kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<47:31, 228kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<33:43, 320kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 216M/862M [01:19<23:48, 453kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<19:32, 550kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<14:47, 726kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<10:35, 1.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:53, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:05, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:48, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<05:00, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:17, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:28, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:05, 2.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:19, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:56, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:41, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<03:24, 3.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<1:17:24, 135kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<55:13, 190kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<38:47, 269kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<29:31, 353kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<21:42, 479kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<15:25, 673kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<13:12, 783kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:16, 1.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<07:23, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:36, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:18, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:02, 2.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:32, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:37, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:09, 2.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:16, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:44, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:31, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:47, 2.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:21, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:18, 3.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:15, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:13, 3.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:36, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<03:09, 3.13MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:33, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:11, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:08, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:28, 2.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:06, 2.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:07, 3.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:59, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:59, 2.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:20, 2.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:01, 2.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:03, 3.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:23, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:02, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:03, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:23, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:02, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:01, 3.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:21, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:00, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:01, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:59, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<03:01, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:19, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:56, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<02:56, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:55, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:58, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:14, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:53, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:55, 3.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:10, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:45, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:42, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<02:42, 3.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:08, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:55, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<04:22, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:09, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:24, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:14, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:23, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:58, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:58, 2.98MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:09, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:48, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<02:52, 3.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:49, 3.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:34, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:37, 2.39MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:55, 2.19MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:37, 2.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:44, 3.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:36, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:43, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:55, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:26, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:32, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:49, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:31, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:40, 3.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:31, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:38, 3.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:47, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:20, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:23, 2.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:27, 3.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:33, 860kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:22, 1.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:22, 1.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<03:51, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<51:18, 159kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<37:32, 217kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<26:38, 305kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<18:37, 434kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<1:11:42, 113kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<50:58, 158kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<35:44, 225kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<26:47, 299kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<19:32, 409kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<13:48, 577kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<11:31, 689kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<09:39, 821kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:05, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<05:02, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<08:00, 983kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:15, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:32, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<03:17, 2.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<44:39, 175kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<32:49, 238kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<23:19, 334kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<17:28, 442kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<13:01, 593kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:16, 830kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:14, 929kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<06:33, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:45, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:06, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:07, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:57, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:58, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:32, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:39, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:09, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:55<02:16, 3.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<23:18, 317kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<17:03, 433kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<12:05, 609kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<10:08, 722kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<08:34, 854kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<06:21, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [02:59<04:29, 1.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<16:23, 442kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<12:12, 593kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<08:42, 829kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<07:44, 929kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<06:51, 1.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:09, 1.39MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<03:40, 1.94MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:25, 845kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:36, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:47, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:59, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:55, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:47, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:42, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<10:22, 673kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<07:58, 874kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<05:44, 1.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:36, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:18, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:03, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:55, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:34, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:23, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:44, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:54, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:06, 3.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:53, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:47, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:30, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:00, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:05, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:08, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:17, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:12, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:37, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:41, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:24, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:02, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<02:15, 2.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:05, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:27, 1.86MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:41, 2.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<01:57, 3.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<05:35, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:34, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:20, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:47, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:55, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:00, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:11, 2.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:04, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:36, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:14, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:30, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:45, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:54, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:39, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:00, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:47, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:14, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:32, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<01:50, 3.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:30, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<04:28, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:38, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:44, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:50, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:09, 2.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:51, 2.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:35, 2.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:56, 2.97MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:41, 2.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:27, 2.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:50, 3.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:36, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:23, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:47, 3.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:56, 1.91MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:20, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:48<01:40, 3.29MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<41:17, 134kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<29:26, 188kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<20:39, 267kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<15:38, 350kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<12:02, 454kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<08:41, 628kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:54, 782kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:22, 1.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:52, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:25, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:56, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:06, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:26, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:32, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:17, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:42, 3.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:24, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:40, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:05, 2.44MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:31, 3.33MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:01, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:19, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:25, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:51, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:00, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:18, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:40, 2.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:17, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:03, 2.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:34, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:46, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:08, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:33, 3.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:38, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:02, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:14, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:39, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:20, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:44, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:18, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:35, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:00, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:30, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:18, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:04, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:32, 2.94MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:08, 2.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:24, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:54, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:02, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:25, 3.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:00, 2.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:17, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:48, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:17, 3.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<37:07, 116kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<26:23, 163kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<18:27, 231kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<13:48, 307kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<10:30, 402kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<07:32, 559kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<05:53, 707kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<04:33, 912kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<03:16, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<02:19, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<17:50, 230kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<12:53, 317kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<09:02, 449kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<07:13, 557kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<05:27, 737kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<03:53, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:38, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:20, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:31, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:22, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:03, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:57, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:45, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:18, 2.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:02, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:35, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:08, 3.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:41, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:15, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:39, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:59, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:09, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<01:12, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<13:42, 259kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<09:53, 358kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<06:58, 505kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<04:52, 713kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<12:04, 288kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<09:08, 380kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<06:31, 531kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<04:33, 751kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<05:05, 670kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:51, 884kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:45, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:34, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:57, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:53<01:23, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<3:10:44, 17.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<2:13:34, 24.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:32:50, 34.9kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<1:05:00, 49.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<46:07, 69.4kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<32:17, 98.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<22:19, 141kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<19:08, 164kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<13:42, 228kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<09:34, 324kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<07:21, 417kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<05:26, 562kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<03:50, 790kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:22, 888kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:39, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:55, 1.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:01, 1.45MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:00, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:31, 1.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:06, 2.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:45, 1.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:31, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:07, 2.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:25, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:33, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:12, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:52, 3.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:40, 1.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:26, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:04, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:29, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:09, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:50, 3.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:45, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:29, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:05, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:20, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:27, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:07, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:48, 3.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:27, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:16, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:56, 2.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:12, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:05, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:48, 2.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:06, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:00, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:45, 3.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:03, 2.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:11, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:56, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:39, 3.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<08:24, 259kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<06:03, 358kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<04:14, 505kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<03:24, 617kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:48, 749kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<02:03, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:44, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:25, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:01, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:10, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:13, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:56, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:40, 2.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:12, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:02, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:45, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:57, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:03, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:48, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:36, 2.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:46, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:34, 3.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:47, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:53, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:42, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:30, 3.25MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<06:14, 261kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<04:30, 359kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:08, 507kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:31, 619kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:54, 811kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:21, 1.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:16, 1.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:02, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:45, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:52, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:28, 2.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:12, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:58, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:42, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:46, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:40, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:29, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:40, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:31, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:22, 3.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<1:06:38, 17.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<46:30, 24.6kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<31:50, 35.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<21:49, 49.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<15:27, 69.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<10:44, 99.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<07:10, 141kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<12:15, 82.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<08:37, 117kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<05:54, 166kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<04:12, 224kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<03:07, 301kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<02:12, 421kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:35, 548kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:17, 676kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:55, 926kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:37, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:26, 559kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:05, 738kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:45, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:40, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:37, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:27, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:18, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:31, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:25, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:10, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<03:59, 133kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<02:48, 187kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<01:52, 265kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<01:19, 348kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:57, 474kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:38, 667kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:30, 776kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:23, 997kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:15, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:14, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:14, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:10, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:04, 2.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:02, 3.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.26MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 34/400000 [00:00<19:37, 339.76it/s]  0%|          | 810/400000 [00:00<13:57, 476.43it/s]  0%|          | 1631/400000 [00:00<09:59, 664.09it/s]  1%|          | 2425/400000 [00:00<07:14, 915.87it/s]  1%|          | 3173/400000 [00:00<05:19, 1243.13it/s]  1%|          | 3918/400000 [00:00<03:58, 1657.30it/s]  1%|          | 4719/400000 [00:00<03:01, 2174.62it/s]  1%|▏         | 5526/400000 [00:00<02:21, 2784.73it/s]  2%|▏         | 6324/400000 [00:00<01:53, 3459.99it/s]  2%|▏         | 7085/400000 [00:01<01:34, 4136.30it/s]  2%|▏         | 7855/400000 [00:01<01:21, 4802.71it/s]  2%|▏         | 8611/400000 [00:01<01:12, 5386.06it/s]  2%|▏         | 9381/400000 [00:01<01:06, 5918.36it/s]  3%|▎         | 10140/400000 [00:01<01:01, 6336.85it/s]  3%|▎         | 10899/400000 [00:01<00:59, 6566.01it/s]  3%|▎         | 11646/400000 [00:01<00:57, 6754.12it/s]  3%|▎         | 12386/400000 [00:01<00:56, 6902.81it/s]  3%|▎         | 13123/400000 [00:01<00:55, 7033.49it/s]  3%|▎         | 13930/400000 [00:01<00:52, 7315.18it/s]  4%|▎         | 14712/400000 [00:02<00:51, 7459.02it/s]  4%|▍         | 15477/400000 [00:02<00:51, 7499.28it/s]  4%|▍         | 16241/400000 [00:02<00:51, 7444.68it/s]  4%|▍         | 16995/400000 [00:02<00:52, 7342.78it/s]  4%|▍         | 17758/400000 [00:02<00:51, 7424.62it/s]  5%|▍         | 18506/400000 [00:02<00:51, 7429.63it/s]  5%|▍         | 19266/400000 [00:02<00:50, 7477.32it/s]  5%|▌         | 20031/400000 [00:02<00:50, 7526.39it/s]  5%|▌         | 20833/400000 [00:02<00:49, 7666.10it/s]  5%|▌         | 21613/400000 [00:02<00:49, 7704.04it/s]  6%|▌         | 22385/400000 [00:03<00:51, 7362.18it/s]  6%|▌         | 23126/400000 [00:03<00:52, 7228.35it/s]  6%|▌         | 23853/400000 [00:03<00:52, 7124.02it/s]  6%|▌         | 24569/400000 [00:03<00:52, 7114.92it/s]  6%|▋         | 25360/400000 [00:03<00:51, 7329.02it/s]  7%|▋         | 26096/400000 [00:03<00:51, 7230.66it/s]  7%|▋         | 26848/400000 [00:03<00:51, 7312.55it/s]  7%|▋         | 27659/400000 [00:03<00:49, 7534.73it/s]  7%|▋         | 28497/400000 [00:03<00:47, 7769.32it/s]  7%|▋         | 29303/400000 [00:03<00:47, 7852.83it/s]  8%|▊         | 30141/400000 [00:04<00:46, 8003.21it/s]  8%|▊         | 30951/400000 [00:04<00:45, 8030.81it/s]  8%|▊         | 31757/400000 [00:04<00:46, 7958.30it/s]  8%|▊         | 32555/400000 [00:04<00:46, 7903.39it/s]  8%|▊         | 33347/400000 [00:04<00:46, 7898.95it/s]  9%|▊         | 34158/400000 [00:04<00:45, 7960.06it/s]  9%|▊         | 34955/400000 [00:04<00:46, 7899.09it/s]  9%|▉         | 35759/400000 [00:04<00:45, 7940.05it/s]  9%|▉         | 36607/400000 [00:04<00:44, 8093.47it/s]  9%|▉         | 37466/400000 [00:04<00:44, 8235.69it/s] 10%|▉         | 38300/400000 [00:05<00:43, 8265.15it/s] 10%|▉         | 39128/400000 [00:05<00:44, 8192.24it/s] 10%|▉         | 39985/400000 [00:05<00:43, 8301.34it/s] 10%|█         | 40881/400000 [00:05<00:42, 8487.59it/s] 10%|█         | 41784/400000 [00:05<00:41, 8640.11it/s] 11%|█         | 42657/400000 [00:05<00:41, 8665.27it/s] 11%|█         | 43525/400000 [00:05<00:41, 8535.11it/s] 11%|█         | 44380/400000 [00:05<00:42, 8354.12it/s] 11%|█▏        | 45219/400000 [00:05<00:42, 8364.02it/s] 12%|█▏        | 46092/400000 [00:05<00:41, 8469.37it/s] 12%|█▏        | 46941/400000 [00:06<00:43, 8201.12it/s] 12%|█▏        | 47764/400000 [00:06<00:43, 8115.67it/s] 12%|█▏        | 48578/400000 [00:06<00:43, 8065.09it/s] 12%|█▏        | 49442/400000 [00:06<00:42, 8227.36it/s] 13%|█▎        | 50300/400000 [00:06<00:41, 8329.70it/s] 13%|█▎        | 51135/400000 [00:06<00:42, 8268.49it/s] 13%|█▎        | 51968/400000 [00:06<00:41, 8286.84it/s] 13%|█▎        | 52798/400000 [00:06<00:42, 8205.51it/s] 13%|█▎        | 53633/400000 [00:06<00:42, 8246.36it/s] 14%|█▎        | 54474/400000 [00:07<00:41, 8294.60it/s] 14%|█▍        | 55304/400000 [00:07<00:42, 8204.20it/s] 14%|█▍        | 56125/400000 [00:07<00:42, 8086.91it/s] 14%|█▍        | 56935/400000 [00:07<00:42, 8028.10it/s] 14%|█▍        | 57777/400000 [00:07<00:42, 8139.91it/s] 15%|█▍        | 58592/400000 [00:07<00:42, 7977.81it/s] 15%|█▍        | 59422/400000 [00:07<00:42, 8070.10it/s] 15%|█▌        | 60231/400000 [00:07<00:42, 7935.51it/s] 15%|█▌        | 61051/400000 [00:07<00:42, 8012.76it/s] 15%|█▌        | 61876/400000 [00:07<00:41, 8081.69it/s] 16%|█▌        | 62700/400000 [00:08<00:41, 8126.92it/s] 16%|█▌        | 63542/400000 [00:08<00:40, 8212.12it/s] 16%|█▌        | 64388/400000 [00:08<00:40, 8281.51it/s] 16%|█▋        | 65236/400000 [00:08<00:40, 8338.51it/s] 17%|█▋        | 66093/400000 [00:08<00:39, 8405.39it/s] 17%|█▋        | 66935/400000 [00:08<00:39, 8391.12it/s] 17%|█▋        | 67775/400000 [00:08<00:40, 8247.27it/s] 17%|█▋        | 68601/400000 [00:08<00:40, 8112.58it/s] 17%|█▋        | 69414/400000 [00:08<00:41, 8046.87it/s] 18%|█▊        | 70220/400000 [00:08<00:41, 8038.45it/s] 18%|█▊        | 71040/400000 [00:09<00:40, 8085.87it/s] 18%|█▊        | 71850/400000 [00:09<00:41, 7946.29it/s] 18%|█▊        | 72646/400000 [00:09<00:41, 7903.00it/s] 18%|█▊        | 73447/400000 [00:09<00:41, 7933.04it/s] 19%|█▊        | 74294/400000 [00:09<00:40, 8085.63it/s] 19%|█▉        | 75194/400000 [00:09<00:38, 8337.23it/s] 19%|█▉        | 76031/400000 [00:09<00:39, 8227.38it/s] 19%|█▉        | 76856/400000 [00:09<00:39, 8126.61it/s] 19%|█▉        | 77681/400000 [00:09<00:39, 8161.14it/s] 20%|█▉        | 78499/400000 [00:09<00:40, 7909.49it/s] 20%|█▉        | 79293/400000 [00:10<00:41, 7730.97it/s] 20%|██        | 80085/400000 [00:10<00:41, 7785.76it/s] 20%|██        | 80866/400000 [00:10<00:41, 7767.63it/s] 20%|██        | 81650/400000 [00:10<00:40, 7788.19it/s] 21%|██        | 82440/400000 [00:10<00:40, 7820.38it/s] 21%|██        | 83242/400000 [00:10<00:40, 7878.18it/s] 21%|██        | 84031/400000 [00:10<00:40, 7817.70it/s] 21%|██        | 84814/400000 [00:10<00:40, 7694.34it/s] 21%|██▏       | 85595/400000 [00:10<00:40, 7726.60it/s] 22%|██▏       | 86414/400000 [00:10<00:39, 7857.69it/s] 22%|██▏       | 87206/400000 [00:11<00:39, 7876.20it/s] 22%|██▏       | 88052/400000 [00:11<00:38, 8042.46it/s] 22%|██▏       | 88858/400000 [00:11<00:39, 7798.45it/s] 22%|██▏       | 89713/400000 [00:11<00:38, 8008.18it/s] 23%|██▎       | 90518/400000 [00:11<00:38, 7955.31it/s] 23%|██▎       | 91317/400000 [00:11<00:38, 7965.04it/s] 23%|██▎       | 92116/400000 [00:11<00:38, 7947.15it/s] 23%|██▎       | 92912/400000 [00:11<00:39, 7794.85it/s] 23%|██▎       | 93712/400000 [00:11<00:38, 7854.66it/s] 24%|██▎       | 94522/400000 [00:12<00:38, 7925.35it/s] 24%|██▍       | 95339/400000 [00:12<00:38, 7995.41it/s] 24%|██▍       | 96140/400000 [00:12<00:38, 7949.52it/s] 24%|██▍       | 96936/400000 [00:12<00:38, 7772.65it/s] 24%|██▍       | 97720/400000 [00:12<00:38, 7791.64it/s] 25%|██▍       | 98502/400000 [00:12<00:38, 7799.24it/s] 25%|██▍       | 99292/400000 [00:12<00:38, 7828.76it/s] 25%|██▌       | 100076/400000 [00:12<00:38, 7786.06it/s] 25%|██▌       | 100855/400000 [00:12<00:38, 7706.48it/s] 25%|██▌       | 101688/400000 [00:12<00:37, 7882.61it/s] 26%|██▌       | 102535/400000 [00:13<00:36, 8048.23it/s] 26%|██▌       | 103342/400000 [00:13<00:36, 8024.95it/s] 26%|██▌       | 104146/400000 [00:13<00:36, 7998.49it/s] 26%|██▌       | 104947/400000 [00:13<00:37, 7949.23it/s] 26%|██▋       | 105757/400000 [00:13<00:36, 7991.55it/s] 27%|██▋       | 106557/400000 [00:13<00:37, 7901.07it/s] 27%|██▋       | 107348/400000 [00:13<00:37, 7746.08it/s] 27%|██▋       | 108124/400000 [00:13<00:38, 7607.73it/s] 27%|██▋       | 108889/400000 [00:13<00:38, 7619.67it/s] 27%|██▋       | 109700/400000 [00:13<00:37, 7760.28it/s] 28%|██▊       | 110478/400000 [00:14<00:37, 7739.51it/s] 28%|██▊       | 111322/400000 [00:14<00:36, 7932.78it/s] 28%|██▊       | 112118/400000 [00:14<00:36, 7882.97it/s] 28%|██▊       | 112908/400000 [00:14<00:36, 7819.33it/s] 28%|██▊       | 113729/400000 [00:14<00:36, 7931.71it/s] 29%|██▊       | 114524/400000 [00:14<00:36, 7861.77it/s] 29%|██▉       | 115350/400000 [00:14<00:35, 7974.63it/s] 29%|██▉       | 116168/400000 [00:14<00:35, 8028.33it/s] 29%|██▉       | 116972/400000 [00:14<00:35, 7889.63it/s] 29%|██▉       | 117871/400000 [00:14<00:34, 8189.78it/s] 30%|██▉       | 118694/400000 [00:15<00:34, 8129.70it/s] 30%|██▉       | 119510/400000 [00:15<00:34, 8031.25it/s] 30%|███       | 120316/400000 [00:15<00:34, 7992.23it/s] 30%|███       | 121117/400000 [00:15<00:35, 7832.67it/s] 30%|███       | 121948/400000 [00:15<00:34, 7967.68it/s] 31%|███       | 122759/400000 [00:15<00:34, 8008.54it/s] 31%|███       | 123625/400000 [00:15<00:33, 8190.92it/s] 31%|███       | 124447/400000 [00:15<00:33, 8134.28it/s] 31%|███▏      | 125262/400000 [00:15<00:34, 8016.93it/s] 32%|███▏      | 126138/400000 [00:15<00:33, 8226.11it/s] 32%|███▏      | 126963/400000 [00:16<00:33, 8065.55it/s] 32%|███▏      | 127787/400000 [00:16<00:34, 7969.74it/s] 32%|███▏      | 128586/400000 [00:16<00:34, 7832.92it/s] 32%|███▏      | 129379/400000 [00:16<00:34, 7859.78it/s] 33%|███▎      | 130173/400000 [00:16<00:34, 7882.83it/s] 33%|███▎      | 130980/400000 [00:16<00:33, 7935.69it/s] 33%|███▎      | 131795/400000 [00:16<00:33, 7998.41it/s] 33%|███▎      | 132607/400000 [00:16<00:33, 8031.98it/s] 33%|███▎      | 133411/400000 [00:16<00:33, 7936.57it/s] 34%|███▎      | 134213/400000 [00:17<00:33, 7958.36it/s] 34%|███▍      | 135019/400000 [00:17<00:33, 7986.28it/s] 34%|███▍      | 135844/400000 [00:17<00:32, 8063.08it/s] 34%|███▍      | 136700/400000 [00:17<00:32, 8205.76it/s] 34%|███▍      | 137522/400000 [00:17<00:32, 8115.78it/s] 35%|███▍      | 138335/400000 [00:17<00:32, 8089.29it/s] 35%|███▍      | 139167/400000 [00:17<00:31, 8154.64it/s] 35%|███▌      | 140005/400000 [00:17<00:31, 8220.17it/s] 35%|███▌      | 140828/400000 [00:17<00:31, 8201.96it/s] 35%|███▌      | 141649/400000 [00:17<00:31, 8113.30it/s] 36%|███▌      | 142465/400000 [00:18<00:31, 8124.62it/s] 36%|███▌      | 143325/400000 [00:18<00:31, 8259.85it/s] 36%|███▌      | 144189/400000 [00:18<00:30, 8367.67it/s] 36%|███▋      | 145056/400000 [00:18<00:30, 8454.70it/s] 36%|███▋      | 145932/400000 [00:18<00:29, 8542.63it/s] 37%|███▋      | 146788/400000 [00:18<00:29, 8517.77it/s] 37%|███▋      | 147663/400000 [00:18<00:29, 8585.68it/s] 37%|███▋      | 148547/400000 [00:18<00:29, 8658.25it/s] 37%|███▋      | 149414/400000 [00:18<00:29, 8598.73it/s] 38%|███▊      | 150288/400000 [00:18<00:28, 8638.33it/s] 38%|███▊      | 151178/400000 [00:19<00:28, 8714.85it/s] 38%|███▊      | 152054/400000 [00:19<00:28, 8727.09it/s] 38%|███▊      | 152928/400000 [00:19<00:28, 8629.77it/s] 38%|███▊      | 153813/400000 [00:19<00:28, 8692.08it/s] 39%|███▊      | 154692/400000 [00:19<00:28, 8718.46it/s] 39%|███▉      | 155565/400000 [00:19<00:28, 8620.36it/s] 39%|███▉      | 156462/400000 [00:19<00:27, 8719.61it/s] 39%|███▉      | 157335/400000 [00:19<00:28, 8634.33it/s] 40%|███▉      | 158214/400000 [00:19<00:27, 8678.95it/s] 40%|███▉      | 159084/400000 [00:19<00:27, 8684.20it/s] 40%|███▉      | 159953/400000 [00:20<00:27, 8580.30it/s] 40%|████      | 160812/400000 [00:20<00:28, 8337.26it/s] 40%|████      | 161650/400000 [00:20<00:28, 8349.32it/s] 41%|████      | 162494/400000 [00:20<00:28, 8371.34it/s] 41%|████      | 163390/400000 [00:20<00:27, 8537.33it/s] 41%|████      | 164266/400000 [00:20<00:27, 8601.25it/s] 41%|████▏     | 165151/400000 [00:20<00:27, 8672.24it/s] 42%|████▏     | 166020/400000 [00:20<00:27, 8626.66it/s] 42%|████▏     | 166884/400000 [00:20<00:27, 8417.92it/s] 42%|████▏     | 167728/400000 [00:20<00:27, 8350.28it/s] 42%|████▏     | 168595/400000 [00:21<00:27, 8442.59it/s] 42%|████▏     | 169520/400000 [00:21<00:26, 8667.14it/s] 43%|████▎     | 170395/400000 [00:21<00:26, 8690.43it/s] 43%|████▎     | 171268/400000 [00:21<00:26, 8700.37it/s] 43%|████▎     | 172141/400000 [00:21<00:26, 8708.62it/s] 43%|████▎     | 173013/400000 [00:21<00:26, 8711.42it/s] 43%|████▎     | 173885/400000 [00:21<00:25, 8702.77it/s] 44%|████▎     | 174756/400000 [00:21<00:26, 8616.44it/s] 44%|████▍     | 175619/400000 [00:21<00:26, 8565.69it/s] 44%|████▍     | 176486/400000 [00:21<00:26, 8595.20it/s] 44%|████▍     | 177346/400000 [00:22<00:26, 8439.90it/s] 45%|████▍     | 178258/400000 [00:22<00:25, 8632.47it/s] 45%|████▍     | 179132/400000 [00:22<00:25, 8663.90it/s] 45%|████▌     | 180010/400000 [00:22<00:25, 8695.90it/s] 45%|████▌     | 180915/400000 [00:22<00:24, 8796.37it/s] 45%|████▌     | 181796/400000 [00:22<00:25, 8710.73it/s] 46%|████▌     | 182723/400000 [00:22<00:24, 8870.54it/s] 46%|████▌     | 183623/400000 [00:22<00:24, 8907.95it/s] 46%|████▌     | 184515/400000 [00:22<00:24, 8767.32it/s] 46%|████▋     | 185393/400000 [00:22<00:24, 8676.21it/s] 47%|████▋     | 186262/400000 [00:23<00:25, 8535.99it/s] 47%|████▋     | 187137/400000 [00:23<00:24, 8598.27it/s] 47%|████▋     | 188030/400000 [00:23<00:24, 8693.29it/s] 47%|████▋     | 188901/400000 [00:23<00:24, 8653.59it/s] 47%|████▋     | 189768/400000 [00:23<00:24, 8608.67it/s] 48%|████▊     | 190644/400000 [00:23<00:24, 8651.09it/s] 48%|████▊     | 191543/400000 [00:23<00:23, 8749.53it/s] 48%|████▊     | 192419/400000 [00:23<00:23, 8696.32it/s] 48%|████▊     | 193328/400000 [00:23<00:23, 8809.31it/s] 49%|████▊     | 194210/400000 [00:24<00:23, 8716.43it/s] 49%|████▉     | 195086/400000 [00:24<00:23, 8726.71it/s] 49%|████▉     | 196004/400000 [00:24<00:23, 8856.77it/s] 49%|████▉     | 196919/400000 [00:24<00:22, 8940.64it/s] 49%|████▉     | 197814/400000 [00:24<00:22, 8940.70it/s] 50%|████▉     | 198709/400000 [00:24<00:23, 8710.87it/s] 50%|████▉     | 199616/400000 [00:24<00:22, 8813.72it/s] 50%|█████     | 200540/400000 [00:24<00:22, 8936.96it/s] 50%|█████     | 201470/400000 [00:24<00:21, 9040.97it/s] 51%|█████     | 202376/400000 [00:24<00:21, 9001.70it/s] 51%|█████     | 203278/400000 [00:25<00:22, 8793.52it/s] 51%|█████     | 204168/400000 [00:25<00:22, 8822.65it/s] 51%|█████▏    | 205052/400000 [00:25<00:22, 8781.36it/s] 51%|█████▏    | 205943/400000 [00:25<00:22, 8817.79it/s] 52%|█████▏    | 206826/400000 [00:25<00:22, 8763.03it/s] 52%|█████▏    | 207703/400000 [00:25<00:22, 8624.82it/s] 52%|█████▏    | 208567/400000 [00:25<00:22, 8629.00it/s] 52%|█████▏    | 209431/400000 [00:25<00:22, 8588.65it/s] 53%|█████▎    | 210291/400000 [00:25<00:22, 8577.88it/s] 53%|█████▎    | 211150/400000 [00:25<00:22, 8467.90it/s] 53%|█████▎    | 211998/400000 [00:26<00:22, 8314.89it/s] 53%|█████▎    | 212831/400000 [00:26<00:22, 8291.09it/s] 53%|█████▎    | 213661/400000 [00:26<00:22, 8293.49it/s] 54%|█████▎    | 214569/400000 [00:26<00:21, 8512.37it/s] 54%|█████▍    | 215423/400000 [00:26<00:21, 8478.00it/s] 54%|█████▍    | 216285/400000 [00:26<00:21, 8519.48it/s] 54%|█████▍    | 217188/400000 [00:26<00:21, 8665.66it/s] 55%|█████▍    | 218078/400000 [00:26<00:20, 8734.13it/s] 55%|█████▍    | 218994/400000 [00:26<00:20, 8855.13it/s] 55%|█████▍    | 219881/400000 [00:26<00:20, 8700.47it/s] 55%|█████▌    | 220753/400000 [00:27<00:21, 8488.42it/s] 55%|█████▌    | 221643/400000 [00:27<00:20, 8605.28it/s] 56%|█████▌    | 222506/400000 [00:27<00:20, 8517.57it/s] 56%|█████▌    | 223405/400000 [00:27<00:20, 8653.19it/s] 56%|█████▌    | 224272/400000 [00:27<00:20, 8522.15it/s] 56%|█████▋    | 225126/400000 [00:27<00:20, 8431.03it/s] 57%|█████▋    | 226003/400000 [00:27<00:20, 8525.79it/s] 57%|█████▋    | 226858/400000 [00:27<00:20, 8532.73it/s] 57%|█████▋    | 227757/400000 [00:27<00:19, 8662.53it/s] 57%|█████▋    | 228625/400000 [00:27<00:19, 8574.86it/s] 57%|█████▋    | 229497/400000 [00:28<00:19, 8616.64it/s] 58%|█████▊    | 230421/400000 [00:28<00:19, 8792.27it/s] 58%|█████▊    | 231308/400000 [00:28<00:19, 8813.99it/s] 58%|█████▊    | 232227/400000 [00:28<00:18, 8920.75it/s] 58%|█████▊    | 233121/400000 [00:28<00:18, 8853.07it/s] 59%|█████▊    | 234008/400000 [00:28<00:19, 8646.27it/s] 59%|█████▊    | 234930/400000 [00:28<00:18, 8810.23it/s] 59%|█████▉    | 235813/400000 [00:28<00:18, 8770.81it/s] 59%|█████▉    | 236692/400000 [00:28<00:18, 8707.60it/s] 59%|█████▉    | 237585/400000 [00:29<00:18, 8772.74it/s] 60%|█████▉    | 238464/400000 [00:29<00:18, 8680.10it/s] 60%|█████▉    | 239333/400000 [00:29<00:18, 8641.59it/s] 60%|██████    | 240198/400000 [00:29<00:18, 8587.89it/s] 60%|██████    | 241101/400000 [00:29<00:18, 8714.38it/s] 60%|██████    | 241974/400000 [00:29<00:18, 8674.48it/s] 61%|██████    | 242848/400000 [00:29<00:18, 8693.10it/s] 61%|██████    | 243737/400000 [00:29<00:17, 8749.47it/s] 61%|██████    | 244638/400000 [00:29<00:17, 8824.85it/s] 61%|██████▏   | 245521/400000 [00:29<00:18, 8557.49it/s] 62%|██████▏   | 246379/400000 [00:30<00:18, 8340.80it/s] 62%|██████▏   | 247216/400000 [00:30<00:18, 8317.70it/s] 62%|██████▏   | 248105/400000 [00:30<00:17, 8478.97it/s] 62%|██████▏   | 248974/400000 [00:30<00:17, 8538.69it/s] 62%|██████▏   | 249830/400000 [00:30<00:17, 8449.39it/s] 63%|██████▎   | 250677/400000 [00:30<00:18, 8220.57it/s] 63%|██████▎   | 251502/400000 [00:30<00:18, 8221.66it/s] 63%|██████▎   | 252345/400000 [00:30<00:17, 8282.11it/s] 63%|██████▎   | 253175/400000 [00:30<00:17, 8237.43it/s] 64%|██████▎   | 254000/400000 [00:30<00:17, 8230.05it/s] 64%|██████▎   | 254824/400000 [00:31<00:18, 7808.78it/s] 64%|██████▍   | 255612/400000 [00:31<00:18, 7828.63it/s] 64%|██████▍   | 256474/400000 [00:31<00:17, 8049.47it/s] 64%|██████▍   | 257345/400000 [00:31<00:17, 8235.42it/s] 65%|██████▍   | 258238/400000 [00:31<00:16, 8431.43it/s] 65%|██████▍   | 259099/400000 [00:31<00:16, 8483.34it/s] 65%|██████▍   | 259974/400000 [00:31<00:16, 8560.12it/s] 65%|██████▌   | 260895/400000 [00:31<00:15, 8744.88it/s] 65%|██████▌   | 261801/400000 [00:31<00:15, 8836.57it/s] 66%|██████▌   | 262687/400000 [00:31<00:15, 8757.82it/s] 66%|██████▌   | 263565/400000 [00:32<00:15, 8708.67it/s] 66%|██████▌   | 264441/400000 [00:32<00:15, 8721.20it/s] 66%|██████▋   | 265314/400000 [00:32<00:15, 8702.95it/s] 67%|██████▋   | 266204/400000 [00:32<00:15, 8758.65it/s] 67%|██████▋   | 267096/400000 [00:32<00:15, 8803.76it/s] 67%|██████▋   | 267977/400000 [00:32<00:15, 8667.20it/s] 67%|██████▋   | 268877/400000 [00:32<00:14, 8761.66it/s] 67%|██████▋   | 269768/400000 [00:32<00:14, 8805.62it/s] 68%|██████▊   | 270650/400000 [00:32<00:14, 8724.57it/s] 68%|██████▊   | 271533/400000 [00:32<00:14, 8755.30it/s] 68%|██████▊   | 272409/400000 [00:33<00:14, 8607.55it/s] 68%|██████▊   | 273299/400000 [00:33<00:14, 8691.18it/s] 69%|██████▊   | 274182/400000 [00:33<00:14, 8730.91it/s] 69%|██████▉   | 275066/400000 [00:33<00:14, 8763.23it/s] 69%|██████▉   | 275943/400000 [00:33<00:14, 8743.01it/s] 69%|██████▉   | 276818/400000 [00:33<00:14, 8589.93it/s] 69%|██████▉   | 277705/400000 [00:33<00:14, 8669.09it/s] 70%|██████▉   | 278573/400000 [00:33<00:14, 8659.72it/s] 70%|██████▉   | 279493/400000 [00:33<00:13, 8813.71it/s] 70%|███████   | 280376/400000 [00:33<00:13, 8817.70it/s] 70%|███████   | 281259/400000 [00:34<00:14, 8448.39it/s] 71%|███████   | 282108/400000 [00:34<00:13, 8446.10it/s] 71%|███████   | 282956/400000 [00:34<00:14, 8180.48it/s] 71%|███████   | 283795/400000 [00:34<00:14, 8239.58it/s] 71%|███████   | 284631/400000 [00:34<00:13, 8272.79it/s] 71%|███████▏  | 285461/400000 [00:34<00:13, 8227.85it/s] 72%|███████▏  | 286347/400000 [00:34<00:13, 8405.88it/s] 72%|███████▏  | 287192/400000 [00:34<00:13, 8417.83it/s] 72%|███████▏  | 288036/400000 [00:34<00:13, 8393.57it/s] 72%|███████▏  | 288877/400000 [00:35<00:13, 8296.68it/s] 72%|███████▏  | 289708/400000 [00:35<00:13, 7894.25it/s] 73%|███████▎  | 290503/400000 [00:35<00:13, 7835.94it/s] 73%|███████▎  | 291360/400000 [00:35<00:13, 8041.92it/s] 73%|███████▎  | 292231/400000 [00:35<00:13, 8228.73it/s] 73%|███████▎  | 293115/400000 [00:35<00:12, 8402.50it/s] 73%|███████▎  | 293987/400000 [00:35<00:12, 8494.47it/s] 74%|███████▎  | 294859/400000 [00:35<00:12, 8557.64it/s] 74%|███████▍  | 295759/400000 [00:35<00:12, 8684.01it/s] 74%|███████▍  | 296646/400000 [00:35<00:11, 8735.71it/s] 74%|███████▍  | 297521/400000 [00:36<00:11, 8717.42it/s] 75%|███████▍  | 298394/400000 [00:36<00:11, 8492.27it/s] 75%|███████▍  | 299263/400000 [00:36<00:11, 8549.60it/s] 75%|███████▌  | 300125/400000 [00:36<00:11, 8570.17it/s] 75%|███████▌  | 301034/400000 [00:36<00:11, 8718.13it/s] 75%|███████▌  | 301908/400000 [00:36<00:11, 8720.20it/s] 76%|███████▌  | 302781/400000 [00:36<00:11, 8584.80it/s] 76%|███████▌  | 303649/400000 [00:36<00:11, 8611.77it/s] 76%|███████▌  | 304527/400000 [00:36<00:11, 8659.45it/s] 76%|███████▋  | 305394/400000 [00:36<00:10, 8606.01it/s] 77%|███████▋  | 306289/400000 [00:37<00:10, 8704.79it/s] 77%|███████▋  | 307161/400000 [00:37<00:11, 8245.02it/s] 77%|███████▋  | 308015/400000 [00:37<00:11, 8330.53it/s] 77%|███████▋  | 308853/400000 [00:37<00:11, 8234.29it/s] 77%|███████▋  | 309680/400000 [00:37<00:10, 8242.84it/s] 78%|███████▊  | 310517/400000 [00:37<00:10, 8278.08it/s] 78%|███████▊  | 311347/400000 [00:37<00:10, 8196.69it/s] 78%|███████▊  | 312171/400000 [00:37<00:10, 8207.67it/s] 78%|███████▊  | 312993/400000 [00:37<00:10, 8139.72it/s] 78%|███████▊  | 313833/400000 [00:38<00:10, 8215.34it/s] 79%|███████▊  | 314659/400000 [00:38<00:10, 8228.58it/s] 79%|███████▉  | 315483/400000 [00:38<00:10, 8158.11it/s] 79%|███████▉  | 316300/400000 [00:38<00:10, 7980.82it/s] 79%|███████▉  | 317100/400000 [00:38<00:10, 7902.57it/s] 79%|███████▉  | 317902/400000 [00:38<00:10, 7936.12it/s] 80%|███████▉  | 318732/400000 [00:38<00:10, 8039.06it/s] 80%|███████▉  | 319542/400000 [00:38<00:09, 8056.27it/s] 80%|████████  | 320413/400000 [00:38<00:09, 8241.37it/s] 80%|████████  | 321275/400000 [00:38<00:09, 8348.40it/s] 81%|████████  | 322112/400000 [00:39<00:09, 8317.05it/s] 81%|████████  | 322945/400000 [00:39<00:09, 8312.21it/s] 81%|████████  | 323777/400000 [00:39<00:09, 8116.10it/s] 81%|████████  | 324609/400000 [00:39<00:09, 8176.12it/s] 81%|████████▏ | 325428/400000 [00:39<00:09, 8090.05it/s] 82%|████████▏ | 326239/400000 [00:39<00:09, 8069.85it/s] 82%|████████▏ | 327093/400000 [00:39<00:08, 8205.16it/s] 82%|████████▏ | 327915/400000 [00:39<00:08, 8015.61it/s] 82%|████████▏ | 328763/400000 [00:39<00:08, 8148.24it/s] 82%|████████▏ | 329619/400000 [00:39<00:08, 8265.17it/s] 83%|████████▎ | 330448/400000 [00:40<00:08, 8087.78it/s] 83%|████████▎ | 331265/400000 [00:40<00:08, 8110.55it/s] 83%|████████▎ | 332078/400000 [00:40<00:08, 8042.91it/s] 83%|████████▎ | 332884/400000 [00:40<00:08, 7976.99it/s] 83%|████████▎ | 333683/400000 [00:40<00:08, 7803.94it/s] 84%|████████▎ | 334465/400000 [00:40<00:08, 7720.40it/s] 84%|████████▍ | 335239/400000 [00:40<00:08, 7580.41it/s] 84%|████████▍ | 336016/400000 [00:40<00:08, 7634.73it/s] 84%|████████▍ | 336781/400000 [00:40<00:08, 7634.89it/s] 84%|████████▍ | 337556/400000 [00:40<00:08, 7666.32it/s] 85%|████████▍ | 338324/400000 [00:41<00:08, 7658.40it/s] 85%|████████▍ | 339091/400000 [00:41<00:07, 7655.61it/s] 85%|████████▍ | 339914/400000 [00:41<00:07, 7817.76it/s] 85%|████████▌ | 340725/400000 [00:41<00:07, 7900.45it/s] 85%|████████▌ | 341516/400000 [00:41<00:07, 7739.79it/s] 86%|████████▌ | 342292/400000 [00:41<00:07, 7593.87it/s] 86%|████████▌ | 343053/400000 [00:41<00:07, 7476.20it/s] 86%|████████▌ | 343803/400000 [00:41<00:07, 7376.01it/s] 86%|████████▌ | 344650/400000 [00:41<00:07, 7671.48it/s] 86%|████████▋ | 345422/400000 [00:41<00:07, 7634.27it/s] 87%|████████▋ | 346189/400000 [00:42<00:07, 7613.94it/s] 87%|████████▋ | 346953/400000 [00:42<00:06, 7613.53it/s] 87%|████████▋ | 347716/400000 [00:42<00:06, 7587.76it/s] 87%|████████▋ | 348605/400000 [00:42<00:06, 7936.12it/s] 87%|████████▋ | 349493/400000 [00:42<00:06, 8195.66it/s] 88%|████████▊ | 350350/400000 [00:42<00:05, 8302.89it/s] 88%|████████▊ | 351189/400000 [00:42<00:05, 8328.46it/s] 88%|████████▊ | 352031/400000 [00:42<00:05, 8354.85it/s] 88%|████████▊ | 352895/400000 [00:42<00:05, 8437.77it/s] 88%|████████▊ | 353774/400000 [00:43<00:05, 8538.85it/s] 89%|████████▊ | 354647/400000 [00:43<00:05, 8592.94it/s] 89%|████████▉ | 355508/400000 [00:43<00:05, 8544.76it/s] 89%|████████▉ | 356364/400000 [00:43<00:05, 8477.02it/s] 89%|████████▉ | 357250/400000 [00:43<00:04, 8586.05it/s] 90%|████████▉ | 358110/400000 [00:43<00:04, 8579.77it/s] 90%|████████▉ | 358969/400000 [00:43<00:04, 8426.63it/s] 90%|████████▉ | 359847/400000 [00:43<00:04, 8529.43it/s] 90%|█████████ | 360703/400000 [00:43<00:04, 8536.57it/s] 90%|█████████ | 361589/400000 [00:43<00:04, 8627.97it/s] 91%|█████████ | 362476/400000 [00:44<00:04, 8698.77it/s] 91%|█████████ | 363347/400000 [00:44<00:04, 8673.43it/s] 91%|█████████ | 364218/400000 [00:44<00:04, 8683.77it/s] 91%|█████████▏| 365087/400000 [00:44<00:04, 8664.40it/s] 91%|█████████▏| 365954/400000 [00:44<00:03, 8598.45it/s] 92%|█████████▏| 366815/400000 [00:44<00:03, 8424.02it/s] 92%|█████████▏| 367659/400000 [00:44<00:03, 8207.66it/s] 92%|█████████▏| 368496/400000 [00:44<00:03, 8251.73it/s] 92%|█████████▏| 369383/400000 [00:44<00:03, 8425.79it/s] 93%|█████████▎| 370261/400000 [00:44<00:03, 8527.38it/s] 93%|█████████▎| 371147/400000 [00:45<00:03, 8621.56it/s] 93%|█████████▎| 372011/400000 [00:45<00:03, 8522.82it/s] 93%|█████████▎| 372865/400000 [00:45<00:03, 8474.71it/s] 93%|█████████▎| 373714/400000 [00:45<00:03, 8117.93it/s] 94%|█████████▎| 374530/400000 [00:45<00:03, 7935.44it/s] 94%|█████████▍| 375328/400000 [00:45<00:03, 7745.31it/s] 94%|█████████▍| 376107/400000 [00:45<00:03, 7608.02it/s] 94%|█████████▍| 376906/400000 [00:45<00:02, 7716.68it/s] 94%|█████████▍| 377741/400000 [00:45<00:02, 7894.92it/s] 95%|█████████▍| 378586/400000 [00:45<00:02, 8052.71it/s] 95%|█████████▍| 379394/400000 [00:46<00:02, 8039.34it/s] 95%|█████████▌| 380256/400000 [00:46<00:02, 8204.75it/s] 95%|█████████▌| 381112/400000 [00:46<00:02, 8305.71it/s] 95%|█████████▌| 381964/400000 [00:46<00:02, 8368.13it/s] 96%|█████████▌| 382826/400000 [00:46<00:02, 8441.37it/s] 96%|█████████▌| 383694/400000 [00:46<00:01, 8508.64it/s] 96%|█████████▌| 384587/400000 [00:46<00:01, 8629.74it/s] 96%|█████████▋| 385452/400000 [00:46<00:01, 8587.12it/s] 97%|█████████▋| 386313/400000 [00:46<00:01, 8591.67it/s] 97%|█████████▋| 387195/400000 [00:46<00:01, 8657.06it/s] 97%|█████████▋| 388062/400000 [00:47<00:01, 8558.92it/s] 97%|█████████▋| 388924/400000 [00:47<00:01, 8576.37it/s] 97%|█████████▋| 389783/400000 [00:47<00:01, 8532.26it/s] 98%|█████████▊| 390637/400000 [00:47<00:01, 8428.92it/s] 98%|█████████▊| 391492/400000 [00:47<00:01, 8463.29it/s] 98%|█████████▊| 392339/400000 [00:47<00:00, 8464.90it/s] 98%|█████████▊| 393186/400000 [00:47<00:00, 8384.01it/s] 99%|█████████▊| 394027/400000 [00:47<00:00, 8389.45it/s] 99%|█████████▊| 394911/400000 [00:47<00:00, 8517.45it/s] 99%|█████████▉| 395807/400000 [00:47<00:00, 8644.84it/s] 99%|█████████▉| 396678/400000 [00:48<00:00, 8662.59it/s] 99%|█████████▉| 397545/400000 [00:48<00:00, 8647.43it/s]100%|█████████▉| 398411/400000 [00:48<00:00, 8488.13it/s]100%|█████████▉| 399261/400000 [00:48<00:00, 8416.59it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8248.39it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5e5b4f4f60>, <torchtext.data.dataset.TabularDataset object at 0x7f5e5b4f4160>, <torchtext.vocab.Vocab object at 0x7f5e5b4f4fd0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.18 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.03 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.03 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.26 file/s]2020-07-14 00:17:59.633031: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-14 00:17:59.637146: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-14 00:17:59.637315: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a77c172730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 00:17:59.637331: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 143368.08it/s] 85%|████████▌ | 8429568/9912422 [00:00<00:07, 204662.07it/s]9920512it [00:00, 43819135.43it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1067959.82it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469632.72it/s]1654784it [00:00, 11661008.35it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 79341.20it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
