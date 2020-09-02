
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9f47d296a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9f47d296a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9fb2a52488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9fb2a52488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9fd1c71ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9fd1c71ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9f5fd88488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9f5fd88488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9f5fd88488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2752512/9912422 [00:00<00:00, 27521340.52it/s]9920512it [00:00, 29677082.02it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1178357.91it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 482245.33it/s]1654784it [00:00, 11917763.57it/s]                         
0it [00:00, ?it/s]8192it [00:00, 229153.53it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9f47c5ceb8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9f46fa4320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9f5fd88158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9f5fd88158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9f5fd88158> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:30,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:39,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:26,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:18,  6.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:18,  6.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:17,  6.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 22.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 29.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 37.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 37.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 46.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 55.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 63.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.57 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.25s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.57 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.42 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.33s/ url]
0 examples [00:00, ? examples/s]2020-09-02 00:07:39.138675: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-02 00:07:39.153497: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-09-02 00:07:39.153732: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f21aca1ce0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-02 00:07:39.153753: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.67 examples/s]108 examples [00:00,  3.81 examples/s]207 examples [00:00,  5.43 examples/s]309 examples [00:00,  7.75 examples/s]422 examples [00:00, 11.03 examples/s]531 examples [00:00, 15.69 examples/s]642 examples [00:00, 22.28 examples/s]749 examples [00:01, 31.55 examples/s]859 examples [00:01, 44.52 examples/s]965 examples [00:01, 62.47 examples/s]1071 examples [00:01, 87.03 examples/s]1175 examples [00:01, 119.83 examples/s]1283 examples [00:01, 163.36 examples/s]1392 examples [00:01, 219.20 examples/s]1502 examples [00:01, 288.46 examples/s]1612 examples [00:01, 370.25 examples/s]1723 examples [00:01, 462.66 examples/s]1834 examples [00:02, 560.67 examples/s]1944 examples [00:02, 656.01 examples/s]2054 examples [00:02, 739.81 examples/s]2162 examples [00:02, 803.12 examples/s]2271 examples [00:02, 871.38 examples/s]2381 examples [00:02, 928.94 examples/s]2491 examples [00:02, 973.39 examples/s]2603 examples [00:02, 1011.32 examples/s]2714 examples [00:02, 1038.15 examples/s]2824 examples [00:03, 1036.43 examples/s]2932 examples [00:03, 1032.36 examples/s]3044 examples [00:03, 1056.74 examples/s]3153 examples [00:03, 1065.73 examples/s]3262 examples [00:03, 1060.12 examples/s]3370 examples [00:03, 1059.36 examples/s]3482 examples [00:03, 1075.13 examples/s]3591 examples [00:03, 1078.74 examples/s]3700 examples [00:03, 1080.68 examples/s]3809 examples [00:03, 1082.86 examples/s]3918 examples [00:04, 1084.71 examples/s]4030 examples [00:04, 1093.91 examples/s]4141 examples [00:04, 1094.05 examples/s]4251 examples [00:04, 1093.46 examples/s]4361 examples [00:04, 1091.59 examples/s]4472 examples [00:04, 1095.40 examples/s]4583 examples [00:04, 1097.90 examples/s]4693 examples [00:04, 1085.26 examples/s]4803 examples [00:04, 1088.13 examples/s]4912 examples [00:04, 1074.48 examples/s]5021 examples [00:05, 1077.94 examples/s]5130 examples [00:05, 1078.36 examples/s]5240 examples [00:05, 1083.50 examples/s]5349 examples [00:05, 1083.27 examples/s]5458 examples [00:05, 1051.58 examples/s]5564 examples [00:05, 1040.62 examples/s]5674 examples [00:05, 1057.11 examples/s]5784 examples [00:05, 1067.36 examples/s]5891 examples [00:05, 1063.96 examples/s]5998 examples [00:05, 1029.32 examples/s]6105 examples [00:06, 1038.63 examples/s]6213 examples [00:06, 1047.32 examples/s]6322 examples [00:06, 1058.00 examples/s]6431 examples [00:06, 1065.96 examples/s]6540 examples [00:06, 1071.91 examples/s]6648 examples [00:06, 1057.76 examples/s]6755 examples [00:06, 1061.16 examples/s]6865 examples [00:06, 1071.03 examples/s]6974 examples [00:06, 1075.88 examples/s]7082 examples [00:06, 1070.45 examples/s]7190 examples [00:07, 1066.38 examples/s]7298 examples [00:07, 1068.07 examples/s]7405 examples [00:07, 1055.69 examples/s]7511 examples [00:07, 1025.72 examples/s]7614 examples [00:07, 970.98 examples/s] 7723 examples [00:07, 1002.64 examples/s]7835 examples [00:07, 1033.02 examples/s]7942 examples [00:07, 1043.21 examples/s]8049 examples [00:07, 1050.89 examples/s]8158 examples [00:08, 1059.85 examples/s]8270 examples [00:08, 1075.70 examples/s]8378 examples [00:08, 1070.24 examples/s]8486 examples [00:08, 1068.16 examples/s]8595 examples [00:08, 1074.11 examples/s]8703 examples [00:08, 1070.90 examples/s]8814 examples [00:08, 1079.69 examples/s]8923 examples [00:08, 1078.68 examples/s]9031 examples [00:08, 1077.71 examples/s]9139 examples [00:08, 1062.76 examples/s]9246 examples [00:09, 1052.22 examples/s]9356 examples [00:09, 1065.06 examples/s]9464 examples [00:09, 1067.73 examples/s]9571 examples [00:09, 1064.94 examples/s]9679 examples [00:09, 1068.90 examples/s]9786 examples [00:09, 1062.11 examples/s]9893 examples [00:09, 1060.34 examples/s]10000 examples [00:09, 1060.52 examples/s]10107 examples [00:09, 1016.42 examples/s]10211 examples [00:09, 1022.05 examples/s]10317 examples [00:10, 1030.98 examples/s]10421 examples [00:10, 1026.55 examples/s]10528 examples [00:10, 1039.13 examples/s]10639 examples [00:10, 1058.56 examples/s]10746 examples [00:10, 1056.70 examples/s]10853 examples [00:10, 1058.07 examples/s]10963 examples [00:10, 1067.78 examples/s]11072 examples [00:10, 1072.00 examples/s]11186 examples [00:10, 1084.32 examples/s]11297 examples [00:10, 1090.31 examples/s]11407 examples [00:11, 1091.56 examples/s]11519 examples [00:11, 1097.95 examples/s]11629 examples [00:11, 1070.78 examples/s]11737 examples [00:11, 1052.76 examples/s]11844 examples [00:11, 1055.70 examples/s]11950 examples [00:11, 1047.62 examples/s]12059 examples [00:11, 1059.84 examples/s]12166 examples [00:11, 1046.75 examples/s]12271 examples [00:11, 1038.06 examples/s]12381 examples [00:11, 1055.86 examples/s]12487 examples [00:12, 1030.19 examples/s]12596 examples [00:12, 1046.29 examples/s]12701 examples [00:12, 1037.90 examples/s]12809 examples [00:12, 1047.81 examples/s]12914 examples [00:12, 1030.88 examples/s]13018 examples [00:12, 1029.02 examples/s]13125 examples [00:12, 1040.54 examples/s]13232 examples [00:12, 1047.58 examples/s]13338 examples [00:12, 1048.45 examples/s]13443 examples [00:13, 1039.17 examples/s]13550 examples [00:13, 1045.66 examples/s]13655 examples [00:13, 1046.21 examples/s]13760 examples [00:13, 1041.29 examples/s]13869 examples [00:13, 1054.31 examples/s]13977 examples [00:13, 1059.21 examples/s]14083 examples [00:13, 1057.20 examples/s]14190 examples [00:13, 1060.86 examples/s]14297 examples [00:13, 1061.56 examples/s]14405 examples [00:13, 1065.49 examples/s]14512 examples [00:14, 1064.61 examples/s]14619 examples [00:14, 1061.20 examples/s]14726 examples [00:14, 1051.65 examples/s]14832 examples [00:14, 1026.79 examples/s]14942 examples [00:14, 1046.43 examples/s]15050 examples [00:14, 1054.46 examples/s]15156 examples [00:14, 1018.52 examples/s]15266 examples [00:14, 1041.32 examples/s]15377 examples [00:14, 1058.80 examples/s]15486 examples [00:14, 1066.38 examples/s]15598 examples [00:15, 1080.59 examples/s]15707 examples [00:15, 1077.28 examples/s]15817 examples [00:15, 1082.91 examples/s]15928 examples [00:15, 1088.83 examples/s]16037 examples [00:15, 1079.08 examples/s]16145 examples [00:15, 1071.25 examples/s]16253 examples [00:15, 1065.72 examples/s]16363 examples [00:15, 1074.27 examples/s]16471 examples [00:15, 1075.42 examples/s]16579 examples [00:15, 1074.88 examples/s]16689 examples [00:16, 1080.34 examples/s]16798 examples [00:16, 1073.41 examples/s]16906 examples [00:16, 1056.28 examples/s]17012 examples [00:16, 1052.63 examples/s]17121 examples [00:16, 1062.24 examples/s]17234 examples [00:16, 1079.31 examples/s]17343 examples [00:16, 1074.08 examples/s]17451 examples [00:16, 1064.29 examples/s]17563 examples [00:16, 1079.67 examples/s]17673 examples [00:16, 1083.81 examples/s]17782 examples [00:17, 1080.34 examples/s]17891 examples [00:17, 1078.28 examples/s]18001 examples [00:17, 1083.34 examples/s]18110 examples [00:17, 1073.71 examples/s]18220 examples [00:17, 1080.39 examples/s]18329 examples [00:17, 1083.02 examples/s]18438 examples [00:17, 1084.52 examples/s]18549 examples [00:17, 1089.75 examples/s]18659 examples [00:17, 1090.91 examples/s]18770 examples [00:17, 1096.16 examples/s]18884 examples [00:18, 1106.55 examples/s]18995 examples [00:18, 1097.22 examples/s]19105 examples [00:18, 1097.53 examples/s]19217 examples [00:18, 1103.39 examples/s]19328 examples [00:18, 1100.50 examples/s]19439 examples [00:18, 1082.61 examples/s]19548 examples [00:18, 1076.33 examples/s]19657 examples [00:18, 1080.16 examples/s]19766 examples [00:18, 1078.50 examples/s]19876 examples [00:19, 1084.30 examples/s]19986 examples [00:19, 1087.94 examples/s]20095 examples [00:19, 1028.13 examples/s]20206 examples [00:19, 1048.86 examples/s]20314 examples [00:19, 1056.24 examples/s]20423 examples [00:19, 1063.49 examples/s]20531 examples [00:19, 1067.45 examples/s]20638 examples [00:19, 1059.10 examples/s]20749 examples [00:19, 1073.28 examples/s]20857 examples [00:19, 1072.98 examples/s]20969 examples [00:20, 1086.09 examples/s]21080 examples [00:20, 1091.43 examples/s]21190 examples [00:20, 1089.82 examples/s]21302 examples [00:20, 1097.16 examples/s]21414 examples [00:20, 1102.24 examples/s]21527 examples [00:20, 1107.78 examples/s]21638 examples [00:20, 1104.37 examples/s]21749 examples [00:20, 1041.65 examples/s]21854 examples [00:20, 1032.60 examples/s]21958 examples [00:20, 1033.59 examples/s]22064 examples [00:21, 1039.47 examples/s]22169 examples [00:21, 1037.69 examples/s]22273 examples [00:21, 1021.64 examples/s]22377 examples [00:21, 1025.23 examples/s]22486 examples [00:21, 1041.74 examples/s]22595 examples [00:21, 1054.40 examples/s]22701 examples [00:21, 1054.71 examples/s]22808 examples [00:21, 1057.66 examples/s]22920 examples [00:21, 1073.81 examples/s]23028 examples [00:21, 1069.79 examples/s]23136 examples [00:22, 1067.83 examples/s]23243 examples [00:22, 1041.73 examples/s]23348 examples [00:22, 1041.33 examples/s]23457 examples [00:22, 1054.62 examples/s]23563 examples [00:22, 1053.86 examples/s]23669 examples [00:22, 1052.82 examples/s]23778 examples [00:22, 1062.25 examples/s]23886 examples [00:22, 1065.15 examples/s]23994 examples [00:22, 1069.28 examples/s]24103 examples [00:22, 1072.59 examples/s]24215 examples [00:23, 1083.52 examples/s]24327 examples [00:23, 1091.40 examples/s]24437 examples [00:23, 1072.96 examples/s]24545 examples [00:23, 1071.62 examples/s]24653 examples [00:23, 1071.95 examples/s]24763 examples [00:23, 1078.38 examples/s]24873 examples [00:23, 1083.99 examples/s]24983 examples [00:23, 1086.95 examples/s]25092 examples [00:23, 1083.14 examples/s]25201 examples [00:24, 1080.57 examples/s]25312 examples [00:24, 1088.68 examples/s]25421 examples [00:24, 1085.49 examples/s]25530 examples [00:24, 1083.48 examples/s]25641 examples [00:24, 1090.82 examples/s]25751 examples [00:24, 1091.76 examples/s]25861 examples [00:24, 1091.31 examples/s]25971 examples [00:24, 1090.71 examples/s]26081 examples [00:24, 1091.54 examples/s]26191 examples [00:24, 1086.48 examples/s]26300 examples [00:25, 1064.72 examples/s]26412 examples [00:25, 1080.00 examples/s]26521 examples [00:25, 1082.92 examples/s]26630 examples [00:25, 1080.82 examples/s]26741 examples [00:25, 1088.91 examples/s]26852 examples [00:25, 1093.30 examples/s]26962 examples [00:25, 1092.86 examples/s]27074 examples [00:25, 1100.20 examples/s]27185 examples [00:25, 1074.95 examples/s]27293 examples [00:25, 1074.40 examples/s]27403 examples [00:26, 1081.88 examples/s]27513 examples [00:26, 1085.17 examples/s]27622 examples [00:26, 1082.30 examples/s]27731 examples [00:26, 1059.33 examples/s]27838 examples [00:26, 1059.50 examples/s]27949 examples [00:26, 1072.01 examples/s]28059 examples [00:26, 1079.15 examples/s]28171 examples [00:26, 1089.35 examples/s]28281 examples [00:26, 1085.54 examples/s]28390 examples [00:26, 1080.01 examples/s]28501 examples [00:27, 1086.98 examples/s]28610 examples [00:27, 1081.43 examples/s]28719 examples [00:27, 1079.34 examples/s]28827 examples [00:27, 1073.66 examples/s]28935 examples [00:27, 1062.06 examples/s]29042 examples [00:27, 1062.51 examples/s]29149 examples [00:27, 1058.48 examples/s]29255 examples [00:27, 1043.06 examples/s]29363 examples [00:27, 1051.38 examples/s]29469 examples [00:27, 1048.14 examples/s]29574 examples [00:28, 1038.14 examples/s]29682 examples [00:28, 1049.20 examples/s]29790 examples [00:28, 1056.17 examples/s]29900 examples [00:28, 1067.04 examples/s]30007 examples [00:28, 1023.77 examples/s]30118 examples [00:28, 1046.51 examples/s]30224 examples [00:28, 1037.35 examples/s]30334 examples [00:28, 1053.61 examples/s]30441 examples [00:28, 1057.98 examples/s]30552 examples [00:29, 1071.68 examples/s]30663 examples [00:29, 1081.00 examples/s]30772 examples [00:29, 1082.41 examples/s]30881 examples [00:29, 1065.55 examples/s]30988 examples [00:29, 1023.82 examples/s]31092 examples [00:29, 1026.24 examples/s]31195 examples [00:29, 1026.00 examples/s]31304 examples [00:29, 1043.03 examples/s]31414 examples [00:29, 1057.84 examples/s]31522 examples [00:29, 1062.21 examples/s]31629 examples [00:30, 1063.86 examples/s]31740 examples [00:30, 1075.49 examples/s]31850 examples [00:30, 1081.31 examples/s]31961 examples [00:30, 1089.11 examples/s]32070 examples [00:30, 1078.33 examples/s]32178 examples [00:30, 1065.39 examples/s]32288 examples [00:30, 1075.10 examples/s]32398 examples [00:30, 1081.09 examples/s]32508 examples [00:30, 1086.47 examples/s]32617 examples [00:30, 1080.91 examples/s]32726 examples [00:31, 1066.80 examples/s]32833 examples [00:31, 1049.94 examples/s]32939 examples [00:31, 1037.54 examples/s]33047 examples [00:31, 1048.47 examples/s]33155 examples [00:31, 1055.27 examples/s]33261 examples [00:31, 1055.03 examples/s]33371 examples [00:31, 1065.66 examples/s]33478 examples [00:31, 1066.51 examples/s]33585 examples [00:31, 1063.61 examples/s]33696 examples [00:31, 1075.34 examples/s]33808 examples [00:32, 1087.18 examples/s]33917 examples [00:32, 1086.50 examples/s]34026 examples [00:32, 1087.07 examples/s]34135 examples [00:32, 1082.38 examples/s]34244 examples [00:32, 1080.39 examples/s]34353 examples [00:32, 1078.19 examples/s]34461 examples [00:32, 1061.89 examples/s]34568 examples [00:32, 1061.44 examples/s]34675 examples [00:32, 1063.53 examples/s]34782 examples [00:32, 1062.79 examples/s]34891 examples [00:33, 1070.07 examples/s]34999 examples [00:33, 1023.29 examples/s]35102 examples [00:33, 1022.02 examples/s]35209 examples [00:33, 1033.48 examples/s]35314 examples [00:33, 1037.91 examples/s]35423 examples [00:33, 1050.53 examples/s]35529 examples [00:33, 1039.28 examples/s]35636 examples [00:33, 1046.16 examples/s]35743 examples [00:33, 1052.16 examples/s]35851 examples [00:33, 1058.43 examples/s]35957 examples [00:34, 1044.49 examples/s]36063 examples [00:34, 1048.20 examples/s]36168 examples [00:34, 1047.13 examples/s]36277 examples [00:34, 1057.31 examples/s]36384 examples [00:34, 1060.23 examples/s]36493 examples [00:34, 1067.44 examples/s]36600 examples [00:34, 1062.10 examples/s]36707 examples [00:34, 1059.77 examples/s]36814 examples [00:34, 1060.89 examples/s]36921 examples [00:35, 1058.27 examples/s]37030 examples [00:35, 1067.32 examples/s]37138 examples [00:35, 1068.90 examples/s]37246 examples [00:35, 1071.88 examples/s]37354 examples [00:35, 1066.11 examples/s]37462 examples [00:35, 1067.83 examples/s]37569 examples [00:35, 1065.41 examples/s]37678 examples [00:35, 1070.62 examples/s]37786 examples [00:35, 1066.98 examples/s]37894 examples [00:35, 1068.09 examples/s]38002 examples [00:36, 1069.89 examples/s]38109 examples [00:36, 1066.89 examples/s]38216 examples [00:36, 1063.65 examples/s]38323 examples [00:36, 1062.52 examples/s]38433 examples [00:36, 1072.06 examples/s]38542 examples [00:36, 1077.29 examples/s]38651 examples [00:36, 1080.76 examples/s]38760 examples [00:36, 1076.61 examples/s]38868 examples [00:36, 1070.38 examples/s]38976 examples [00:36, 1067.02 examples/s]39083 examples [00:37, 1064.20 examples/s]39192 examples [00:37, 1070.44 examples/s]39300 examples [00:37, 1066.86 examples/s]39407 examples [00:37, 1066.18 examples/s]39514 examples [00:37, 1065.68 examples/s]39621 examples [00:37, 1055.91 examples/s]39729 examples [00:37, 1060.39 examples/s]39836 examples [00:37, 1056.89 examples/s]39944 examples [00:37, 1062.22 examples/s]40051 examples [00:37, 1000.05 examples/s]40155 examples [00:38, 1009.90 examples/s]40265 examples [00:38, 1032.78 examples/s]40371 examples [00:38, 1040.14 examples/s]40479 examples [00:38, 1050.56 examples/s]40585 examples [00:38, 1052.46 examples/s]40692 examples [00:38, 1057.32 examples/s]40799 examples [00:38, 1060.29 examples/s]40907 examples [00:38, 1065.59 examples/s]41016 examples [00:38, 1070.84 examples/s]41124 examples [00:38, 1061.31 examples/s]41234 examples [00:39, 1070.95 examples/s]41344 examples [00:39, 1076.52 examples/s]41452 examples [00:39, 1069.75 examples/s]41562 examples [00:39, 1078.50 examples/s]41673 examples [00:39, 1085.70 examples/s]41782 examples [00:39, 1079.85 examples/s]41891 examples [00:39, 1080.52 examples/s]42000 examples [00:39, 1069.54 examples/s]42107 examples [00:39, 1068.45 examples/s]42216 examples [00:39, 1073.40 examples/s]42324 examples [00:40, 1071.70 examples/s]42432 examples [00:40, 1057.13 examples/s]42541 examples [00:40, 1066.46 examples/s]42649 examples [00:40, 1068.50 examples/s]42756 examples [00:40, 1054.40 examples/s]42862 examples [00:40, 1039.94 examples/s]42967 examples [00:40, 1032.08 examples/s]43075 examples [00:40, 1044.97 examples/s]43183 examples [00:40, 1054.32 examples/s]43291 examples [00:41, 1059.98 examples/s]43398 examples [00:41, 1060.68 examples/s]43505 examples [00:41, 1059.13 examples/s]43611 examples [00:41, 1050.61 examples/s]43719 examples [00:41, 1057.64 examples/s]43829 examples [00:41, 1066.76 examples/s]43936 examples [00:41, 1067.27 examples/s]44044 examples [00:41, 1069.32 examples/s]44152 examples [00:41, 1072.05 examples/s]44260 examples [00:41, 1071.78 examples/s]44368 examples [00:42, 1073.29 examples/s]44476 examples [00:42, 1067.67 examples/s]44586 examples [00:42, 1075.61 examples/s]44694 examples [00:42, 1068.63 examples/s]44802 examples [00:42, 1069.43 examples/s]44909 examples [00:42, 1057.99 examples/s]45016 examples [00:42, 1061.47 examples/s]45125 examples [00:42, 1069.69 examples/s]45233 examples [00:42, 1070.14 examples/s]45341 examples [00:42, 1054.92 examples/s]45447 examples [00:43, 1038.44 examples/s]45554 examples [00:43, 1046.42 examples/s]45660 examples [00:43, 1048.59 examples/s]45768 examples [00:43, 1055.91 examples/s]45878 examples [00:43, 1068.61 examples/s]45985 examples [00:43, 1055.41 examples/s]46093 examples [00:43, 1062.15 examples/s]46204 examples [00:43, 1073.77 examples/s]46313 examples [00:43, 1077.08 examples/s]46424 examples [00:43, 1084.94 examples/s]46533 examples [00:44, 1080.94 examples/s]46642 examples [00:44, 1080.23 examples/s]46751 examples [00:44, 1064.99 examples/s]46859 examples [00:44, 1066.73 examples/s]46969 examples [00:44, 1074.55 examples/s]47077 examples [00:44, 1038.77 examples/s]47182 examples [00:44, 1042.07 examples/s]47290 examples [00:44, 1052.53 examples/s]47399 examples [00:44, 1062.09 examples/s]47507 examples [00:44, 1066.74 examples/s]47616 examples [00:45, 1072.11 examples/s]47724 examples [00:45, 1067.05 examples/s]47831 examples [00:45, 1058.79 examples/s]47937 examples [00:45, 1054.23 examples/s]48043 examples [00:45, 1043.35 examples/s]48149 examples [00:45, 1045.55 examples/s]48254 examples [00:45, 1045.07 examples/s]48363 examples [00:45, 1055.78 examples/s]48471 examples [00:45, 1060.36 examples/s]48580 examples [00:45, 1068.50 examples/s]48688 examples [00:46, 1071.49 examples/s]48796 examples [00:46, 1050.83 examples/s]48902 examples [00:46, 1052.25 examples/s]49010 examples [00:46, 1057.61 examples/s]49116 examples [00:46, 1042.52 examples/s]49224 examples [00:46, 1051.76 examples/s]49332 examples [00:46, 1058.01 examples/s]49445 examples [00:46, 1075.99 examples/s]49554 examples [00:46, 1078.67 examples/s]49662 examples [00:47, 1065.07 examples/s]49769 examples [00:47, 1063.95 examples/s]49876 examples [00:47, 1062.02 examples/s]49985 examples [00:47, 1067.66 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6537/50000 [00:00<00:00, 65369.44 examples/s] 39%|███▉      | 19582/50000 [00:00<00:00, 76874.45 examples/s] 65%|██████▌   | 32727/50000 [00:00<00:00, 87810.84 examples/s] 91%|█████████ | 45393/50000 [00:00<00:00, 96709.28 examples/s]                                                               0 examples [00:00, ? examples/s]89 examples [00:00, 887.97 examples/s]197 examples [00:00, 936.27 examples/s]304 examples [00:00, 971.35 examples/s]410 examples [00:00, 995.03 examples/s]517 examples [00:00, 1015.21 examples/s]624 examples [00:00, 1028.63 examples/s]736 examples [00:00, 1052.37 examples/s]848 examples [00:00, 1069.35 examples/s]958 examples [00:00, 1075.88 examples/s]1068 examples [00:01, 1080.32 examples/s]1177 examples [00:01, 1083.00 examples/s]1287 examples [00:01, 1086.81 examples/s]1398 examples [00:01, 1093.38 examples/s]1507 examples [00:01, 1070.26 examples/s]1614 examples [00:01, 1055.57 examples/s]1721 examples [00:01, 1057.58 examples/s]1829 examples [00:01, 1061.77 examples/s]1936 examples [00:01, 1036.74 examples/s]2041 examples [00:01, 1039.82 examples/s]2148 examples [00:02, 1047.13 examples/s]2253 examples [00:02, 1044.92 examples/s]2359 examples [00:02, 1048.10 examples/s]2469 examples [00:02, 1061.67 examples/s]2576 examples [00:02, 1054.70 examples/s]2682 examples [00:02, 1044.28 examples/s]2791 examples [00:02, 1056.97 examples/s]2901 examples [00:02, 1068.82 examples/s]3010 examples [00:02, 1074.64 examples/s]3120 examples [00:02, 1079.49 examples/s]3234 examples [00:03, 1096.60 examples/s]3349 examples [00:03, 1111.81 examples/s]3463 examples [00:03, 1119.80 examples/s]3576 examples [00:03, 1118.97 examples/s]3688 examples [00:03, 1107.64 examples/s]3799 examples [00:03, 1096.55 examples/s]3909 examples [00:03, 1095.53 examples/s]4020 examples [00:03, 1098.96 examples/s]4130 examples [00:03, 1097.79 examples/s]4240 examples [00:03, 1088.76 examples/s]4350 examples [00:04, 1090.82 examples/s]4463 examples [00:04, 1100.17 examples/s]4577 examples [00:04, 1109.29 examples/s]4688 examples [00:04, 1093.91 examples/s]4798 examples [00:04, 1048.60 examples/s]4904 examples [00:04, 1049.83 examples/s]5013 examples [00:04, 1059.51 examples/s]5120 examples [00:04, 1054.05 examples/s]5231 examples [00:04, 1069.29 examples/s]5339 examples [00:04, 1070.56 examples/s]5447 examples [00:05, 853.33 examples/s] 5559 examples [00:05, 917.96 examples/s]5670 examples [00:05, 967.60 examples/s]5772 examples [00:05, 977.99 examples/s]5880 examples [00:05, 1004.42 examples/s]5992 examples [00:05, 1034.94 examples/s]6098 examples [00:05, 1039.17 examples/s]6209 examples [00:05, 1057.82 examples/s]6317 examples [00:05, 1063.36 examples/s]6426 examples [00:06, 1069.36 examples/s]6534 examples [00:06, 1068.88 examples/s]6642 examples [00:06, 1061.64 examples/s]6751 examples [00:06, 1069.68 examples/s]6859 examples [00:06, 1052.98 examples/s]6966 examples [00:06, 1055.71 examples/s]7077 examples [00:06, 1068.83 examples/s]7187 examples [00:06, 1076.85 examples/s]7297 examples [00:06, 1081.34 examples/s]7406 examples [00:06, 1079.90 examples/s]7516 examples [00:07, 1083.24 examples/s]7625 examples [00:07, 1075.06 examples/s]7734 examples [00:07, 1077.45 examples/s]7842 examples [00:07, 1071.48 examples/s]7950 examples [00:07, 1047.93 examples/s]8058 examples [00:07, 1057.08 examples/s]8167 examples [00:07, 1066.48 examples/s]8274 examples [00:07, 1052.47 examples/s]8380 examples [00:07, 1042.17 examples/s]8486 examples [00:08, 1045.27 examples/s]8596 examples [00:08, 1059.32 examples/s]8705 examples [00:08, 1067.88 examples/s]8817 examples [00:08, 1079.86 examples/s]8926 examples [00:08, 1063.39 examples/s]9033 examples [00:08, 1057.69 examples/s]9139 examples [00:08, 1052.97 examples/s]9249 examples [00:08, 1065.78 examples/s]9361 examples [00:08, 1080.00 examples/s]9472 examples [00:08, 1088.58 examples/s]9582 examples [00:09, 1089.47 examples/s]9692 examples [00:09, 1037.75 examples/s]9800 examples [00:09, 1048.53 examples/s]9912 examples [00:09, 1066.93 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY61G55/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY61G55/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9f5fd88488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9f5fd88488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9f5fd88488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9f5ddeea20>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9ee9fe25f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9f5fd88488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9f5fd88488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9f5fd88488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9ee9fe2fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9f46fcd128>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.70 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.70 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.00 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.76 file/s]2020-09-02 00:08:43.323715: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-02 00:08:43.327828: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-09-02 00:08:43.327994: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557e85408130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-02 00:08:43.328009: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 469855.35it/s] 95%|█████████▍| 9379840/9912422 [00:00<00:00, 669729.28it/s]9920512it [00:00, 47151927.92it/s]                           
0it [00:00, ?it/s]32768it [00:00, 495580.89it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479272.14it/s]1654784it [00:00, 11908970.59it/s]                         
0it [00:00, ?it/s]8192it [00:00, 178861.95it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
