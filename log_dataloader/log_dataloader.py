
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '77c2c9f427106a88e922f586a543b21e10cfaf22', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5ec19896a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5ec19896a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5f2cf510d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5f2cf510d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5f46c7dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5f46c7dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5eda7ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5eda7ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5eda7ce950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3866624/9912422 [00:00<00:00, 38646190.40it/s]9920512it [00:00, 34524721.49it/s]                             
0it [00:00, ?it/s]32768it [00:00, 624427.33it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467116.12it/s]1654784it [00:00, 11468329.06it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205143.79it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ed7d22208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ed7d14da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5eda7ce598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5eda7ce598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5eda7ce598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:13,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.52 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.52 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.21 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.03s/ url]
0 examples [00:00, ? examples/s]2020-06-13 18:09:28.495160: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 18:09:28.508029: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-13 18:09:28.508218: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562610d7d3e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 18:09:28.508241: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 188.68 examples/s]102 examples [00:00, 245.58 examples/s]187 examples [00:00, 311.86 examples/s]272 examples [00:00, 384.52 examples/s]357 examples [00:00, 459.75 examples/s]437 examples [00:00, 526.93 examples/s]532 examples [00:00, 607.78 examples/s]627 examples [00:00, 680.86 examples/s]720 examples [00:00, 739.51 examples/s]810 examples [00:01, 779.54 examples/s]906 examples [00:01, 824.57 examples/s]999 examples [00:01, 852.75 examples/s]1090 examples [00:01, 851.19 examples/s]1179 examples [00:01, 854.06 examples/s]1267 examples [00:01, 846.49 examples/s]1354 examples [00:01, 843.13 examples/s]1440 examples [00:01, 840.55 examples/s]1525 examples [00:01, 834.75 examples/s]1610 examples [00:01, 831.23 examples/s]1701 examples [00:02, 853.16 examples/s]1787 examples [00:02, 854.50 examples/s]1877 examples [00:02, 864.92 examples/s]1964 examples [00:02, 858.66 examples/s]2051 examples [00:02, 859.70 examples/s]2138 examples [00:02, 858.12 examples/s]2227 examples [00:02, 866.73 examples/s]2316 examples [00:02, 871.93 examples/s]2404 examples [00:02, 869.39 examples/s]2492 examples [00:02, 871.31 examples/s]2580 examples [00:03, 851.13 examples/s]2673 examples [00:03, 872.35 examples/s]2764 examples [00:03, 881.28 examples/s]2853 examples [00:03, 873.03 examples/s]2941 examples [00:03, 873.70 examples/s]3029 examples [00:03, 870.55 examples/s]3117 examples [00:03, 863.28 examples/s]3209 examples [00:03, 878.14 examples/s]3298 examples [00:03, 880.30 examples/s]3388 examples [00:03, 884.07 examples/s]3477 examples [00:04, 880.23 examples/s]3566 examples [00:04, 875.95 examples/s]3654 examples [00:04, 854.02 examples/s]3743 examples [00:04, 863.61 examples/s]3833 examples [00:04, 873.08 examples/s]3921 examples [00:04, 871.52 examples/s]4009 examples [00:04, 873.18 examples/s]4097 examples [00:04, 869.60 examples/s]4192 examples [00:04, 890.81 examples/s]4282 examples [00:04, 883.35 examples/s]4371 examples [00:05, 874.54 examples/s]4459 examples [00:05, 867.03 examples/s]4546 examples [00:05, 854.54 examples/s]4632 examples [00:05, 852.66 examples/s]4725 examples [00:05, 872.54 examples/s]4813 examples [00:05, 867.03 examples/s]4904 examples [00:05, 877.30 examples/s]4997 examples [00:05, 890.39 examples/s]5094 examples [00:05, 910.37 examples/s]5186 examples [00:06, 903.30 examples/s]5277 examples [00:06, 877.15 examples/s]5365 examples [00:06, 874.66 examples/s]5453 examples [00:06, 866.11 examples/s]5548 examples [00:06, 887.42 examples/s]5637 examples [00:06, 885.58 examples/s]5726 examples [00:06, 876.18 examples/s]5820 examples [00:06, 893.57 examples/s]5910 examples [00:06, 838.75 examples/s]5995 examples [00:06, 835.37 examples/s]6081 examples [00:07, 840.22 examples/s]6170 examples [00:07, 852.53 examples/s]6260 examples [00:07, 865.16 examples/s]6356 examples [00:07, 890.14 examples/s]6449 examples [00:07, 900.72 examples/s]6546 examples [00:07, 917.80 examples/s]6639 examples [00:07, 903.30 examples/s]6730 examples [00:07, 898.37 examples/s]6821 examples [00:07, 851.23 examples/s]6908 examples [00:07, 854.94 examples/s]6994 examples [00:08, 846.33 examples/s]7088 examples [00:08, 870.34 examples/s]7178 examples [00:08, 878.57 examples/s]7267 examples [00:08, 863.01 examples/s]7354 examples [00:08, 852.16 examples/s]7440 examples [00:08, 850.60 examples/s]7527 examples [00:08, 854.71 examples/s]7616 examples [00:08, 864.62 examples/s]7703 examples [00:08, 862.74 examples/s]7790 examples [00:09, 857.92 examples/s]7876 examples [00:09, 847.94 examples/s]7967 examples [00:09, 861.55 examples/s]8054 examples [00:09, 862.63 examples/s]8146 examples [00:09, 877.01 examples/s]8238 examples [00:09, 889.31 examples/s]8328 examples [00:09, 866.25 examples/s]8415 examples [00:09, 856.88 examples/s]8501 examples [00:09, 842.61 examples/s]8590 examples [00:09, 854.79 examples/s]8681 examples [00:10, 870.32 examples/s]8769 examples [00:10, 866.71 examples/s]8856 examples [00:10, 853.18 examples/s]8945 examples [00:10, 862.65 examples/s]9032 examples [00:10, 855.94 examples/s]9119 examples [00:10, 858.70 examples/s]9206 examples [00:10, 861.98 examples/s]9293 examples [00:10, 855.47 examples/s]9383 examples [00:10, 866.30 examples/s]9472 examples [00:10, 872.38 examples/s]9560 examples [00:11, 865.55 examples/s]9647 examples [00:11, 863.48 examples/s]9734 examples [00:11, 860.94 examples/s]9823 examples [00:11, 867.88 examples/s]9910 examples [00:11, 859.19 examples/s]10001 examples [00:11, 832.35 examples/s]10093 examples [00:11, 854.57 examples/s]10187 examples [00:11, 877.54 examples/s]10281 examples [00:11, 895.19 examples/s]10374 examples [00:11, 904.39 examples/s]10467 examples [00:12, 909.64 examples/s]10559 examples [00:12, 887.19 examples/s]10653 examples [00:12, 902.00 examples/s]10748 examples [00:12, 913.56 examples/s]10842 examples [00:12, 920.58 examples/s]10935 examples [00:12, 918.46 examples/s]11029 examples [00:12, 923.43 examples/s]11122 examples [00:12, 923.91 examples/s]11215 examples [00:12, 917.03 examples/s]11309 examples [00:13, 922.25 examples/s]11402 examples [00:13, 918.86 examples/s]11494 examples [00:13, 884.41 examples/s]11583 examples [00:13, 867.55 examples/s]11677 examples [00:13, 885.76 examples/s]11770 examples [00:13, 895.68 examples/s]11863 examples [00:13, 903.26 examples/s]11957 examples [00:13, 909.95 examples/s]12051 examples [00:13, 917.83 examples/s]12148 examples [00:13, 931.89 examples/s]12242 examples [00:14, 916.83 examples/s]12334 examples [00:14, 909.37 examples/s]12426 examples [00:14, 892.58 examples/s]12516 examples [00:14, 851.63 examples/s]12613 examples [00:14, 881.92 examples/s]12702 examples [00:14, 883.99 examples/s]12791 examples [00:14, 881.31 examples/s]12882 examples [00:14, 887.30 examples/s]12977 examples [00:14, 900.29 examples/s]13070 examples [00:14, 907.00 examples/s]13164 examples [00:15, 916.45 examples/s]13257 examples [00:15, 918.49 examples/s]13352 examples [00:15, 925.28 examples/s]13445 examples [00:15, 896.02 examples/s]13535 examples [00:15, 896.23 examples/s]13633 examples [00:15, 917.86 examples/s]13726 examples [00:15, 895.28 examples/s]13816 examples [00:15, 883.89 examples/s]13905 examples [00:15, 885.02 examples/s]13999 examples [00:16, 899.12 examples/s]14093 examples [00:16, 910.30 examples/s]14185 examples [00:16, 903.15 examples/s]14281 examples [00:16, 916.88 examples/s]14376 examples [00:16, 926.31 examples/s]14469 examples [00:16, 926.74 examples/s]14562 examples [00:16, 904.07 examples/s]14653 examples [00:16, 891.25 examples/s]14748 examples [00:16, 906.87 examples/s]14839 examples [00:16, 883.15 examples/s]14930 examples [00:17, 889.31 examples/s]15025 examples [00:17, 904.23 examples/s]15116 examples [00:17, 896.58 examples/s]15206 examples [00:17, 893.51 examples/s]15297 examples [00:17, 897.69 examples/s]15389 examples [00:17, 902.22 examples/s]15480 examples [00:17, 903.82 examples/s]15571 examples [00:17, 877.69 examples/s]15660 examples [00:17, 879.83 examples/s]15749 examples [00:17, 877.98 examples/s]15842 examples [00:18, 890.85 examples/s]15935 examples [00:18, 901.16 examples/s]16026 examples [00:18, 900.40 examples/s]16117 examples [00:18, 894.59 examples/s]16212 examples [00:18, 910.37 examples/s]16306 examples [00:18, 918.79 examples/s]16400 examples [00:18, 922.81 examples/s]16493 examples [00:18, 921.12 examples/s]16588 examples [00:18, 929.28 examples/s]16685 examples [00:18, 941.09 examples/s]16780 examples [00:19, 941.07 examples/s]16875 examples [00:19, 935.92 examples/s]16969 examples [00:19, 913.52 examples/s]17061 examples [00:19, 899.50 examples/s]17152 examples [00:19, 895.82 examples/s]17242 examples [00:19, 884.50 examples/s]17332 examples [00:19, 887.73 examples/s]17421 examples [00:19, 876.93 examples/s]17509 examples [00:19, 847.69 examples/s]17595 examples [00:20, 836.84 examples/s]17687 examples [00:20, 859.27 examples/s]17775 examples [00:20, 864.82 examples/s]17865 examples [00:20, 870.10 examples/s]17955 examples [00:20, 877.71 examples/s]18043 examples [00:20, 857.39 examples/s]18130 examples [00:20, 858.57 examples/s]18219 examples [00:20, 865.03 examples/s]18306 examples [00:20, 854.09 examples/s]18401 examples [00:20, 878.90 examples/s]18493 examples [00:21, 886.79 examples/s]18582 examples [00:21, 886.61 examples/s]18671 examples [00:21, 880.56 examples/s]18760 examples [00:21, 859.55 examples/s]18847 examples [00:21, 857.07 examples/s]18934 examples [00:21, 858.97 examples/s]19020 examples [00:21, 856.95 examples/s]19106 examples [00:21, 848.74 examples/s]19191 examples [00:21, 840.66 examples/s]19276 examples [00:21, 841.54 examples/s]19362 examples [00:22, 845.69 examples/s]19448 examples [00:22, 849.20 examples/s]19533 examples [00:22, 838.02 examples/s]19618 examples [00:22, 839.10 examples/s]19711 examples [00:22, 862.27 examples/s]19800 examples [00:22, 865.30 examples/s]19890 examples [00:22, 874.59 examples/s]19978 examples [00:22, 865.68 examples/s]20065 examples [00:22, 810.65 examples/s]20153 examples [00:23, 829.77 examples/s]20237 examples [00:23, 830.92 examples/s]20321 examples [00:23, 832.47 examples/s]20408 examples [00:23, 843.27 examples/s]20496 examples [00:23, 852.80 examples/s]20589 examples [00:23, 873.01 examples/s]20679 examples [00:23, 879.97 examples/s]20769 examples [00:23, 883.12 examples/s]20860 examples [00:23, 889.70 examples/s]20950 examples [00:23, 876.92 examples/s]21038 examples [00:24, 869.04 examples/s]21132 examples [00:24, 886.73 examples/s]21221 examples [00:24, 882.36 examples/s]21310 examples [00:24, 864.69 examples/s]21403 examples [00:24, 880.35 examples/s]21496 examples [00:24, 893.56 examples/s]21586 examples [00:24, 882.05 examples/s]21675 examples [00:24, 881.81 examples/s]21764 examples [00:24, 865.93 examples/s]21852 examples [00:24, 867.85 examples/s]21943 examples [00:25, 879.96 examples/s]22033 examples [00:25, 885.57 examples/s]22122 examples [00:25, 871.28 examples/s]22211 examples [00:25, 874.39 examples/s]22301 examples [00:25, 880.30 examples/s]22392 examples [00:25, 887.62 examples/s]22485 examples [00:25, 898.75 examples/s]22580 examples [00:25, 910.77 examples/s]22672 examples [00:25, 894.77 examples/s]22762 examples [00:25, 872.28 examples/s]22852 examples [00:26, 879.76 examples/s]22942 examples [00:26, 885.69 examples/s]23035 examples [00:26, 897.37 examples/s]23125 examples [00:26, 888.40 examples/s]23216 examples [00:26, 893.58 examples/s]23306 examples [00:26, 870.83 examples/s]23395 examples [00:26, 874.35 examples/s]23485 examples [00:26, 878.97 examples/s]23576 examples [00:26, 886.83 examples/s]23667 examples [00:26, 893.42 examples/s]23757 examples [00:27, 887.45 examples/s]23850 examples [00:27, 899.47 examples/s]23945 examples [00:27, 913.32 examples/s]24037 examples [00:27, 891.12 examples/s]24128 examples [00:27, 895.50 examples/s]24220 examples [00:27, 900.49 examples/s]24316 examples [00:27, 915.75 examples/s]24410 examples [00:27, 920.07 examples/s]24503 examples [00:27, 909.93 examples/s]24595 examples [00:28, 909.33 examples/s]24686 examples [00:28, 899.27 examples/s]24776 examples [00:28, 889.57 examples/s]24867 examples [00:28, 895.32 examples/s]24957 examples [00:28, 891.55 examples/s]25047 examples [00:28, 892.24 examples/s]25137 examples [00:28, 892.29 examples/s]25228 examples [00:28, 896.16 examples/s]25321 examples [00:28, 905.91 examples/s]25418 examples [00:28, 922.41 examples/s]25511 examples [00:29, 923.14 examples/s]25605 examples [00:29, 926.95 examples/s]25698 examples [00:29, 919.76 examples/s]25791 examples [00:29, 912.99 examples/s]25883 examples [00:29, 908.66 examples/s]25974 examples [00:29, 908.91 examples/s]26067 examples [00:29, 913.76 examples/s]26159 examples [00:29, 897.62 examples/s]26249 examples [00:29, 891.69 examples/s]26339 examples [00:29, 891.13 examples/s]26429 examples [00:30, 889.31 examples/s]26518 examples [00:30, 888.07 examples/s]26609 examples [00:30, 891.55 examples/s]26703 examples [00:30, 905.45 examples/s]26794 examples [00:30, 889.31 examples/s]26884 examples [00:30, 877.08 examples/s]26972 examples [00:30, 874.72 examples/s]27060 examples [00:30, 864.35 examples/s]27149 examples [00:30, 871.14 examples/s]27237 examples [00:30, 868.71 examples/s]27326 examples [00:31, 873.56 examples/s]27415 examples [00:31, 877.42 examples/s]27511 examples [00:31, 900.28 examples/s]27602 examples [00:31, 882.85 examples/s]27697 examples [00:31, 901.44 examples/s]27788 examples [00:31, 883.29 examples/s]27877 examples [00:31, 868.45 examples/s]27966 examples [00:31, 874.76 examples/s]28054 examples [00:31, 874.50 examples/s]28146 examples [00:31, 886.83 examples/s]28237 examples [00:32, 892.14 examples/s]28333 examples [00:32, 909.86 examples/s]28426 examples [00:32, 914.23 examples/s]28520 examples [00:32, 921.77 examples/s]28613 examples [00:32, 924.00 examples/s]28706 examples [00:32, 923.56 examples/s]28801 examples [00:32, 929.08 examples/s]28899 examples [00:32, 941.23 examples/s]28994 examples [00:32, 924.25 examples/s]29087 examples [00:33, 898.33 examples/s]29178 examples [00:33, 882.17 examples/s]29272 examples [00:33, 897.56 examples/s]29362 examples [00:33, 895.98 examples/s]29452 examples [00:33, 893.72 examples/s]29542 examples [00:33, 889.93 examples/s]29632 examples [00:33, 883.50 examples/s]29721 examples [00:33, 877.94 examples/s]29810 examples [00:33, 878.86 examples/s]29901 examples [00:33, 887.16 examples/s]29995 examples [00:34, 901.01 examples/s]30086 examples [00:34, 864.35 examples/s]30182 examples [00:34, 890.80 examples/s]30273 examples [00:34, 895.71 examples/s]30365 examples [00:34, 900.83 examples/s]30456 examples [00:34, 880.65 examples/s]30547 examples [00:34, 888.63 examples/s]30638 examples [00:34, 894.73 examples/s]30729 examples [00:34, 896.57 examples/s]30820 examples [00:34, 900.52 examples/s]30911 examples [00:35, 889.94 examples/s]31001 examples [00:35, 879.76 examples/s]31090 examples [00:35, 875.88 examples/s]31178 examples [00:35, 870.70 examples/s]31266 examples [00:35, 871.62 examples/s]31356 examples [00:35, 877.76 examples/s]31448 examples [00:35, 889.61 examples/s]31538 examples [00:35, 863.90 examples/s]31625 examples [00:35, 862.18 examples/s]31713 examples [00:35, 865.10 examples/s]31800 examples [00:36, 855.61 examples/s]31886 examples [00:36, 846.56 examples/s]31971 examples [00:36, 839.53 examples/s]32064 examples [00:36, 864.43 examples/s]32161 examples [00:36, 891.77 examples/s]32256 examples [00:36, 907.27 examples/s]32348 examples [00:36, 888.27 examples/s]32443 examples [00:36, 903.74 examples/s]32538 examples [00:36, 916.99 examples/s]32631 examples [00:37, 919.33 examples/s]32725 examples [00:37, 923.26 examples/s]32818 examples [00:37, 922.52 examples/s]32912 examples [00:37, 925.57 examples/s]33005 examples [00:37, 926.49 examples/s]33098 examples [00:37, 908.97 examples/s]33190 examples [00:37, 908.92 examples/s]33284 examples [00:37, 916.29 examples/s]33377 examples [00:37, 913.21 examples/s]33469 examples [00:37, 900.12 examples/s]33560 examples [00:38, 877.66 examples/s]33652 examples [00:38, 887.47 examples/s]33747 examples [00:38, 904.22 examples/s]33839 examples [00:38, 906.19 examples/s]33932 examples [00:38, 912.67 examples/s]34025 examples [00:38, 917.20 examples/s]34117 examples [00:38, 916.97 examples/s]34209 examples [00:38, 905.61 examples/s]34300 examples [00:38, 904.34 examples/s]34391 examples [00:38, 889.77 examples/s]34482 examples [00:39, 895.13 examples/s]34579 examples [00:39, 915.62 examples/s]34677 examples [00:39, 931.50 examples/s]34771 examples [00:39, 913.94 examples/s]34863 examples [00:39, 897.73 examples/s]34953 examples [00:39, 863.23 examples/s]35041 examples [00:39, 868.17 examples/s]35129 examples [00:39, 867.10 examples/s]35220 examples [00:39, 877.40 examples/s]35311 examples [00:39, 884.38 examples/s]35400 examples [00:40, 870.81 examples/s]35493 examples [00:40, 887.65 examples/s]35590 examples [00:40, 908.34 examples/s]35684 examples [00:40, 916.18 examples/s]35776 examples [00:40, 909.93 examples/s]35868 examples [00:40, 909.31 examples/s]35962 examples [00:40, 917.20 examples/s]36055 examples [00:40, 920.03 examples/s]36148 examples [00:40, 894.49 examples/s]36241 examples [00:41, 903.55 examples/s]36332 examples [00:41, 897.87 examples/s]36422 examples [00:41, 887.35 examples/s]36511 examples [00:41, 851.51 examples/s]36597 examples [00:41, 849.45 examples/s]36683 examples [00:41, 848.25 examples/s]36776 examples [00:41, 869.10 examples/s]36871 examples [00:41, 889.63 examples/s]36961 examples [00:41, 890.24 examples/s]37051 examples [00:41, 879.36 examples/s]37140 examples [00:42, 876.71 examples/s]37228 examples [00:42, 875.65 examples/s]37316 examples [00:42, 874.45 examples/s]37405 examples [00:42, 876.73 examples/s]37496 examples [00:42, 884.04 examples/s]37585 examples [00:42, 879.00 examples/s]37673 examples [00:42, 858.86 examples/s]37764 examples [00:42, 873.39 examples/s]37852 examples [00:42, 874.72 examples/s]37947 examples [00:42, 891.81 examples/s]38037 examples [00:43, 884.98 examples/s]38130 examples [00:43, 896.17 examples/s]38226 examples [00:43, 912.88 examples/s]38318 examples [00:43, 909.64 examples/s]38413 examples [00:43, 920.18 examples/s]38508 examples [00:43, 928.60 examples/s]38601 examples [00:43, 920.19 examples/s]38694 examples [00:43, 914.55 examples/s]38786 examples [00:43, 913.78 examples/s]38882 examples [00:43, 925.42 examples/s]38975 examples [00:44, 918.99 examples/s]39067 examples [00:44, 915.54 examples/s]39162 examples [00:44, 923.09 examples/s]39255 examples [00:44, 919.70 examples/s]39348 examples [00:44, 901.52 examples/s]39439 examples [00:44, 901.81 examples/s]39530 examples [00:44, 885.55 examples/s]39622 examples [00:44, 895.35 examples/s]39712 examples [00:44, 887.36 examples/s]39806 examples [00:45, 901.11 examples/s]39899 examples [00:45, 908.48 examples/s]39990 examples [00:45, 884.19 examples/s]40079 examples [00:45, 824.28 examples/s]40167 examples [00:45, 840.03 examples/s]40257 examples [00:45, 856.77 examples/s]40344 examples [00:45, 854.31 examples/s]40438 examples [00:45, 875.21 examples/s]40529 examples [00:45, 883.17 examples/s]40618 examples [00:45, 873.81 examples/s]40714 examples [00:46, 896.88 examples/s]40811 examples [00:46, 916.58 examples/s]40903 examples [00:46, 916.10 examples/s]40996 examples [00:46, 920.04 examples/s]41091 examples [00:46, 928.09 examples/s]41184 examples [00:46, 905.21 examples/s]41275 examples [00:46, 895.64 examples/s]41370 examples [00:46, 910.02 examples/s]41463 examples [00:46, 914.06 examples/s]41555 examples [00:46, 910.32 examples/s]41649 examples [00:47, 917.00 examples/s]41741 examples [00:47, 914.78 examples/s]41833 examples [00:47, 889.81 examples/s]41923 examples [00:47, 888.02 examples/s]42012 examples [00:47, 881.49 examples/s]42101 examples [00:47, 876.50 examples/s]42189 examples [00:47, 873.83 examples/s]42277 examples [00:47, 854.58 examples/s]42367 examples [00:47, 867.45 examples/s]42461 examples [00:47, 885.58 examples/s]42554 examples [00:48, 895.81 examples/s]42644 examples [00:48, 891.30 examples/s]42734 examples [00:48, 873.87 examples/s]42822 examples [00:48, 849.97 examples/s]42911 examples [00:48, 860.44 examples/s]42998 examples [00:48, 857.44 examples/s]43084 examples [00:48, 856.28 examples/s]43171 examples [00:48, 857.89 examples/s]43260 examples [00:48, 864.79 examples/s]43347 examples [00:49, 847.79 examples/s]43437 examples [00:49, 860.84 examples/s]43525 examples [00:49, 865.78 examples/s]43612 examples [00:49, 862.48 examples/s]43701 examples [00:49, 870.53 examples/s]43795 examples [00:49, 889.99 examples/s]43885 examples [00:49, 891.30 examples/s]43977 examples [00:49, 898.94 examples/s]44072 examples [00:49, 911.36 examples/s]44167 examples [00:49, 920.88 examples/s]44261 examples [00:50, 924.89 examples/s]44355 examples [00:50, 928.47 examples/s]44452 examples [00:50, 939.15 examples/s]44546 examples [00:50, 933.72 examples/s]44642 examples [00:50, 940.82 examples/s]44738 examples [00:50, 944.12 examples/s]44833 examples [00:50, 940.65 examples/s]44928 examples [00:50, 917.64 examples/s]45023 examples [00:50, 925.46 examples/s]45116 examples [00:50, 923.08 examples/s]45209 examples [00:51, 923.15 examples/s]45302 examples [00:51, 921.79 examples/s]45395 examples [00:51, 919.56 examples/s]45487 examples [00:51, 919.57 examples/s]45579 examples [00:51, 914.51 examples/s]45671 examples [00:51, 885.00 examples/s]45760 examples [00:51, 847.62 examples/s]45846 examples [00:51, 834.45 examples/s]45930 examples [00:51, 836.02 examples/s]46016 examples [00:51, 841.80 examples/s]46103 examples [00:52, 848.64 examples/s]46196 examples [00:52, 869.25 examples/s]46287 examples [00:52, 879.83 examples/s]46382 examples [00:52, 898.37 examples/s]46476 examples [00:52, 908.51 examples/s]46568 examples [00:52, 897.18 examples/s]46658 examples [00:52, 893.55 examples/s]46750 examples [00:52, 898.74 examples/s]46840 examples [00:52, 894.66 examples/s]46933 examples [00:53, 904.70 examples/s]47030 examples [00:53, 922.71 examples/s]47123 examples [00:53, 921.26 examples/s]47216 examples [00:53, 905.27 examples/s]47307 examples [00:53, 902.15 examples/s]47402 examples [00:53, 913.62 examples/s]47497 examples [00:53, 923.12 examples/s]47590 examples [00:53, 924.85 examples/s]47685 examples [00:53, 930.40 examples/s]47781 examples [00:53, 936.38 examples/s]47881 examples [00:54, 952.26 examples/s]47977 examples [00:54, 948.60 examples/s]48072 examples [00:54, 917.06 examples/s]48164 examples [00:54, 895.10 examples/s]48261 examples [00:54, 914.96 examples/s]48356 examples [00:54, 924.09 examples/s]48449 examples [00:54, 907.84 examples/s]48541 examples [00:54, 897.69 examples/s]48631 examples [00:54, 888.82 examples/s]48721 examples [00:54, 889.31 examples/s]48811 examples [00:55, 880.87 examples/s]48900 examples [00:55, 865.73 examples/s]48995 examples [00:55, 888.95 examples/s]49087 examples [00:55, 896.74 examples/s]49178 examples [00:55, 898.76 examples/s]49273 examples [00:55, 913.35 examples/s]49365 examples [00:55, 909.93 examples/s]49457 examples [00:55, 901.17 examples/s]49548 examples [00:55, 887.63 examples/s]49644 examples [00:55, 907.25 examples/s]49739 examples [00:56, 917.48 examples/s]49831 examples [00:56, 912.05 examples/s]49925 examples [00:56, 915.35 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6741/50000 [00:00<00:00, 67409.74 examples/s] 37%|███▋      | 18386/50000 [00:00<00:00, 77156.18 examples/s] 60%|█████▉    | 29772/50000 [00:00<00:00, 85415.32 examples/s] 83%|████████▎ | 41499/50000 [00:00<00:00, 92991.75 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 692.90 examples/s]162 examples [00:00, 746.66 examples/s]252 examples [00:00, 786.64 examples/s]341 examples [00:00, 814.50 examples/s]427 examples [00:00, 825.39 examples/s]515 examples [00:00, 839.79 examples/s]605 examples [00:00, 856.76 examples/s]696 examples [00:00, 870.05 examples/s]785 examples [00:00, 875.57 examples/s]874 examples [00:01, 878.23 examples/s]963 examples [00:01, 880.69 examples/s]1055 examples [00:01, 891.16 examples/s]1144 examples [00:01, 887.10 examples/s]1235 examples [00:01, 893.60 examples/s]1324 examples [00:01, 885.13 examples/s]1413 examples [00:01, 687.95 examples/s]1503 examples [00:01, 738.23 examples/s]1591 examples [00:01, 775.61 examples/s]1685 examples [00:02, 817.39 examples/s]1780 examples [00:02, 852.78 examples/s]1874 examples [00:02, 875.96 examples/s]1964 examples [00:02, 880.70 examples/s]2054 examples [00:02, 863.06 examples/s]2142 examples [00:02, 864.23 examples/s]2231 examples [00:02, 871.43 examples/s]2319 examples [00:02, 862.95 examples/s]2409 examples [00:02, 872.62 examples/s]2500 examples [00:02, 880.35 examples/s]2589 examples [00:03, 876.15 examples/s]2680 examples [00:03, 885.33 examples/s]2771 examples [00:03, 890.91 examples/s]2864 examples [00:03, 900.09 examples/s]2955 examples [00:03, 894.12 examples/s]3045 examples [00:03, 890.62 examples/s]3135 examples [00:03, 880.92 examples/s]3224 examples [00:03, 869.91 examples/s]3312 examples [00:03, 864.75 examples/s]3401 examples [00:03, 869.99 examples/s]3490 examples [00:04, 875.24 examples/s]3581 examples [00:04, 883.87 examples/s]3675 examples [00:04, 899.65 examples/s]3772 examples [00:04, 919.66 examples/s]3865 examples [00:04, 914.55 examples/s]3959 examples [00:04, 921.67 examples/s]4059 examples [00:04, 943.68 examples/s]4158 examples [00:04, 954.73 examples/s]4254 examples [00:04, 942.78 examples/s]4349 examples [00:04, 927.22 examples/s]4442 examples [00:05, 902.23 examples/s]4536 examples [00:05, 912.78 examples/s]4628 examples [00:05, 908.72 examples/s]4720 examples [00:05, 903.19 examples/s]4811 examples [00:05, 890.80 examples/s]4901 examples [00:05, 879.43 examples/s]4993 examples [00:05, 890.80 examples/s]5083 examples [00:05, 838.58 examples/s]5173 examples [00:05, 852.77 examples/s]5262 examples [00:06, 861.42 examples/s]5350 examples [00:06, 864.86 examples/s]5437 examples [00:06, 840.57 examples/s]5529 examples [00:06, 860.76 examples/s]5626 examples [00:06, 888.91 examples/s]5720 examples [00:06, 903.36 examples/s]5813 examples [00:06, 909.40 examples/s]5909 examples [00:06, 922.38 examples/s]6005 examples [00:06, 930.64 examples/s]6100 examples [00:06, 935.12 examples/s]6197 examples [00:07, 943.03 examples/s]6292 examples [00:07, 929.15 examples/s]6386 examples [00:07, 917.06 examples/s]6480 examples [00:07, 923.77 examples/s]6573 examples [00:07, 911.61 examples/s]6665 examples [00:07, 896.57 examples/s]6760 examples [00:07, 909.66 examples/s]6853 examples [00:07, 913.96 examples/s]6945 examples [00:07, 906.55 examples/s]7036 examples [00:07, 901.18 examples/s]7127 examples [00:08, 896.83 examples/s]7217 examples [00:08, 887.71 examples/s]7306 examples [00:08, 881.34 examples/s]7396 examples [00:08, 884.94 examples/s]7485 examples [00:08, 877.42 examples/s]7573 examples [00:08, 870.53 examples/s]7663 examples [00:08, 877.20 examples/s]7757 examples [00:08, 893.86 examples/s]7850 examples [00:08, 903.13 examples/s]7941 examples [00:08, 902.18 examples/s]8032 examples [00:09, 903.11 examples/s]8123 examples [00:09, 898.64 examples/s]8218 examples [00:09, 912.47 examples/s]8313 examples [00:09, 922.34 examples/s]8408 examples [00:09, 929.87 examples/s]8502 examples [00:09, 930.93 examples/s]8596 examples [00:09, 932.72 examples/s]8690 examples [00:09, 924.44 examples/s]8783 examples [00:09, 889.91 examples/s]8873 examples [00:10, 869.70 examples/s]8963 examples [00:10, 876.76 examples/s]9052 examples [00:10, 879.63 examples/s]9142 examples [00:10, 885.15 examples/s]9231 examples [00:10, 856.85 examples/s]9317 examples [00:10, 835.84 examples/s]9408 examples [00:10, 856.76 examples/s]9497 examples [00:10, 866.01 examples/s]9587 examples [00:10, 874.65 examples/s]9679 examples [00:10, 885.55 examples/s]9775 examples [00:11, 904.03 examples/s]9867 examples [00:11, 908.54 examples/s]9959 examples [00:11, 909.94 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTMSCHX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTMSCHX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5eda7ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5eda7ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5eda7ce950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5e64a3cf98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5eda7b9c50>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5eda7ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5eda7ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5eda7ce950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f261b93c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5eb83c8128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5e5c1852f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5e5c1852f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f5e5c1852f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5e5c185400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5e5c185400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5e5c185400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e6f659b9a1f9cfc4633713ce5a9c7c3f4a27f147a0e60f2ee02344447822598f
  Stored in directory: /tmp/pip-ephem-wheel-cache-ek38h3pi/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:26:34, 10.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:56:28, 15.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:12:42, 21.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:51:22, 30.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:29:07, 43.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.90M/862M [00:01<3:48:48, 62.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:39:17, 88.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.6M/862M [00:01<1:51:09, 126kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<1:17:21, 181kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:01<54:04, 257kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:01<37:42, 367kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.0M/862M [00:02<26:24, 521kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<18:26, 742kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.6M/862M [00:02<12:58, 1.05MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<09:06, 1.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<06:58, 1.94MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<06:46, 1.98MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<08:22, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<06:42, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<04:54, 2.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<09:01, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<08:09, 1.64MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<06:09, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<06:51, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:19, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<04:48, 2.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<06:07, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<06:58, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:26, 2.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:10<04:03, 3.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:10, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<04:39, 2.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<06:19, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<07:11, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<05:36, 2.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:14<04:07, 3.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<07:49, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:49, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<05:06, 2.55MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:36, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<07:14, 1.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:37, 2.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:18<04:07, 3.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<08:11, 1.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<07:02, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<05:14, 2.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<06:40, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<04:29, 2.85MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<06:07, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<05:34, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<04:10, 3.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:56, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:43, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:20, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<03:51, 3.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<1:34:01, 134kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:07:03, 188kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<47:10, 267kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<35:52, 350kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<27:38, 454kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<19:54, 629kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<14:02, 888kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:39:43, 125kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:11:03, 175kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<49:54, 249kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<37:45, 329kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<28:55, 429kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<20:46, 597kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<14:40, 842kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<16:17, 757kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<12:38, 975kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<09:09, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:16, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:43, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:51, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:16, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:06, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<1:30:01, 135kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:04:14, 189kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<45:07, 268kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<34:20, 351kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<26:28, 455kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<19:07, 630kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<15:16, 785kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:54, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:34, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:47, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:31, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:47, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:19, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:46, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:25, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:39, 3.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:13, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:58, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<05:06, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:19, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:46, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:19, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:35, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:06, 2.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<03:52, 2.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:45, 2.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:34, 3.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<02:40, 4.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<1:10:21, 163kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<50:23, 227kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<35:28, 322kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<27:27, 414kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<21:30, 529kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<15:31, 732kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<11:01, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<12:11, 928kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<09:41, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<07:03, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:34, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:34, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:51, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<04:12, 2.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:24:03, 133kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<59:57, 186kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<42:09, 264kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<32:01, 347kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<24:33, 452kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<17:44, 625kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<14:09, 780kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<11:02, 999kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<07:59, 1.38MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:09, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:49, 1.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:02, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:06, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:28, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:59, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:36, 3.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<11:11, 968kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:55, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:30, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:03, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:08, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:26, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:58, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:58, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:59, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:27, 2.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:34, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:04, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:47, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:02, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:35, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:26, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:52, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:21, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:11, 3.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:46, 1.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:49, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:17, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:54, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:36, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:20, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<9:55:48, 17.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<6:57:41, 24.6kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<4:51:54, 35.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<3:26:01, 49.6kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<2:26:14, 69.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<1:42:45, 99.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:13:12, 139kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<52:14, 194kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<36:44, 275kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<27:59, 360kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<20:25, 493kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<14:33, 691kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<10:17, 973kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<56:58, 176kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<41:53, 239kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<29:47, 335kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<22:21, 444kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<16:29, 602kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<11:48, 839kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<10:33, 935kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:22, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<06:03, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:33, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:34, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:02, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:36, 2.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<29:50, 326kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<21:51, 445kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<15:30, 626kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<13:05, 738kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<11:06, 869kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:15, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<05:51, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<16:01, 599kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<12:16, 782kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<08:48, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<08:11, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:56, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:05, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<12:23, 763kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<09:44, 970kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<07:04, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:56, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:01, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:23, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<03:53, 2.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:25, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:32, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:08, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:51, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:15, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:11, 2.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<02:23, 3.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<10:29, 874kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<09:28, 969kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:03, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<05:06, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:59, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:13, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:52, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:37, 1.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:19, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:10, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:01, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:49, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:04, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:46, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:33, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:12, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:08, 2.82MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:52, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:54, 3.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<02:09, 4.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<18:32, 473kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<13:56, 628kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<09:56, 878kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<08:48, 987kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:08, 1.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:28, 1.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:48, 1.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:35, 2.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:56, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:52, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:48, 3.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:14, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:19, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:32, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:07, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:01, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<02:55, 2.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<09:21, 891kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:20, 1.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:18, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:51, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:45, 848kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<08:44, 946kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<06:30, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<04:38, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:03, 1.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:52, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:18, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:46, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:07, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:03, 2.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<02:55, 2.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<09:21, 863kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:27, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:25, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:30, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:37, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:18, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:08, 2.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:42, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:09, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:05, 2.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:51, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:28, 2.26MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:33, 3.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:08, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:47, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:51, 2.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:37, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:23, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:28, 3.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<10:38, 719kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<08:18, 920kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<06:01, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:48, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:49, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:26, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:12, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:56, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:17, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:49, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:23, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:30, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:51, 1.52MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:07, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:47, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:11, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:19, 2.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:24, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<12:46, 566kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<09:47, 739kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:59, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<04:58, 1.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<26:08, 274kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<19:55, 360kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:18, 500kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<10:03, 706kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<13:12, 537kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<10:02, 706kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<07:11, 984kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:30, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:02, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:32, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<03:15, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:13, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:10, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:46, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:05, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:23, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:24, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:27, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:28, 1.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:36, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:24, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:49, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:25, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:33, 2.63MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:13, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:45, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:58, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:09, 3.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:59, 946kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:38, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:06, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:15, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:25, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:23, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:28, 2.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:01, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:25, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:34, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:09, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:02, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<16:38, 380kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<12:19, 513kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:44, 721kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:28, 838kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<06:40, 937kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:58, 1.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:33, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:58, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:10, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:03, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:26, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:49, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:58, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:11, 2.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:13, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:55, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:11, 2.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:16, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<01:51, 3.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:49, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:12, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:24, 2.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:40, 2.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<01:59, 2.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:10, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:30, 2.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:48, 3.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:07, 1.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:32, 1.61MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:37, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:02, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:26, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:40, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:55, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:30, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:46, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:46, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:06, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:20, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:37, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:52, 2.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<12:31, 433kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<09:20, 580kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<06:39, 811kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:49, 920kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:18, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:58, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:49, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:49, 1.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:55, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:51, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:10, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:48, 1.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:04, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:33, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:21, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:47, 2.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:20, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:45, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:11, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:03<01:35, 3.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:06, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:26, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<02:32, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:49, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:01, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:22, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:41, 2.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<09:49, 497kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<07:24, 657kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<05:18, 914kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:43, 1.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:24, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<03:19, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<02:22, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:49, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:12, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:22, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:38, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:52, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:14, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:36, 2.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:24, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<02:04, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:31, 2.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:34, 990kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:14, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:13, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<02:17, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<06:00, 743kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:40, 954kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:22, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:19, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:14, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:29, 1.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:46, 2.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<10:31, 411kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<07:50, 550kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<05:33, 771kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:46, 890kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:49, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:45, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:49, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:54, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:13, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:29<01:36, 2.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:09, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:41, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:58, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:13, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:29, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:57, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:25, 2.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:54, 810kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:52, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:48, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:47, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:46, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:06, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:32, 2.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:10, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:27, 2.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:40, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:15, 2.96MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:40, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:08, 3.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:14, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:36, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:55, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:23, 2.56MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:48, 934kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:29, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:36, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:50, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:20, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:43, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:59, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:06, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:14, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:43, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:14, 2.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:32, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:09, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:34, 2.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:47, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:56, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:30, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:06, 2.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:51, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:39, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:13, 2.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:32, 2.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:46, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:23, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:59, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:37, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:11, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:35, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:44, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:54, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:29, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:03, 2.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:26, 1.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:02, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:29, 1.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:38, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:49, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:24, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:01, 2.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:34, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:24, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:02, 2.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:32, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:12, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:53, 3.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:23, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:16, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:57, 2.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:12, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:50, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:09, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:48, 3.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:06, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:46, 3.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:03, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:17, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:01, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:43, 3.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:12, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:48, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:18, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:22, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:28, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:49, 2.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:43, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:26, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:02, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:11, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:16, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:59, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:42, 2.89MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:38, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:22, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:59, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:06, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:13, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:57, 2.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:40, 2.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:11, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:02, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:46, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:55, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:01, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:47, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:34, 3.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:03, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:55, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:40, 2.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:56, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:44, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:32, 3.04MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:49, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:44, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:33, 2.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:43, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:51, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:40, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:29, 3.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:49, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:44, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:33, 2.60MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:40, 2.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:36, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:27, 3.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:43, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:24, 3.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:39, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:29, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:36, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:40, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:31, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:22, 3.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:47, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:40, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:33, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:38, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:21, 2.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:31, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:21, 2.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:24, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:16, 3.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:33, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:28, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:20, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:24, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:27, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:21, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:14, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:42, 421kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<01:15, 564kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:52, 791kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:42, 903kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:38, 1.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:28, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:18, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:37, 933kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:29, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:20, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:19, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:20, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:15, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:10, 2.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:18, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:11, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:05, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:03, 3.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:26, 365kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:18, 494kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:11, 695kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:06, 811kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 922kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:01, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 762kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 970kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<32:01:59,  3.47it/s]  0%|          | 665/400000 [00:00<22:23:27,  4.95it/s]  0%|          | 1333/400000 [00:00<15:39:09,  7.07it/s]  1%|          | 2007/400000 [00:00<10:56:35, 10.10it/s]  1%|          | 2677/400000 [00:00<7:39:08, 14.42it/s]   1%|          | 3307/400000 [00:00<5:21:12, 20.58it/s]  1%|          | 3929/400000 [00:00<3:44:48, 29.36it/s]  1%|          | 4596/400000 [00:00<2:37:23, 41.87it/s]  1%|▏         | 5259/400000 [00:01<1:50:17, 59.65it/s]  1%|▏         | 5934/400000 [00:01<1:17:21, 84.90it/s]  2%|▏         | 6579/400000 [00:01<54:22, 120.60it/s]   2%|▏         | 7256/400000 [00:01<38:17, 170.98it/s]  2%|▏         | 7974/400000 [00:01<27:01, 241.78it/s]  2%|▏         | 8716/400000 [00:01<19:08, 340.64it/s]  2%|▏         | 9408/400000 [00:01<13:40, 476.17it/s]  3%|▎         | 10092/400000 [00:01<09:51, 659.49it/s]  3%|▎         | 10774/400000 [00:01<07:10, 904.62it/s]  3%|▎         | 11450/400000 [00:02<05:18, 1219.49it/s]  3%|▎         | 12190/400000 [00:02<03:58, 1627.18it/s]  3%|▎         | 12902/400000 [00:02<03:02, 2117.02it/s]  3%|▎         | 13598/400000 [00:02<02:25, 2658.78it/s]  4%|▎         | 14284/400000 [00:02<02:00, 3201.23it/s]  4%|▎         | 14947/400000 [00:02<01:41, 3788.26it/s]  4%|▍         | 15702/400000 [00:02<01:26, 4453.49it/s]  4%|▍         | 16440/400000 [00:02<01:15, 5054.55it/s]  4%|▍         | 17145/400000 [00:02<01:10, 5423.84it/s]  4%|▍         | 17878/400000 [00:02<01:04, 5881.35it/s]  5%|▍         | 18638/400000 [00:03<01:00, 6308.10it/s]  5%|▍         | 19358/400000 [00:03<00:58, 6457.68it/s]  5%|▌         | 20067/400000 [00:03<00:58, 6546.72it/s]  5%|▌         | 20766/400000 [00:03<00:57, 6595.40it/s]  5%|▌         | 21457/400000 [00:03<00:57, 6531.12it/s]  6%|▌         | 22133/400000 [00:03<00:57, 6556.59it/s]  6%|▌         | 22805/400000 [00:03<00:57, 6599.51it/s]  6%|▌         | 23476/400000 [00:03<00:56, 6629.69it/s]  6%|▌         | 24147/400000 [00:03<00:57, 6587.70it/s]  6%|▌         | 24858/400000 [00:03<00:55, 6734.55it/s]  6%|▋         | 25561/400000 [00:04<00:54, 6818.89it/s]  7%|▋         | 26305/400000 [00:04<00:53, 6992.94it/s]  7%|▋         | 27050/400000 [00:04<00:52, 7095.67it/s]  7%|▋         | 27790/400000 [00:04<00:51, 7181.75it/s]  7%|▋         | 28545/400000 [00:04<00:50, 7287.75it/s]  7%|▋         | 29276/400000 [00:04<00:50, 7286.55it/s]  8%|▊         | 30007/400000 [00:04<00:52, 6994.03it/s]  8%|▊         | 30710/400000 [00:04<00:53, 6910.12it/s]  8%|▊         | 31404/400000 [00:04<00:54, 6792.33it/s]  8%|▊         | 32086/400000 [00:04<00:54, 6736.56it/s]  8%|▊         | 32782/400000 [00:05<00:54, 6799.69it/s]  8%|▊         | 33538/400000 [00:05<00:52, 7009.94it/s]  9%|▊         | 34256/400000 [00:05<00:51, 7054.75it/s]  9%|▊         | 34964/400000 [00:05<00:52, 6923.26it/s]  9%|▉         | 35659/400000 [00:05<00:53, 6828.42it/s]  9%|▉         | 36344/400000 [00:05<00:53, 6779.00it/s]  9%|▉         | 37024/400000 [00:05<00:54, 6715.68it/s]  9%|▉         | 37697/400000 [00:05<00:54, 6679.36it/s] 10%|▉         | 38366/400000 [00:05<00:54, 6659.44it/s] 10%|▉         | 39033/400000 [00:06<00:54, 6614.93it/s] 10%|▉         | 39703/400000 [00:06<00:54, 6638.32it/s] 10%|█         | 40395/400000 [00:06<00:53, 6720.03it/s] 10%|█         | 41068/400000 [00:06<00:54, 6642.95it/s] 10%|█         | 41733/400000 [00:06<00:55, 6465.84it/s] 11%|█         | 42419/400000 [00:06<00:54, 6576.59it/s] 11%|█         | 43128/400000 [00:06<00:53, 6720.25it/s] 11%|█         | 43885/400000 [00:06<00:51, 6953.31it/s] 11%|█         | 44620/400000 [00:06<00:50, 7066.80it/s] 11%|█▏        | 45361/400000 [00:06<00:49, 7166.34it/s] 12%|█▏        | 46106/400000 [00:07<00:48, 7247.10it/s] 12%|█▏        | 46852/400000 [00:07<00:48, 7308.72it/s] 12%|█▏        | 47585/400000 [00:07<00:48, 7263.92it/s] 12%|█▏        | 48313/400000 [00:07<00:48, 7259.44it/s] 12%|█▏        | 49068/400000 [00:07<00:47, 7342.55it/s] 12%|█▏        | 49810/400000 [00:07<00:47, 7364.80it/s] 13%|█▎        | 50560/400000 [00:07<00:47, 7401.83it/s] 13%|█▎        | 51314/400000 [00:07<00:46, 7440.43it/s] 13%|█▎        | 52059/400000 [00:07<00:47, 7302.65it/s] 13%|█▎        | 52805/400000 [00:07<00:47, 7348.18it/s] 13%|█▎        | 53551/400000 [00:08<00:46, 7379.30it/s] 14%|█▎        | 54290/400000 [00:08<00:47, 7290.89it/s] 14%|█▍        | 55020/400000 [00:08<00:47, 7206.14it/s] 14%|█▍        | 55742/400000 [00:08<00:48, 7144.71it/s] 14%|█▍        | 56458/400000 [00:08<00:49, 6939.12it/s] 14%|█▍        | 57154/400000 [00:08<00:50, 6839.48it/s] 14%|█▍        | 57840/400000 [00:08<00:50, 6797.10it/s] 15%|█▍        | 58521/400000 [00:08<00:50, 6758.23it/s] 15%|█▍        | 59207/400000 [00:08<00:50, 6787.69it/s] 15%|█▍        | 59951/400000 [00:08<00:48, 6970.53it/s] 15%|█▌        | 60669/400000 [00:09<00:48, 7030.82it/s] 15%|█▌        | 61374/400000 [00:09<00:48, 6978.68it/s] 16%|█▌        | 62073/400000 [00:09<00:50, 6695.27it/s] 16%|█▌        | 62746/400000 [00:09<00:50, 6694.20it/s] 16%|█▌        | 63442/400000 [00:09<00:49, 6771.15it/s] 16%|█▌        | 64121/400000 [00:09<00:50, 6591.14it/s] 16%|█▌        | 64783/400000 [00:09<00:52, 6332.14it/s] 16%|█▋        | 65502/400000 [00:09<00:50, 6566.06it/s] 17%|█▋        | 66223/400000 [00:09<00:49, 6745.85it/s] 17%|█▋        | 66954/400000 [00:10<00:48, 6904.02it/s] 17%|█▋        | 67714/400000 [00:10<00:46, 7097.58it/s] 17%|█▋        | 68429/400000 [00:10<00:46, 7079.35it/s] 17%|█▋        | 69140/400000 [00:10<00:46, 7049.90it/s] 17%|█▋        | 69880/400000 [00:10<00:46, 7149.02it/s] 18%|█▊        | 70629/400000 [00:10<00:45, 7246.20it/s] 18%|█▊        | 71383/400000 [00:10<00:44, 7330.76it/s] 18%|█▊        | 72118/400000 [00:10<00:45, 7262.23it/s] 18%|█▊        | 72846/400000 [00:10<00:46, 7004.61it/s] 18%|█▊        | 73550/400000 [00:10<00:47, 6915.31it/s] 19%|█▊        | 74289/400000 [00:11<00:46, 7049.54it/s] 19%|█▉        | 75036/400000 [00:11<00:45, 7167.13it/s] 19%|█▉        | 75783/400000 [00:11<00:44, 7254.53it/s] 19%|█▉        | 76531/400000 [00:11<00:44, 7318.22it/s] 19%|█▉        | 77265/400000 [00:11<00:44, 7277.25it/s] 19%|█▉        | 77994/400000 [00:11<00:45, 7014.04it/s] 20%|█▉        | 78699/400000 [00:11<00:46, 6940.54it/s] 20%|█▉        | 79396/400000 [00:11<00:46, 6911.90it/s] 20%|██        | 80089/400000 [00:11<00:46, 6902.99it/s] 20%|██        | 80781/400000 [00:11<00:46, 6815.75it/s] 20%|██        | 81530/400000 [00:12<00:45, 7004.88it/s] 21%|██        | 82233/400000 [00:12<00:45, 6922.45it/s] 21%|██        | 82978/400000 [00:12<00:44, 7071.77it/s] 21%|██        | 83721/400000 [00:12<00:44, 7174.45it/s] 21%|██        | 84453/400000 [00:12<00:43, 7217.35it/s] 21%|██▏       | 85189/400000 [00:12<00:43, 7257.70it/s] 21%|██▏       | 85941/400000 [00:12<00:42, 7332.34it/s] 22%|██▏       | 86676/400000 [00:12<00:43, 7261.23it/s] 22%|██▏       | 87419/400000 [00:12<00:42, 7308.79it/s] 22%|██▏       | 88153/400000 [00:12<00:42, 7317.18it/s] 22%|██▏       | 88886/400000 [00:13<00:43, 7204.66it/s] 22%|██▏       | 89608/400000 [00:13<00:44, 6942.37it/s] 23%|██▎       | 90305/400000 [00:13<00:45, 6875.97it/s] 23%|██▎       | 90995/400000 [00:13<00:46, 6700.05it/s] 23%|██▎       | 91696/400000 [00:13<00:45, 6790.09it/s] 23%|██▎       | 92413/400000 [00:13<00:44, 6897.94it/s] 23%|██▎       | 93105/400000 [00:13<00:44, 6842.03it/s] 23%|██▎       | 93791/400000 [00:13<00:44, 6812.94it/s] 24%|██▎       | 94474/400000 [00:13<00:44, 6809.36it/s] 24%|██▍       | 95156/400000 [00:14<00:46, 6567.78it/s] 24%|██▍       | 95816/400000 [00:14<00:46, 6500.55it/s] 24%|██▍       | 96485/400000 [00:14<00:46, 6553.50it/s] 24%|██▍       | 97142/400000 [00:14<00:46, 6542.27it/s] 24%|██▍       | 97800/400000 [00:14<00:46, 6552.91it/s] 25%|██▍       | 98462/400000 [00:14<00:45, 6571.39it/s] 25%|██▍       | 99158/400000 [00:14<00:45, 6682.41it/s] 25%|██▍       | 99882/400000 [00:14<00:43, 6839.34it/s] 25%|██▌       | 100568/400000 [00:14<00:44, 6788.27it/s] 25%|██▌       | 101290/400000 [00:14<00:43, 6911.00it/s] 25%|██▌       | 101983/400000 [00:15<00:44, 6756.68it/s] 26%|██▌       | 102661/400000 [00:15<00:44, 6748.93it/s] 26%|██▌       | 103338/400000 [00:15<00:44, 6729.34it/s] 26%|██▌       | 104049/400000 [00:15<00:43, 6837.80it/s] 26%|██▌       | 104785/400000 [00:15<00:42, 6985.37it/s] 26%|██▋       | 105498/400000 [00:15<00:41, 7026.78it/s] 27%|██▋       | 106247/400000 [00:15<00:41, 7158.39it/s] 27%|██▋       | 106965/400000 [00:15<00:41, 7047.50it/s] 27%|██▋       | 107715/400000 [00:15<00:40, 7176.26it/s] 27%|██▋       | 108440/400000 [00:15<00:40, 7196.35it/s] 27%|██▋       | 109161/400000 [00:16<00:41, 7079.30it/s] 27%|██▋       | 109889/400000 [00:16<00:40, 7137.25it/s] 28%|██▊       | 110604/400000 [00:16<00:41, 7051.54it/s] 28%|██▊       | 111326/400000 [00:16<00:40, 7099.48it/s] 28%|██▊       | 112037/400000 [00:16<00:41, 6923.86it/s] 28%|██▊       | 112784/400000 [00:16<00:40, 7076.82it/s] 28%|██▊       | 113520/400000 [00:16<00:40, 7158.89it/s] 29%|██▊       | 114238/400000 [00:16<00:40, 7033.95it/s] 29%|██▊       | 114943/400000 [00:16<00:40, 6985.88it/s] 29%|██▉       | 115664/400000 [00:16<00:40, 7049.21it/s] 29%|██▉       | 116370/400000 [00:17<00:40, 6930.86it/s] 29%|██▉       | 117133/400000 [00:17<00:39, 7124.03it/s] 29%|██▉       | 117885/400000 [00:17<00:38, 7237.52it/s] 30%|██▉       | 118634/400000 [00:17<00:38, 7309.81it/s] 30%|██▉       | 119378/400000 [00:17<00:38, 7345.57it/s] 30%|███       | 120125/400000 [00:17<00:37, 7381.30it/s] 30%|███       | 120864/400000 [00:17<00:38, 7210.30it/s] 30%|███       | 121625/400000 [00:17<00:38, 7325.46it/s] 31%|███       | 122360/400000 [00:17<00:37, 7330.91it/s] 31%|███       | 123098/400000 [00:17<00:37, 7345.51it/s] 31%|███       | 123834/400000 [00:18<00:37, 7349.20it/s] 31%|███       | 124570/400000 [00:18<00:38, 7204.72it/s] 31%|███▏      | 125292/400000 [00:18<00:38, 7189.40it/s] 32%|███▏      | 126026/400000 [00:18<00:37, 7233.53it/s] 32%|███▏      | 126776/400000 [00:18<00:37, 7311.11it/s] 32%|███▏      | 127508/400000 [00:18<00:37, 7199.84it/s] 32%|███▏      | 128250/400000 [00:18<00:37, 7264.46it/s] 32%|███▏      | 129009/400000 [00:18<00:36, 7358.19it/s] 32%|███▏      | 129746/400000 [00:18<00:37, 7238.93it/s] 33%|███▎      | 130487/400000 [00:19<00:36, 7287.74it/s] 33%|███▎      | 131217/400000 [00:19<00:36, 7267.30it/s] 33%|███▎      | 131956/400000 [00:19<00:36, 7303.17it/s] 33%|███▎      | 132687/400000 [00:19<00:37, 7153.17it/s] 33%|███▎      | 133404/400000 [00:19<00:37, 7027.86it/s] 34%|███▎      | 134108/400000 [00:19<00:39, 6815.98it/s] 34%|███▎      | 134792/400000 [00:19<00:38, 6811.95it/s] 34%|███▍      | 135481/400000 [00:19<00:38, 6833.53it/s] 34%|███▍      | 136174/400000 [00:19<00:38, 6859.50it/s] 34%|███▍      | 136874/400000 [00:19<00:38, 6898.82it/s] 34%|███▍      | 137565/400000 [00:20<00:38, 6816.04it/s] 35%|███▍      | 138248/400000 [00:20<00:39, 6610.77it/s] 35%|███▍      | 138911/400000 [00:20<00:40, 6518.24it/s] 35%|███▍      | 139565/400000 [00:20<00:39, 6523.91it/s] 35%|███▌      | 140294/400000 [00:20<00:38, 6735.98it/s] 35%|███▌      | 141052/400000 [00:20<00:37, 6967.25it/s] 35%|███▌      | 141790/400000 [00:20<00:36, 7083.72it/s] 36%|███▌      | 142507/400000 [00:20<00:36, 7106.84it/s] 36%|███▌      | 143267/400000 [00:20<00:35, 7247.71it/s] 36%|███▌      | 144015/400000 [00:20<00:34, 7314.63it/s] 36%|███▌      | 144749/400000 [00:21<00:35, 7167.08it/s] 36%|███▋      | 145468/400000 [00:21<00:37, 6834.74it/s] 37%|███▋      | 146157/400000 [00:21<00:37, 6823.35it/s] 37%|███▋      | 146843/400000 [00:21<00:38, 6513.33it/s] 37%|███▋      | 147516/400000 [00:21<00:38, 6576.72it/s] 37%|███▋      | 148178/400000 [00:21<00:38, 6556.80it/s] 37%|███▋      | 148925/400000 [00:21<00:36, 6804.87it/s] 37%|███▋      | 149667/400000 [00:21<00:35, 6978.29it/s] 38%|███▊      | 150432/400000 [00:21<00:34, 7166.99it/s] 38%|███▊      | 151153/400000 [00:21<00:35, 7058.85it/s] 38%|███▊      | 151863/400000 [00:22<00:35, 6958.54it/s] 38%|███▊      | 152562/400000 [00:22<00:35, 6895.23it/s] 38%|███▊      | 153285/400000 [00:22<00:35, 6990.34it/s] 39%|███▊      | 154037/400000 [00:22<00:34, 7139.77it/s] 39%|███▊      | 154798/400000 [00:22<00:33, 7272.29it/s] 39%|███▉      | 155528/400000 [00:22<00:34, 7185.29it/s] 39%|███▉      | 156249/400000 [00:22<00:34, 7016.06it/s] 39%|███▉      | 156953/400000 [00:22<00:35, 6841.58it/s] 39%|███▉      | 157640/400000 [00:22<00:35, 6793.34it/s] 40%|███▉      | 158322/400000 [00:23<00:35, 6789.38it/s] 40%|███▉      | 159003/400000 [00:23<00:35, 6780.50it/s] 40%|███▉      | 159682/400000 [00:23<00:35, 6696.49it/s] 40%|████      | 160353/400000 [00:23<00:35, 6698.24it/s] 40%|████      | 161035/400000 [00:23<00:35, 6733.37it/s] 40%|████      | 161714/400000 [00:23<00:35, 6748.37it/s] 41%|████      | 162435/400000 [00:23<00:34, 6879.02it/s] 41%|████      | 163179/400000 [00:23<00:33, 7036.53it/s] 41%|████      | 163900/400000 [00:23<00:33, 7085.34it/s] 41%|████      | 164631/400000 [00:23<00:32, 7149.84it/s] 41%|████▏     | 165350/400000 [00:24<00:32, 7160.13it/s] 42%|████▏     | 166091/400000 [00:24<00:32, 7231.28it/s] 42%|████▏     | 166830/400000 [00:24<00:32, 7277.23it/s] 42%|████▏     | 167571/400000 [00:24<00:31, 7316.17it/s] 42%|████▏     | 168304/400000 [00:24<00:32, 7104.09it/s] 42%|████▏     | 169017/400000 [00:24<00:32, 7083.86it/s] 42%|████▏     | 169727/400000 [00:24<00:33, 6892.35it/s] 43%|████▎     | 170419/400000 [00:24<00:33, 6842.79it/s] 43%|████▎     | 171105/400000 [00:24<00:33, 6834.17it/s] 43%|████▎     | 171847/400000 [00:24<00:32, 6998.33it/s] 43%|████▎     | 172560/400000 [00:25<00:32, 7035.73it/s] 43%|████▎     | 173288/400000 [00:25<00:31, 7105.53it/s] 44%|████▎     | 174036/400000 [00:25<00:31, 7212.10it/s] 44%|████▎     | 174771/400000 [00:25<00:31, 7250.75it/s] 44%|████▍     | 175497/400000 [00:25<00:32, 6991.47it/s] 44%|████▍     | 176199/400000 [00:25<00:32, 6909.39it/s] 44%|████▍     | 176892/400000 [00:25<00:33, 6705.51it/s] 44%|████▍     | 177566/400000 [00:25<00:33, 6565.26it/s] 45%|████▍     | 178226/400000 [00:25<00:34, 6517.90it/s] 45%|████▍     | 178932/400000 [00:25<00:33, 6669.39it/s] 45%|████▍     | 179602/400000 [00:26<00:33, 6628.45it/s] 45%|████▌     | 180267/400000 [00:26<00:33, 6589.96it/s] 45%|████▌     | 180928/400000 [00:26<00:33, 6550.87it/s] 45%|████▌     | 181683/400000 [00:26<00:32, 6820.42it/s] 46%|████▌     | 182443/400000 [00:26<00:30, 7036.80it/s] 46%|████▌     | 183196/400000 [00:26<00:30, 7177.31it/s] 46%|████▌     | 183918/400000 [00:26<00:30, 6971.30it/s] 46%|████▌     | 184619/400000 [00:26<00:31, 6756.12it/s] 46%|████▋     | 185299/400000 [00:26<00:31, 6716.97it/s] 46%|████▋     | 185978/400000 [00:27<00:31, 6736.91it/s] 47%|████▋     | 186655/400000 [00:27<00:31, 6744.15it/s] 47%|████▋     | 187331/400000 [00:27<00:31, 6697.01it/s] 47%|████▋     | 188003/400000 [00:27<00:31, 6703.66it/s] 47%|████▋     | 188717/400000 [00:27<00:30, 6827.71it/s] 47%|████▋     | 189401/400000 [00:27<00:31, 6715.29it/s] 48%|████▊     | 190074/400000 [00:27<00:31, 6573.08it/s] 48%|████▊     | 190795/400000 [00:27<00:30, 6751.76it/s] 48%|████▊     | 191533/400000 [00:27<00:30, 6927.14it/s] 48%|████▊     | 192252/400000 [00:27<00:29, 7002.16it/s] 48%|████▊     | 192965/400000 [00:28<00:29, 7039.87it/s] 48%|████▊     | 193687/400000 [00:28<00:29, 7091.49it/s] 49%|████▊     | 194423/400000 [00:28<00:28, 7169.09it/s] 49%|████▉     | 195170/400000 [00:28<00:28, 7255.62it/s] 49%|████▉     | 195897/400000 [00:28<00:28, 7068.86it/s] 49%|████▉     | 196606/400000 [00:28<00:29, 6969.53it/s] 49%|████▉     | 197317/400000 [00:28<00:28, 7009.29it/s] 50%|████▉     | 198020/400000 [00:28<00:28, 6971.54it/s] 50%|████▉     | 198774/400000 [00:28<00:28, 7132.34it/s] 50%|████▉     | 199531/400000 [00:28<00:27, 7257.49it/s] 50%|█████     | 200290/400000 [00:29<00:27, 7352.48it/s] 50%|█████     | 201035/400000 [00:29<00:26, 7376.80it/s] 50%|█████     | 201774/400000 [00:29<00:27, 7267.85it/s] 51%|█████     | 202502/400000 [00:29<00:28, 6992.65it/s] 51%|█████     | 203205/400000 [00:29<00:28, 6845.25it/s] 51%|█████     | 203893/400000 [00:29<00:29, 6724.36it/s] 51%|█████     | 204579/400000 [00:29<00:28, 6764.44it/s] 51%|█████▏    | 205259/400000 [00:29<00:28, 6774.04it/s] 51%|█████▏    | 205938/400000 [00:29<00:28, 6750.99it/s] 52%|█████▏    | 206614/400000 [00:30<00:29, 6613.89it/s] 52%|█████▏    | 207284/400000 [00:30<00:29, 6637.38it/s] 52%|█████▏    | 207996/400000 [00:30<00:28, 6774.10it/s] 52%|█████▏    | 208711/400000 [00:30<00:27, 6880.50it/s] 52%|█████▏    | 209458/400000 [00:30<00:27, 7046.42it/s] 53%|█████▎    | 210213/400000 [00:30<00:26, 7188.78it/s] 53%|█████▎    | 210934/400000 [00:30<00:26, 7136.89it/s] 53%|█████▎    | 211669/400000 [00:30<00:26, 7198.27it/s] 53%|█████▎    | 212391/400000 [00:30<00:26, 7151.40it/s] 53%|█████▎    | 213142/400000 [00:30<00:25, 7253.20it/s] 53%|█████▎    | 213898/400000 [00:31<00:25, 7339.92it/s] 54%|█████▎    | 214654/400000 [00:31<00:25, 7403.92it/s] 54%|█████▍    | 215396/400000 [00:31<00:25, 7270.02it/s] 54%|█████▍    | 216129/400000 [00:31<00:25, 7286.48it/s] 54%|█████▍    | 216875/400000 [00:31<00:24, 7336.54it/s] 54%|█████▍    | 217635/400000 [00:31<00:24, 7411.94it/s] 55%|█████▍    | 218377/400000 [00:31<00:24, 7316.75it/s] 55%|█████▍    | 219110/400000 [00:31<00:24, 7312.78it/s] 55%|█████▍    | 219842/400000 [00:31<00:24, 7250.93it/s] 55%|█████▌    | 220596/400000 [00:31<00:24, 7332.29it/s] 55%|█████▌    | 221356/400000 [00:32<00:24, 7408.54it/s] 56%|█████▌    | 222113/400000 [00:32<00:23, 7455.23it/s] 56%|█████▌    | 222859/400000 [00:32<00:24, 7283.21it/s] 56%|█████▌    | 223589/400000 [00:32<00:24, 7143.41it/s] 56%|█████▌    | 224348/400000 [00:32<00:24, 7269.00it/s] 56%|█████▋    | 225091/400000 [00:32<00:23, 7314.23it/s] 56%|█████▋    | 225844/400000 [00:32<00:23, 7375.79it/s] 57%|█████▋    | 226588/400000 [00:32<00:23, 7394.11it/s] 57%|█████▋    | 227329/400000 [00:32<00:24, 7174.83it/s] 57%|█████▋    | 228049/400000 [00:32<00:24, 7154.49it/s] 57%|█████▋    | 228795/400000 [00:33<00:23, 7240.60it/s] 57%|█████▋    | 229536/400000 [00:33<00:23, 7288.86it/s] 58%|█████▊    | 230291/400000 [00:33<00:23, 7364.78it/s] 58%|█████▊    | 231030/400000 [00:33<00:22, 7370.52it/s] 58%|█████▊    | 231768/400000 [00:33<00:22, 7358.71it/s] 58%|█████▊    | 232505/400000 [00:33<00:23, 7279.08it/s] 58%|█████▊    | 233258/400000 [00:33<00:22, 7351.11it/s] 58%|█████▊    | 233994/400000 [00:33<00:22, 7259.76it/s] 59%|█████▊    | 234721/400000 [00:33<00:23, 7175.36it/s] 59%|█████▉    | 235472/400000 [00:33<00:22, 7272.58it/s] 59%|█████▉    | 236201/400000 [00:34<00:23, 6925.95it/s] 59%|█████▉    | 236898/400000 [00:34<00:23, 6797.27it/s] 59%|█████▉    | 237582/400000 [00:34<00:24, 6697.15it/s] 60%|█████▉    | 238255/400000 [00:34<00:24, 6696.70it/s] 60%|█████▉    | 238927/400000 [00:34<00:24, 6652.18it/s] 60%|█████▉    | 239602/400000 [00:34<00:24, 6679.25it/s] 60%|██████    | 240341/400000 [00:34<00:23, 6876.22it/s] 60%|██████    | 241049/400000 [00:34<00:22, 6934.87it/s] 60%|██████    | 241773/400000 [00:34<00:22, 7021.21it/s] 61%|██████    | 242522/400000 [00:34<00:22, 7152.80it/s] 61%|██████    | 243282/400000 [00:35<00:21, 7280.00it/s] 61%|██████    | 244038/400000 [00:35<00:21, 7359.18it/s] 61%|██████    | 244776/400000 [00:35<00:21, 7269.75it/s] 61%|██████▏   | 245505/400000 [00:35<00:22, 6876.66it/s] 62%|██████▏   | 246198/400000 [00:35<00:22, 6853.33it/s] 62%|██████▏   | 246889/400000 [00:35<00:22, 6870.19it/s] 62%|██████▏   | 247579/400000 [00:35<00:22, 6835.37it/s] 62%|██████▏   | 248265/400000 [00:35<00:23, 6576.40it/s] 62%|██████▏   | 248927/400000 [00:35<00:22, 6569.31it/s] 62%|██████▏   | 249587/400000 [00:36<00:23, 6483.94it/s] 63%|██████▎   | 250271/400000 [00:36<00:22, 6585.66it/s] 63%|██████▎   | 250989/400000 [00:36<00:22, 6750.82it/s] 63%|██████▎   | 251687/400000 [00:36<00:21, 6816.43it/s] 63%|██████▎   | 252404/400000 [00:36<00:21, 6915.66it/s] 63%|██████▎   | 253098/400000 [00:36<00:21, 6849.22it/s] 63%|██████▎   | 253785/400000 [00:36<00:21, 6754.14it/s] 64%|██████▎   | 254473/400000 [00:36<00:21, 6789.67it/s] 64%|██████▍   | 255189/400000 [00:36<00:20, 6896.09it/s] 64%|██████▍   | 255937/400000 [00:36<00:20, 7060.09it/s] 64%|██████▍   | 256645/400000 [00:37<00:20, 6984.79it/s] 64%|██████▍   | 257345/400000 [00:37<00:20, 6986.86it/s] 65%|██████▍   | 258045/400000 [00:37<00:20, 6892.05it/s] 65%|██████▍   | 258736/400000 [00:37<00:20, 6877.78it/s] 65%|██████▍   | 259425/400000 [00:37<00:20, 6822.45it/s] 65%|██████▌   | 260108/400000 [00:37<00:20, 6800.46it/s] 65%|██████▌   | 260789/400000 [00:37<00:20, 6772.20it/s] 65%|██████▌   | 261467/400000 [00:37<00:20, 6744.88it/s] 66%|██████▌   | 262142/400000 [00:37<00:20, 6617.35it/s] 66%|██████▌   | 262896/400000 [00:37<00:19, 6867.15it/s] 66%|██████▌   | 263637/400000 [00:38<00:19, 7020.11it/s] 66%|██████▌   | 264342/400000 [00:38<00:19, 7012.92it/s] 66%|██████▋   | 265100/400000 [00:38<00:18, 7172.93it/s] 66%|██████▋   | 265821/400000 [00:38<00:18, 7182.42it/s] 67%|██████▋   | 266541/400000 [00:38<00:18, 7139.22it/s] 67%|██████▋   | 267257/400000 [00:38<00:18, 7071.09it/s] 67%|██████▋   | 267966/400000 [00:38<00:19, 6948.48it/s] 67%|██████▋   | 268711/400000 [00:38<00:18, 7089.18it/s] 67%|██████▋   | 269422/400000 [00:38<00:19, 6809.46it/s] 68%|██████▊   | 270107/400000 [00:39<00:19, 6586.69it/s] 68%|██████▊   | 270793/400000 [00:39<00:19, 6664.37it/s] 68%|██████▊   | 271548/400000 [00:39<00:18, 6905.69it/s] 68%|██████▊   | 272289/400000 [00:39<00:18, 7049.01it/s] 68%|██████▊   | 272998/400000 [00:39<00:18, 6987.70it/s] 68%|██████▊   | 273700/400000 [00:39<00:18, 6938.28it/s] 69%|██████▊   | 274396/400000 [00:39<00:18, 6838.08it/s] 69%|██████▉   | 275082/400000 [00:39<00:18, 6669.22it/s] 69%|██████▉   | 275754/400000 [00:39<00:18, 6682.49it/s] 69%|██████▉   | 276442/400000 [00:39<00:18, 6737.95it/s] 69%|██████▉   | 277151/400000 [00:40<00:17, 6838.82it/s] 69%|██████▉   | 277909/400000 [00:40<00:17, 7044.33it/s] 70%|██████▉   | 278669/400000 [00:40<00:16, 7201.23it/s] 70%|██████▉   | 279392/400000 [00:40<00:16, 7198.02it/s] 70%|███████   | 280136/400000 [00:40<00:16, 7268.88it/s] 70%|███████   | 280881/400000 [00:40<00:16, 7320.40it/s] 70%|███████   | 281615/400000 [00:40<00:16, 7164.65it/s] 71%|███████   | 282333/400000 [00:40<00:16, 7033.00it/s] 71%|███████   | 283038/400000 [00:40<00:17, 6752.97it/s] 71%|███████   | 283717/400000 [00:40<00:17, 6692.79it/s] 71%|███████   | 284389/400000 [00:41<00:17, 6430.67it/s] 71%|███████▏  | 285059/400000 [00:41<00:17, 6508.89it/s] 71%|███████▏  | 285765/400000 [00:41<00:17, 6663.82it/s] 72%|███████▏  | 286438/400000 [00:41<00:16, 6682.88it/s] 72%|███████▏  | 287137/400000 [00:41<00:16, 6771.28it/s] 72%|███████▏  | 287816/400000 [00:41<00:16, 6721.14it/s] 72%|███████▏  | 288545/400000 [00:41<00:16, 6881.89it/s] 72%|███████▏  | 289292/400000 [00:41<00:15, 7046.90it/s] 72%|███████▏  | 289999/400000 [00:41<00:15, 7018.49it/s] 73%|███████▎  | 290703/400000 [00:42<00:16, 6811.71it/s] 73%|███████▎  | 291395/400000 [00:42<00:15, 6841.62it/s] 73%|███████▎  | 292087/400000 [00:42<00:15, 6862.58it/s] 73%|███████▎  | 292838/400000 [00:42<00:15, 7044.75it/s] 73%|███████▎  | 293565/400000 [00:42<00:14, 7109.74it/s] 74%|███████▎  | 294300/400000 [00:42<00:14, 7178.23it/s] 74%|███████▍  | 295020/400000 [00:42<00:15, 6903.35it/s] 74%|███████▍  | 295714/400000 [00:42<00:15, 6638.87it/s] 74%|███████▍  | 296383/400000 [00:42<00:15, 6491.31it/s] 74%|███████▍  | 297124/400000 [00:42<00:15, 6741.88it/s] 74%|███████▍  | 297848/400000 [00:43<00:14, 6883.65it/s] 75%|███████▍  | 298595/400000 [00:43<00:14, 7046.98it/s] 75%|███████▍  | 299304/400000 [00:43<00:14, 7051.31it/s] 75%|███████▌  | 300012/400000 [00:43<00:14, 6801.06it/s] 75%|███████▌  | 300696/400000 [00:43<00:15, 6565.65it/s] 75%|███████▌  | 301446/400000 [00:43<00:14, 6818.38it/s] 76%|███████▌  | 302134/400000 [00:43<00:14, 6776.13it/s] 76%|███████▌  | 302816/400000 [00:43<00:14, 6770.74it/s] 76%|███████▌  | 303496/400000 [00:43<00:14, 6729.92it/s] 76%|███████▌  | 304172/400000 [00:43<00:14, 6664.04it/s] 76%|███████▌  | 304840/400000 [00:44<00:14, 6597.26it/s] 76%|███████▋  | 305522/400000 [00:44<00:14, 6662.23it/s] 77%|███████▋  | 306191/400000 [00:44<00:14, 6669.78it/s] 77%|███████▋  | 306889/400000 [00:44<00:13, 6758.11it/s] 77%|███████▋  | 307621/400000 [00:44<00:13, 6915.79it/s] 77%|███████▋  | 308373/400000 [00:44<00:12, 7086.07it/s] 77%|███████▋  | 309094/400000 [00:44<00:12, 7122.23it/s] 77%|███████▋  | 309812/400000 [00:44<00:12, 7138.03it/s] 78%|███████▊  | 310560/400000 [00:44<00:12, 7236.58it/s] 78%|███████▊  | 311301/400000 [00:44<00:12, 7279.82it/s] 78%|███████▊  | 312030/400000 [00:45<00:12, 7126.24it/s] 78%|███████▊  | 312744/400000 [00:45<00:12, 6999.69it/s] 78%|███████▊  | 313446/400000 [00:45<00:12, 6974.23it/s] 79%|███████▊  | 314195/400000 [00:45<00:12, 7119.63it/s] 79%|███████▊  | 314909/400000 [00:45<00:12, 7005.18it/s] 79%|███████▉  | 315611/400000 [00:45<00:12, 6957.66it/s] 79%|███████▉  | 316308/400000 [00:45<00:12, 6937.50it/s] 79%|███████▉  | 317003/400000 [00:45<00:12, 6902.97it/s] 79%|███████▉  | 317694/400000 [00:45<00:12, 6821.10it/s] 80%|███████▉  | 318377/400000 [00:46<00:12, 6732.75it/s] 80%|███████▉  | 319080/400000 [00:46<00:11, 6818.76it/s] 80%|███████▉  | 319787/400000 [00:46<00:11, 6891.26it/s] 80%|████████  | 320553/400000 [00:46<00:11, 7102.79it/s] 80%|████████  | 321302/400000 [00:46<00:10, 7214.16it/s] 81%|████████  | 322026/400000 [00:46<00:10, 7160.25it/s] 81%|████████  | 322766/400000 [00:46<00:10, 7230.37it/s] 81%|████████  | 323515/400000 [00:46<00:10, 7305.63it/s] 81%|████████  | 324276/400000 [00:46<00:10, 7391.94it/s] 81%|████████▏ | 325034/400000 [00:46<00:10, 7444.85it/s] 81%|████████▏ | 325780/400000 [00:47<00:10, 7230.37it/s] 82%|████████▏ | 326505/400000 [00:47<00:10, 7162.61it/s] 82%|████████▏ | 327223/400000 [00:47<00:10, 7059.01it/s] 82%|████████▏ | 327931/400000 [00:47<00:10, 6976.40it/s] 82%|████████▏ | 328630/400000 [00:47<00:10, 6928.00it/s] 82%|████████▏ | 329324/400000 [00:47<00:10, 6834.89it/s] 83%|████████▎ | 330068/400000 [00:47<00:09, 7003.83it/s] 83%|████████▎ | 330780/400000 [00:47<00:09, 7037.06it/s] 83%|████████▎ | 331525/400000 [00:47<00:09, 7153.12it/s] 83%|████████▎ | 332242/400000 [00:47<00:09, 7112.63it/s] 83%|████████▎ | 332961/400000 [00:48<00:09, 7134.72it/s] 83%|████████▎ | 333676/400000 [00:48<00:09, 7004.45it/s] 84%|████████▎ | 334378/400000 [00:48<00:09, 6909.31it/s] 84%|████████▍ | 335070/400000 [00:48<00:09, 6889.09it/s] 84%|████████▍ | 335763/400000 [00:48<00:09, 6900.39it/s] 84%|████████▍ | 336454/400000 [00:48<00:09, 6832.60it/s] 84%|████████▍ | 337182/400000 [00:48<00:09, 6952.93it/s] 84%|████████▍ | 337935/400000 [00:48<00:08, 7116.03it/s] 85%|████████▍ | 338686/400000 [00:48<00:08, 7228.01it/s] 85%|████████▍ | 339411/400000 [00:48<00:08, 7223.42it/s] 85%|████████▌ | 340135/400000 [00:49<00:08, 7104.52it/s] 85%|████████▌ | 340860/400000 [00:49<00:08, 7145.47it/s] 85%|████████▌ | 341592/400000 [00:49<00:08, 7196.31it/s] 86%|████████▌ | 342342/400000 [00:49<00:07, 7284.48it/s] 86%|████████▌ | 343072/400000 [00:49<00:07, 7240.62it/s] 86%|████████▌ | 343804/400000 [00:49<00:07, 7261.03it/s] 86%|████████▌ | 344536/400000 [00:49<00:07, 7278.15it/s] 86%|████████▋ | 345291/400000 [00:49<00:07, 7348.03it/s] 87%|████████▋ | 346052/400000 [00:49<00:07, 7422.69it/s] 87%|████████▋ | 346813/400000 [00:49<00:07, 7476.16it/s] 87%|████████▋ | 347562/400000 [00:50<00:07, 7347.97it/s] 87%|████████▋ | 348314/400000 [00:50<00:06, 7396.67it/s] 87%|████████▋ | 349055/400000 [00:50<00:07, 7253.21it/s] 87%|████████▋ | 349782/400000 [00:50<00:07, 7153.95it/s] 88%|████████▊ | 350499/400000 [00:50<00:06, 7081.83it/s] 88%|████████▊ | 351209/400000 [00:50<00:06, 7027.88it/s] 88%|████████▊ | 351948/400000 [00:50<00:06, 7131.37it/s] 88%|████████▊ | 352689/400000 [00:50<00:06, 7212.02it/s] 88%|████████▊ | 353414/400000 [00:50<00:06, 7220.66it/s] 89%|████████▊ | 354161/400000 [00:50<00:06, 7292.67it/s] 89%|████████▊ | 354891/400000 [00:51<00:06, 7172.19it/s] 89%|████████▉ | 355610/400000 [00:51<00:06, 7005.60it/s] 89%|████████▉ | 356313/400000 [00:51<00:06, 6841.79it/s] 89%|████████▉ | 357000/400000 [00:51<00:06, 6646.61it/s] 89%|████████▉ | 357707/400000 [00:51<00:06, 6766.48it/s] 90%|████████▉ | 358405/400000 [00:51<00:06, 6826.71it/s] 90%|████████▉ | 359154/400000 [00:51<00:05, 7011.78it/s] 90%|████████▉ | 359908/400000 [00:51<00:05, 7162.02it/s] 90%|█████████ | 360627/400000 [00:51<00:05, 7135.84it/s] 90%|█████████ | 361378/400000 [00:52<00:05, 7242.28it/s] 91%|█████████ | 362104/400000 [00:52<00:05, 7113.08it/s] 91%|█████████ | 362817/400000 [00:52<00:05, 7067.37it/s] 91%|█████████ | 363525/400000 [00:52<00:05, 6961.26it/s] 91%|█████████ | 364223/400000 [00:52<00:05, 6913.80it/s] 91%|█████████ | 364916/400000 [00:52<00:05, 6890.36it/s] 91%|█████████▏| 365654/400000 [00:52<00:04, 7030.01it/s] 92%|█████████▏| 366403/400000 [00:52<00:04, 7160.96it/s] 92%|█████████▏| 367125/400000 [00:52<00:04, 7175.59it/s] 92%|█████████▏| 367844/400000 [00:52<00:04, 6960.62it/s] 92%|█████████▏| 368543/400000 [00:53<00:04, 6910.32it/s] 92%|█████████▏| 369236/400000 [00:53<00:04, 6862.10it/s] 93%|█████████▎| 370005/400000 [00:53<00:04, 7091.03it/s] 93%|█████████▎| 370765/400000 [00:53<00:04, 7234.04it/s] 93%|█████████▎| 371524/400000 [00:53<00:03, 7334.60it/s] 93%|█████████▎| 372296/400000 [00:53<00:03, 7444.37it/s] 93%|█████████▎| 373043/400000 [00:53<00:03, 7265.99it/s] 93%|█████████▎| 373772/400000 [00:53<00:03, 7102.39it/s] 94%|█████████▎| 374485/400000 [00:53<00:03, 6995.55it/s] 94%|█████████▍| 375187/400000 [00:53<00:03, 6906.59it/s] 94%|█████████▍| 375880/400000 [00:54<00:03, 6809.44it/s] 94%|█████████▍| 376563/400000 [00:54<00:03, 6812.80it/s] 94%|█████████▍| 377246/400000 [00:54<00:03, 6753.44it/s] 94%|█████████▍| 377926/400000 [00:54<00:03, 6767.21it/s] 95%|█████████▍| 378608/400000 [00:54<00:03, 6782.11it/s] 95%|█████████▍| 379340/400000 [00:54<00:02, 6932.69it/s] 95%|█████████▌| 380079/400000 [00:54<00:02, 7062.87it/s] 95%|█████████▌| 380839/400000 [00:54<00:02, 7214.23it/s] 95%|█████████▌| 381565/400000 [00:54<00:02, 7226.08it/s] 96%|█████████▌| 382311/400000 [00:55<00:02, 7292.83it/s] 96%|█████████▌| 383057/400000 [00:55<00:02, 7340.42it/s] 96%|█████████▌| 383797/400000 [00:55<00:02, 7357.22it/s] 96%|█████████▌| 384562/400000 [00:55<00:02, 7441.96it/s] 96%|█████████▋| 385321/400000 [00:55<00:01, 7483.16it/s] 97%|█████████▋| 386070/400000 [00:55<00:01, 7410.70it/s] 97%|█████████▋| 386812/400000 [00:55<00:01, 7316.34it/s] 97%|█████████▋| 387545/400000 [00:55<00:01, 7174.70it/s] 97%|█████████▋| 388264/400000 [00:55<00:01, 7081.67it/s] 97%|█████████▋| 388974/400000 [00:55<00:01, 7082.97it/s] 97%|█████████▋| 389684/400000 [00:56<00:01, 7069.21it/s] 98%|█████████▊| 390408/400000 [00:56<00:01, 7119.22it/s] 98%|█████████▊| 391135/400000 [00:56<00:01, 7163.15it/s] 98%|█████████▊| 391886/400000 [00:56<00:01, 7262.79it/s] 98%|█████████▊| 392624/400000 [00:56<00:01, 7293.93it/s] 98%|█████████▊| 393359/400000 [00:56<00:00, 7307.58it/s] 99%|█████████▊| 394091/400000 [00:56<00:00, 7097.94it/s] 99%|█████████▊| 394803/400000 [00:56<00:00, 7007.07it/s] 99%|█████████▉| 395514/400000 [00:56<00:00, 7035.71it/s] 99%|█████████▉| 396219/400000 [00:56<00:00, 7033.69it/s] 99%|█████████▉| 396924/400000 [00:57<00:00, 7006.71it/s] 99%|█████████▉| 397631/400000 [00:57<00:00, 7023.07it/s]100%|█████████▉| 398334/400000 [00:57<00:00, 6932.46it/s]100%|█████████▉| 399029/400000 [00:57<00:00, 6936.22it/s]100%|█████████▉| 399723/400000 [00:57<00:00, 6923.65it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 6958.39it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5f469fe160>, <torchtext.data.dataset.TabularDataset object at 0x7f5f469fe2b0>, <torchtext.vocab.Vocab object at 0x7f5f469fe1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.37 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.37 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.37 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.83 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.14 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.23 file/s]2020-06-13 18:18:58.889195: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 18:18:58.894595: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-13 18:18:58.894820: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd4ab061d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 18:18:58.894840: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3366912/9912422 [00:00<00:00, 33650619.12it/s]9920512it [00:00, 37437058.38it/s]                             
0it [00:00, ?it/s]32768it [00:00, 624498.26it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157695.58it/s]1654784it [00:00, 11242932.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202511.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
