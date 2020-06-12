
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f310f7f76a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f310f7f76a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f317adbf0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f317adbf0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3194aebea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3194aebea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f312863a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f312863a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f312863a950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3686400/9912422 [00:00<00:00, 36624864.13it/s]9920512it [00:00, 34604254.14it/s]                             
0it [00:00, ?it/s]32768it [00:00, 546225.02it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475542.08it/s]1654784it [00:00, 11850795.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202546.22it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3128627a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3125b80e10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f312863a598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f312863a598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f312863a598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:29,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:29,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:28,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:28,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:27,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:27,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:26,  1.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:00,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:59,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:58,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:58,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:58,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:57,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:57,  2.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:40,  3.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:39,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:38,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:38,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:38,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:38,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:37,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:26,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:26,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:26,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:25,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:25,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:25,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:24,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  7.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10,  9.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 13.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 55.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 55.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.94 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.37s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.94 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.20s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.61 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.20s/ url]
0 examples [00:00, ? examples/s]2020-06-12 18:08:57.693622: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 18:08:57.707556: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-12 18:08:57.708292: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559982f61270 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 18:08:57.708308: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
6 examples [00:00, 59.50 examples/s]115 examples [00:00, 83.05 examples/s]226 examples [00:00, 114.96 examples/s]332 examples [00:00, 156.93 examples/s]439 examples [00:00, 210.87 examples/s]555 examples [00:00, 279.42 examples/s]671 examples [00:00, 361.82 examples/s]793 examples [00:00, 458.24 examples/s]902 examples [00:00, 554.62 examples/s]1014 examples [00:01, 653.11 examples/s]1139 examples [00:01, 762.14 examples/s]1265 examples [00:01, 863.77 examples/s]1382 examples [00:01, 932.54 examples/s]1503 examples [00:01, 1000.63 examples/s]1625 examples [00:01, 1057.27 examples/s]1754 examples [00:01, 1115.41 examples/s]1876 examples [00:01, 1132.57 examples/s]1997 examples [00:01, 1144.65 examples/s]2117 examples [00:01, 1138.08 examples/s]2243 examples [00:02, 1168.50 examples/s]2372 examples [00:02, 1200.29 examples/s]2495 examples [00:02, 1199.36 examples/s]2617 examples [00:02, 1199.01 examples/s]2738 examples [00:02, 1177.03 examples/s]2857 examples [00:02, 1168.86 examples/s]2980 examples [00:02, 1184.50 examples/s]3103 examples [00:02, 1195.53 examples/s]3225 examples [00:02, 1202.23 examples/s]3349 examples [00:02, 1211.21 examples/s]3471 examples [00:03, 1204.72 examples/s]3592 examples [00:03, 1192.61 examples/s]3712 examples [00:03, 1186.26 examples/s]3832 examples [00:03, 1187.06 examples/s]3951 examples [00:03, 1185.29 examples/s]4070 examples [00:03, 1182.95 examples/s]4189 examples [00:03, 1171.14 examples/s]4307 examples [00:03, 1167.48 examples/s]4424 examples [00:03, 1159.33 examples/s]4547 examples [00:03, 1178.30 examples/s]4665 examples [00:04, 1135.52 examples/s]4788 examples [00:04, 1159.89 examples/s]4907 examples [00:04, 1167.12 examples/s]5025 examples [00:04, 1009.14 examples/s]5130 examples [00:04, 966.85 examples/s] 5236 examples [00:04, 990.14 examples/s]5338 examples [00:04, 994.62 examples/s]5440 examples [00:04, 995.22 examples/s]5541 examples [00:04, 972.86 examples/s]5640 examples [00:05, 956.17 examples/s]5745 examples [00:05, 981.91 examples/s]5850 examples [00:05, 1000.44 examples/s]5968 examples [00:05, 1046.70 examples/s]6081 examples [00:05, 1070.17 examples/s]6190 examples [00:05, 1074.95 examples/s]6299 examples [00:05, 1073.75 examples/s]6413 examples [00:05, 1090.05 examples/s]6528 examples [00:05, 1101.57 examples/s]6644 examples [00:05, 1115.93 examples/s]6766 examples [00:06, 1143.53 examples/s]6889 examples [00:06, 1167.70 examples/s]7008 examples [00:06, 1172.54 examples/s]7126 examples [00:06, 1160.75 examples/s]7243 examples [00:06, 1147.18 examples/s]7362 examples [00:06, 1158.42 examples/s]7478 examples [00:06, 1152.69 examples/s]7594 examples [00:06, 1151.95 examples/s]7710 examples [00:06, 1136.08 examples/s]7824 examples [00:06, 1117.91 examples/s]7939 examples [00:07, 1114.17 examples/s]8057 examples [00:07, 1132.04 examples/s]8174 examples [00:07, 1142.85 examples/s]8289 examples [00:07, 1144.70 examples/s]8404 examples [00:07, 1120.77 examples/s]8520 examples [00:07, 1129.99 examples/s]8637 examples [00:07, 1141.13 examples/s]8757 examples [00:07, 1157.83 examples/s]8879 examples [00:07, 1174.76 examples/s]8997 examples [00:08, 1166.65 examples/s]9117 examples [00:08, 1176.08 examples/s]9237 examples [00:08, 1180.52 examples/s]9358 examples [00:08, 1186.75 examples/s]9478 examples [00:08, 1188.39 examples/s]9597 examples [00:08, 1174.07 examples/s]9715 examples [00:08, 1169.61 examples/s]9836 examples [00:08, 1179.62 examples/s]9960 examples [00:08, 1195.71 examples/s]10080 examples [00:08, 1128.98 examples/s]10198 examples [00:09, 1142.88 examples/s]10318 examples [00:09, 1158.63 examples/s]10437 examples [00:09, 1165.20 examples/s]10559 examples [00:09, 1180.13 examples/s]10678 examples [00:09, 1182.95 examples/s]10797 examples [00:09, 1171.72 examples/s]10916 examples [00:09, 1175.16 examples/s]11034 examples [00:09, 1169.53 examples/s]11153 examples [00:09, 1173.04 examples/s]11272 examples [00:09, 1176.98 examples/s]11390 examples [00:10, 1152.82 examples/s]11506 examples [00:10, 1120.41 examples/s]11620 examples [00:10, 1124.34 examples/s]11737 examples [00:10, 1134.99 examples/s]11851 examples [00:10, 1118.42 examples/s]11972 examples [00:10, 1143.40 examples/s]12090 examples [00:10, 1153.58 examples/s]12210 examples [00:10, 1166.34 examples/s]12327 examples [00:10, 1149.83 examples/s]12443 examples [00:10, 1133.40 examples/s]12557 examples [00:11, 1129.96 examples/s]12671 examples [00:11, 1132.80 examples/s]12785 examples [00:11, 1131.28 examples/s]12902 examples [00:11, 1141.62 examples/s]13017 examples [00:11, 1140.43 examples/s]13139 examples [00:11, 1160.66 examples/s]13257 examples [00:11, 1164.54 examples/s]13374 examples [00:11, 1156.35 examples/s]13490 examples [00:11, 1149.05 examples/s]13605 examples [00:12, 1123.88 examples/s]13718 examples [00:12, 1089.03 examples/s]13828 examples [00:12, 1078.09 examples/s]13942 examples [00:12, 1095.15 examples/s]14062 examples [00:12, 1124.15 examples/s]14183 examples [00:12, 1145.50 examples/s]14302 examples [00:12, 1158.28 examples/s]14430 examples [00:12, 1189.85 examples/s]14550 examples [00:12, 1190.63 examples/s]14670 examples [00:12, 1178.13 examples/s]14789 examples [00:13, 1165.34 examples/s]14906 examples [00:13, 1109.52 examples/s]15018 examples [00:13, 1108.47 examples/s]15136 examples [00:13, 1128.91 examples/s]15258 examples [00:13, 1152.43 examples/s]15377 examples [00:13, 1160.69 examples/s]15495 examples [00:13, 1164.57 examples/s]15612 examples [00:13, 1160.54 examples/s]15733 examples [00:13, 1172.35 examples/s]15851 examples [00:13, 1173.22 examples/s]15969 examples [00:14, 1163.73 examples/s]16088 examples [00:14, 1169.16 examples/s]16211 examples [00:14, 1184.94 examples/s]16335 examples [00:14, 1200.65 examples/s]16458 examples [00:14, 1207.12 examples/s]16579 examples [00:14, 1194.10 examples/s]16699 examples [00:14, 1172.24 examples/s]16817 examples [00:14, 1150.44 examples/s]16933 examples [00:14, 1114.51 examples/s]17045 examples [00:14, 1112.14 examples/s]17157 examples [00:15, 1110.87 examples/s]17272 examples [00:15, 1120.16 examples/s]17390 examples [00:15, 1137.13 examples/s]17520 examples [00:15, 1180.21 examples/s]17639 examples [00:15, 1178.96 examples/s]17758 examples [00:15, 1161.87 examples/s]17875 examples [00:15, 1153.04 examples/s]17992 examples [00:15, 1156.63 examples/s]18108 examples [00:15, 1151.60 examples/s]18229 examples [00:16, 1167.90 examples/s]18352 examples [00:16, 1184.97 examples/s]18472 examples [00:16, 1189.28 examples/s]18592 examples [00:16, 1145.64 examples/s]18707 examples [00:16, 1085.34 examples/s]18817 examples [00:16, 1054.85 examples/s]18925 examples [00:16, 1062.08 examples/s]19044 examples [00:16, 1095.35 examples/s]19157 examples [00:16, 1105.19 examples/s]19269 examples [00:16, 1104.35 examples/s]19387 examples [00:17, 1124.24 examples/s]19511 examples [00:17, 1155.77 examples/s]19641 examples [00:17, 1193.88 examples/s]19764 examples [00:17, 1201.98 examples/s]19888 examples [00:17, 1211.69 examples/s]20010 examples [00:17, 1166.17 examples/s]20130 examples [00:17, 1175.13 examples/s]20248 examples [00:17, 1167.52 examples/s]20373 examples [00:17, 1189.46 examples/s]20493 examples [00:17, 1179.40 examples/s]20614 examples [00:18, 1186.08 examples/s]20733 examples [00:18, 1177.64 examples/s]20856 examples [00:18, 1190.21 examples/s]20976 examples [00:18, 1181.82 examples/s]21095 examples [00:18, 1161.99 examples/s]21213 examples [00:18, 1166.16 examples/s]21333 examples [00:18, 1173.75 examples/s]21455 examples [00:18, 1186.09 examples/s]21574 examples [00:18, 1180.89 examples/s]21694 examples [00:18, 1183.79 examples/s]21822 examples [00:19, 1210.59 examples/s]21944 examples [00:19, 1211.43 examples/s]22066 examples [00:19, 1204.47 examples/s]22187 examples [00:19, 1174.47 examples/s]22305 examples [00:19, 1167.33 examples/s]22422 examples [00:19, 1165.89 examples/s]22541 examples [00:19, 1171.85 examples/s]22665 examples [00:19, 1190.36 examples/s]22791 examples [00:19, 1208.86 examples/s]22917 examples [00:20, 1221.59 examples/s]23040 examples [00:20, 1221.29 examples/s]23163 examples [00:20, 1215.73 examples/s]23285 examples [00:20, 1214.23 examples/s]23407 examples [00:20, 1205.63 examples/s]23528 examples [00:20, 1164.49 examples/s]23645 examples [00:20, 1157.40 examples/s]23761 examples [00:20, 1150.08 examples/s]23877 examples [00:20, 1143.30 examples/s]24005 examples [00:20, 1180.92 examples/s]24127 examples [00:21, 1190.46 examples/s]24253 examples [00:21, 1210.32 examples/s]24375 examples [00:21, 1208.46 examples/s]24503 examples [00:21, 1226.65 examples/s]24626 examples [00:21, 1225.14 examples/s]24751 examples [00:21, 1229.37 examples/s]24875 examples [00:21, 1223.73 examples/s]24998 examples [00:21, 1171.34 examples/s]25116 examples [00:21, 1152.26 examples/s]25232 examples [00:21, 1136.48 examples/s]25360 examples [00:22, 1174.42 examples/s]25487 examples [00:22, 1199.80 examples/s]25609 examples [00:22, 1204.49 examples/s]25730 examples [00:22, 1173.15 examples/s]25852 examples [00:22, 1185.86 examples/s]25978 examples [00:22, 1207.03 examples/s]26100 examples [00:22, 1208.37 examples/s]26222 examples [00:22, 1175.72 examples/s]26348 examples [00:22, 1196.95 examples/s]26470 examples [00:22, 1202.30 examples/s]26591 examples [00:23, 1200.11 examples/s]26715 examples [00:23, 1209.18 examples/s]26837 examples [00:23, 1202.95 examples/s]26966 examples [00:23, 1227.64 examples/s]27089 examples [00:23, 1209.41 examples/s]27211 examples [00:23, 1210.74 examples/s]27334 examples [00:23, 1214.00 examples/s]27457 examples [00:23, 1215.86 examples/s]27580 examples [00:23, 1218.17 examples/s]27702 examples [00:24, 1216.10 examples/s]27826 examples [00:24, 1221.24 examples/s]27954 examples [00:24, 1237.20 examples/s]28078 examples [00:24, 1228.00 examples/s]28201 examples [00:24, 1212.22 examples/s]28323 examples [00:24, 1197.46 examples/s]28443 examples [00:24, 1193.04 examples/s]28566 examples [00:24, 1196.31 examples/s]28686 examples [00:24, 1176.14 examples/s]28804 examples [00:24, 1162.69 examples/s]28921 examples [00:25, 1147.36 examples/s]29037 examples [00:25, 1150.39 examples/s]29154 examples [00:25, 1154.62 examples/s]29277 examples [00:25, 1173.95 examples/s]29396 examples [00:25, 1177.24 examples/s]29516 examples [00:25, 1182.42 examples/s]29639 examples [00:25, 1195.87 examples/s]29766 examples [00:25, 1215.93 examples/s]29888 examples [00:25, 1207.24 examples/s]30009 examples [00:25, 1147.68 examples/s]30130 examples [00:26, 1165.57 examples/s]30259 examples [00:26, 1199.15 examples/s]30381 examples [00:26, 1204.84 examples/s]30502 examples [00:26, 1205.37 examples/s]30623 examples [00:26, 1188.05 examples/s]30743 examples [00:26, 1161.45 examples/s]30861 examples [00:26, 1166.13 examples/s]30983 examples [00:26, 1179.16 examples/s]31102 examples [00:26, 1173.17 examples/s]31220 examples [00:26, 1165.38 examples/s]31337 examples [00:27, 1154.02 examples/s]31453 examples [00:27, 1147.77 examples/s]31568 examples [00:27, 1134.26 examples/s]31682 examples [00:27, 1120.14 examples/s]31804 examples [00:27, 1145.98 examples/s]31919 examples [00:27, 1137.36 examples/s]32036 examples [00:27, 1144.67 examples/s]32151 examples [00:27, 1143.19 examples/s]32266 examples [00:27, 1136.69 examples/s]32380 examples [00:28, 1123.39 examples/s]32496 examples [00:28, 1132.16 examples/s]32613 examples [00:28, 1140.88 examples/s]32732 examples [00:28, 1154.08 examples/s]32852 examples [00:28, 1165.41 examples/s]32969 examples [00:28, 1156.37 examples/s]33085 examples [00:28, 1142.98 examples/s]33200 examples [00:28, 1109.58 examples/s]33312 examples [00:28, 1112.14 examples/s]33424 examples [00:28, 1105.80 examples/s]33535 examples [00:29, 1082.12 examples/s]33653 examples [00:29, 1108.13 examples/s]33771 examples [00:29, 1127.44 examples/s]33890 examples [00:29, 1143.09 examples/s]34012 examples [00:29, 1164.71 examples/s]34129 examples [00:29, 1165.69 examples/s]34246 examples [00:29, 1144.92 examples/s]34374 examples [00:29, 1180.15 examples/s]34495 examples [00:29, 1187.82 examples/s]34616 examples [00:29, 1193.57 examples/s]34746 examples [00:30, 1220.86 examples/s]34869 examples [00:30, 1216.28 examples/s]34991 examples [00:30, 1209.13 examples/s]35113 examples [00:30, 1186.98 examples/s]35236 examples [00:30, 1197.41 examples/s]35356 examples [00:30, 1193.74 examples/s]35481 examples [00:30, 1208.86 examples/s]35610 examples [00:30, 1230.14 examples/s]35734 examples [00:30, 1230.53 examples/s]35858 examples [00:30, 1225.00 examples/s]35981 examples [00:31, 1216.49 examples/s]36103 examples [00:31, 1205.74 examples/s]36224 examples [00:31, 1137.92 examples/s]36341 examples [00:31, 1144.94 examples/s]36461 examples [00:31, 1159.34 examples/s]36578 examples [00:31, 1134.71 examples/s]36697 examples [00:31, 1148.87 examples/s]36817 examples [00:31, 1163.16 examples/s]36934 examples [00:31, 1155.82 examples/s]37050 examples [00:32, 1146.56 examples/s]37167 examples [00:32, 1151.71 examples/s]37290 examples [00:32, 1172.72 examples/s]37411 examples [00:32, 1182.54 examples/s]37530 examples [00:32, 1156.35 examples/s]37646 examples [00:32, 1146.24 examples/s]37761 examples [00:32, 1111.43 examples/s]37873 examples [00:32, 1102.12 examples/s]37985 examples [00:32, 1107.11 examples/s]38099 examples [00:32, 1114.71 examples/s]38216 examples [00:33, 1128.77 examples/s]38330 examples [00:33, 1125.51 examples/s]38451 examples [00:33, 1147.62 examples/s]38567 examples [00:33, 1149.56 examples/s]38687 examples [00:33, 1161.01 examples/s]38804 examples [00:33, 1158.13 examples/s]38920 examples [00:33, 1147.46 examples/s]39038 examples [00:33, 1154.43 examples/s]39162 examples [00:33, 1176.14 examples/s]39280 examples [00:33, 1172.45 examples/s]39398 examples [00:34, 1165.97 examples/s]39522 examples [00:34, 1185.01 examples/s]39644 examples [00:34, 1193.77 examples/s]39766 examples [00:34, 1198.37 examples/s]39886 examples [00:34, 1173.10 examples/s]40004 examples [00:34, 1110.71 examples/s]40123 examples [00:34, 1131.45 examples/s]40250 examples [00:34, 1167.94 examples/s]40374 examples [00:34, 1186.01 examples/s]40495 examples [00:34, 1190.43 examples/s]40622 examples [00:35, 1211.33 examples/s]40749 examples [00:35, 1227.50 examples/s]40877 examples [00:35, 1240.01 examples/s]41005 examples [00:35, 1251.66 examples/s]41131 examples [00:35, 1234.98 examples/s]41255 examples [00:35, 1229.07 examples/s]41379 examples [00:35, 1184.81 examples/s]41504 examples [00:35, 1200.83 examples/s]41625 examples [00:35, 1200.95 examples/s]41746 examples [00:36, 1179.22 examples/s]41871 examples [00:36, 1196.66 examples/s]41995 examples [00:36, 1207.61 examples/s]42120 examples [00:36, 1217.53 examples/s]42245 examples [00:36, 1225.62 examples/s]42368 examples [00:36, 1204.96 examples/s]42491 examples [00:36, 1210.23 examples/s]42620 examples [00:36, 1232.08 examples/s]42749 examples [00:36, 1246.26 examples/s]42874 examples [00:36, 1224.71 examples/s]42997 examples [00:37, 1196.32 examples/s]43120 examples [00:37, 1204.38 examples/s]43253 examples [00:37, 1238.51 examples/s]43378 examples [00:37, 1240.46 examples/s]43503 examples [00:37, 1232.55 examples/s]43627 examples [00:37, 1224.05 examples/s]43750 examples [00:37, 1215.76 examples/s]43872 examples [00:37, 1197.32 examples/s]43992 examples [00:37, 1184.78 examples/s]44111 examples [00:37, 1175.10 examples/s]44229 examples [00:38, 1174.24 examples/s]44355 examples [00:38, 1196.72 examples/s]44488 examples [00:38, 1231.71 examples/s]44612 examples [00:38, 1232.47 examples/s]44739 examples [00:38, 1240.77 examples/s]44864 examples [00:38, 1220.84 examples/s]44991 examples [00:38, 1232.48 examples/s]45115 examples [00:38, 1201.25 examples/s]45241 examples [00:38, 1217.20 examples/s]45369 examples [00:38, 1235.37 examples/s]45493 examples [00:39, 1219.83 examples/s]45617 examples [00:39, 1224.93 examples/s]45740 examples [00:39, 1220.26 examples/s]45863 examples [00:39, 1210.31 examples/s]45985 examples [00:39, 1210.50 examples/s]46107 examples [00:39, 1191.45 examples/s]46229 examples [00:39, 1198.77 examples/s]46351 examples [00:39, 1202.85 examples/s]46472 examples [00:39, 1178.44 examples/s]46591 examples [00:40, 1162.63 examples/s]46711 examples [00:40, 1170.41 examples/s]46829 examples [00:40, 1169.73 examples/s]46947 examples [00:40, 1162.98 examples/s]47064 examples [00:40, 1164.38 examples/s]47181 examples [00:40, 1161.96 examples/s]47298 examples [00:40, 1163.90 examples/s]47417 examples [00:40, 1170.63 examples/s]47539 examples [00:40, 1184.26 examples/s]47668 examples [00:40, 1212.74 examples/s]47791 examples [00:41, 1217.82 examples/s]47915 examples [00:41, 1223.83 examples/s]48040 examples [00:41, 1229.64 examples/s]48165 examples [00:41, 1232.82 examples/s]48289 examples [00:41, 1232.19 examples/s]48413 examples [00:41, 1216.01 examples/s]48535 examples [00:41, 1166.97 examples/s]48657 examples [00:41, 1182.04 examples/s]48776 examples [00:41, 1162.52 examples/s]48896 examples [00:41, 1171.83 examples/s]49019 examples [00:42, 1186.86 examples/s]49138 examples [00:42, 1187.60 examples/s]49258 examples [00:42, 1189.09 examples/s]49380 examples [00:42, 1196.93 examples/s]49500 examples [00:42, 1159.92 examples/s]49617 examples [00:42, 1146.50 examples/s]49732 examples [00:42, 1120.66 examples/s]49860 examples [00:42, 1163.32 examples/s]49984 examples [00:42, 1184.50 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 20%|█▉        | 9859/50000 [00:00<00:00, 98589.86 examples/s] 49%|████▊     | 24339/50000 [00:00<00:00, 109027.98 examples/s] 78%|███████▊  | 38856/50000 [00:00<00:00, 117827.27 examples/s]                                                                0 examples [00:00, ? examples/s]100 examples [00:00, 997.57 examples/s]224 examples [00:00, 1058.06 examples/s]349 examples [00:00, 1108.32 examples/s]466 examples [00:00, 1124.89 examples/s]588 examples [00:00, 1151.79 examples/s]711 examples [00:00, 1172.97 examples/s]836 examples [00:00, 1194.88 examples/s]953 examples [00:00, 1186.20 examples/s]1076 examples [00:00, 1197.36 examples/s]1202 examples [00:01, 1214.69 examples/s]1331 examples [00:01, 1233.85 examples/s]1453 examples [00:01, 998.54 examples/s] 1568 examples [00:01, 1038.52 examples/s]1694 examples [00:01, 1094.45 examples/s]1815 examples [00:01, 1124.73 examples/s]1942 examples [00:01, 1163.29 examples/s]2070 examples [00:01, 1194.94 examples/s]2192 examples [00:01, 1196.08 examples/s]2313 examples [00:01, 1192.80 examples/s]2436 examples [00:02, 1202.04 examples/s]2564 examples [00:02, 1222.47 examples/s]2689 examples [00:02, 1228.95 examples/s]2813 examples [00:02, 1209.78 examples/s]2941 examples [00:02, 1228.44 examples/s]3068 examples [00:02, 1238.47 examples/s]3193 examples [00:02, 1238.03 examples/s]3317 examples [00:02, 1233.70 examples/s]3441 examples [00:02, 1231.58 examples/s]3568 examples [00:02, 1241.64 examples/s]3693 examples [00:03, 1232.26 examples/s]3817 examples [00:03, 1228.02 examples/s]3940 examples [00:03, 1176.99 examples/s]4059 examples [00:03, 1152.07 examples/s]4179 examples [00:03, 1165.14 examples/s]4296 examples [00:03, 1163.08 examples/s]4413 examples [00:03, 1151.04 examples/s]4529 examples [00:03, 1145.42 examples/s]4644 examples [00:03, 1137.18 examples/s]4761 examples [00:04, 1144.13 examples/s]4880 examples [00:04, 1155.93 examples/s]5001 examples [00:04, 1171.30 examples/s]5119 examples [00:04, 1159.11 examples/s]5236 examples [00:04, 1159.68 examples/s]5353 examples [00:04, 1157.35 examples/s]5469 examples [00:04, 1109.32 examples/s]5581 examples [00:04, 1107.97 examples/s]5695 examples [00:04, 1114.82 examples/s]5807 examples [00:04, 1111.06 examples/s]5927 examples [00:05, 1135.78 examples/s]6041 examples [00:05, 1136.86 examples/s]6155 examples [00:05, 1135.68 examples/s]6269 examples [00:05, 1133.57 examples/s]6383 examples [00:05, 1109.14 examples/s]6496 examples [00:05, 1113.01 examples/s]6608 examples [00:05, 1104.25 examples/s]6724 examples [00:05, 1120.29 examples/s]6846 examples [00:05, 1146.42 examples/s]6963 examples [00:05, 1153.01 examples/s]7092 examples [00:06, 1189.05 examples/s]7216 examples [00:06, 1203.83 examples/s]7339 examples [00:06, 1209.00 examples/s]7465 examples [00:06, 1221.13 examples/s]7588 examples [00:06, 1219.36 examples/s]7713 examples [00:06, 1227.56 examples/s]7840 examples [00:06, 1239.28 examples/s]7965 examples [00:06, 1222.76 examples/s]8090 examples [00:06, 1228.61 examples/s]8214 examples [00:06, 1230.85 examples/s]8338 examples [00:07, 1203.47 examples/s]8464 examples [00:07, 1218.21 examples/s]8586 examples [00:07, 1217.47 examples/s]8711 examples [00:07, 1223.94 examples/s]8834 examples [00:07, 1220.30 examples/s]8957 examples [00:07, 1222.17 examples/s]9080 examples [00:07, 1201.85 examples/s]9208 examples [00:07, 1222.46 examples/s]9332 examples [00:07, 1226.09 examples/s]9455 examples [00:08, 1213.22 examples/s]9578 examples [00:08, 1216.93 examples/s]9700 examples [00:08, 1208.36 examples/s]9822 examples [00:08, 1209.10 examples/s]9945 examples [00:08, 1212.69 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1JR6FX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1JR6FX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f312863a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f312863a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f312863a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f30adaf0a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f30ada8ce10>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f312863a950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f312863a950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f312863a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3128627a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f310e9f90f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f30a00412f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f30a00412f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f30a00412f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f30a0041400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f30a0041400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f30a0041400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=6fc11cca26b4db40232046e1fa363ca0e714b83822046c06b17c9fa4f38c7d0c
  Stored in directory: /tmp/pip-ephem-wheel-cache-p3uzvucn/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:47:39, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:47:14, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:24:16, 23.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:17:22, 32.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.01M/862M [00:01<5:05:36, 46.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.59M/862M [00:01<3:33:10, 66.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:28:20, 95.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:43:14, 136kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:11:53, 194kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.4M/862M [00:01<50:07, 277kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:01<35:02, 395kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.5M/862M [00:02<24:30, 562kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<17:08, 798kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<12:03, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<08:28, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<06:50, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<06:41, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:51, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:20, 2.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<06:04, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:38, 2.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:49, 2.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:10, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:39, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:09<04:10, 3.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<07:13, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<06:25, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<04:49, 2.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:27, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<07:11, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<05:35, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<04:05, 3.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<09:03, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<07:29, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<05:34, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:55, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:29, 1.74MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<05:48, 2.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<04:13, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<09:58, 1.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<08:18, 1.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<06:05, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<07:23, 1.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<07:41, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<05:56, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<04:18, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<11:42, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<09:31, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<06:58, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:52, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<06:49, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:24<05:05, 2.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:31, 1.94MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:08, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:38, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:58, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:27, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:08, 3.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:11, 2.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:44, 3.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<32:01, 389kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<23:41, 526kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<16:52, 737kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<14:41, 844kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:47, 968kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:35, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:41, 1.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:19, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<05:26, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:40, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:09, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:33, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<04:00, 3.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<50:37, 241kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<36:41, 332kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<25:57, 468kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<20:56, 579kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:53, 763kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<11:24, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:48, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:47, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:26, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:20, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:34, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:12, 2.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<12:05, 986kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:41, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:02, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:41, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:35, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:54, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:12, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:32, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:07, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:38, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:09, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:53, 3.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:28, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:11, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:51, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:31, 3.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<11:00, 1.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:52, 1.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<06:27, 1.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:13, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:11, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:37, 2.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:55, 1.93MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:30, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:01, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:41, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:48, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:55, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:26, 2.56MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:44, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:18, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:35, 3.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:25, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:20, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:43, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:54, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:38, 3.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:43:52, 17.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<7:31:35, 24.5kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<5:15:37, 35.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<3:42:48, 49.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<2:37:00, 70.2kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<1:49:53, 100kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<1:19:16, 138kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<56:33, 194kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<39:44, 275kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<30:19, 359kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<23:26, 464kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<16:56, 641kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<13:34, 797kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:35, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<07:40, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:52, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:43, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:52, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<04:13, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:53, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:00, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:52, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:34, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:52, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:16, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:48, 2.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<09:13, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:33, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:32, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:19, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:44, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:17, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<03:49, 2.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<19:02, 546kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<14:27, 720kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<10:22, 999kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:30, 1.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:56, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:45, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:51, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:44, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:33, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:51, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:37, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:11, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:53, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:33, 2.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<14:45, 686kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<11:25, 885kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<08:15, 1.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:58, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:48, 1.29MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:01, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<04:18, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<13:43, 728kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<10:42, 932kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<07:42, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:34, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:30, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:42, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:10, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:45, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:06, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:48, 2.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:49, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:32, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:24, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:12, 3.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<13:47, 704kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<10:43, 905kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<07:45, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:31, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:19, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:39, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:48, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:37, 2.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:36, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:15, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:12, 2.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:19, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:03, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:03, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:12, 2.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:01, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:01, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:55, 3.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<10:53, 854kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:27, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<06:06, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:24, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<12:14, 754kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<10:36, 869kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:55, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<05:37, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<13:43, 667kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<10:37, 861kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<07:37, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:19, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:09, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:26, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:54, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:05, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:40, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:52, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:22, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:45, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:26, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:13, 2.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:50, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:03, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:44, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:33, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:09, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:00, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<02:54, 3.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:10, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:16, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:53, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:37, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:05, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:57, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<10:14, 840kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:05, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:53, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<05:58, 1.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:06, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:47, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:30, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:01, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<02:51, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:03, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:28, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:20, 2.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:09, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:48, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:44, 3.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<10:46, 765kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:25, 978kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:04, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<06:02, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:03, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:37, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:20, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<05:33, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:46, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:33, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:14, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:51, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<02:52, 2.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<03:45, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:24, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:26, 2.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:32, 3.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:12, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:46, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:49, 2.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:44, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:25, 2.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:29, 3.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:54, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:59, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:40, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:14, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:40, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:38, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:38, 2.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:24, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:36, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:24, 2.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:03, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:30, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:34, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:35, 2.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<10:53, 688kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<08:26, 886kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<06:04, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:51, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:44, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:21, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<03:08, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:56, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:47, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:29, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:33, 2.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<19:52, 366kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<15:26, 471kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<11:07, 653kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<07:50, 922kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<09:39, 746kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:31, 958kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<05:24, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:22, 1.33MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:30, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:56, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:19, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:27, 2.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<09:21, 748kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:18, 957kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<05:17, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:12, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:10, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:00, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:52, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<09:19, 736kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<07:17, 941kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:16, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:09, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:06, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:56, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:50, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<10:08, 663kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:49, 859kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:36, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:23, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:10, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:56, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:49, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<16:34, 397kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<12:17, 535kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<08:43, 752kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<07:32, 863kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:43, 969kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:00, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:33, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:15, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:57, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:38, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:56, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:09, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:15, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:20, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:22, 753kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<06:32, 963kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<04:42, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:39, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:38, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:31, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:34, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:44, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:17, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:26, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:02, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:25, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:39, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:56, 3.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:40, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:55, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:52, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:18, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:57, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:13, 2.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:50, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:34, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:51, 3.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<18:47, 310kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<13:46, 422kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<09:45, 594kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<08:03, 714kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:55, 831kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:09, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<03:38, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<08:38, 657kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<06:40, 851kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<04:48, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:35, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:27, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:22, 1.66MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<02:25, 2.29MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:05, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:28, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:34, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:59, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:41, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:35, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:01, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:24, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<01:44, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:56, 769kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:25, 982kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:54, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:52, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:53, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:58, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:10, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:53, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:35, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:56, 2.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:28, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:52, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:17, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:39, 3.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<06:42, 753kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:49, 868kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<04:18, 1.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<03:03, 1.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:44, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:44, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:42, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<01:58, 2.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<14:06, 349kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<10:57, 448kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<07:55, 618kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<05:33, 874kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<09:04, 534kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<06:47, 713kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<04:50, 997kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<03:26, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:44, 617kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:28, 738kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:45, 1.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<03:21, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:15, 895kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:10, 1.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:00, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:10, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:42, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:59, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:25, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:40, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:04, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:29, 3.04MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:00, 899kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:54, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:49, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:01, 2.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<05:49, 762kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:32, 974kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:17, 1.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:14, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:14, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:46, 2.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:55, 1.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:31, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:51, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:26, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:55, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:23, 3.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<09:32, 436kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<07:07, 583kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<05:04, 814kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<04:24, 926kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:59, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:00, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<02:08, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<06:22, 631kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<04:54, 819kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:31, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:18, 1.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<03:08, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:24, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:43, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<07:09, 543kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:25, 713kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:53, 991kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:31, 1.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<03:19, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:31, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:47, 646kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:27, 839kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:11, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:01, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:55, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:14, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:36, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<05:29, 656kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:13, 850kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:01, 1.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:52, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:47, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:08, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:31, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<06:30, 532kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:54, 704kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:29, 984kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<03:12, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:58, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:14, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:14, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:56, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:31, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:05, 2.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<04:10, 764kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:15, 977kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:20, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:17, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:15, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:17, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:56, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:25, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:38, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:46, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:22, 2.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:59, 2.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:34, 1.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:07, 1.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:32, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:41, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:48, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:25, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:00, 2.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:40, 756kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:52, 963kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:04, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:59, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:30, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:04, 2.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:04, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:43, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:15, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:27, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:16, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:56, 2.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:13, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:05, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:47, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<07:51, 309kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<05:41, 425kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:00, 597kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<03:17, 715kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:49, 833kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:04, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:27, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:59, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:38, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:11, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:18, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:24, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:05, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:46, 2.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<03:50, 559kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<02:54, 736kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:03, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:53, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:32, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:06, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:13, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:18, 1.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:01, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:43, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<02:48, 692kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:10, 891kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:33, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:28, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:27, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:46, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:28, 728kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:55, 932kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:22, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:19, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:18, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:00, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:42, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:04, 801kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:37, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:10, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:09, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:09, 1.38MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:53, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:37, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:17, 667kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:45, 862kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:15, 1.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:10, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:08, 1.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:52, 1.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:36, 2.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:54, 727kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:28, 932kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:03, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:00, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:58, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:44, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<04:07, 303kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<03:00, 414kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<02:05, 583kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:41, 700kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:26, 819kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:03, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:43, 1.55MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:37, 681kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:15, 882kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:52, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:50, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:48, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:25, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:20, 727kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:02, 931kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:43, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:41, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:40, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:21, 2.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:08, 733kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:52, 939kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:37, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:34, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:29, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:22, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:13, 2.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:15, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:10, 2.98MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:22, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:18, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 2.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:14, 1.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:12, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:04, 2.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:18, 688kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:13, 888kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:05, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:20, 415kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:17, 478kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:12, 637kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:07, 891kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:03, 1.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<01:56, 36.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<01:20, 51.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:51, 72.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:21, 104kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 148kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:02, 43.4kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<49:22:23,  2.25it/s]  0%|          | 925/400000 [00:00<34:29:05,  3.21it/s]  0%|          | 1870/400000 [00:00<24:05:08,  4.59it/s]  1%|          | 2807/400000 [00:00<16:49:26,  6.56it/s]  1%|          | 3726/400000 [00:00<11:45:11,  9.37it/s]  1%|          | 4660/400000 [00:00<8:12:40, 13.37it/s]   1%|▏         | 5678/400000 [00:01<5:44:10, 19.09it/s]  2%|▏         | 6659/400000 [00:01<4:00:31, 27.26it/s]  2%|▏         | 7653/400000 [00:01<2:48:08, 38.89it/s]  2%|▏         | 8726/400000 [00:01<1:57:33, 55.47it/s]  2%|▏         | 9790/400000 [00:01<1:22:15, 79.07it/s]  3%|▎         | 10811/400000 [00:01<57:36, 112.58it/s]  3%|▎         | 11930/400000 [00:01<40:23, 160.14it/s]  3%|▎         | 12988/400000 [00:01<28:22, 227.30it/s]  4%|▎         | 14055/400000 [00:01<19:59, 321.77it/s]  4%|▍         | 15106/400000 [00:01<14:09, 453.35it/s]  4%|▍         | 16175/400000 [00:02<10:03, 636.08it/s]  4%|▍         | 17218/400000 [00:02<07:12, 885.30it/s]  5%|▍         | 18278/400000 [00:02<05:12, 1220.98it/s]  5%|▍         | 19335/400000 [00:02<03:49, 1661.93it/s]  5%|▌         | 20384/400000 [00:02<02:50, 2220.59it/s]  5%|▌         | 21428/400000 [00:02<02:10, 2905.31it/s]  6%|▌         | 22542/400000 [00:02<01:41, 3733.00it/s]  6%|▌         | 23605/400000 [00:02<01:21, 4630.80it/s]  6%|▌         | 24666/400000 [00:02<01:08, 5468.13it/s]  6%|▋         | 25693/400000 [00:02<00:59, 6305.46it/s]  7%|▋         | 26722/400000 [00:03<00:52, 7133.20it/s]  7%|▋         | 27807/400000 [00:03<00:46, 7950.22it/s]  7%|▋         | 28884/400000 [00:03<00:43, 8627.57it/s]  7%|▋         | 29934/400000 [00:03<00:40, 9071.25it/s]  8%|▊         | 30978/400000 [00:03<00:39, 9249.37it/s]  8%|▊         | 32036/400000 [00:03<00:38, 9609.46it/s]  8%|▊         | 33088/400000 [00:03<00:37, 9864.12it/s]  9%|▊         | 34126/400000 [00:03<00:36, 9968.90it/s]  9%|▉         | 35171/400000 [00:03<00:36, 10106.76it/s]  9%|▉         | 36253/400000 [00:03<00:35, 10308.79it/s]  9%|▉         | 37311/400000 [00:04<00:34, 10387.74it/s] 10%|▉         | 38364/400000 [00:04<00:34, 10414.55it/s] 10%|▉         | 39419/400000 [00:04<00:34, 10452.13it/s] 10%|█         | 40471/400000 [00:04<00:35, 10121.80it/s] 10%|█         | 41491/400000 [00:04<00:35, 10126.39it/s] 11%|█         | 42509/400000 [00:04<00:35, 10065.79it/s] 11%|█         | 43520/400000 [00:04<00:35, 9957.30it/s]  11%|█         | 44536/400000 [00:04<00:35, 10015.43it/s] 11%|█▏        | 45602/400000 [00:04<00:34, 10198.64it/s] 12%|█▏        | 46625/400000 [00:05<00:35, 9980.74it/s]  12%|█▏        | 47626/400000 [00:05<00:36, 9733.93it/s] 12%|█▏        | 48605/400000 [00:05<00:36, 9750.61it/s] 12%|█▏        | 49688/400000 [00:05<00:34, 10048.02it/s] 13%|█▎        | 50768/400000 [00:05<00:34, 10262.00it/s] 13%|█▎        | 51798/400000 [00:05<00:34, 10013.99it/s] 13%|█▎        | 52818/400000 [00:05<00:34, 10066.69it/s] 13%|█▎        | 53852/400000 [00:05<00:34, 10146.81it/s] 14%|█▎        | 54963/400000 [00:05<00:33, 10417.49it/s] 14%|█▍        | 56009/400000 [00:05<00:34, 10085.68it/s] 14%|█▍        | 57023/400000 [00:06<00:34, 9915.61it/s]  15%|█▍        | 58019/400000 [00:06<00:34, 9823.58it/s] 15%|█▍        | 59005/400000 [00:06<00:34, 9815.07it/s] 15%|█▌        | 60035/400000 [00:06<00:34, 9955.21it/s] 15%|█▌        | 61088/400000 [00:06<00:33, 10120.86it/s] 16%|█▌        | 62103/400000 [00:06<00:33, 10127.06it/s] 16%|█▌        | 63166/400000 [00:06<00:32, 10270.10it/s] 16%|█▌        | 64195/400000 [00:06<00:33, 10089.77it/s] 16%|█▋        | 65250/400000 [00:06<00:32, 10220.75it/s] 17%|█▋        | 66350/400000 [00:06<00:31, 10442.11it/s] 17%|█▋        | 67397/400000 [00:07<00:32, 10321.87it/s] 17%|█▋        | 68466/400000 [00:07<00:31, 10426.71it/s] 17%|█▋        | 69518/400000 [00:07<00:31, 10453.44it/s] 18%|█▊        | 70628/400000 [00:07<00:30, 10636.32it/s] 18%|█▊        | 71731/400000 [00:07<00:30, 10750.57it/s] 18%|█▊        | 72808/400000 [00:07<00:30, 10657.38it/s] 18%|█▊        | 73875/400000 [00:07<00:30, 10555.75it/s] 19%|█▊        | 74937/400000 [00:07<00:30, 10573.02it/s] 19%|█▉        | 75996/400000 [00:07<00:30, 10543.66it/s] 19%|█▉        | 77051/400000 [00:07<00:30, 10511.18it/s] 20%|█▉        | 78103/400000 [00:08<00:31, 10076.58it/s] 20%|█▉        | 79115/400000 [00:08<00:32, 9996.22it/s]  20%|██        | 80118/400000 [00:08<00:32, 9872.91it/s] 20%|██        | 81136/400000 [00:08<00:32, 9963.02it/s] 21%|██        | 82152/400000 [00:08<00:31, 10018.42it/s] 21%|██        | 83172/400000 [00:08<00:31, 10070.79it/s] 21%|██        | 84202/400000 [00:08<00:31, 10136.16it/s] 21%|██▏       | 85279/400000 [00:08<00:30, 10318.29it/s] 22%|██▏       | 86313/400000 [00:08<00:30, 10309.19it/s] 22%|██▏       | 87352/400000 [00:08<00:30, 10333.20it/s] 22%|██▏       | 88387/400000 [00:09<00:30, 10142.49it/s] 22%|██▏       | 89404/400000 [00:09<00:30, 10147.55it/s] 23%|██▎       | 90434/400000 [00:09<00:30, 10192.22it/s] 23%|██▎       | 91454/400000 [00:09<00:30, 10170.23it/s] 23%|██▎       | 92511/400000 [00:09<00:29, 10284.53it/s] 23%|██▎       | 93561/400000 [00:09<00:29, 10347.85it/s] 24%|██▎       | 94597/400000 [00:09<00:29, 10246.25it/s] 24%|██▍       | 95642/400000 [00:09<00:29, 10306.48it/s] 24%|██▍       | 96674/400000 [00:09<00:29, 10192.23it/s] 24%|██▍       | 97734/400000 [00:10<00:29, 10309.69it/s] 25%|██▍       | 98766/400000 [00:10<00:29, 10049.29it/s] 25%|██▍       | 99773/400000 [00:10<00:29, 10025.08it/s] 25%|██▌       | 100850/400000 [00:10<00:29, 10235.84it/s] 25%|██▌       | 101919/400000 [00:10<00:28, 10365.43it/s] 26%|██▌       | 102958/400000 [00:10<00:28, 10282.60it/s] 26%|██▌       | 103988/400000 [00:10<00:29, 10182.25it/s] 26%|██▋       | 105008/400000 [00:10<00:29, 10139.57it/s] 27%|██▋       | 106047/400000 [00:10<00:28, 10213.07it/s] 27%|██▋       | 107072/400000 [00:10<00:28, 10222.91it/s] 27%|██▋       | 108131/400000 [00:11<00:28, 10328.32it/s] 27%|██▋       | 109165/400000 [00:11<00:28, 10156.23it/s] 28%|██▊       | 110197/400000 [00:11<00:28, 10201.88it/s] 28%|██▊       | 111218/400000 [00:11<00:28, 10018.36it/s] 28%|██▊       | 112222/400000 [00:11<00:29, 9856.52it/s]  28%|██▊       | 113210/400000 [00:11<00:29, 9659.88it/s] 29%|██▊       | 114186/400000 [00:11<00:29, 9687.75it/s] 29%|██▉       | 115239/400000 [00:11<00:28, 9924.92it/s] 29%|██▉       | 116272/400000 [00:11<00:28, 10041.10it/s] 29%|██▉       | 117357/400000 [00:11<00:27, 10270.20it/s] 30%|██▉       | 118457/400000 [00:12<00:26, 10477.26it/s] 30%|██▉       | 119508/400000 [00:12<00:27, 10226.70it/s] 30%|███       | 120551/400000 [00:12<00:27, 10286.05it/s] 30%|███       | 121629/400000 [00:12<00:26, 10429.37it/s] 31%|███       | 122707/400000 [00:12<00:26, 10530.34it/s] 31%|███       | 123786/400000 [00:12<00:26, 10604.97it/s] 31%|███       | 124848/400000 [00:12<00:26, 10542.46it/s] 31%|███▏      | 125918/400000 [00:12<00:25, 10587.77it/s] 32%|███▏      | 126994/400000 [00:12<00:25, 10638.17it/s] 32%|███▏      | 128059/400000 [00:12<00:25, 10629.69it/s] 32%|███▏      | 129123/400000 [00:13<00:25, 10500.78it/s] 33%|███▎      | 130174/400000 [00:13<00:25, 10387.48it/s] 33%|███▎      | 131214/400000 [00:13<00:25, 10344.06it/s] 33%|███▎      | 132249/400000 [00:13<00:25, 10311.64it/s] 33%|███▎      | 133316/400000 [00:13<00:25, 10415.40it/s] 34%|███▎      | 134362/400000 [00:13<00:25, 10425.76it/s] 34%|███▍      | 135405/400000 [00:13<00:25, 10264.77it/s] 34%|███▍      | 136495/400000 [00:13<00:25, 10443.47it/s] 34%|███▍      | 137541/400000 [00:13<00:25, 10386.65it/s] 35%|███▍      | 138606/400000 [00:13<00:24, 10463.23it/s] 35%|███▍      | 139654/400000 [00:14<00:24, 10465.68it/s] 35%|███▌      | 140702/400000 [00:14<00:25, 10335.13it/s] 35%|███▌      | 141772/400000 [00:14<00:24, 10439.30it/s] 36%|███▌      | 142817/400000 [00:14<00:25, 10124.89it/s] 36%|███▌      | 143871/400000 [00:14<00:25, 10244.55it/s] 36%|███▌      | 144898/400000 [00:14<00:25, 10139.78it/s] 36%|███▋      | 145914/400000 [00:14<00:25, 10076.37it/s] 37%|███▋      | 146923/400000 [00:14<00:25, 9873.44it/s]  37%|███▋      | 147936/400000 [00:14<00:25, 9947.81it/s] 37%|███▋      | 148933/400000 [00:15<00:25, 9782.70it/s] 37%|███▋      | 149913/400000 [00:15<00:25, 9624.97it/s] 38%|███▊      | 150878/400000 [00:15<00:26, 9548.28it/s] 38%|███▊      | 151896/400000 [00:15<00:25, 9728.43it/s] 38%|███▊      | 152918/400000 [00:15<00:25, 9869.88it/s] 38%|███▊      | 153921/400000 [00:15<00:24, 9915.18it/s] 39%|███▊      | 154914/400000 [00:15<00:24, 9886.39it/s] 39%|███▉      | 155904/400000 [00:15<00:25, 9735.07it/s] 39%|███▉      | 156911/400000 [00:15<00:24, 9831.26it/s] 39%|███▉      | 157896/400000 [00:15<00:25, 9443.43it/s] 40%|███▉      | 158882/400000 [00:16<00:25, 9564.47it/s] 40%|███▉      | 159843/400000 [00:16<00:25, 9575.73it/s] 40%|████      | 160860/400000 [00:16<00:24, 9746.12it/s] 40%|████      | 161841/400000 [00:16<00:24, 9763.34it/s] 41%|████      | 162830/400000 [00:16<00:24, 9800.55it/s] 41%|████      | 163812/400000 [00:16<00:24, 9748.35it/s] 41%|████      | 164788/400000 [00:16<00:24, 9456.71it/s] 41%|████▏     | 165737/400000 [00:16<00:26, 8903.87it/s] 42%|████▏     | 166662/400000 [00:16<00:25, 9004.41it/s] 42%|████▏     | 167636/400000 [00:16<00:25, 9213.14it/s] 42%|████▏     | 168726/400000 [00:17<00:23, 9660.62it/s] 42%|████▏     | 169702/400000 [00:17<00:24, 9491.32it/s] 43%|████▎     | 170727/400000 [00:17<00:23, 9705.63it/s] 43%|████▎     | 171706/400000 [00:17<00:23, 9730.68it/s] 43%|████▎     | 172684/400000 [00:17<00:23, 9704.42it/s] 43%|████▎     | 173706/400000 [00:17<00:22, 9851.77it/s] 44%|████▎     | 174694/400000 [00:17<00:22, 9812.84it/s] 44%|████▍     | 175698/400000 [00:17<00:22, 9879.80it/s] 44%|████▍     | 176713/400000 [00:17<00:22, 9957.16it/s] 44%|████▍     | 177720/400000 [00:17<00:22, 9988.92it/s] 45%|████▍     | 178722/400000 [00:18<00:22, 9995.92it/s] 45%|████▍     | 179723/400000 [00:18<00:22, 9785.72it/s] 45%|████▌     | 180703/400000 [00:18<00:22, 9593.71it/s] 45%|████▌     | 181665/400000 [00:18<00:23, 9447.15it/s] 46%|████▌     | 182612/400000 [00:18<00:23, 9368.33it/s] 46%|████▌     | 183582/400000 [00:18<00:22, 9463.34it/s] 46%|████▌     | 184530/400000 [00:18<00:22, 9452.61it/s] 46%|████▋     | 185534/400000 [00:18<00:22, 9620.24it/s] 47%|████▋     | 186527/400000 [00:18<00:21, 9709.46it/s] 47%|████▋     | 187500/400000 [00:19<00:22, 9523.05it/s] 47%|████▋     | 188476/400000 [00:19<00:22, 9592.03it/s] 47%|████▋     | 189437/400000 [00:19<00:22, 9428.10it/s] 48%|████▊     | 190382/400000 [00:19<00:22, 9330.79it/s] 48%|████▊     | 191317/400000 [00:19<00:22, 9193.54it/s] 48%|████▊     | 192298/400000 [00:19<00:22, 9370.10it/s] 48%|████▊     | 193311/400000 [00:19<00:21, 9585.01it/s] 49%|████▊     | 194272/400000 [00:19<00:21, 9579.28it/s] 49%|████▉     | 195322/400000 [00:19<00:20, 9837.03it/s] 49%|████▉     | 196347/400000 [00:19<00:20, 9954.43it/s] 49%|████▉     | 197399/400000 [00:20<00:20, 10115.58it/s] 50%|████▉     | 198413/400000 [00:20<00:20, 9862.40it/s]  50%|████▉     | 199407/400000 [00:20<00:20, 9883.82it/s] 50%|█████     | 200427/400000 [00:20<00:20, 9975.96it/s] 50%|█████     | 201452/400000 [00:20<00:19, 10055.40it/s] 51%|█████     | 202473/400000 [00:20<00:19, 10099.20it/s] 51%|█████     | 203592/400000 [00:20<00:18, 10403.07it/s] 51%|█████     | 204636/400000 [00:20<00:19, 10117.73it/s] 51%|█████▏    | 205652/400000 [00:20<00:19, 10101.35it/s] 52%|█████▏    | 206694/400000 [00:20<00:18, 10193.28it/s] 52%|█████▏    | 207716/400000 [00:21<00:19, 10117.49it/s] 52%|█████▏    | 208730/400000 [00:21<00:19, 10014.16it/s] 52%|█████▏    | 209733/400000 [00:21<00:19, 9802.75it/s]  53%|█████▎    | 210766/400000 [00:21<00:19, 9954.86it/s] 53%|█████▎    | 211788/400000 [00:21<00:18, 10032.37it/s] 53%|█████▎    | 212793/400000 [00:21<00:18, 9915.96it/s]  53%|█████▎    | 213874/400000 [00:21<00:18, 10166.70it/s] 54%|█████▎    | 214894/400000 [00:21<00:18, 10057.10it/s] 54%|█████▍    | 215948/400000 [00:21<00:18, 10195.17it/s] 54%|█████▍    | 216982/400000 [00:21<00:17, 10236.61it/s] 55%|█████▍    | 218008/400000 [00:22<00:18, 10048.62it/s] 55%|█████▍    | 219015/400000 [00:22<00:18, 9667.54it/s]  55%|█████▍    | 219987/400000 [00:22<00:18, 9552.53it/s] 55%|█████▌    | 220946/400000 [00:22<00:18, 9511.27it/s] 55%|█████▌    | 221981/400000 [00:22<00:18, 9746.30it/s] 56%|█████▌    | 223076/400000 [00:22<00:17, 10078.56it/s] 56%|█████▌    | 224113/400000 [00:22<00:17, 10163.05it/s] 56%|█████▋    | 225133/400000 [00:22<00:17, 10165.12it/s] 57%|█████▋    | 226176/400000 [00:22<00:16, 10242.48it/s] 57%|█████▋    | 227275/400000 [00:23<00:16, 10454.72it/s] 57%|█████▋    | 228323/400000 [00:23<00:16, 10225.35it/s] 57%|█████▋    | 229372/400000 [00:23<00:16, 10301.24it/s] 58%|█████▊    | 230405/400000 [00:23<00:16, 10119.25it/s] 58%|█████▊    | 231430/400000 [00:23<00:16, 10156.89it/s] 58%|█████▊    | 232448/400000 [00:23<00:16, 10144.76it/s] 58%|█████▊    | 233464/400000 [00:23<00:16, 10054.22it/s] 59%|█████▊    | 234531/400000 [00:23<00:16, 10229.07it/s] 59%|█████▉    | 235556/400000 [00:23<00:16, 9851.90it/s]  59%|█████▉    | 236573/400000 [00:23<00:16, 9942.96it/s] 59%|█████▉    | 237636/400000 [00:24<00:16, 10139.27it/s] 60%|█████▉    | 238653/400000 [00:24<00:15, 10145.94it/s] 60%|█████▉    | 239670/400000 [00:24<00:15, 10140.86it/s] 60%|██████    | 240686/400000 [00:24<00:15, 10096.77it/s] 60%|██████    | 241697/400000 [00:24<00:16, 9752.59it/s]  61%|██████    | 242681/400000 [00:24<00:16, 9778.58it/s] 61%|██████    | 243662/400000 [00:24<00:16, 9493.50it/s] 61%|██████    | 244615/400000 [00:24<00:16, 9274.30it/s] 61%|██████▏   | 245546/400000 [00:24<00:16, 9200.33it/s] 62%|██████▏   | 246477/400000 [00:24<00:16, 9230.25it/s] 62%|██████▏   | 247439/400000 [00:25<00:16, 9341.13it/s] 62%|██████▏   | 248375/400000 [00:25<00:16, 9229.05it/s] 62%|██████▏   | 249300/400000 [00:25<00:16, 9149.61it/s] 63%|██████▎   | 250217/400000 [00:25<00:16, 9099.16it/s] 63%|██████▎   | 251139/400000 [00:25<00:16, 9133.63it/s] 63%|██████▎   | 252079/400000 [00:25<00:16, 9209.21it/s] 63%|██████▎   | 253001/400000 [00:25<00:16, 9161.86it/s] 63%|██████▎   | 253918/400000 [00:25<00:16, 9009.44it/s] 64%|██████▎   | 254820/400000 [00:25<00:16, 8994.61it/s] 64%|██████▍   | 255754/400000 [00:25<00:15, 9092.61it/s] 64%|██████▍   | 256720/400000 [00:26<00:15, 9253.95it/s] 64%|██████▍   | 257652/400000 [00:26<00:15, 9271.00it/s] 65%|██████▍   | 258610/400000 [00:26<00:15, 9361.58it/s] 65%|██████▍   | 259547/400000 [00:26<00:15, 9308.60it/s] 65%|██████▌   | 260511/400000 [00:26<00:14, 9404.98it/s] 65%|██████▌   | 261482/400000 [00:26<00:14, 9493.36it/s] 66%|██████▌   | 262433/400000 [00:26<00:14, 9457.47it/s] 66%|██████▌   | 263397/400000 [00:26<00:14, 9509.35it/s] 66%|██████▌   | 264349/400000 [00:26<00:14, 9447.58it/s] 66%|██████▋   | 265303/400000 [00:26<00:14, 9472.87it/s] 67%|██████▋   | 266251/400000 [00:27<00:14, 9332.25it/s] 67%|██████▋   | 267209/400000 [00:27<00:14, 9404.03it/s] 67%|██████▋   | 268186/400000 [00:27<00:13, 9510.27it/s] 67%|██████▋   | 269138/400000 [00:27<00:14, 9298.23it/s] 68%|██████▊   | 270070/400000 [00:27<00:14, 9206.78it/s] 68%|██████▊   | 270992/400000 [00:27<00:14, 9066.99it/s] 68%|██████▊   | 271942/400000 [00:27<00:13, 9192.52it/s] 68%|██████▊   | 272897/400000 [00:27<00:13, 9295.48it/s] 68%|██████▊   | 273828/400000 [00:27<00:13, 9292.88it/s] 69%|██████▊   | 274778/400000 [00:28<00:13, 9352.97it/s] 69%|██████▉   | 275742/400000 [00:28<00:13, 9436.97it/s] 69%|██████▉   | 276715/400000 [00:28<00:12, 9521.58it/s] 69%|██████▉   | 277718/400000 [00:28<00:12, 9668.12it/s] 70%|██████▉   | 278721/400000 [00:28<00:12, 9772.58it/s] 70%|██████▉   | 279788/400000 [00:28<00:11, 10025.43it/s] 70%|███████   | 280893/400000 [00:28<00:11, 10310.49it/s] 70%|███████   | 281928/400000 [00:28<00:11, 10202.84it/s] 71%|███████   | 282952/400000 [00:28<00:11, 9981.62it/s]  71%|███████   | 283960/400000 [00:28<00:11, 10009.66it/s] 71%|███████▏  | 285076/400000 [00:29<00:11, 10327.56it/s] 72%|███████▏  | 286136/400000 [00:29<00:10, 10405.99it/s] 72%|███████▏  | 287180/400000 [00:29<00:10, 10330.30it/s] 72%|███████▏  | 288261/400000 [00:29<00:10, 10466.96it/s] 72%|███████▏  | 289310/400000 [00:29<00:10, 10337.09it/s] 73%|███████▎  | 290402/400000 [00:29<00:10, 10503.25it/s] 73%|███████▎  | 291456/400000 [00:29<00:10, 10511.91it/s] 73%|███████▎  | 292509/400000 [00:29<00:10, 10443.73it/s] 73%|███████▎  | 293605/400000 [00:29<00:10, 10591.23it/s] 74%|███████▎  | 294666/400000 [00:29<00:10, 10359.22it/s] 74%|███████▍  | 295740/400000 [00:30<00:09, 10468.23it/s] 74%|███████▍  | 296789/400000 [00:30<00:10, 10301.40it/s] 74%|███████▍  | 297821/400000 [00:30<00:10, 10056.69it/s] 75%|███████▍  | 298830/400000 [00:30<00:10, 9983.95it/s]  75%|███████▍  | 299831/400000 [00:30<00:10, 9653.60it/s] 75%|███████▌  | 300807/400000 [00:30<00:10, 9682.12it/s] 75%|███████▌  | 301778/400000 [00:30<00:10, 9572.62it/s] 76%|███████▌  | 302776/400000 [00:30<00:10, 9690.01it/s] 76%|███████▌  | 303751/400000 [00:30<00:09, 9707.62it/s] 76%|███████▌  | 304736/400000 [00:30<00:09, 9747.54it/s] 76%|███████▋  | 305733/400000 [00:31<00:09, 9812.71it/s] 77%|███████▋  | 306722/400000 [00:31<00:09, 9832.80it/s] 77%|███████▋  | 307706/400000 [00:31<00:09, 9777.86it/s] 77%|███████▋  | 308700/400000 [00:31<00:09, 9824.87it/s] 77%|███████▋  | 309683/400000 [00:31<00:09, 9612.07it/s] 78%|███████▊  | 310662/400000 [00:31<00:09, 9661.96it/s] 78%|███████▊  | 311696/400000 [00:31<00:08, 9855.74it/s] 78%|███████▊  | 312725/400000 [00:31<00:08, 9979.53it/s] 78%|███████▊  | 313725/400000 [00:31<00:08, 9984.43it/s] 79%|███████▊  | 314745/400000 [00:31<00:08, 10047.45it/s] 79%|███████▉  | 315795/400000 [00:32<00:08, 10177.96it/s] 79%|███████▉  | 316814/400000 [00:32<00:08, 10114.87it/s] 79%|███████▉  | 317828/400000 [00:32<00:08, 10120.15it/s] 80%|███████▉  | 318841/400000 [00:32<00:08, 10021.48it/s] 80%|███████▉  | 319844/400000 [00:32<00:08, 9979.07it/s]  80%|████████  | 320894/400000 [00:32<00:07, 10127.53it/s] 80%|████████  | 321925/400000 [00:32<00:07, 10180.01it/s] 81%|████████  | 322944/400000 [00:32<00:07, 10001.31it/s] 81%|████████  | 323946/400000 [00:32<00:07, 9821.39it/s]  81%|████████  | 324930/400000 [00:33<00:07, 9709.48it/s] 81%|████████▏ | 325903/400000 [00:33<00:07, 9678.82it/s] 82%|████████▏ | 326899/400000 [00:33<00:07, 9759.13it/s] 82%|████████▏ | 327916/400000 [00:33<00:07, 9878.76it/s] 82%|████████▏ | 328953/400000 [00:33<00:07, 10019.27it/s] 82%|████████▏ | 329957/400000 [00:33<00:07, 9829.59it/s]  83%|████████▎ | 330970/400000 [00:33<00:06, 9917.36it/s] 83%|████████▎ | 332076/400000 [00:33<00:06, 10234.05it/s] 83%|████████▎ | 333103/400000 [00:33<00:06, 10232.95it/s] 84%|████████▎ | 334129/400000 [00:33<00:06, 10140.25it/s] 84%|████████▍ | 335168/400000 [00:34<00:06, 10212.60it/s] 84%|████████▍ | 336191/400000 [00:34<00:06, 10217.73it/s] 84%|████████▍ | 337240/400000 [00:34<00:06, 10297.90it/s] 85%|████████▍ | 338357/400000 [00:34<00:05, 10543.85it/s] 85%|████████▍ | 339414/400000 [00:34<00:05, 10154.72it/s] 85%|████████▌ | 340492/400000 [00:34<00:05, 10331.92it/s] 85%|████████▌ | 341530/400000 [00:34<00:05, 10272.43it/s] 86%|████████▌ | 342574/400000 [00:34<00:05, 10321.41it/s] 86%|████████▌ | 343609/400000 [00:34<00:05, 10190.33it/s] 86%|████████▌ | 344630/400000 [00:34<00:05, 9879.73it/s]  86%|████████▋ | 345622/400000 [00:35<00:05, 9887.97it/s] 87%|████████▋ | 346622/400000 [00:35<00:05, 9918.95it/s] 87%|████████▋ | 347657/400000 [00:35<00:05, 10043.17it/s] 87%|████████▋ | 348727/400000 [00:35<00:05, 10229.46it/s] 87%|████████▋ | 349752/400000 [00:35<00:04, 10207.51it/s] 88%|████████▊ | 350775/400000 [00:35<00:04, 10003.10it/s] 88%|████████▊ | 351778/400000 [00:35<00:04, 9845.30it/s]  88%|████████▊ | 352835/400000 [00:35<00:04, 10051.19it/s] 88%|████████▊ | 353927/400000 [00:35<00:04, 10295.53it/s] 89%|████████▊ | 354960/400000 [00:35<00:04, 10301.88it/s] 89%|████████▉ | 355993/400000 [00:36<00:04, 10202.05it/s] 89%|████████▉ | 357044/400000 [00:36<00:04, 10291.26it/s] 90%|████████▉ | 358076/400000 [00:36<00:04, 10298.00it/s] 90%|████████▉ | 359179/400000 [00:36<00:03, 10506.25it/s] 90%|█████████ | 360232/400000 [00:36<00:03, 10439.81it/s] 90%|█████████ | 361278/400000 [00:36<00:03, 10229.04it/s] 91%|█████████ | 362329/400000 [00:36<00:03, 10308.90it/s] 91%|█████████ | 363362/400000 [00:36<00:03, 10284.44it/s] 91%|█████████ | 364394/400000 [00:36<00:03, 10292.71it/s] 91%|█████████▏| 365424/400000 [00:36<00:03, 10262.26it/s] 92%|█████████▏| 366451/400000 [00:37<00:03, 10210.78it/s] 92%|█████████▏| 367473/400000 [00:37<00:03, 9810.50it/s]  92%|█████████▏| 368458/400000 [00:37<00:03, 9590.13it/s] 92%|█████████▏| 369473/400000 [00:37<00:03, 9751.15it/s] 93%|█████████▎| 370473/400000 [00:37<00:03, 9823.52it/s] 93%|█████████▎| 371478/400000 [00:37<00:02, 9888.61it/s] 93%|█████████▎| 372503/400000 [00:37<00:02, 9993.58it/s] 93%|█████████▎| 373504/400000 [00:37<00:02, 9990.87it/s] 94%|█████████▎| 374522/400000 [00:37<00:02, 10046.71it/s] 94%|█████████▍| 375528/400000 [00:38<00:02, 9964.11it/s]  94%|█████████▍| 376526/400000 [00:38<00:02, 9819.63it/s] 94%|█████████▍| 377509/400000 [00:38<00:02, 9645.64it/s] 95%|█████████▍| 378475/400000 [00:38<00:02, 9646.08it/s] 95%|█████████▍| 379485/400000 [00:38<00:02, 9775.45it/s] 95%|█████████▌| 380533/400000 [00:38<00:01, 9976.59it/s] 95%|█████████▌| 381544/400000 [00:38<00:01, 10014.93it/s] 96%|█████████▌| 382646/400000 [00:38<00:01, 10295.65it/s] 96%|█████████▌| 383679/400000 [00:38<00:01, 10150.21it/s] 96%|█████████▌| 384697/400000 [00:38<00:01, 10123.79it/s] 96%|█████████▋| 385755/400000 [00:39<00:01, 10254.83it/s] 97%|█████████▋| 386783/400000 [00:39<00:01, 10127.58it/s] 97%|█████████▋| 387812/400000 [00:39<00:01, 10172.80it/s] 97%|█████████▋| 388880/400000 [00:39<00:01, 10318.24it/s] 97%|█████████▋| 389914/400000 [00:39<00:00, 10311.91it/s] 98%|█████████▊| 390965/400000 [00:39<00:00, 10367.64it/s] 98%|█████████▊| 392003/400000 [00:39<00:00, 10334.07it/s] 98%|█████████▊| 393042/400000 [00:39<00:00, 10350.35it/s] 99%|█████████▊| 394102/400000 [00:39<00:00, 10422.41it/s] 99%|█████████▉| 395177/400000 [00:39<00:00, 10516.48it/s] 99%|█████████▉| 396258/400000 [00:40<00:00, 10602.69it/s] 99%|█████████▉| 397319/400000 [00:40<00:00, 10493.76it/s]100%|█████████▉| 398369/400000 [00:40<00:00, 10216.67it/s]100%|█████████▉| 399393/400000 [00:40<00:00, 10165.30it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9898.38it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f319486d128>, <torchtext.data.dataset.TabularDataset object at 0x7f319486d278>, <torchtext.vocab.Vocab object at 0x7f319486d198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.76 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.26 file/s]2020-06-12 18:17:47.978040: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 18:17:48.129479: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-12 18:17:48.129679: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56297b54aa90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 18:17:48.129695: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3457024/9912422 [00:00<00:00, 34063258.87it/s]9920512it [00:00, 35617711.22it/s]                             
0it [00:00, ?it/s]32768it [00:00, 582644.96it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470345.23it/s]1654784it [00:00, 11673953.19it/s]                         
0it [00:00, ?it/s]8192it [00:00, 212339.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
