
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f84968bb620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f84968bb620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8501e820d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8501e820d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f851bbafea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f851bbafea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84af6fd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84af6fd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84af6fd950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3768320/9912422 [00:00<00:00, 37676498.85it/s]9920512it [00:00, 35856410.40it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 275902.06it/s]           
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 454507.92it/s]1654784it [00:00, 11634385.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 124102.86it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84acd15a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f84acd09d30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f84af6fd598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f84af6fd598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f84af6fd598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:18,  7.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12,  9.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 21.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 32.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 41.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 45.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 49.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 49.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 54.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 57.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 57.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 59.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 59.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 60.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 60.10 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 60.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 60.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 60.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.82 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.63s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 60.82 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.63s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.63s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.77 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.63s/ url]
0 examples [00:00, ? examples/s]2020-06-09 06:09:24.439191: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-09 06:09:24.454651: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-09 06:09:24.454956: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562755927460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 06:09:24.454977: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
27 examples [00:00, 267.65 examples/s]115 examples [00:00, 338.25 examples/s]210 examples [00:00, 418.67 examples/s]296 examples [00:00, 494.21 examples/s]389 examples [00:00, 574.07 examples/s]478 examples [00:00, 641.28 examples/s]567 examples [00:00, 697.78 examples/s]650 examples [00:00, 731.25 examples/s]731 examples [00:00, 722.45 examples/s]809 examples [00:01, 731.22 examples/s]888 examples [00:01, 746.60 examples/s]966 examples [00:01, 755.62 examples/s]1054 examples [00:01, 786.83 examples/s]1145 examples [00:01, 818.45 examples/s]1236 examples [00:01, 841.60 examples/s]1322 examples [00:01, 844.01 examples/s]1414 examples [00:01, 864.01 examples/s]1508 examples [00:01, 884.61 examples/s]1601 examples [00:01, 895.57 examples/s]1692 examples [00:02, 866.24 examples/s]1780 examples [00:02, 866.88 examples/s]1871 examples [00:02, 878.90 examples/s]1962 examples [00:02, 885.95 examples/s]2051 examples [00:02, 864.49 examples/s]2140 examples [00:02, 869.44 examples/s]2228 examples [00:02, 870.00 examples/s]2318 examples [00:02, 877.92 examples/s]2411 examples [00:02, 890.03 examples/s]2502 examples [00:02, 894.57 examples/s]2594 examples [00:03, 899.27 examples/s]2684 examples [00:03, 889.66 examples/s]2774 examples [00:03, 889.01 examples/s]2863 examples [00:03, 881.90 examples/s]2952 examples [00:03, 876.22 examples/s]3040 examples [00:03, 867.97 examples/s]3127 examples [00:03, 848.43 examples/s]3212 examples [00:03, 827.15 examples/s]3300 examples [00:03, 840.49 examples/s]3389 examples [00:03, 853.71 examples/s]3484 examples [00:04, 879.57 examples/s]3573 examples [00:04, 862.71 examples/s]3660 examples [00:04, 863.66 examples/s]3748 examples [00:04, 867.69 examples/s]3839 examples [00:04, 878.47 examples/s]3930 examples [00:04, 885.79 examples/s]4019 examples [00:04, 876.10 examples/s]4109 examples [00:04, 880.75 examples/s]4198 examples [00:04, 874.50 examples/s]4289 examples [00:05, 884.55 examples/s]4380 examples [00:05, 889.75 examples/s]4470 examples [00:05, 883.34 examples/s]4559 examples [00:05, 870.93 examples/s]4647 examples [00:05, 857.55 examples/s]4735 examples [00:05, 864.16 examples/s]4822 examples [00:05, 859.98 examples/s]4910 examples [00:05, 865.67 examples/s]4997 examples [00:05, 864.36 examples/s]5085 examples [00:05, 866.81 examples/s]5179 examples [00:06, 884.78 examples/s]5268 examples [00:06, 872.74 examples/s]5356 examples [00:06, 867.57 examples/s]5443 examples [00:06, 858.11 examples/s]5531 examples [00:06, 863.13 examples/s]5621 examples [00:06, 872.94 examples/s]5710 examples [00:06, 877.95 examples/s]5800 examples [00:06, 881.83 examples/s]5889 examples [00:06, 864.31 examples/s]5976 examples [00:06, 858.20 examples/s]6067 examples [00:07, 873.10 examples/s]6157 examples [00:07, 878.70 examples/s]6247 examples [00:07, 883.25 examples/s]6336 examples [00:07, 874.57 examples/s]6429 examples [00:07, 888.44 examples/s]6519 examples [00:07, 888.41 examples/s]6608 examples [00:07, 882.05 examples/s]6697 examples [00:07, 880.21 examples/s]6787 examples [00:07, 883.85 examples/s]6880 examples [00:07, 896.54 examples/s]6974 examples [00:08, 906.56 examples/s]7065 examples [00:08, 906.87 examples/s]7156 examples [00:08, 896.64 examples/s]7246 examples [00:08, 879.44 examples/s]7335 examples [00:08, 873.77 examples/s]7428 examples [00:08, 887.84 examples/s]7519 examples [00:08, 894.31 examples/s]7609 examples [00:08, 852.19 examples/s]7695 examples [00:08, 842.48 examples/s]7780 examples [00:09, 830.71 examples/s]7873 examples [00:09, 856.63 examples/s]7960 examples [00:09, 855.05 examples/s]8049 examples [00:09, 862.45 examples/s]8140 examples [00:09, 875.02 examples/s]8231 examples [00:09, 884.33 examples/s]8324 examples [00:09, 896.60 examples/s]8417 examples [00:09, 902.85 examples/s]8508 examples [00:09, 886.00 examples/s]8597 examples [00:09, 869.61 examples/s]8692 examples [00:10, 889.99 examples/s]8784 examples [00:10, 896.85 examples/s]8875 examples [00:10, 899.20 examples/s]8966 examples [00:10, 899.95 examples/s]9057 examples [00:10, 894.45 examples/s]9147 examples [00:10, 894.55 examples/s]9237 examples [00:10, 895.76 examples/s]9327 examples [00:10, 863.68 examples/s]9416 examples [00:10, 869.14 examples/s]9505 examples [00:10, 873.25 examples/s]9595 examples [00:11, 880.85 examples/s]9688 examples [00:11, 888.83 examples/s]9781 examples [00:11, 900.55 examples/s]9872 examples [00:11, 900.25 examples/s]9963 examples [00:11, 897.25 examples/s]10053 examples [00:11, 819.52 examples/s]10144 examples [00:11, 843.64 examples/s]10232 examples [00:11, 854.08 examples/s]10322 examples [00:11, 866.49 examples/s]10413 examples [00:11, 877.69 examples/s]10502 examples [00:12, 863.59 examples/s]10593 examples [00:12, 874.76 examples/s]10682 examples [00:12, 878.10 examples/s]10773 examples [00:12, 886.45 examples/s]10862 examples [00:12, 851.54 examples/s]10952 examples [00:12, 863.57 examples/s]11042 examples [00:12, 873.91 examples/s]11130 examples [00:12, 872.43 examples/s]11219 examples [00:12, 873.52 examples/s]11307 examples [00:13, 874.47 examples/s]11395 examples [00:13, 864.17 examples/s]11484 examples [00:13, 870.69 examples/s]11572 examples [00:13, 865.86 examples/s]11661 examples [00:13, 872.42 examples/s]11749 examples [00:13, 858.07 examples/s]11839 examples [00:13, 867.29 examples/s]11929 examples [00:13, 874.81 examples/s]12017 examples [00:13, 875.96 examples/s]12105 examples [00:13, 875.70 examples/s]12197 examples [00:14, 885.74 examples/s]12286 examples [00:14, 849.72 examples/s]12372 examples [00:14, 839.69 examples/s]12460 examples [00:14, 849.23 examples/s]12549 examples [00:14, 859.15 examples/s]12636 examples [00:14, 850.53 examples/s]12727 examples [00:14, 865.79 examples/s]12814 examples [00:14, 853.04 examples/s]12900 examples [00:14, 840.32 examples/s]12991 examples [00:14, 858.60 examples/s]13082 examples [00:15, 871.89 examples/s]13172 examples [00:15, 879.00 examples/s]13263 examples [00:15, 885.49 examples/s]13353 examples [00:15, 887.46 examples/s]13442 examples [00:15, 874.46 examples/s]13531 examples [00:15, 877.65 examples/s]13619 examples [00:15, 856.61 examples/s]13709 examples [00:15, 867.35 examples/s]13798 examples [00:15, 872.17 examples/s]13886 examples [00:16, 871.87 examples/s]13974 examples [00:16, 867.83 examples/s]14064 examples [00:16, 876.49 examples/s]14154 examples [00:16, 881.33 examples/s]14243 examples [00:16, 862.27 examples/s]14333 examples [00:16, 870.52 examples/s]14421 examples [00:16, 854.94 examples/s]14514 examples [00:16, 873.89 examples/s]14607 examples [00:16, 887.81 examples/s]14696 examples [00:16, 884.89 examples/s]14790 examples [00:17, 899.54 examples/s]14884 examples [00:17, 909.31 examples/s]14977 examples [00:17, 915.23 examples/s]15070 examples [00:17, 918.30 examples/s]15162 examples [00:17, 911.05 examples/s]15254 examples [00:17, 909.53 examples/s]15345 examples [00:17, 896.86 examples/s]15435 examples [00:17, 888.25 examples/s]15526 examples [00:17, 894.23 examples/s]15616 examples [00:17, 891.51 examples/s]15707 examples [00:18, 895.08 examples/s]15797 examples [00:18, 871.83 examples/s]15885 examples [00:18, 870.79 examples/s]15975 examples [00:18, 878.70 examples/s]16063 examples [00:18, 873.59 examples/s]16151 examples [00:18, 869.64 examples/s]16239 examples [00:18, 864.88 examples/s]16327 examples [00:18, 868.24 examples/s]16415 examples [00:18, 869.85 examples/s]16503 examples [00:18, 854.53 examples/s]16589 examples [00:19, 848.37 examples/s]16674 examples [00:19, 839.55 examples/s]16764 examples [00:19, 856.24 examples/s]16857 examples [00:19, 877.03 examples/s]16945 examples [00:19, 857.35 examples/s]17031 examples [00:19, 846.56 examples/s]17120 examples [00:19, 858.39 examples/s]17208 examples [00:19, 862.51 examples/s]17295 examples [00:19, 863.76 examples/s]17384 examples [00:19, 871.47 examples/s]17474 examples [00:20, 878.75 examples/s]17566 examples [00:20, 887.83 examples/s]17655 examples [00:20, 866.90 examples/s]17743 examples [00:20, 868.22 examples/s]17832 examples [00:20, 871.28 examples/s]17920 examples [00:20, 833.46 examples/s]18004 examples [00:20, 815.33 examples/s]18094 examples [00:20, 838.47 examples/s]18184 examples [00:20, 853.28 examples/s]18270 examples [00:21, 850.35 examples/s]18356 examples [00:21, 851.23 examples/s]18445 examples [00:21, 860.04 examples/s]18537 examples [00:21, 875.52 examples/s]18625 examples [00:21, 874.40 examples/s]18713 examples [00:21, 874.08 examples/s]18803 examples [00:21, 879.52 examples/s]18892 examples [00:21, 875.65 examples/s]18980 examples [00:21, 872.78 examples/s]19069 examples [00:21, 876.66 examples/s]19157 examples [00:22, 868.83 examples/s]19244 examples [00:22, 824.15 examples/s]19331 examples [00:22, 835.20 examples/s]19419 examples [00:22, 846.24 examples/s]19504 examples [00:22, 846.96 examples/s]19590 examples [00:22, 849.02 examples/s]19677 examples [00:22, 854.97 examples/s]19763 examples [00:22, 854.36 examples/s]19852 examples [00:22, 862.42 examples/s]19940 examples [00:22, 864.67 examples/s]20027 examples [00:23, 778.20 examples/s]20109 examples [00:23, 789.78 examples/s]20199 examples [00:23, 817.49 examples/s]20288 examples [00:23, 836.31 examples/s]20374 examples [00:23, 841.95 examples/s]20462 examples [00:23, 851.68 examples/s]20549 examples [00:23, 856.48 examples/s]20637 examples [00:23, 862.94 examples/s]20729 examples [00:23, 877.40 examples/s]20817 examples [00:24, 869.81 examples/s]20907 examples [00:24, 877.81 examples/s]20998 examples [00:24, 885.90 examples/s]21087 examples [00:24, 864.22 examples/s]21176 examples [00:24, 867.43 examples/s]21263 examples [00:24, 859.25 examples/s]21353 examples [00:24, 870.37 examples/s]21441 examples [00:24, 861.01 examples/s]21530 examples [00:24, 868.89 examples/s]21619 examples [00:24, 872.85 examples/s]21709 examples [00:25, 879.55 examples/s]21801 examples [00:25, 890.64 examples/s]21891 examples [00:25, 868.60 examples/s]21984 examples [00:25, 886.15 examples/s]22073 examples [00:25, 861.81 examples/s]22160 examples [00:25, 857.87 examples/s]22252 examples [00:25, 873.85 examples/s]22340 examples [00:25, 858.47 examples/s]22428 examples [00:25, 862.66 examples/s]22517 examples [00:25, 869.35 examples/s]22605 examples [00:26, 872.13 examples/s]22696 examples [00:26, 881.54 examples/s]22785 examples [00:26, 864.79 examples/s]22872 examples [00:26, 857.72 examples/s]22958 examples [00:26, 856.80 examples/s]23044 examples [00:26, 847.37 examples/s]23130 examples [00:26, 850.05 examples/s]23220 examples [00:26, 861.81 examples/s]23309 examples [00:26, 868.09 examples/s]23397 examples [00:26, 870.90 examples/s]23485 examples [00:27, 871.87 examples/s]23573 examples [00:27, 873.84 examples/s]23661 examples [00:27, 863.27 examples/s]23748 examples [00:27, 862.21 examples/s]23835 examples [00:27, 855.74 examples/s]23921 examples [00:27, 845.68 examples/s]24006 examples [00:27, 842.73 examples/s]24091 examples [00:27, 833.62 examples/s]24180 examples [00:27, 846.80 examples/s]24271 examples [00:28, 863.27 examples/s]24360 examples [00:28, 870.29 examples/s]24448 examples [00:28, 847.53 examples/s]24533 examples [00:28, 839.77 examples/s]24619 examples [00:28, 844.89 examples/s]24706 examples [00:28, 851.27 examples/s]24792 examples [00:28, 853.65 examples/s]24879 examples [00:28, 857.38 examples/s]24966 examples [00:28, 860.93 examples/s]25054 examples [00:28, 865.36 examples/s]25142 examples [00:29, 869.46 examples/s]25229 examples [00:29, 869.55 examples/s]25319 examples [00:29, 878.05 examples/s]25407 examples [00:29, 875.14 examples/s]25495 examples [00:29, 872.10 examples/s]25585 examples [00:29, 879.42 examples/s]25673 examples [00:29, 869.59 examples/s]25761 examples [00:29, 858.75 examples/s]25847 examples [00:29, 858.29 examples/s]25935 examples [00:29, 862.54 examples/s]26023 examples [00:30, 864.99 examples/s]26110 examples [00:30, 830.42 examples/s]26194 examples [00:30, 817.62 examples/s]26280 examples [00:30, 829.35 examples/s]26369 examples [00:30, 845.04 examples/s]26460 examples [00:30, 863.21 examples/s]26547 examples [00:30, 844.12 examples/s]26632 examples [00:30, 829.71 examples/s]26718 examples [00:30, 838.36 examples/s]26804 examples [00:30, 844.23 examples/s]26891 examples [00:31, 851.12 examples/s]26977 examples [00:31, 848.29 examples/s]27062 examples [00:31, 843.61 examples/s]27147 examples [00:31, 825.38 examples/s]27235 examples [00:31, 838.69 examples/s]27320 examples [00:31, 840.90 examples/s]27406 examples [00:31, 846.23 examples/s]27492 examples [00:31, 848.04 examples/s]27579 examples [00:31, 852.44 examples/s]27665 examples [00:31, 854.35 examples/s]27752 examples [00:32, 858.41 examples/s]27838 examples [00:32, 833.04 examples/s]27928 examples [00:32, 850.71 examples/s]28015 examples [00:32, 843.29 examples/s]28100 examples [00:32, 837.67 examples/s]28184 examples [00:32, 821.83 examples/s]28268 examples [00:32, 824.57 examples/s]28353 examples [00:32, 829.48 examples/s]28437 examples [00:32, 820.28 examples/s]28520 examples [00:33, 821.26 examples/s]28606 examples [00:33, 831.30 examples/s]28692 examples [00:33, 839.52 examples/s]28777 examples [00:33, 823.91 examples/s]28866 examples [00:33, 840.28 examples/s]28951 examples [00:33, 832.74 examples/s]29042 examples [00:33, 852.79 examples/s]29128 examples [00:33, 850.75 examples/s]29214 examples [00:33, 847.26 examples/s]29299 examples [00:33, 847.69 examples/s]29384 examples [00:34, 846.06 examples/s]29470 examples [00:34, 830.97 examples/s]29555 examples [00:34, 835.43 examples/s]29640 examples [00:34, 838.63 examples/s]29729 examples [00:34, 850.65 examples/s]29815 examples [00:34, 850.07 examples/s]29901 examples [00:34, 839.28 examples/s]29985 examples [00:34, 834.99 examples/s]30069 examples [00:34, 801.55 examples/s]30162 examples [00:34, 836.16 examples/s]30255 examples [00:35, 861.33 examples/s]30342 examples [00:35, 853.01 examples/s]30430 examples [00:35, 859.02 examples/s]30520 examples [00:35, 868.60 examples/s]30608 examples [00:35, 867.34 examples/s]30695 examples [00:35, 863.02 examples/s]30782 examples [00:35, 825.27 examples/s]30866 examples [00:35, 828.93 examples/s]30958 examples [00:35, 852.44 examples/s]31048 examples [00:36, 863.69 examples/s]31138 examples [00:36, 872.21 examples/s]31227 examples [00:36, 876.05 examples/s]31315 examples [00:36, 854.27 examples/s]31402 examples [00:36, 857.64 examples/s]31489 examples [00:36, 861.24 examples/s]31577 examples [00:36, 865.44 examples/s]31664 examples [00:36, 845.95 examples/s]31752 examples [00:36, 853.61 examples/s]31839 examples [00:36, 856.58 examples/s]31928 examples [00:37, 864.93 examples/s]32017 examples [00:37, 870.54 examples/s]32105 examples [00:37, 869.81 examples/s]32193 examples [00:37, 866.38 examples/s]32283 examples [00:37, 874.01 examples/s]32371 examples [00:37, 862.32 examples/s]32458 examples [00:37, 857.77 examples/s]32544 examples [00:37, 838.11 examples/s]32628 examples [00:37, 826.28 examples/s]32718 examples [00:37, 846.81 examples/s]32807 examples [00:38, 856.70 examples/s]32897 examples [00:38, 867.05 examples/s]32984 examples [00:38, 856.98 examples/s]33072 examples [00:38, 860.69 examples/s]33159 examples [00:38, 862.73 examples/s]33250 examples [00:38, 874.82 examples/s]33338 examples [00:38, 838.06 examples/s]33426 examples [00:38, 847.61 examples/s]33512 examples [00:38, 841.47 examples/s]33597 examples [00:38, 843.52 examples/s]33687 examples [00:39, 858.65 examples/s]33777 examples [00:39, 868.85 examples/s]33865 examples [00:39, 854.25 examples/s]33957 examples [00:39, 872.10 examples/s]34045 examples [00:39, 873.79 examples/s]34133 examples [00:39, 874.17 examples/s]34224 examples [00:39, 881.92 examples/s]34313 examples [00:39, 879.63 examples/s]34402 examples [00:39, 882.31 examples/s]34494 examples [00:39, 891.72 examples/s]34584 examples [00:40, 893.28 examples/s]34675 examples [00:40, 897.15 examples/s]34765 examples [00:40, 877.12 examples/s]34855 examples [00:40, 883.10 examples/s]34946 examples [00:40, 890.20 examples/s]35036 examples [00:40, 892.30 examples/s]35126 examples [00:40, 880.56 examples/s]35215 examples [00:40, 840.34 examples/s]35300 examples [00:40, 836.91 examples/s]35385 examples [00:41, 823.15 examples/s]35476 examples [00:41, 844.58 examples/s]35562 examples [00:41, 848.45 examples/s]35650 examples [00:41, 855.90 examples/s]35736 examples [00:41, 854.22 examples/s]35822 examples [00:41, 849.20 examples/s]35908 examples [00:41, 807.40 examples/s]35990 examples [00:41, 799.48 examples/s]36074 examples [00:41, 810.92 examples/s]36163 examples [00:41, 832.61 examples/s]36252 examples [00:42, 847.55 examples/s]36340 examples [00:42, 855.52 examples/s]36429 examples [00:42, 863.52 examples/s]36516 examples [00:42, 863.67 examples/s]36604 examples [00:42, 867.11 examples/s]36695 examples [00:42, 878.65 examples/s]36786 examples [00:42, 886.34 examples/s]36875 examples [00:42, 885.36 examples/s]36964 examples [00:42, 876.74 examples/s]37052 examples [00:42, 876.11 examples/s]37142 examples [00:43, 880.80 examples/s]37231 examples [00:43, 864.97 examples/s]37321 examples [00:43, 874.04 examples/s]37409 examples [00:43, 868.49 examples/s]37496 examples [00:43, 847.72 examples/s]37584 examples [00:43, 856.75 examples/s]37670 examples [00:43, 845.71 examples/s]37760 examples [00:43, 860.67 examples/s]37847 examples [00:43, 841.41 examples/s]37933 examples [00:44, 846.43 examples/s]38019 examples [00:44, 850.04 examples/s]38106 examples [00:44, 853.87 examples/s]38197 examples [00:44, 866.90 examples/s]38284 examples [00:44, 858.12 examples/s]38372 examples [00:44, 862.22 examples/s]38459 examples [00:44, 854.29 examples/s]38546 examples [00:44, 857.40 examples/s]38634 examples [00:44, 861.50 examples/s]38726 examples [00:44, 876.12 examples/s]38816 examples [00:45, 881.64 examples/s]38906 examples [00:45, 883.68 examples/s]38997 examples [00:45, 890.52 examples/s]39087 examples [00:45, 862.46 examples/s]39178 examples [00:45, 875.49 examples/s]39267 examples [00:45, 877.93 examples/s]39357 examples [00:45, 883.22 examples/s]39446 examples [00:45, 859.03 examples/s]39539 examples [00:45, 876.21 examples/s]39627 examples [00:45, 876.68 examples/s]39715 examples [00:46, 869.60 examples/s]39803 examples [00:46, 867.11 examples/s]39891 examples [00:46, 869.58 examples/s]39979 examples [00:46, 857.22 examples/s]40065 examples [00:46, 773.50 examples/s]40148 examples [00:46, 789.27 examples/s]40232 examples [00:46, 802.88 examples/s]40318 examples [00:46, 816.18 examples/s]40403 examples [00:46, 825.14 examples/s]40487 examples [00:47, 828.52 examples/s]40575 examples [00:47, 841.68 examples/s]40660 examples [00:47, 843.21 examples/s]40748 examples [00:47, 852.27 examples/s]40835 examples [00:47, 855.21 examples/s]40922 examples [00:47, 857.24 examples/s]41008 examples [00:47, 826.39 examples/s]41097 examples [00:47, 843.79 examples/s]41184 examples [00:47, 849.30 examples/s]41270 examples [00:47, 841.40 examples/s]41360 examples [00:48, 858.11 examples/s]41448 examples [00:48, 863.51 examples/s]41535 examples [00:48, 850.76 examples/s]41624 examples [00:48, 860.69 examples/s]41711 examples [00:48, 856.16 examples/s]41800 examples [00:48, 863.54 examples/s]41888 examples [00:48, 866.78 examples/s]41976 examples [00:48, 869.44 examples/s]42063 examples [00:48, 839.07 examples/s]42148 examples [00:48, 841.25 examples/s]42238 examples [00:49, 856.20 examples/s]42324 examples [00:49, 855.45 examples/s]42412 examples [00:49, 862.13 examples/s]42501 examples [00:49, 868.62 examples/s]42589 examples [00:49, 870.44 examples/s]42681 examples [00:49, 882.52 examples/s]42770 examples [00:49, 880.48 examples/s]42859 examples [00:49, 860.20 examples/s]42948 examples [00:49, 868.13 examples/s]43035 examples [00:49, 849.45 examples/s]43121 examples [00:50, 846.23 examples/s]43215 examples [00:50, 870.22 examples/s]43305 examples [00:50, 878.59 examples/s]43395 examples [00:50, 883.92 examples/s]43484 examples [00:50, 876.40 examples/s]43576 examples [00:50, 887.19 examples/s]43665 examples [00:50, 878.10 examples/s]43753 examples [00:50, 875.13 examples/s]43842 examples [00:50, 878.35 examples/s]43930 examples [00:50, 872.45 examples/s]44018 examples [00:51, 874.20 examples/s]44106 examples [00:51, 873.39 examples/s]44194 examples [00:51, 872.72 examples/s]44282 examples [00:51, 852.22 examples/s]44368 examples [00:51, 842.36 examples/s]44456 examples [00:51, 851.82 examples/s]44543 examples [00:51, 854.56 examples/s]44629 examples [00:51, 849.87 examples/s]44716 examples [00:51, 853.06 examples/s]44804 examples [00:52, 859.09 examples/s]44895 examples [00:52, 871.11 examples/s]44985 examples [00:52, 879.26 examples/s]45077 examples [00:52, 890.37 examples/s]45167 examples [00:52, 888.45 examples/s]45257 examples [00:52, 891.04 examples/s]45347 examples [00:52, 871.69 examples/s]45436 examples [00:52, 876.63 examples/s]45526 examples [00:52, 881.46 examples/s]45615 examples [00:52, 881.21 examples/s]45704 examples [00:53, 876.53 examples/s]45792 examples [00:53, 856.26 examples/s]45882 examples [00:53, 866.33 examples/s]45976 examples [00:53, 884.90 examples/s]46069 examples [00:53, 897.71 examples/s]46159 examples [00:53, 892.61 examples/s]46249 examples [00:53, 879.67 examples/s]46341 examples [00:53, 891.08 examples/s]46431 examples [00:53, 887.78 examples/s]46520 examples [00:53, 888.07 examples/s]46609 examples [00:54, 871.27 examples/s]46702 examples [00:54, 885.97 examples/s]46791 examples [00:54, 886.09 examples/s]46882 examples [00:54, 890.55 examples/s]46972 examples [00:54, 889.86 examples/s]47062 examples [00:54, 886.15 examples/s]47151 examples [00:54, 884.22 examples/s]47240 examples [00:54, 871.28 examples/s]47332 examples [00:54, 883.59 examples/s]47425 examples [00:54, 896.01 examples/s]47515 examples [00:55, 894.78 examples/s]47605 examples [00:55, 893.98 examples/s]47698 examples [00:55, 903.66 examples/s]47789 examples [00:55, 896.92 examples/s]47884 examples [00:55, 909.59 examples/s]47976 examples [00:55, 884.34 examples/s]48065 examples [00:55, 878.28 examples/s]48153 examples [00:55, 863.35 examples/s]48244 examples [00:55, 874.96 examples/s]48335 examples [00:55, 884.48 examples/s]48427 examples [00:56, 893.90 examples/s]48517 examples [00:56, 862.63 examples/s]48604 examples [00:56, 864.16 examples/s]48695 examples [00:56, 875.01 examples/s]48783 examples [00:56, 872.50 examples/s]48871 examples [00:56, 870.18 examples/s]48959 examples [00:56, 852.15 examples/s]49049 examples [00:56, 863.49 examples/s]49136 examples [00:56, 865.37 examples/s]49223 examples [00:57, 853.91 examples/s]49309 examples [00:57, 845.07 examples/s]49401 examples [00:57, 865.01 examples/s]49493 examples [00:57, 878.38 examples/s]49585 examples [00:57, 888.96 examples/s]49679 examples [00:57, 902.78 examples/s]49770 examples [00:57, 894.01 examples/s]49860 examples [00:57, 892.95 examples/s]49951 examples [00:57, 895.58 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5668/50000 [00:00<00:00, 56679.38 examples/s] 34%|███▎      | 16773/50000 [00:00<00:00, 66437.65 examples/s] 55%|█████▌    | 27729/50000 [00:00<00:00, 75327.84 examples/s] 78%|███████▊  | 39009/50000 [00:00<00:00, 83665.56 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.34 examples/s]159 examples [00:00, 746.54 examples/s]249 examples [00:00, 785.60 examples/s]334 examples [00:00, 803.79 examples/s]424 examples [00:00, 827.66 examples/s]510 examples [00:00, 836.10 examples/s]600 examples [00:00, 853.14 examples/s]681 examples [00:00, 834.04 examples/s]761 examples [00:00, 655.58 examples/s]854 examples [00:01, 718.44 examples/s]946 examples [00:01, 766.90 examples/s]1029 examples [00:01, 784.20 examples/s]1120 examples [00:01, 813.29 examples/s]1212 examples [00:01, 841.56 examples/s]1304 examples [00:01, 862.42 examples/s]1397 examples [00:01, 879.86 examples/s]1488 examples [00:01, 886.10 examples/s]1578 examples [00:01, 883.49 examples/s]1670 examples [00:01, 893.65 examples/s]1760 examples [00:02, 889.14 examples/s]1852 examples [00:02, 896.43 examples/s]1942 examples [00:02, 875.10 examples/s]2030 examples [00:02, 832.13 examples/s]2114 examples [00:02, 793.05 examples/s]2201 examples [00:02, 812.21 examples/s]2288 examples [00:02, 826.68 examples/s]2373 examples [00:02, 832.84 examples/s]2457 examples [00:02, 818.81 examples/s]2547 examples [00:03, 841.21 examples/s]2638 examples [00:03, 858.72 examples/s]2726 examples [00:03, 862.81 examples/s]2813 examples [00:03, 861.49 examples/s]2900 examples [00:03, 859.86 examples/s]2987 examples [00:03, 846.22 examples/s]3078 examples [00:03, 862.22 examples/s]3165 examples [00:03, 842.89 examples/s]3250 examples [00:03, 826.94 examples/s]3340 examples [00:03, 845.93 examples/s]3431 examples [00:04, 863.09 examples/s]3520 examples [00:04, 870.63 examples/s]3608 examples [00:04, 848.02 examples/s]3698 examples [00:04, 861.83 examples/s]3787 examples [00:04, 869.26 examples/s]3875 examples [00:04, 866.73 examples/s]3968 examples [00:04, 884.44 examples/s]4057 examples [00:04, 868.88 examples/s]4149 examples [00:04, 882.57 examples/s]4240 examples [00:05, 888.91 examples/s]4330 examples [00:05, 880.41 examples/s]4422 examples [00:05, 889.25 examples/s]4512 examples [00:05, 878.84 examples/s]4600 examples [00:05, 854.41 examples/s]4688 examples [00:05, 861.90 examples/s]4775 examples [00:05, 811.00 examples/s]4864 examples [00:05, 833.15 examples/s]4948 examples [00:05, 811.70 examples/s]5037 examples [00:05, 832.75 examples/s]5129 examples [00:06, 856.05 examples/s]5217 examples [00:06, 861.02 examples/s]5306 examples [00:06, 867.31 examples/s]5394 examples [00:06, 864.58 examples/s]5483 examples [00:06, 870.99 examples/s]5573 examples [00:06, 878.51 examples/s]5664 examples [00:06, 885.93 examples/s]5755 examples [00:06, 892.31 examples/s]5845 examples [00:06, 891.94 examples/s]5935 examples [00:06, 874.98 examples/s]6027 examples [00:07, 885.83 examples/s]6116 examples [00:07, 870.30 examples/s]6204 examples [00:07, 863.04 examples/s]6291 examples [00:07, 851.66 examples/s]6379 examples [00:07, 858.89 examples/s]6471 examples [00:07, 874.41 examples/s]6561 examples [00:07, 879.45 examples/s]6650 examples [00:07, 873.07 examples/s]6738 examples [00:07, 872.51 examples/s]6828 examples [00:07, 878.45 examples/s]6919 examples [00:08, 887.24 examples/s]7008 examples [00:08, 886.98 examples/s]7097 examples [00:08, 885.56 examples/s]7186 examples [00:08, 879.55 examples/s]7274 examples [00:08, 839.40 examples/s]7367 examples [00:08, 863.86 examples/s]7457 examples [00:08, 874.07 examples/s]7547 examples [00:08, 881.07 examples/s]7636 examples [00:08, 877.91 examples/s]7724 examples [00:09, 870.57 examples/s]7812 examples [00:09, 855.33 examples/s]7904 examples [00:09, 871.47 examples/s]7992 examples [00:09, 871.08 examples/s]8080 examples [00:09, 868.61 examples/s]8167 examples [00:09, 863.73 examples/s]8254 examples [00:09, 863.55 examples/s]8342 examples [00:09, 866.38 examples/s]8431 examples [00:09, 870.73 examples/s]8519 examples [00:09, 850.14 examples/s]8605 examples [00:10, 828.49 examples/s]8693 examples [00:10, 843.21 examples/s]8780 examples [00:10, 850.98 examples/s]8866 examples [00:10, 844.07 examples/s]8952 examples [00:10, 848.59 examples/s]9037 examples [00:10, 838.57 examples/s]9124 examples [00:10, 845.70 examples/s]9214 examples [00:10, 860.09 examples/s]9302 examples [00:10, 863.10 examples/s]9394 examples [00:10, 877.68 examples/s]9483 examples [00:11, 880.54 examples/s]9575 examples [00:11, 891.15 examples/s]9665 examples [00:11, 876.12 examples/s]9753 examples [00:11, 869.37 examples/s]9841 examples [00:11, 872.38 examples/s]9929 examples [00:11, 868.01 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]100%|█████████▉| 9995/10000 [00:00<00:00, 99945.81 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3U13ZA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete3U13ZA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84af6fd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84af6fd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84af6fd950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f851598bac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8434bc2e80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f84af6fd950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f84af6fd950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f84af6fd950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8434bc2e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8495abc080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f8430161268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f8430161268> 

  function with postional parmater data_info <function split_train_valid at 0x7f8430161268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f8430161378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f8430161378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f8430161378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=0d83f5c813a35533d5c4e198df5e04e7a2acef10162b6afbabfa165d47fcbab4
  Stored in directory: /tmp/pip-ephem-wheel-cache-fgi2i51v/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:35:05, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:44:42, 14.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:46:32, 20.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:15:02, 29.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:45:38, 41.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.91M/862M [00:01<4:00:32, 59.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:47:40, 84.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:56:49, 120kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:01<1:21:28, 172kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<56:46, 245kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<39:42, 349kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:02<27:41, 497kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<19:26, 706kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.3M/862M [00:02<13:37, 1.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.9M/862M [00:02<09:34, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<06:52, 1.96MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<06:42, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<06:40, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<05:09, 2.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<06:04, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:47, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:25, 3.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<05:55, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<05:34, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<04:11, 3.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<05:56, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<06:52, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:30, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:10<04:00, 3.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<1:36:35, 136kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<1:08:41, 192kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<48:21, 272kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:12<33:54, 386kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<1:16:27, 171kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<54:48, 239kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:14<38:37, 338kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<30:02, 434kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<22:21, 583kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<15:57, 815kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<14:12, 913kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<12:36, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<09:22, 1.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:18<06:44, 1.92MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<10:26, 1.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<08:37, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<06:21, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<07:26, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<06:31, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:53, 2.62MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<06:25, 1.98MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<05:48, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<04:20, 2.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:03, 2.10MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:32, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:11, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:54, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:44, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:21, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:20, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:01, 3.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:15, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:40, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:16, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<03:57, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:35, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:28, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:03, 2.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<03:42, 3.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<08:03, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:55, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:06, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:25, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:47, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:19, 2.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:52, 2.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:21, 2.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:00, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:40, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<03:54, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<04:53, 2.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<03:45, 3.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<02:45, 4.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<47:02, 254kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<35:22, 337kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<25:20, 470kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<19:35, 606kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<14:55, 794kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<10:44, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<10:14, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<09:37, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:15, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:11, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:35, 804kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<11:26, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<08:17, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:31, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:09, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<05:18, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:26, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:41, 2.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:13, 2.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:40, 2.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:09, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:54, 2.94MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:25, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:09, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:48, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<03:30, 3.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:30, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:07, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:16, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<06:19, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<06:46, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:13, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:48, 2.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:46, 1.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:37, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:51, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:01, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:21, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:01, 2.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:26, 2.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:57, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:43, 2.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:13, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:46, 2.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:37, 3.04MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:07, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:50, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:34, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:20, 3.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<09:43, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<09:03, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:51, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:17<04:55, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:07, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<08:12, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:57, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:18, 2.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<45:40, 236kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<33:04, 325kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<23:22, 459kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<18:49, 568kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<14:16, 749kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:15, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<09:39, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:51, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:45, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:37, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:12, 2.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:24, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:40, 2.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:01, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:35, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:27, 3.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:52, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:28, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:23, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:45, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:23, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:19, 3.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:43, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:20, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:17, 3.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:48, 2.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:27, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:15, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:08, 3.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:32, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:54, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<03:38, 2.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:52, 2.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:25, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:18, 2.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:39, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:16, 2.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:14, 3.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:34, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:12, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:08, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:28, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:08, 3.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:25, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:07, 2.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:05, 3.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:24, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:03, 2.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:05, 3.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:03, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:04, 3.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:23, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:02, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:03, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:21, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:59, 3.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:17, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:56, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:59, 3.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:58, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:57, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<03:52, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:56, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<04:48, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:46, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:48, 3.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:44, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:15, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:12, 2.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:17, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<03:53, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<02:56, 2.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:07, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:41, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:40, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:39, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<08:59, 969kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:10, 1.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:39, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:44, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:27, 1.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:29, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:01, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:01, 2.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:44, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:49, 3.00MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:57, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:38, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:45, 3.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:54, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:35, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:43, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:51, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:32, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<02:41, 3.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:49, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:31, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<02:39, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:28, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:37, 3.09MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<03:27, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:36, 3.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:35, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:41, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:13, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:18, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:23, 3.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<19:31, 404kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<14:28, 544kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:18, 762kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<09:00, 867kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:54, 988kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:56, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<04:13, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<58:13, 133kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<41:31, 186kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<29:10, 265kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<22:07, 347kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<16:15, 472kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<11:30, 664kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<09:49, 775kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:38, 995kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:29, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:37, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:42, 1.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:28, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:11, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:28, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:38, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:18, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:30, 2.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:27, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:55, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:07, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:20, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:06, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:21, 3.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:19, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:03, 2.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:17, 3.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:16, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:01, 2.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:17, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:16, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:01, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:17, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:15, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:00, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:16, 3.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:13, 2.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:40, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:53, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:06, 3.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<1:07:08, 102kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<47:40, 144kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<33:25, 204kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<24:51, 273kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<18:47, 361kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<13:26, 504kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<09:24, 715kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<31:16, 215kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<22:33, 298kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<15:54, 420kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<12:38, 526kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:19<09:31, 698kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:46, 976kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<06:17, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:04, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:42, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<04:07, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:33, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:36, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:21, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:14, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:04, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:28, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<01:57, 3.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:57, 1.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:49, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:54, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:22, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:30, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:12, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:52, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:09, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:56, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:41, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:00, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:50, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:36, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:58, 3.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:47, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:33, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<01:54, 3.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:32, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<01:55, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:42, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:29, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<01:52, 3.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:40, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:21, 2.43MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:53, 3.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<01:22, 4.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<23:46, 239kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<17:12, 330kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<12:08, 466kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<09:46, 575kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<07:24, 757kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<05:18, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:00, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:04, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:57, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:22, 1.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:55, 1.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:10, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:47, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:30, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:53, 2.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:34, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:21, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:46, 2.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:29, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:16, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:43, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<02:25, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:13, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<01:40, 3.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:23, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:11, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:39, 3.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:09, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:38, 3.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:18, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:41, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:32, 3.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<32:35, 151kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<23:18, 212kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<16:22, 300kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<11:25, 426kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<26:03, 187kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<18:43, 260kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<13:08, 368kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<09:11, 522kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<1:03:24, 75.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<44:49, 107kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<31:19, 152kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<22:52, 207kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<16:58, 278kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<12:03, 391kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<08:25, 554kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<08:47, 531kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<06:37, 702kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:44, 978kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<04:21, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:59, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:59, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:08, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:33, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:56, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:10, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:33, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:42, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:05, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:30, 2.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:19, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:47, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:02, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:25, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:08, 2.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:35, 2.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:06, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:54, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:26, 2.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:59, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:48, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:22, 3.03MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:55, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:46, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:20, 3.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:53, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:43, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:18, 3.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:50, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:42, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:17, 3.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:48, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:35, 2.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:14, 3.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<00:54, 4.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<16:07, 238kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<12:03, 318kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<08:36, 444kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<06:33, 575kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:58, 757kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:32, 1.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:19, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:04, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:19, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:38, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<13:00, 279kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<09:28, 383kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<06:40, 539kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<05:27, 653kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:33, 782kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:21, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<02:21, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<26:32, 132kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<18:53, 185kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<13:12, 262kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<09:57, 344kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<07:18, 468kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:09, 658kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:22, 769kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:24, 986kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:26, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:24, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:50, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:47, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:13, 2.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:52, 3.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<13:17, 237kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<09:57, 316kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<07:06, 442kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:23, 573kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<04:05, 753kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:54, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:39, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:29, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:53, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:46, 1.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:33, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:08, 2.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:27, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:19, 2.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<00:59, 2.88MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:21, 2.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:13, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<00:55, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:17, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:10, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:52, 3.09MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:14, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:08, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:51, 3.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:12, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:06, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:50, 3.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:48, 3.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:03, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:47, 3.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:06, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:16, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:59, 2.40MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:42, 3.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<04:06, 568kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:06, 749kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:12, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:03, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:39, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:21, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:07, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:49, 2.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<00:36, 3.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<08:25, 252kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:06, 347kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<04:16, 490kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:25, 601kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:48, 730kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:03, 989kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:25, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<15:21, 129kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<10:55, 182kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<07:35, 258kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<05:39, 339kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<04:08, 461kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<02:54, 649kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:26, 760kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:53, 977kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:21, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:20, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:07, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:48, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:01, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:48, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:44, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:32, 2.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:44, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:40, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:30, 3.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:42, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:47, 1.88MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:26, 3.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<02:00, 719kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:32, 929kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:05, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:04, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:53, 1.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:38, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:38, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:28, 2.66MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:41, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:32, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:30, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:22, 3.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:26, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:19, 3.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:36, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.91MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:23, 2.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:25, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:17, 3.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:59, 899kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:46, 1.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:33, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:33, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:28, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:26, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 3.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:27, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:23, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:12, 2.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:18, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:09, 3.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<03:31, 136kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<02:29, 190kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<01:39, 270kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:09, 354kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:50, 480kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:33, 674kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:26, 785kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:20, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:13, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.62MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<14:37:11,  7.60it/s]  0%|          | 777/400000 [00:00<10:13:06, 10.85it/s]  0%|          | 1533/400000 [00:00<7:08:37, 15.49it/s]  1%|          | 2223/400000 [00:00<4:59:48, 22.11it/s]  1%|          | 2971/400000 [00:00<3:29:44, 31.55it/s]  1%|          | 3676/400000 [00:00<2:26:50, 44.99it/s]  1%|          | 4444/400000 [00:00<1:42:50, 64.10it/s]  1%|▏         | 5235/400000 [00:00<1:12:05, 91.26it/s]  2%|▏         | 6034/400000 [00:00<50:36, 129.73it/s]   2%|▏         | 6782/400000 [00:01<35:37, 183.97it/s]  2%|▏         | 7546/400000 [00:01<25:08, 260.12it/s]  2%|▏         | 8301/400000 [00:01<17:49, 366.20it/s]  2%|▏         | 9091/400000 [00:01<12:42, 512.94it/s]  2%|▏         | 9875/400000 [00:01<09:07, 712.74it/s]  3%|▎         | 10641/400000 [00:01<06:37, 978.81it/s]  3%|▎         | 11405/400000 [00:01<04:54, 1317.94it/s]  3%|▎         | 12146/400000 [00:01<03:41, 1747.11it/s]  3%|▎         | 12894/400000 [00:01<02:50, 2268.57it/s]  3%|▎         | 13671/400000 [00:01<02:14, 2880.20it/s]  4%|▎         | 14429/400000 [00:02<01:48, 3538.00it/s]  4%|▍         | 15188/400000 [00:02<01:31, 4212.34it/s]  4%|▍         | 15943/400000 [00:02<01:19, 4840.89it/s]  4%|▍         | 16695/400000 [00:02<01:12, 5296.39it/s]  4%|▍         | 17463/400000 [00:02<01:05, 5839.66it/s]  5%|▍         | 18204/400000 [00:02<01:01, 6190.92it/s]  5%|▍         | 18944/400000 [00:02<00:58, 6509.14it/s]  5%|▍         | 19692/400000 [00:02<00:56, 6772.52it/s]  5%|▌         | 20477/400000 [00:02<00:53, 7061.44it/s]  5%|▌         | 21231/400000 [00:02<00:52, 7174.40it/s]  5%|▌         | 21983/400000 [00:03<00:51, 7271.94it/s]  6%|▌         | 22734/400000 [00:03<00:51, 7308.02it/s]  6%|▌         | 23482/400000 [00:03<00:53, 7075.01it/s]  6%|▌         | 24220/400000 [00:03<00:52, 7162.13it/s]  6%|▌         | 24946/400000 [00:03<00:52, 7146.72it/s]  6%|▋         | 25674/400000 [00:03<00:52, 7183.42it/s]  7%|▋         | 26431/400000 [00:03<00:51, 7293.35it/s]  7%|▋         | 27195/400000 [00:03<00:50, 7392.27it/s]  7%|▋         | 27963/400000 [00:03<00:49, 7471.30it/s]  7%|▋         | 28723/400000 [00:03<00:49, 7509.10it/s]  7%|▋         | 29476/400000 [00:04<00:50, 7387.74it/s]  8%|▊         | 30241/400000 [00:04<00:49, 7462.14it/s]  8%|▊         | 31011/400000 [00:04<00:49, 7528.41it/s]  8%|▊         | 31799/400000 [00:04<00:48, 7630.51it/s]  8%|▊         | 32569/400000 [00:04<00:48, 7650.48it/s]  8%|▊         | 33335/400000 [00:04<00:48, 7636.41it/s]  9%|▊         | 34100/400000 [00:04<00:48, 7493.94it/s]  9%|▊         | 34851/400000 [00:04<00:48, 7460.45it/s]  9%|▉         | 35598/400000 [00:04<00:48, 7456.21it/s]  9%|▉         | 36377/400000 [00:04<00:48, 7552.97it/s]  9%|▉         | 37157/400000 [00:05<00:47, 7623.44it/s]  9%|▉         | 37920/400000 [00:05<00:47, 7553.47it/s] 10%|▉         | 38676/400000 [00:05<00:49, 7329.89it/s] 10%|▉         | 39458/400000 [00:05<00:48, 7469.88it/s] 10%|█         | 40228/400000 [00:05<00:47, 7535.15it/s] 10%|█         | 40989/400000 [00:05<00:47, 7555.46it/s] 10%|█         | 41746/400000 [00:05<00:48, 7428.35it/s] 11%|█         | 42502/400000 [00:05<00:47, 7464.59it/s] 11%|█         | 43250/400000 [00:05<00:47, 7456.01it/s] 11%|█         | 44012/400000 [00:06<00:47, 7502.03it/s] 11%|█         | 44767/400000 [00:06<00:47, 7515.54it/s] 11%|█▏        | 45519/400000 [00:06<00:47, 7395.46it/s] 12%|█▏        | 46267/400000 [00:06<00:47, 7418.29it/s] 12%|█▏        | 47010/400000 [00:06<00:47, 7409.10it/s] 12%|█▏        | 47752/400000 [00:06<00:48, 7324.07it/s] 12%|█▏        | 48530/400000 [00:06<00:47, 7453.30it/s] 12%|█▏        | 49277/400000 [00:06<00:47, 7406.38it/s] 13%|█▎        | 50053/400000 [00:06<00:46, 7508.33it/s] 13%|█▎        | 50821/400000 [00:06<00:46, 7558.24it/s] 13%|█▎        | 51578/400000 [00:07<00:46, 7496.42it/s] 13%|█▎        | 52329/400000 [00:07<00:46, 7426.54it/s] 13%|█▎        | 53073/400000 [00:07<00:47, 7301.44it/s] 13%|█▎        | 53811/400000 [00:07<00:47, 7323.80it/s] 14%|█▎        | 54544/400000 [00:07<00:47, 7311.56it/s] 14%|█▍        | 55282/400000 [00:07<00:47, 7329.74it/s] 14%|█▍        | 56033/400000 [00:07<00:46, 7381.76it/s] 14%|█▍        | 56772/400000 [00:07<00:46, 7373.72it/s] 14%|█▍        | 57533/400000 [00:07<00:46, 7440.46it/s] 15%|█▍        | 58295/400000 [00:07<00:45, 7493.23it/s] 15%|█▍        | 59080/400000 [00:08<00:44, 7594.62it/s] 15%|█▍        | 59841/400000 [00:08<00:44, 7588.94it/s] 15%|█▌        | 60601/400000 [00:08<00:45, 7450.36it/s] 15%|█▌        | 61347/400000 [00:08<00:46, 7245.86it/s] 16%|█▌        | 62110/400000 [00:08<00:45, 7355.65it/s] 16%|█▌        | 62881/400000 [00:08<00:45, 7456.42it/s] 16%|█▌        | 63646/400000 [00:08<00:44, 7511.39it/s] 16%|█▌        | 64399/400000 [00:08<00:45, 7455.31it/s] 16%|█▋        | 65146/400000 [00:08<00:44, 7459.44it/s] 16%|█▋        | 65912/400000 [00:08<00:44, 7515.98it/s] 17%|█▋        | 66665/400000 [00:09<00:44, 7501.67it/s] 17%|█▋        | 67423/400000 [00:09<00:44, 7522.50it/s] 17%|█▋        | 68176/400000 [00:09<00:44, 7452.02it/s] 17%|█▋        | 68922/400000 [00:09<00:45, 7284.81it/s] 17%|█▋        | 69652/400000 [00:09<00:45, 7226.17it/s] 18%|█▊        | 70395/400000 [00:09<00:45, 7283.64it/s] 18%|█▊        | 71158/400000 [00:09<00:44, 7379.88it/s] 18%|█▊        | 71903/400000 [00:09<00:44, 7400.31it/s] 18%|█▊        | 72686/400000 [00:09<00:43, 7524.04it/s] 18%|█▊        | 73440/400000 [00:09<00:43, 7471.07it/s] 19%|█▊        | 74206/400000 [00:10<00:43, 7524.86it/s] 19%|█▊        | 74960/400000 [00:10<00:43, 7437.26it/s] 19%|█▉        | 75705/400000 [00:10<00:44, 7360.92it/s] 19%|█▉        | 76458/400000 [00:10<00:43, 7409.27it/s] 19%|█▉        | 77200/400000 [00:10<00:43, 7403.24it/s] 19%|█▉        | 77941/400000 [00:10<00:44, 7259.79it/s] 20%|█▉        | 78668/400000 [00:10<00:44, 7243.66it/s] 20%|█▉        | 79393/400000 [00:10<00:44, 7215.67it/s] 20%|██        | 80160/400000 [00:10<00:43, 7344.27it/s] 20%|██        | 80928/400000 [00:10<00:42, 7439.65it/s] 20%|██        | 81679/400000 [00:11<00:42, 7459.76it/s] 21%|██        | 82426/400000 [00:11<00:42, 7448.80it/s] 21%|██        | 83172/400000 [00:11<00:42, 7421.78it/s] 21%|██        | 83915/400000 [00:11<00:42, 7415.31it/s] 21%|██        | 84669/400000 [00:11<00:42, 7452.26it/s] 21%|██▏       | 85415/400000 [00:11<00:42, 7365.87it/s] 22%|██▏       | 86152/400000 [00:11<00:42, 7341.54it/s] 22%|██▏       | 86887/400000 [00:11<00:42, 7330.13it/s] 22%|██▏       | 87651/400000 [00:11<00:42, 7418.36it/s] 22%|██▏       | 88394/400000 [00:12<00:42, 7263.30it/s] 22%|██▏       | 89163/400000 [00:12<00:42, 7385.36it/s] 22%|██▏       | 89906/400000 [00:12<00:41, 7397.50it/s] 23%|██▎       | 90647/400000 [00:12<00:42, 7212.16it/s] 23%|██▎       | 91370/400000 [00:12<00:42, 7200.92it/s] 23%|██▎       | 92105/400000 [00:12<00:42, 7243.72it/s] 23%|██▎       | 92860/400000 [00:12<00:41, 7331.39it/s] 23%|██▎       | 93594/400000 [00:12<00:41, 7328.28it/s] 24%|██▎       | 94340/400000 [00:12<00:41, 7366.67it/s] 24%|██▍       | 95090/400000 [00:12<00:41, 7404.27it/s] 24%|██▍       | 95831/400000 [00:13<00:41, 7396.50it/s] 24%|██▍       | 96589/400000 [00:13<00:40, 7448.42it/s] 24%|██▍       | 97335/400000 [00:13<00:40, 7417.60it/s] 25%|██▍       | 98077/400000 [00:13<00:40, 7379.00it/s] 25%|██▍       | 98816/400000 [00:13<00:41, 7254.80it/s] 25%|██▍       | 99549/400000 [00:13<00:41, 7276.37it/s] 25%|██▌       | 100282/400000 [00:13<00:41, 7291.57it/s] 25%|██▌       | 101024/400000 [00:13<00:40, 7327.47it/s] 25%|██▌       | 101758/400000 [00:13<00:40, 7291.44it/s] 26%|██▌       | 102488/400000 [00:13<00:40, 7257.27it/s] 26%|██▌       | 103214/400000 [00:14<00:41, 7180.81it/s] 26%|██▌       | 103938/400000 [00:14<00:41, 7197.48it/s] 26%|██▌       | 104678/400000 [00:14<00:40, 7256.99it/s] 26%|██▋       | 105414/400000 [00:14<00:40, 7285.98it/s] 27%|██▋       | 106187/400000 [00:14<00:39, 7413.02it/s] 27%|██▋       | 106948/400000 [00:14<00:39, 7468.76it/s] 27%|██▋       | 107696/400000 [00:14<00:39, 7416.62it/s] 27%|██▋       | 108442/400000 [00:14<00:39, 7428.71it/s] 27%|██▋       | 109193/400000 [00:14<00:39, 7451.44it/s] 27%|██▋       | 109942/400000 [00:14<00:38, 7461.25it/s] 28%|██▊       | 110695/400000 [00:15<00:38, 7480.20it/s] 28%|██▊       | 111451/400000 [00:15<00:38, 7501.24it/s] 28%|██▊       | 112202/400000 [00:15<00:38, 7475.27it/s] 28%|██▊       | 112950/400000 [00:15<00:38, 7375.00it/s] 28%|██▊       | 113688/400000 [00:15<00:38, 7362.68it/s] 29%|██▊       | 114425/400000 [00:15<00:39, 7166.74it/s] 29%|██▉       | 115157/400000 [00:15<00:39, 7210.71it/s] 29%|██▉       | 115896/400000 [00:15<00:39, 7260.06it/s] 29%|██▉       | 116625/400000 [00:15<00:38, 7268.72it/s] 29%|██▉       | 117388/400000 [00:15<00:38, 7371.37it/s] 30%|██▉       | 118157/400000 [00:16<00:37, 7462.60it/s] 30%|██▉       | 118905/400000 [00:16<00:37, 7467.12it/s] 30%|██▉       | 119667/400000 [00:16<00:37, 7510.00it/s] 30%|███       | 120424/400000 [00:16<00:37, 7525.66it/s] 30%|███       | 121177/400000 [00:16<00:37, 7489.93it/s] 30%|███       | 121927/400000 [00:16<00:37, 7441.43it/s] 31%|███       | 122672/400000 [00:16<00:37, 7435.68it/s] 31%|███       | 123427/400000 [00:16<00:37, 7468.21it/s] 31%|███       | 124174/400000 [00:16<00:37, 7435.80it/s] 31%|███       | 124918/400000 [00:16<00:37, 7417.95it/s] 31%|███▏      | 125660/400000 [00:17<00:37, 7332.11it/s] 32%|███▏      | 126413/400000 [00:17<00:37, 7389.30it/s] 32%|███▏      | 127175/400000 [00:17<00:36, 7456.52it/s] 32%|███▏      | 127922/400000 [00:17<00:37, 7341.25it/s] 32%|███▏      | 128664/400000 [00:17<00:36, 7362.01it/s] 32%|███▏      | 129401/400000 [00:17<00:37, 7223.03it/s] 33%|███▎      | 130164/400000 [00:17<00:36, 7339.84it/s] 33%|███▎      | 130941/400000 [00:17<00:36, 7461.49it/s] 33%|███▎      | 131689/400000 [00:17<00:36, 7435.32it/s] 33%|███▎      | 132435/400000 [00:17<00:35, 7442.53it/s] 33%|███▎      | 133184/400000 [00:18<00:35, 7456.02it/s] 33%|███▎      | 133931/400000 [00:18<00:36, 7295.03it/s] 34%|███▎      | 134684/400000 [00:18<00:36, 7363.19it/s] 34%|███▍      | 135422/400000 [00:18<00:35, 7365.44it/s] 34%|███▍      | 136160/400000 [00:18<00:35, 7349.62it/s] 34%|███▍      | 136918/400000 [00:18<00:35, 7416.20it/s] 34%|███▍      | 137672/400000 [00:18<00:35, 7452.17it/s] 35%|███▍      | 138434/400000 [00:18<00:34, 7501.24it/s] 35%|███▍      | 139190/400000 [00:18<00:34, 7517.88it/s] 35%|███▍      | 139964/400000 [00:18<00:34, 7582.51it/s] 35%|███▌      | 140735/400000 [00:19<00:34, 7617.93it/s] 35%|███▌      | 141499/400000 [00:19<00:33, 7622.65it/s] 36%|███▌      | 142262/400000 [00:19<00:34, 7525.77it/s] 36%|███▌      | 143015/400000 [00:19<00:34, 7516.31it/s] 36%|███▌      | 143767/400000 [00:19<00:34, 7448.94it/s] 36%|███▌      | 144568/400000 [00:19<00:33, 7604.13it/s] 36%|███▋      | 145359/400000 [00:19<00:33, 7691.33it/s] 37%|███▋      | 146130/400000 [00:19<00:33, 7661.43it/s] 37%|███▋      | 146897/400000 [00:19<00:33, 7595.56it/s] 37%|███▋      | 147658/400000 [00:20<00:33, 7490.54it/s] 37%|███▋      | 148408/400000 [00:20<00:33, 7449.29it/s] 37%|███▋      | 149154/400000 [00:20<00:33, 7442.68it/s] 37%|███▋      | 149936/400000 [00:20<00:33, 7550.86it/s] 38%|███▊      | 150692/400000 [00:20<00:33, 7466.73it/s] 38%|███▊      | 151451/400000 [00:20<00:33, 7501.56it/s] 38%|███▊      | 152219/400000 [00:20<00:32, 7550.75it/s] 38%|███▊      | 152992/400000 [00:20<00:32, 7602.52it/s] 38%|███▊      | 153770/400000 [00:20<00:32, 7653.63it/s] 39%|███▊      | 154536/400000 [00:20<00:32, 7594.81it/s] 39%|███▉      | 155297/400000 [00:21<00:32, 7598.34it/s] 39%|███▉      | 156071/400000 [00:21<00:31, 7639.55it/s] 39%|███▉      | 156836/400000 [00:21<00:32, 7545.89it/s] 39%|███▉      | 157592/400000 [00:21<00:32, 7513.36it/s] 40%|███▉      | 158352/400000 [00:21<00:32, 7536.14it/s] 40%|███▉      | 159106/400000 [00:21<00:32, 7526.96it/s] 40%|███▉      | 159859/400000 [00:21<00:32, 7412.98it/s] 40%|████      | 160614/400000 [00:21<00:32, 7452.64it/s] 40%|████      | 161368/400000 [00:21<00:31, 7478.43it/s] 41%|████      | 162117/400000 [00:21<00:31, 7461.80it/s] 41%|████      | 162896/400000 [00:22<00:31, 7555.52it/s] 41%|████      | 163652/400000 [00:22<00:31, 7546.47it/s] 41%|████      | 164427/400000 [00:22<00:30, 7604.90it/s] 41%|████▏     | 165188/400000 [00:22<00:31, 7551.15it/s] 41%|████▏     | 165944/400000 [00:22<00:31, 7470.22it/s] 42%|████▏     | 166692/400000 [00:22<00:31, 7468.08it/s] 42%|████▏     | 167440/400000 [00:22<00:31, 7442.12it/s] 42%|████▏     | 168185/400000 [00:22<00:31, 7327.46it/s] 42%|████▏     | 168944/400000 [00:22<00:31, 7400.61it/s] 42%|████▏     | 169692/400000 [00:22<00:31, 7421.71it/s] 43%|████▎     | 170481/400000 [00:23<00:30, 7554.55it/s] 43%|████▎     | 171238/400000 [00:23<00:30, 7537.84it/s] 43%|████▎     | 171993/400000 [00:23<00:30, 7507.69it/s] 43%|████▎     | 172750/400000 [00:23<00:30, 7525.16it/s] 43%|████▎     | 173520/400000 [00:23<00:29, 7574.49it/s] 44%|████▎     | 174278/400000 [00:23<00:29, 7574.75it/s] 44%|████▍     | 175046/400000 [00:23<00:29, 7604.18it/s] 44%|████▍     | 175838/400000 [00:23<00:29, 7694.00it/s] 44%|████▍     | 176608/400000 [00:23<00:29, 7580.75it/s] 44%|████▍     | 177367/400000 [00:23<00:29, 7466.60it/s] 45%|████▍     | 178116/400000 [00:24<00:29, 7473.09it/s] 45%|████▍     | 178887/400000 [00:24<00:29, 7539.48it/s] 45%|████▍     | 179642/400000 [00:24<00:29, 7484.75it/s] 45%|████▌     | 180391/400000 [00:24<00:29, 7466.51it/s] 45%|████▌     | 181138/400000 [00:24<00:29, 7392.65it/s] 45%|████▌     | 181889/400000 [00:24<00:29, 7427.01it/s] 46%|████▌     | 182646/400000 [00:24<00:29, 7469.00it/s] 46%|████▌     | 183394/400000 [00:24<00:29, 7459.60it/s] 46%|████▌     | 184163/400000 [00:24<00:28, 7526.37it/s] 46%|████▌     | 184922/400000 [00:24<00:28, 7542.76it/s] 46%|████▋     | 185691/400000 [00:25<00:28, 7583.65it/s] 47%|████▋     | 186450/400000 [00:25<00:28, 7575.05it/s] 47%|████▋     | 187225/400000 [00:25<00:27, 7625.55it/s] 47%|████▋     | 188012/400000 [00:25<00:27, 7693.98it/s] 47%|████▋     | 188789/400000 [00:25<00:27, 7714.27it/s] 47%|████▋     | 189561/400000 [00:25<00:27, 7672.37it/s] 48%|████▊     | 190329/400000 [00:25<00:27, 7602.11it/s] 48%|████▊     | 191090/400000 [00:25<00:27, 7592.93it/s] 48%|████▊     | 191858/400000 [00:25<00:27, 7618.39it/s] 48%|████▊     | 192620/400000 [00:25<00:27, 7577.41it/s] 48%|████▊     | 193380/400000 [00:26<00:27, 7582.85it/s] 49%|████▊     | 194139/400000 [00:26<00:27, 7536.99it/s] 49%|████▊     | 194902/400000 [00:26<00:27, 7564.65it/s] 49%|████▉     | 195684/400000 [00:26<00:26, 7638.07it/s] 49%|████▉     | 196449/400000 [00:26<00:27, 7495.79it/s] 49%|████▉     | 197200/400000 [00:26<00:27, 7492.35it/s] 49%|████▉     | 197950/400000 [00:26<00:27, 7417.10it/s] 50%|████▉     | 198706/400000 [00:26<00:26, 7457.61it/s] 50%|████▉     | 199475/400000 [00:26<00:26, 7524.49it/s] 50%|█████     | 200233/400000 [00:26<00:26, 7538.27it/s] 50%|█████     | 200988/400000 [00:27<00:26, 7515.92it/s] 50%|█████     | 201753/400000 [00:27<00:26, 7553.35it/s] 51%|█████     | 202509/400000 [00:27<00:26, 7352.69it/s] 51%|█████     | 203248/400000 [00:27<00:26, 7363.17it/s] 51%|█████     | 203986/400000 [00:27<00:26, 7327.16it/s] 51%|█████     | 204720/400000 [00:27<00:26, 7283.98it/s] 51%|█████▏    | 205466/400000 [00:27<00:26, 7333.29it/s] 52%|█████▏    | 206232/400000 [00:27<00:26, 7427.12it/s] 52%|█████▏    | 206995/400000 [00:27<00:25, 7486.43it/s] 52%|█████▏    | 207745/400000 [00:28<00:26, 7348.24it/s] 52%|█████▏    | 208481/400000 [00:28<00:26, 7248.63it/s] 52%|█████▏    | 209250/400000 [00:28<00:25, 7375.05it/s] 53%|█████▎    | 210042/400000 [00:28<00:25, 7529.05it/s] 53%|█████▎    | 210797/400000 [00:28<00:25, 7487.96it/s] 53%|█████▎    | 211585/400000 [00:28<00:24, 7600.99it/s] 53%|█████▎    | 212347/400000 [00:28<00:24, 7593.15it/s] 53%|█████▎    | 213108/400000 [00:28<00:24, 7586.02it/s] 53%|█████▎    | 213871/400000 [00:28<00:24, 7597.10it/s] 54%|█████▎    | 214641/400000 [00:28<00:24, 7627.09it/s] 54%|█████▍    | 215405/400000 [00:29<00:24, 7622.37it/s] 54%|█████▍    | 216172/400000 [00:29<00:24, 7635.41it/s] 54%|█████▍    | 216949/400000 [00:29<00:23, 7672.95it/s] 54%|█████▍    | 217717/400000 [00:29<00:24, 7575.62it/s] 55%|█████▍    | 218484/400000 [00:29<00:23, 7602.65it/s] 55%|█████▍    | 219245/400000 [00:29<00:23, 7575.15it/s] 55%|█████▌    | 220005/400000 [00:29<00:23, 7578.39it/s] 55%|█████▌    | 220792/400000 [00:29<00:23, 7661.42it/s] 55%|█████▌    | 221589/400000 [00:29<00:23, 7748.47it/s] 56%|█████▌    | 222369/400000 [00:29<00:22, 7763.32it/s] 56%|█████▌    | 223146/400000 [00:30<00:22, 7752.40it/s] 56%|█████▌    | 223922/400000 [00:30<00:23, 7645.16it/s] 56%|█████▌    | 224702/400000 [00:30<00:22, 7689.58it/s] 56%|█████▋    | 225472/400000 [00:30<00:22, 7673.45it/s] 57%|█████▋    | 226248/400000 [00:30<00:22, 7697.16it/s] 57%|█████▋    | 227018/400000 [00:30<00:22, 7607.14it/s] 57%|█████▋    | 227780/400000 [00:30<00:22, 7495.99it/s] 57%|█████▋    | 228531/400000 [00:30<00:23, 7443.87it/s] 57%|█████▋    | 229280/400000 [00:30<00:22, 7456.83it/s] 58%|█████▊    | 230039/400000 [00:30<00:22, 7495.61it/s] 58%|█████▊    | 230792/400000 [00:31<00:22, 7503.32it/s] 58%|█████▊    | 231563/400000 [00:31<00:22, 7563.63it/s] 58%|█████▊    | 232320/400000 [00:31<00:22, 7486.47it/s] 58%|█████▊    | 233086/400000 [00:31<00:22, 7536.03it/s] 58%|█████▊    | 233858/400000 [00:31<00:21, 7589.57it/s] 59%|█████▊    | 234618/400000 [00:31<00:21, 7537.08it/s] 59%|█████▉    | 235373/400000 [00:31<00:21, 7494.82it/s] 59%|█████▉    | 236141/400000 [00:31<00:21, 7548.80it/s] 59%|█████▉    | 236897/400000 [00:31<00:21, 7479.52it/s] 59%|█████▉    | 237654/400000 [00:31<00:21, 7506.07it/s] 60%|█████▉    | 238432/400000 [00:32<00:21, 7584.13it/s] 60%|█████▉    | 239191/400000 [00:32<00:21, 7525.87it/s] 60%|█████▉    | 239944/400000 [00:32<00:21, 7513.66it/s] 60%|██████    | 240696/400000 [00:32<00:21, 7434.01it/s] 60%|██████    | 241454/400000 [00:32<00:21, 7474.45it/s] 61%|██████    | 242234/400000 [00:32<00:20, 7569.19it/s] 61%|██████    | 242992/400000 [00:32<00:20, 7508.92it/s] 61%|██████    | 243744/400000 [00:32<00:20, 7473.70it/s] 61%|██████    | 244492/400000 [00:32<00:20, 7464.22it/s] 61%|██████▏   | 245239/400000 [00:32<00:21, 7273.06it/s] 61%|██████▏   | 245969/400000 [00:33<00:21, 7279.92it/s] 62%|██████▏   | 246699/400000 [00:33<00:21, 7284.32it/s] 62%|██████▏   | 247462/400000 [00:33<00:20, 7384.24it/s] 62%|██████▏   | 248226/400000 [00:33<00:20, 7458.71it/s] 62%|██████▏   | 248975/400000 [00:33<00:20, 7467.30it/s] 62%|██████▏   | 249723/400000 [00:33<00:20, 7430.52it/s] 63%|██████▎   | 250467/400000 [00:33<00:20, 7401.93it/s] 63%|██████▎   | 251232/400000 [00:33<00:19, 7472.18it/s] 63%|██████▎   | 251980/400000 [00:33<00:19, 7444.17it/s] 63%|██████▎   | 252726/400000 [00:33<00:19, 7445.91it/s] 63%|██████▎   | 253485/400000 [00:34<00:19, 7486.05it/s] 64%|██████▎   | 254234/400000 [00:34<00:19, 7476.11it/s] 64%|██████▎   | 254992/400000 [00:34<00:19, 7506.50it/s] 64%|██████▍   | 255750/400000 [00:34<00:19, 7528.09it/s] 64%|██████▍   | 256507/400000 [00:34<00:19, 7540.03it/s] 64%|██████▍   | 257272/400000 [00:34<00:18, 7571.95it/s] 65%|██████▍   | 258030/400000 [00:34<00:19, 7423.70it/s] 65%|██████▍   | 258774/400000 [00:34<00:19, 7417.79it/s] 65%|██████▍   | 259527/400000 [00:34<00:18, 7450.45it/s] 65%|██████▌   | 260273/400000 [00:34<00:18, 7398.68it/s] 65%|██████▌   | 261042/400000 [00:35<00:18, 7482.32it/s] 65%|██████▌   | 261791/400000 [00:35<00:18, 7418.27it/s] 66%|██████▌   | 262534/400000 [00:35<00:18, 7377.14it/s] 66%|██████▌   | 263287/400000 [00:35<00:18, 7419.80it/s] 66%|██████▌   | 264065/400000 [00:35<00:18, 7522.23it/s] 66%|██████▌   | 264825/400000 [00:35<00:17, 7543.65it/s] 66%|██████▋   | 265580/400000 [00:35<00:17, 7520.17it/s] 67%|██████▋   | 266351/400000 [00:35<00:17, 7573.77it/s] 67%|██████▋   | 267109/400000 [00:35<00:17, 7573.98it/s] 67%|██████▋   | 267883/400000 [00:35<00:17, 7620.51it/s] 67%|██████▋   | 268646/400000 [00:36<00:17, 7563.36it/s] 67%|██████▋   | 269403/400000 [00:36<00:17, 7457.17it/s] 68%|██████▊   | 270166/400000 [00:36<00:17, 7505.99it/s] 68%|██████▊   | 270919/400000 [00:36<00:17, 7511.00it/s] 68%|██████▊   | 271680/400000 [00:36<00:17, 7538.49it/s] 68%|██████▊   | 272435/400000 [00:36<00:17, 7489.73it/s] 68%|██████▊   | 273185/400000 [00:36<00:17, 7380.25it/s] 68%|██████▊   | 273963/400000 [00:36<00:16, 7494.28it/s] 69%|██████▊   | 274714/400000 [00:36<00:16, 7406.00it/s] 69%|██████▉   | 275471/400000 [00:37<00:16, 7453.35it/s] 69%|██████▉   | 276244/400000 [00:37<00:16, 7533.24it/s] 69%|██████▉   | 276998/400000 [00:37<00:16, 7454.25it/s] 69%|██████▉   | 277745/400000 [00:37<00:17, 7176.85it/s] 70%|██████▉   | 278498/400000 [00:37<00:16, 7277.01it/s] 70%|██████▉   | 279246/400000 [00:37<00:16, 7336.46it/s] 70%|██████▉   | 279995/400000 [00:37<00:16, 7380.45it/s] 70%|███████   | 280756/400000 [00:37<00:16, 7445.55it/s] 70%|███████   | 281556/400000 [00:37<00:15, 7602.78it/s] 71%|███████   | 282329/400000 [00:37<00:15, 7640.24it/s] 71%|███████   | 283095/400000 [00:38<00:15, 7644.97it/s] 71%|███████   | 283861/400000 [00:38<00:15, 7562.51it/s] 71%|███████   | 284618/400000 [00:38<00:15, 7517.14it/s] 71%|███████▏  | 285381/400000 [00:38<00:15, 7548.45it/s] 72%|███████▏  | 286137/400000 [00:38<00:15, 7432.07it/s] 72%|███████▏  | 286910/400000 [00:38<00:15, 7518.30it/s] 72%|███████▏  | 287670/400000 [00:38<00:14, 7540.75it/s] 72%|███████▏  | 288437/400000 [00:38<00:14, 7576.99it/s] 72%|███████▏  | 289196/400000 [00:38<00:14, 7568.48it/s] 72%|███████▏  | 289954/400000 [00:38<00:14, 7530.56it/s] 73%|███████▎  | 290740/400000 [00:39<00:14, 7626.17it/s] 73%|███████▎  | 291504/400000 [00:39<00:14, 7536.72it/s] 73%|███████▎  | 292259/400000 [00:39<00:14, 7273.67it/s] 73%|███████▎  | 292989/400000 [00:39<00:14, 7252.12it/s] 73%|███████▎  | 293716/400000 [00:39<00:14, 7227.23it/s] 74%|███████▎  | 294440/400000 [00:39<00:15, 6976.96it/s] 74%|███████▍  | 295176/400000 [00:39<00:14, 7086.06it/s] 74%|███████▍  | 295912/400000 [00:39<00:14, 7164.17it/s] 74%|███████▍  | 296654/400000 [00:39<00:14, 7234.74it/s] 74%|███████▍  | 297414/400000 [00:39<00:13, 7338.57it/s] 75%|███████▍  | 298193/400000 [00:40<00:13, 7466.18it/s] 75%|███████▍  | 298950/400000 [00:40<00:13, 7496.31it/s] 75%|███████▍  | 299711/400000 [00:40<00:13, 7527.57it/s] 75%|███████▌  | 300465/400000 [00:40<00:13, 7461.33it/s] 75%|███████▌  | 301245/400000 [00:40<00:13, 7557.49it/s] 76%|███████▌  | 302002/400000 [00:40<00:12, 7549.43it/s] 76%|███████▌  | 302789/400000 [00:40<00:12, 7640.74it/s] 76%|███████▌  | 303554/400000 [00:40<00:12, 7621.17it/s] 76%|███████▌  | 304335/400000 [00:40<00:12, 7675.54it/s] 76%|███████▋  | 305103/400000 [00:40<00:12, 7634.14it/s] 76%|███████▋  | 305867/400000 [00:41<00:12, 7576.08it/s] 77%|███████▋  | 306625/400000 [00:41<00:12, 7321.58it/s] 77%|███████▋  | 307360/400000 [00:41<00:13, 6991.20it/s] 77%|███████▋  | 308064/400000 [00:41<00:13, 6935.06it/s] 77%|███████▋  | 308784/400000 [00:41<00:13, 7009.43it/s] 77%|███████▋  | 309522/400000 [00:41<00:12, 7114.99it/s] 78%|███████▊  | 310251/400000 [00:41<00:12, 7164.11it/s] 78%|███████▊  | 310969/400000 [00:41<00:12, 7168.17it/s] 78%|███████▊  | 311687/400000 [00:41<00:12, 7141.35it/s] 78%|███████▊  | 312413/400000 [00:42<00:12, 7175.60it/s] 78%|███████▊  | 313132/400000 [00:42<00:12, 7088.35it/s] 78%|███████▊  | 313842/400000 [00:42<00:12, 7076.12it/s] 79%|███████▊  | 314551/400000 [00:42<00:12, 7046.58it/s] 79%|███████▉  | 315270/400000 [00:42<00:11, 7087.81it/s] 79%|███████▉  | 316024/400000 [00:42<00:11, 7217.36it/s] 79%|███████▉  | 316780/400000 [00:42<00:11, 7316.51it/s] 79%|███████▉  | 317513/400000 [00:42<00:11, 7312.22it/s] 80%|███████▉  | 318252/400000 [00:42<00:11, 7331.55it/s] 80%|███████▉  | 319001/400000 [00:42<00:10, 7377.65it/s] 80%|███████▉  | 319740/400000 [00:43<00:10, 7381.31it/s] 80%|████████  | 320497/400000 [00:43<00:10, 7434.76it/s] 80%|████████  | 321251/400000 [00:43<00:10, 7465.39it/s] 80%|████████  | 321998/400000 [00:43<00:10, 7291.63it/s] 81%|████████  | 322761/400000 [00:43<00:10, 7388.04it/s] 81%|████████  | 323536/400000 [00:43<00:10, 7492.93it/s] 81%|████████  | 324301/400000 [00:43<00:10, 7538.78it/s] 81%|████████▏ | 325056/400000 [00:43<00:09, 7513.21it/s] 81%|████████▏ | 325808/400000 [00:43<00:09, 7443.41it/s] 82%|████████▏ | 326553/400000 [00:43<00:10, 7199.43it/s] 82%|████████▏ | 327284/400000 [00:44<00:10, 7230.06it/s] 82%|████████▏ | 328035/400000 [00:44<00:09, 7309.12it/s] 82%|████████▏ | 328768/400000 [00:44<00:09, 7295.03it/s] 82%|████████▏ | 329506/400000 [00:44<00:09, 7319.95it/s] 83%|████████▎ | 330239/400000 [00:44<00:09, 7230.92it/s] 83%|████████▎ | 330989/400000 [00:44<00:09, 7308.21it/s] 83%|████████▎ | 331721/400000 [00:44<00:09, 7306.77it/s] 83%|████████▎ | 332453/400000 [00:44<00:09, 7300.32it/s] 83%|████████▎ | 333202/400000 [00:44<00:09, 7354.34it/s] 83%|████████▎ | 333938/400000 [00:44<00:09, 7332.37it/s] 84%|████████▎ | 334672/400000 [00:45<00:08, 7300.09it/s] 84%|████████▍ | 335413/400000 [00:45<00:08, 7325.49it/s] 84%|████████▍ | 336194/400000 [00:45<00:08, 7461.94it/s] 84%|████████▍ | 336953/400000 [00:45<00:08, 7499.36it/s] 84%|████████▍ | 337708/400000 [00:45<00:08, 7513.12it/s] 85%|████████▍ | 338460/400000 [00:45<00:08, 7512.28it/s] 85%|████████▍ | 339212/400000 [00:45<00:08, 7434.70it/s] 85%|████████▍ | 339971/400000 [00:45<00:08, 7479.08it/s] 85%|████████▌ | 340749/400000 [00:45<00:07, 7566.68it/s] 85%|████████▌ | 341507/400000 [00:45<00:07, 7514.83it/s] 86%|████████▌ | 342259/400000 [00:46<00:07, 7320.52it/s] 86%|████████▌ | 342993/400000 [00:46<00:07, 7293.96it/s] 86%|████████▌ | 343771/400000 [00:46<00:07, 7432.10it/s] 86%|████████▌ | 344521/400000 [00:46<00:07, 7450.98it/s] 86%|████████▋ | 345269/400000 [00:46<00:07, 7459.20it/s] 87%|████████▋ | 346018/400000 [00:46<00:07, 7466.73it/s] 87%|████████▋ | 346790/400000 [00:46<00:07, 7539.84it/s] 87%|████████▋ | 347545/400000 [00:46<00:07, 7420.64it/s] 87%|████████▋ | 348316/400000 [00:46<00:06, 7502.74it/s] 87%|████████▋ | 349068/400000 [00:46<00:06, 7493.29it/s] 87%|████████▋ | 349825/400000 [00:47<00:06, 7515.96it/s] 88%|████████▊ | 350600/400000 [00:47<00:06, 7581.83it/s] 88%|████████▊ | 351359/400000 [00:47<00:06, 7553.56it/s] 88%|████████▊ | 352115/400000 [00:47<00:06, 7415.13it/s] 88%|████████▊ | 352858/400000 [00:47<00:06, 7352.92it/s] 88%|████████▊ | 353598/400000 [00:47<00:06, 7366.79it/s] 89%|████████▊ | 354384/400000 [00:47<00:06, 7506.15it/s] 89%|████████▉ | 355166/400000 [00:47<00:05, 7596.07it/s] 89%|████████▉ | 355927/400000 [00:47<00:05, 7496.07it/s] 89%|████████▉ | 356678/400000 [00:48<00:05, 7463.77it/s] 89%|████████▉ | 357452/400000 [00:48<00:05, 7542.51it/s] 90%|████████▉ | 358214/400000 [00:48<00:05, 7563.24it/s] 90%|████████▉ | 358971/400000 [00:48<00:05, 7557.05it/s] 90%|████████▉ | 359744/400000 [00:48<00:05, 7608.10it/s] 90%|█████████ | 360506/400000 [00:48<00:05, 7483.33it/s] 90%|█████████ | 361256/400000 [00:48<00:05, 7414.84it/s] 91%|█████████ | 362042/400000 [00:48<00:05, 7542.40it/s] 91%|█████████ | 362810/400000 [00:48<00:04, 7580.96it/s] 91%|█████████ | 363569/400000 [00:48<00:04, 7554.11it/s] 91%|█████████ | 364325/400000 [00:49<00:04, 7331.19it/s] 91%|█████████▏| 365090/400000 [00:49<00:04, 7423.62it/s] 91%|█████████▏| 365856/400000 [00:49<00:04, 7491.55it/s] 92%|█████████▏| 366623/400000 [00:49<00:04, 7544.07it/s] 92%|█████████▏| 367389/400000 [00:49<00:04, 7574.77it/s] 92%|█████████▏| 368148/400000 [00:49<00:04, 7466.89it/s] 92%|█████████▏| 368914/400000 [00:49<00:04, 7523.33it/s] 92%|█████████▏| 369707/400000 [00:49<00:03, 7638.81it/s] 93%|█████████▎| 370486/400000 [00:49<00:03, 7682.14it/s] 93%|█████████▎| 371255/400000 [00:49<00:03, 7578.86it/s] 93%|█████████▎| 372017/400000 [00:50<00:03, 7588.81it/s] 93%|█████████▎| 372777/400000 [00:50<00:03, 7561.98it/s] 93%|█████████▎| 373553/400000 [00:50<00:03, 7618.15it/s] 94%|█████████▎| 374316/400000 [00:50<00:03, 7561.49it/s] 94%|█████████▍| 375073/400000 [00:50<00:03, 7429.53it/s] 94%|█████████▍| 375817/400000 [00:50<00:03, 7399.09it/s] 94%|█████████▍| 376582/400000 [00:50<00:03, 7469.90it/s] 94%|█████████▍| 377330/400000 [00:50<00:03, 7449.75it/s] 95%|█████████▍| 378130/400000 [00:50<00:02, 7595.57it/s] 95%|█████████▍| 378921/400000 [00:50<00:02, 7686.21it/s] 95%|█████████▍| 379691/400000 [00:51<00:02, 7680.75it/s] 95%|█████████▌| 380481/400000 [00:51<00:02, 7742.47it/s] 95%|█████████▌| 381256/400000 [00:51<00:02, 7523.97it/s] 96%|█████████▌| 382030/400000 [00:51<00:02, 7586.32it/s] 96%|█████████▌| 382790/400000 [00:51<00:02, 7562.49it/s] 96%|█████████▌| 383548/400000 [00:51<00:02, 7201.69it/s] 96%|█████████▌| 384274/400000 [00:51<00:02, 7216.44it/s] 96%|█████████▋| 385053/400000 [00:51<00:02, 7376.77it/s] 96%|█████████▋| 385795/400000 [00:51<00:01, 7389.09it/s] 97%|█████████▋| 386549/400000 [00:51<00:01, 7431.07it/s] 97%|█████████▋| 387316/400000 [00:52<00:01, 7499.29it/s] 97%|█████████▋| 388072/400000 [00:52<00:01, 7516.21it/s] 97%|█████████▋| 388825/400000 [00:52<00:01, 7501.50it/s] 97%|█████████▋| 389595/400000 [00:52<00:01, 7558.95it/s] 98%|█████████▊| 390352/400000 [00:52<00:01, 7502.72it/s] 98%|█████████▊| 391103/400000 [00:52<00:01, 7481.07it/s] 98%|█████████▊| 391862/400000 [00:52<00:01, 7512.52it/s] 98%|█████████▊| 392614/400000 [00:52<00:00, 7502.30it/s] 98%|█████████▊| 393383/400000 [00:52<00:00, 7557.45it/s] 99%|█████████▊| 394139/400000 [00:52<00:00, 7514.23it/s] 99%|█████████▊| 394891/400000 [00:53<00:00, 7432.44it/s] 99%|█████████▉| 395635/400000 [00:53<00:00, 7349.64it/s] 99%|█████████▉| 396383/400000 [00:53<00:00, 7388.12it/s] 99%|█████████▉| 397123/400000 [00:53<00:00, 7332.39it/s] 99%|█████████▉| 397862/400000 [00:53<00:00, 7347.02it/s]100%|█████████▉| 398597/400000 [00:53<00:00, 7160.34it/s]100%|█████████▉| 399328/400000 [00:53<00:00, 7201.35it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7435.57it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f851b9300f0>, <torchtext.data.dataset.TabularDataset object at 0x7f851b930240>, <torchtext.vocab.Vocab object at 0x7f851b930160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.50 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.50 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.50 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.65 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.65 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.86 file/s]2020-06-09 06:18:54.652881: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-09 06:18:54.657728: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-09 06:18:54.658542: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557b14d94310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 06:18:54.658558: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 485991.18it/s] 94%|█████████▍| 9314304/9912422 [00:00<00:00, 692708.93it/s]9920512it [00:00, 48064404.27it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578376.18it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 490265.09it/s]1654784it [00:00, 12240215.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 266674.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
