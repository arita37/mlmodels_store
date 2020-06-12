
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f272a4ac6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f272a4ac6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2795a740d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2795a740d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f27af7a0ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f27af7a0ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27432ef950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27432ef950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27432ef950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 473125.83it/s] 38%|███▊      | 3768320/9912422 [00:00<00:09, 672189.64it/s]9920512it [00:00, 32528346.78it/s]                           
0it [00:00, ?it/s]32768it [00:00, 731708.24it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1059348.94it/s]1654784it [00:00, 12915031.63it/s]                           
0it [00:00, ?it/s]8192it [00:00, 217885.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27432dcb38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f27405bcdd8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f27432ef598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f27432ef598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f27432ef598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:37,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:36,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:35,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:35,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:34,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:34,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:33,  1.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:05,  2.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:05,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:05,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:04,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:04,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:03,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:02,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:02,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<01:01,  2.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:43,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:42,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:42,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:41,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:41,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:41,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:40,  3.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:28,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:28,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:27,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:27,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:27,  4.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:26,  4.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:26,  4.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:18,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:12,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:08, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 16.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 21.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 28.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 35.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 43.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 50.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.15 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.15 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.71 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ url]
0 examples [00:00, ? examples/s]2020-06-12 06:11:37.501950: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 06:11:37.517250: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-12 06:11:37.517900: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d93ce1d750 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 06:11:37.517933: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.66 examples/s]109 examples [00:00,  3.79 examples/s]219 examples [00:00,  5.41 examples/s]327 examples [00:00,  7.72 examples/s]438 examples [00:00, 10.99 examples/s]548 examples [00:00, 15.63 examples/s]657 examples [00:00, 22.19 examples/s]766 examples [00:01, 31.43 examples/s]876 examples [00:01, 44.35 examples/s]983 examples [00:01, 62.25 examples/s]1092 examples [00:01, 86.78 examples/s]1198 examples [00:01, 119.75 examples/s]1307 examples [00:01, 163.34 examples/s]1418 examples [00:01, 219.50 examples/s]1528 examples [00:01, 288.65 examples/s]1636 examples [00:01, 366.40 examples/s]1742 examples [00:01, 454.92 examples/s]1852 examples [00:02, 551.76 examples/s]1963 examples [00:02, 649.19 examples/s]2073 examples [00:02, 739.19 examples/s]2182 examples [00:02, 816.38 examples/s]2291 examples [00:02, 879.98 examples/s]2402 examples [00:02, 937.99 examples/s]2512 examples [00:02, 980.60 examples/s]2623 examples [00:02, 1013.97 examples/s]2733 examples [00:02, 1030.96 examples/s]2842 examples [00:03, 1047.23 examples/s]2951 examples [00:03, 1058.06 examples/s]3060 examples [00:03, 1054.58 examples/s]3168 examples [00:03, 1060.05 examples/s]3278 examples [00:03, 1071.46 examples/s]3387 examples [00:03, 1075.30 examples/s]3497 examples [00:03, 1081.34 examples/s]3606 examples [00:03, 1082.73 examples/s]3715 examples [00:03, 1059.16 examples/s]3822 examples [00:03, 1062.32 examples/s]3932 examples [00:04, 1071.53 examples/s]4042 examples [00:04, 1079.42 examples/s]4152 examples [00:04, 1084.90 examples/s]4262 examples [00:04, 1087.33 examples/s]4371 examples [00:04, 1080.45 examples/s]4480 examples [00:04, 1066.86 examples/s]4587 examples [00:04, 1031.10 examples/s]4696 examples [00:04, 1047.88 examples/s]4803 examples [00:04, 1052.96 examples/s]4909 examples [00:04, 1051.63 examples/s]5015 examples [00:05, 1033.03 examples/s]5119 examples [00:05, 1032.04 examples/s]5229 examples [00:05, 1049.12 examples/s]5339 examples [00:05, 1063.74 examples/s]5446 examples [00:05, 1058.33 examples/s]5553 examples [00:05, 1058.89 examples/s]5659 examples [00:05, 1004.81 examples/s]5761 examples [00:05, 1002.51 examples/s]5869 examples [00:05, 1021.09 examples/s]5975 examples [00:05, 1031.45 examples/s]6083 examples [00:06, 1044.34 examples/s]6189 examples [00:06, 1048.21 examples/s]6295 examples [00:06, 1050.08 examples/s]6404 examples [00:06, 1059.37 examples/s]6511 examples [00:06, 1058.83 examples/s]6617 examples [00:06, 1057.09 examples/s]6723 examples [00:06, 1046.03 examples/s]6828 examples [00:06, 1044.12 examples/s]6933 examples [00:06, 1041.36 examples/s]7040 examples [00:06, 1048.83 examples/s]7148 examples [00:07, 1057.58 examples/s]7255 examples [00:07, 1060.38 examples/s]7364 examples [00:07, 1068.01 examples/s]7472 examples [00:07, 1069.51 examples/s]7579 examples [00:07, 1055.47 examples/s]7685 examples [00:07, 1051.20 examples/s]7791 examples [00:07, 1030.10 examples/s]7900 examples [00:07, 1047.33 examples/s]8005 examples [00:07, 1044.67 examples/s]8113 examples [00:08, 1053.14 examples/s]8219 examples [00:08, 1044.06 examples/s]8324 examples [00:08, 1036.11 examples/s]8428 examples [00:08, 1035.89 examples/s]8532 examples [00:08, 1031.82 examples/s]8636 examples [00:08, 1034.22 examples/s]8744 examples [00:08, 1046.81 examples/s]8852 examples [00:08, 1055.62 examples/s]8960 examples [00:08, 1061.60 examples/s]9067 examples [00:08, 1062.96 examples/s]9174 examples [00:09, 1062.85 examples/s]9284 examples [00:09, 1069.96 examples/s]9392 examples [00:09, 1072.60 examples/s]9502 examples [00:09, 1078.10 examples/s]9610 examples [00:09, 1069.11 examples/s]9717 examples [00:09, 1048.17 examples/s]9827 examples [00:09, 1060.30 examples/s]9936 examples [00:09, 1066.47 examples/s]10043 examples [00:09, 964.69 examples/s]10150 examples [00:09, 990.99 examples/s]10257 examples [00:10, 1011.88 examples/s]10360 examples [00:10, 1002.20 examples/s]10462 examples [00:10, 993.88 examples/s] 10571 examples [00:10, 1018.62 examples/s]10680 examples [00:10, 1036.55 examples/s]10790 examples [00:10, 1053.15 examples/s]10901 examples [00:10, 1067.56 examples/s]11012 examples [00:10, 1078.36 examples/s]11123 examples [00:10, 1085.35 examples/s]11232 examples [00:10, 1085.92 examples/s]11343 examples [00:11, 1090.96 examples/s]11454 examples [00:11, 1096.46 examples/s]11564 examples [00:11, 1053.01 examples/s]11675 examples [00:11, 1068.93 examples/s]11785 examples [00:11, 1069.31 examples/s]11893 examples [00:11, 1028.44 examples/s]12003 examples [00:11, 1048.19 examples/s]12111 examples [00:11, 1057.28 examples/s]12221 examples [00:11, 1067.20 examples/s]12329 examples [00:12, 1068.27 examples/s]12440 examples [00:12, 1077.70 examples/s]12551 examples [00:12, 1085.33 examples/s]12662 examples [00:12, 1090.38 examples/s]12773 examples [00:12, 1093.88 examples/s]12883 examples [00:12, 1095.28 examples/s]12993 examples [00:12, 1095.93 examples/s]13103 examples [00:12, 1094.43 examples/s]13215 examples [00:12, 1098.81 examples/s]13326 examples [00:12, 1101.12 examples/s]13437 examples [00:13, 1102.27 examples/s]13548 examples [00:13, 1090.73 examples/s]13658 examples [00:13, 1091.15 examples/s]13768 examples [00:13, 1087.30 examples/s]13878 examples [00:13, 1091.01 examples/s]13988 examples [00:13, 1091.02 examples/s]14098 examples [00:13, 1091.58 examples/s]14209 examples [00:13, 1094.18 examples/s]14319 examples [00:13, 1092.52 examples/s]14429 examples [00:13, 1091.33 examples/s]14539 examples [00:14, 1089.30 examples/s]14648 examples [00:14, 1071.57 examples/s]14758 examples [00:14, 1077.52 examples/s]14866 examples [00:14, 1050.77 examples/s]14976 examples [00:14, 1064.51 examples/s]15083 examples [00:14, 1064.00 examples/s]15192 examples [00:14, 1071.27 examples/s]15302 examples [00:14, 1078.83 examples/s]15411 examples [00:14, 1081.31 examples/s]15520 examples [00:14, 1082.83 examples/s]15629 examples [00:15, 1065.01 examples/s]15737 examples [00:15, 1067.82 examples/s]15848 examples [00:15, 1078.18 examples/s]15956 examples [00:15, 1075.01 examples/s]16066 examples [00:15, 1081.78 examples/s]16175 examples [00:15, 1080.42 examples/s]16285 examples [00:15, 1083.60 examples/s]16396 examples [00:15, 1088.58 examples/s]16505 examples [00:15, 1084.26 examples/s]16614 examples [00:15, 1047.73 examples/s]16720 examples [00:16, 1047.52 examples/s]16828 examples [00:16, 1055.29 examples/s]16934 examples [00:16, 1038.98 examples/s]17044 examples [00:16, 1056.55 examples/s]17152 examples [00:16, 1063.36 examples/s]17262 examples [00:16, 1071.30 examples/s]17370 examples [00:16, 1068.79 examples/s]17478 examples [00:16, 1071.74 examples/s]17587 examples [00:16, 1075.77 examples/s]17696 examples [00:16, 1077.37 examples/s]17804 examples [00:17, 1076.18 examples/s]17912 examples [00:17, 1074.07 examples/s]18024 examples [00:17, 1085.93 examples/s]18133 examples [00:17, 1053.29 examples/s]18242 examples [00:17, 1062.45 examples/s]18352 examples [00:17, 1071.74 examples/s]18461 examples [00:17, 1075.25 examples/s]18572 examples [00:17, 1083.25 examples/s]18682 examples [00:17, 1088.00 examples/s]18791 examples [00:17, 1087.15 examples/s]18900 examples [00:18, 1084.09 examples/s]19009 examples [00:18, 1076.67 examples/s]19119 examples [00:18, 1082.68 examples/s]19228 examples [00:18, 1081.16 examples/s]19337 examples [00:18, 1083.00 examples/s]19446 examples [00:18, 1079.48 examples/s]19556 examples [00:18, 1083.64 examples/s]19667 examples [00:18, 1089.11 examples/s]19778 examples [00:18, 1094.28 examples/s]19888 examples [00:19, 1095.35 examples/s]19998 examples [00:19, 1096.32 examples/s]20108 examples [00:19, 1036.45 examples/s]20220 examples [00:19, 1058.09 examples/s]20330 examples [00:19, 1069.04 examples/s]20440 examples [00:19, 1075.62 examples/s]20550 examples [00:19, 1082.04 examples/s]20659 examples [00:19, 1080.85 examples/s]20769 examples [00:19, 1085.37 examples/s]20879 examples [00:19, 1089.40 examples/s]20989 examples [00:20, 1082.34 examples/s]21099 examples [00:20, 1087.57 examples/s]21208 examples [00:20, 1067.88 examples/s]21315 examples [00:20, 1038.02 examples/s]21420 examples [00:20, 1028.01 examples/s]21525 examples [00:20, 1032.70 examples/s]21636 examples [00:20, 1052.35 examples/s]21742 examples [00:20, 1021.64 examples/s]21852 examples [00:20, 1042.80 examples/s]21963 examples [00:20, 1060.36 examples/s]22070 examples [00:21, 1060.86 examples/s]22180 examples [00:21, 1070.10 examples/s]22288 examples [00:21, 1064.19 examples/s]22396 examples [00:21, 1067.01 examples/s]22504 examples [00:21, 1069.12 examples/s]22613 examples [00:21, 1073.75 examples/s]22722 examples [00:21, 1077.86 examples/s]22830 examples [00:21, 1076.48 examples/s]22940 examples [00:21, 1080.98 examples/s]23050 examples [00:21, 1086.35 examples/s]23159 examples [00:22, 1083.96 examples/s]23271 examples [00:22, 1091.68 examples/s]23381 examples [00:22, 1092.26 examples/s]23492 examples [00:22, 1095.93 examples/s]23603 examples [00:22, 1098.78 examples/s]23714 examples [00:22, 1099.89 examples/s]23824 examples [00:22, 1093.21 examples/s]23934 examples [00:22, 1084.45 examples/s]24043 examples [00:22, 1066.29 examples/s]24154 examples [00:22, 1076.54 examples/s]24264 examples [00:23, 1083.06 examples/s]24373 examples [00:23, 1055.24 examples/s]24479 examples [00:23, 1038.73 examples/s]24589 examples [00:23, 1055.58 examples/s]24695 examples [00:23, 1011.62 examples/s]24804 examples [00:23, 1031.63 examples/s]24909 examples [00:23, 1033.99 examples/s]25018 examples [00:23, 1047.80 examples/s]25127 examples [00:23, 1058.04 examples/s]25237 examples [00:24, 1069.11 examples/s]25346 examples [00:24, 1073.76 examples/s]25457 examples [00:24, 1082.50 examples/s]25566 examples [00:24, 1083.88 examples/s]25675 examples [00:24, 1056.82 examples/s]25785 examples [00:24, 1067.64 examples/s]25892 examples [00:24, 1041.08 examples/s]26002 examples [00:24, 1057.32 examples/s]26108 examples [00:24, 1054.00 examples/s]26214 examples [00:24, 1054.80 examples/s]26324 examples [00:25, 1067.61 examples/s]26435 examples [00:25, 1078.96 examples/s]26545 examples [00:25, 1084.80 examples/s]26655 examples [00:25, 1088.99 examples/s]26764 examples [00:25, 1046.07 examples/s]26870 examples [00:25, 1024.88 examples/s]26978 examples [00:25, 1038.64 examples/s]27083 examples [00:25, 1035.80 examples/s]27193 examples [00:25, 1052.00 examples/s]27303 examples [00:25, 1065.53 examples/s]27413 examples [00:26, 1073.31 examples/s]27522 examples [00:26, 1074.77 examples/s]27630 examples [00:26, 1059.77 examples/s]27737 examples [00:26, 1031.21 examples/s]27841 examples [00:26, 1032.56 examples/s]27945 examples [00:26, 1030.85 examples/s]28052 examples [00:26, 1040.42 examples/s]28161 examples [00:26, 1054.25 examples/s]28270 examples [00:26, 1063.15 examples/s]28377 examples [00:26, 1059.49 examples/s]28484 examples [00:27, 1023.85 examples/s]28591 examples [00:27, 1035.56 examples/s]28700 examples [00:27, 1050.08 examples/s]28810 examples [00:27, 1062.23 examples/s]28920 examples [00:27, 1071.58 examples/s]29031 examples [00:27, 1081.56 examples/s]29142 examples [00:27, 1087.13 examples/s]29251 examples [00:27, 1082.19 examples/s]29362 examples [00:27, 1088.00 examples/s]29473 examples [00:28, 1093.74 examples/s]29583 examples [00:28, 1083.57 examples/s]29694 examples [00:28, 1090.41 examples/s]29804 examples [00:28, 1087.79 examples/s]29915 examples [00:28, 1092.33 examples/s]30025 examples [00:28, 1022.71 examples/s]30129 examples [00:28, 1027.74 examples/s]30236 examples [00:28, 1038.53 examples/s]30344 examples [00:28, 1049.01 examples/s]30454 examples [00:28, 1061.84 examples/s]30562 examples [00:29, 1065.76 examples/s]30672 examples [00:29, 1073.87 examples/s]30780 examples [00:29, 1067.34 examples/s]30887 examples [00:29, 1044.21 examples/s]30998 examples [00:29, 1060.85 examples/s]31105 examples [00:29, 1035.75 examples/s]31213 examples [00:29, 1047.31 examples/s]31322 examples [00:29, 1059.54 examples/s]31429 examples [00:29, 1057.34 examples/s]31539 examples [00:29, 1067.01 examples/s]31646 examples [00:30, 1062.43 examples/s]31757 examples [00:30, 1073.35 examples/s]31867 examples [00:30, 1080.99 examples/s]31976 examples [00:30, 1079.87 examples/s]32085 examples [00:30, 1081.76 examples/s]32195 examples [00:30, 1085.59 examples/s]32305 examples [00:30, 1087.86 examples/s]32415 examples [00:30, 1090.07 examples/s]32525 examples [00:30, 1083.80 examples/s]32636 examples [00:30, 1090.93 examples/s]32747 examples [00:31, 1095.29 examples/s]32859 examples [00:31, 1100.04 examples/s]32970 examples [00:31, 1085.24 examples/s]33079 examples [00:31, 1075.36 examples/s]33187 examples [00:31, 1071.35 examples/s]33297 examples [00:31, 1077.52 examples/s]33405 examples [00:31, 1072.07 examples/s]33514 examples [00:31, 1076.08 examples/s]33622 examples [00:31, 1075.23 examples/s]33734 examples [00:31, 1085.76 examples/s]33843 examples [00:32, 1074.68 examples/s]33951 examples [00:32, 1069.52 examples/s]34061 examples [00:32, 1076.25 examples/s]34169 examples [00:32, 1063.91 examples/s]34279 examples [00:32, 1073.08 examples/s]34387 examples [00:32, 1061.03 examples/s]34495 examples [00:32, 1064.78 examples/s]34604 examples [00:32, 1071.55 examples/s]34712 examples [00:32, 1070.41 examples/s]34820 examples [00:33, 1048.03 examples/s]34931 examples [00:33, 1065.36 examples/s]35041 examples [00:33, 1073.14 examples/s]35152 examples [00:33, 1083.25 examples/s]35262 examples [00:33, 1087.26 examples/s]35371 examples [00:33, 1050.67 examples/s]35482 examples [00:33, 1065.09 examples/s]35594 examples [00:33, 1078.81 examples/s]35705 examples [00:33, 1086.92 examples/s]35814 examples [00:33, 1081.27 examples/s]35923 examples [00:34, 1082.94 examples/s]36033 examples [00:34, 1085.79 examples/s]36142 examples [00:34, 1079.46 examples/s]36252 examples [00:34, 1083.79 examples/s]36361 examples [00:34, 1076.26 examples/s]36469 examples [00:34, 1074.42 examples/s]36579 examples [00:34, 1080.37 examples/s]36688 examples [00:34, 1080.36 examples/s]36798 examples [00:34, 1084.29 examples/s]36907 examples [00:34, 1073.27 examples/s]37015 examples [00:35, 1056.57 examples/s]37124 examples [00:35, 1064.13 examples/s]37234 examples [00:35, 1074.59 examples/s]37344 examples [00:35, 1081.83 examples/s]37453 examples [00:35, 1079.48 examples/s]37562 examples [00:35, 1081.51 examples/s]37671 examples [00:35, 1036.78 examples/s]37777 examples [00:35, 1041.64 examples/s]37882 examples [00:35, 1043.27 examples/s]37987 examples [00:35, 1021.17 examples/s]38092 examples [00:36, 1028.11 examples/s]38201 examples [00:36, 1043.36 examples/s]38306 examples [00:36, 1040.58 examples/s]38411 examples [00:36, 1036.48 examples/s]38519 examples [00:36, 1047.54 examples/s]38628 examples [00:36, 1059.08 examples/s]38738 examples [00:36, 1068.87 examples/s]38849 examples [00:36, 1079.39 examples/s]38958 examples [00:36, 1076.18 examples/s]39066 examples [00:36, 1073.63 examples/s]39177 examples [00:37, 1081.23 examples/s]39287 examples [00:37, 1085.80 examples/s]39397 examples [00:37, 1088.78 examples/s]39507 examples [00:37, 1091.62 examples/s]39617 examples [00:37, 1087.21 examples/s]39727 examples [00:37, 1088.69 examples/s]39838 examples [00:37, 1092.53 examples/s]39948 examples [00:37, 1082.98 examples/s]40057 examples [00:37, 1024.40 examples/s]40166 examples [00:38, 1040.92 examples/s]40276 examples [00:38, 1057.50 examples/s]40386 examples [00:38, 1067.25 examples/s]40494 examples [00:38, 1035.21 examples/s]40601 examples [00:38, 1043.17 examples/s]40706 examples [00:38, 1036.50 examples/s]40816 examples [00:38, 1053.42 examples/s]40922 examples [00:38, 1027.17 examples/s]41031 examples [00:38, 1043.19 examples/s]41141 examples [00:38, 1059.46 examples/s]41249 examples [00:39, 1063.19 examples/s]41356 examples [00:39, 1051.79 examples/s]41465 examples [00:39, 1061.22 examples/s]41575 examples [00:39, 1071.14 examples/s]41685 examples [00:39, 1078.43 examples/s]41793 examples [00:39, 1078.38 examples/s]41902 examples [00:39, 1080.19 examples/s]42011 examples [00:39, 1080.45 examples/s]42120 examples [00:39, 1081.60 examples/s]42229 examples [00:39, 1068.76 examples/s]42336 examples [00:40, 1060.53 examples/s]42445 examples [00:40, 1068.29 examples/s]42554 examples [00:40, 1074.37 examples/s]42662 examples [00:40, 1072.16 examples/s]42772 examples [00:40, 1078.03 examples/s]42880 examples [00:40, 1076.67 examples/s]42988 examples [00:40, 1073.40 examples/s]43096 examples [00:40, 1027.57 examples/s]43207 examples [00:40, 1048.36 examples/s]43317 examples [00:40, 1062.59 examples/s]43427 examples [00:41, 1071.55 examples/s]43537 examples [00:41, 1076.83 examples/s]43645 examples [00:41, 1067.51 examples/s]43754 examples [00:41, 1071.19 examples/s]43862 examples [00:41, 1072.60 examples/s]43970 examples [00:41, 1074.32 examples/s]44078 examples [00:41, 1071.79 examples/s]44186 examples [00:41, 1035.34 examples/s]44294 examples [00:41, 1047.87 examples/s]44402 examples [00:42, 1057.09 examples/s]44511 examples [00:42, 1065.46 examples/s]44620 examples [00:42, 1071.17 examples/s]44728 examples [00:42, 1069.33 examples/s]44837 examples [00:42, 1074.57 examples/s]44947 examples [00:42, 1079.98 examples/s]45056 examples [00:42, 1076.58 examples/s]45165 examples [00:42, 1078.72 examples/s]45275 examples [00:42, 1083.15 examples/s]45385 examples [00:42, 1086.09 examples/s]45495 examples [00:43, 1089.78 examples/s]45604 examples [00:43, 1086.22 examples/s]45713 examples [00:43, 1085.73 examples/s]45824 examples [00:43, 1090.41 examples/s]45934 examples [00:43, 1082.21 examples/s]46043 examples [00:43, 1083.09 examples/s]46152 examples [00:43, 1084.06 examples/s]46262 examples [00:43, 1086.20 examples/s]46373 examples [00:43, 1091.01 examples/s]46483 examples [00:43, 1089.92 examples/s]46593 examples [00:44, 1071.94 examples/s]46701 examples [00:44, 1060.98 examples/s]46808 examples [00:44, 1060.86 examples/s]46915 examples [00:44, 1056.10 examples/s]47025 examples [00:44, 1067.19 examples/s]47132 examples [00:44, 1065.98 examples/s]47241 examples [00:44, 1071.67 examples/s]47350 examples [00:44, 1075.12 examples/s]47458 examples [00:44, 1058.16 examples/s]47567 examples [00:44, 1066.78 examples/s]47675 examples [00:45, 1067.87 examples/s]47784 examples [00:45, 1072.66 examples/s]47894 examples [00:45, 1080.37 examples/s]48004 examples [00:45, 1084.45 examples/s]48113 examples [00:45, 1083.30 examples/s]48223 examples [00:45, 1086.39 examples/s]48332 examples [00:45, 1086.83 examples/s]48441 examples [00:45, 1085.13 examples/s]48550 examples [00:45, 1075.82 examples/s]48659 examples [00:45, 1079.31 examples/s]48767 examples [00:46, 1079.12 examples/s]48876 examples [00:46, 1081.80 examples/s]48985 examples [00:46, 1082.58 examples/s]49094 examples [00:46, 1070.31 examples/s]49202 examples [00:46, 1065.55 examples/s]49310 examples [00:46, 1068.72 examples/s]49419 examples [00:46, 1073.67 examples/s]49527 examples [00:46, 1070.11 examples/s]49635 examples [00:46, 1046.26 examples/s]49745 examples [00:46, 1060.72 examples/s]49852 examples [00:47, 1042.42 examples/s]49958 examples [00:47, 1046.70 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7114/50000 [00:00<00:00, 71134.30 examples/s] 41%|████      | 20313/50000 [00:00<00:00, 82552.43 examples/s] 64%|██████▍   | 32225/50000 [00:00<00:00, 90925.59 examples/s] 91%|█████████▏| 45667/50000 [00:00<00:00, 100700.28 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 869.45 examples/s]180 examples [00:00, 885.49 examples/s]290 examples [00:00, 939.65 examples/s]403 examples [00:00, 987.77 examples/s]514 examples [00:00, 1019.32 examples/s]628 examples [00:00, 1051.86 examples/s]740 examples [00:00, 1068.75 examples/s]851 examples [00:00, 1079.75 examples/s]964 examples [00:00, 1091.90 examples/s]1075 examples [00:01, 1094.59 examples/s]1187 examples [00:01, 1099.72 examples/s]1296 examples [00:01, 1086.24 examples/s]1404 examples [00:01, 854.68 examples/s] 1514 examples [00:01, 914.76 examples/s]1625 examples [00:01, 965.02 examples/s]1735 examples [00:01, 999.94 examples/s]1846 examples [00:01, 1029.26 examples/s]1955 examples [00:01, 1046.03 examples/s]2065 examples [00:02, 1060.35 examples/s]2175 examples [00:02, 1071.05 examples/s]2284 examples [00:02, 1071.74 examples/s]2392 examples [00:02, 1062.31 examples/s]2501 examples [00:02, 1070.40 examples/s]2610 examples [00:02, 1074.19 examples/s]2721 examples [00:02, 1081.23 examples/s]2831 examples [00:02, 1085.32 examples/s]2942 examples [00:02, 1090.58 examples/s]3052 examples [00:02, 1091.21 examples/s]3162 examples [00:03, 1077.43 examples/s]3272 examples [00:03, 1082.77 examples/s]3381 examples [00:03, 1058.31 examples/s]3488 examples [00:03, 1033.93 examples/s]3593 examples [00:03, 1035.98 examples/s]3704 examples [00:03, 1055.37 examples/s]3814 examples [00:03, 1067.26 examples/s]3921 examples [00:03, 1061.27 examples/s]4028 examples [00:03, 1058.72 examples/s]4137 examples [00:03, 1067.45 examples/s]4244 examples [00:04, 1065.22 examples/s]4354 examples [00:04, 1074.43 examples/s]4462 examples [00:04, 1075.96 examples/s]4570 examples [00:04, 1067.12 examples/s]4681 examples [00:04, 1077.05 examples/s]4789 examples [00:04, 1077.79 examples/s]4899 examples [00:04, 1083.39 examples/s]5009 examples [00:04, 1086.85 examples/s]5118 examples [00:04, 1085.17 examples/s]5227 examples [00:04, 1078.49 examples/s]5335 examples [00:05, 1049.93 examples/s]5441 examples [00:05, 1046.82 examples/s]5552 examples [00:05, 1062.98 examples/s]5663 examples [00:05, 1075.32 examples/s]5773 examples [00:05, 1081.13 examples/s]5883 examples [00:05, 1084.03 examples/s]5992 examples [00:05, 1073.91 examples/s]6102 examples [00:05, 1081.29 examples/s]6211 examples [00:05, 1082.60 examples/s]6322 examples [00:05, 1087.93 examples/s]6431 examples [00:06, 1086.28 examples/s]6540 examples [00:06, 1087.36 examples/s]6649 examples [00:06, 1080.34 examples/s]6758 examples [00:06, 1036.46 examples/s]6866 examples [00:06, 1046.64 examples/s]6971 examples [00:06, 1034.37 examples/s]7079 examples [00:06, 1045.31 examples/s]7186 examples [00:06, 1050.12 examples/s]7295 examples [00:06, 1059.58 examples/s]7404 examples [00:06, 1067.31 examples/s]7512 examples [00:07, 1068.39 examples/s]7620 examples [00:07, 1069.25 examples/s]7727 examples [00:07, 1006.64 examples/s]7834 examples [00:07, 1022.66 examples/s]7940 examples [00:07, 1032.35 examples/s]8044 examples [00:07, 1009.19 examples/s]8146 examples [00:07, 1009.94 examples/s]8257 examples [00:07, 1036.18 examples/s]8368 examples [00:07, 1055.92 examples/s]8477 examples [00:08, 1062.52 examples/s]8586 examples [00:08, 1069.51 examples/s]8694 examples [00:08, 1040.75 examples/s]8804 examples [00:08, 1056.47 examples/s]8914 examples [00:08, 1067.01 examples/s]9025 examples [00:08, 1077.05 examples/s]9133 examples [00:08, 1074.62 examples/s]9241 examples [00:08, 1075.49 examples/s]9352 examples [00:08, 1082.69 examples/s]9461 examples [00:08, 1084.68 examples/s]9570 examples [00:09, 1062.05 examples/s]9677 examples [00:09, 1055.03 examples/s]9785 examples [00:09, 1061.45 examples/s]9892 examples [00:09, 1003.48 examples/s]9996 examples [00:09, 1013.46 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSGOXKL/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSGOXKL/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27432ef950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27432ef950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27432ef950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c9834358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f26c879def0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f27432ef950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f27432ef950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f27432ef950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f272a39f0f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f272a39f0b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f26c40f42f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f26c40f42f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f26c40f42f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f26c40f4400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f26c40f4400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f26c40f4400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c49b996347d25e65c293ce59f3c20f45f196ed32b91161b2b0db32a1f1afa628
  Stored in directory: /tmp/pip-ephem-wheel-cache-phc3vg0h/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:39:13, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:41:31, 16.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:20:20, 23.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:14:45, 33.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:03:34, 47.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.37M/862M [00:01<3:31:10, 67.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:26:56, 96.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:01<1:42:20, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:11:27, 196kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.5M/862M [00:01<49:48, 279kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:01<34:50, 397kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:02<24:18, 565kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<17:05, 801kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.0M/862M [00:02<11:57, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<08:28, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:36, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:31, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<08:14, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:46, 2.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<08:57, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<07:53, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:54, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:53, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:17, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<04:42, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:14, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:41, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<04:19, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<06:08, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<07:02, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<05:35, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:01, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<04:14, 3.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<05:59, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:51, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:16<05:28, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<05:54, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<05:17, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<04:03, 3.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:18<03:00, 4.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<27:06, 475kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<21:37, 596kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<15:47, 815kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<13:04, 979kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<10:28, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:22<07:36, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<08:18, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<07:05, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<05:17, 2.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:41, 1.89MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:16, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:38, 2.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:06, 3.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:02, 1.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:18, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<06:08, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:13, 1.74MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:20, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:42, 2.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:13, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:54, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:22, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<03:55, 3.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:34, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:00, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:51, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:00, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:25, 1.66MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:49, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:03, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:16, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:57, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<02:55, 4.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<51:10, 238kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<37:03, 329kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<26:09, 465kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<21:08, 574kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:01, 756kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<11:30, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:53, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:05, 1.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:40, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:17, 1.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:09, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:37, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:02, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:41, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:11, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:47, 3.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:17, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:02, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:10, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:23, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:40, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:16, 2.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:45, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:13, 2.24MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:57, 2.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:30, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:04, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:48, 3.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:46, 3.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:56, 2.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:41, 3.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:16, 2.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:02, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:42, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:27, 3.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:43, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:33, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:52, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:03, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:33, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:05, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:43, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:07, 1.58MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:09, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<04:35, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:48, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:21, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:58, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<03:36, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:28, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:45, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:42, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:34, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:46, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:18, 2.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:28, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:47, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:43, 2.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<02:44, 3.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<12:21, 881kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:59, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:18, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:20, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:40, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:01, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<04:21, 2.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<09:44, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<08:08, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<05:57, 1.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:21, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:05, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:36, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<04:03, 2.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:26, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:34, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:15, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:33, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:51, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:24, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:14, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:07, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:54, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:34, 2.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:44, 1.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<05:53, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:36, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:14, 2.43MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:04, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:57, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:43, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:25, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:29, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:28, 1.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:46, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:26, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:18, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:01, 2.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:39, 2.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:45, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:49, 1.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:42, 1.76MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<04:09, 2.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:50, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:06, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:55, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:11, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:16, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:48, 2.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:18, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<07:41, 1.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:37, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<04:02, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<21:44, 450kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<16:24, 596kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<11:44, 830kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<10:11, 953kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<08:18, 1.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:03, 1.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:13, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:30, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:08, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:51, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:33, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:28, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:22, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:17, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:11, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:04, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:19, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:51, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:38, 2.59MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:28, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:15, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:13, 2.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:09, 2.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:01, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:03, 3.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:03, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:58, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:00, 3.07MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:12, 4.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<9:11:57, 16.6kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<6:27:14, 23.6kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<4:30:41, 33.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<3:10:29, 47.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<2:15:32, 67.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<1:35:15, 95.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<1:06:29, 136kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<52:29, 172kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<37:48, 238kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<26:40, 337kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<20:26, 438kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<15:24, 581kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<11:01, 809kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<09:31, 932kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:36, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:38, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:00, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:40, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<05:54, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:13, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<08:48, 993kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:12, 1.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:15, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:28, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:59, 1.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:38, 1.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:20, 2.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<08:06, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:39, 1.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:51, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:13, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<05:39, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:25, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:10, 2.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:52, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:40, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:13, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:40, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:08, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:08, 2.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:55, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:24, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:31, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<02:43, 3.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:34, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:25, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:35, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:35, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:25, 2.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:34, 3.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:35, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:30, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:34, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:36, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:04, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:31, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:24, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<04:01, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:46, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<02:52, 2.77MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:37, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:29, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:32, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:39, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:41, 2.13MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:31, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:39, 2.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<01:56, 3.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<1:57:48, 65.9kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<1:24:21, 92.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<59:26, 130kB/s]   .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<41:28, 186kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<34:07, 226kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<24:48, 310kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<17:32, 437kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<13:47, 553kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<10:33, 722kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<07:34, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<05:21, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<1:33:03, 81.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<1:06:00, 114kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<46:18, 163kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<33:46, 222kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<25:29, 294kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<18:11, 411kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<12:49, 581kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<11:04, 670kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<08:39, 856kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<06:13, 1.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<04:27, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<10:42, 686kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<08:22, 877kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<06:02, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:42, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:52, 1.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<03:37, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:01, 1.79MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:36, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:41, 2.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:09, 1.71MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:20, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<02:26, 2.91MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:45, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:31, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:02, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:21, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:53, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:54, 2.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:08<02:06, 3.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<43:05, 161kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<30:59, 224kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<21:50, 316kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<16:35, 414kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<12:18, 557kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<08:48, 777kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<07:31, 904kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:56, 979kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:13, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<03:42, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:15, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:10, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:47, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:00, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:38, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:44, 2.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:15, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:45, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<02:08, 3.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:09, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:42, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:45, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:15, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:03, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:18, 2.79MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:54, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:37, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:51, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:05, 3.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:48, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:25, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:33, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:04, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:41, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:06, 2.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:44, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:59, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:55, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:23, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:41, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:52, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:05, 2.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:19, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:28, 2.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:56, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:47, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:07, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:41, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:35, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<01:58, 2.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:34, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:15, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<01:54, 3.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:10, 1.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:18, 1.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:09, 1.82MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:22, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:41, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:05, 2.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:27, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:58, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:14, 2.50MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<01:38, 3.38MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:28, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:41, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:03, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:05, 2.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:34, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:27, 2.20MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:52, 2.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:24, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:59, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:21, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:44, 3.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:37, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:28, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:58<01:51, 2.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:22, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:17, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:45, 2.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:48, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<01:38, 3.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:31, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:38, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:38, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:54, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<06:54, 723kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:54, 846kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:21, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<03:07, 1.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:32, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:04, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:16, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:35, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:56, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:21, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:41, 2.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<04:10, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:29, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:34, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:26, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:45, 2.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:33, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:40, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:41, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:48, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:05, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:24, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:43, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:20, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:50, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:04, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:24, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:41, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:04, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:30, 2.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:20, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:45, 2.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:06, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:29, 2.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:55, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:51, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:24, 2.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:50, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:47, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:22, 3.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:47, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:15, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:47, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:17, 3.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:50, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:28, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:50, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:56, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:23, 2.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:37, 1.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:00, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:11, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:18, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:30, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:56, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:24, 2.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:13, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:00, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:29, 2.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:41, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:13, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:10, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:55, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:26, 2.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:46, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:40, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:16, 2.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:35, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:28, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:08, 3.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:29, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:27, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:04, 3.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:42, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:35, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:11, 2.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:30, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:51, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:27, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:03, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:49, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:39, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:14, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:29, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:48, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:02, 2.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:43, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:34, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:10, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:20, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:57, 3.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<04:11, 696kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<03:11, 910kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<02:17, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:12, 1.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:53, 1.50MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:23, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:32, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:24, 1.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:02, 2.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:33, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:15, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:54, 2.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:30, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:04, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:31, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:34, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:25, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:03, 2.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:14, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:27, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:09, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<00:50, 2.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:20, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:54, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:23, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:28, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:18, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:58, 2.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:08, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:21, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:03, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:46, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:16, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:09, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:51, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:02, 2.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:13, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:59, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:42, 2.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:52, 1.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:33, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:08, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:11, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:19, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:44, 2.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:54, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:34, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:07, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:09, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:16, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:59, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<00:42, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:36, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:20, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:58, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:01, 1.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:07, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:37, 2.72MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:18, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:06, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:48, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:53, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:48, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:36, 2.59MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:43, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:52, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:41, 2.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:28, 3.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:17, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:04, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:46, 1.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:49, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:44, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:32, 2.48MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:38, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:36, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:27, 2.86MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:33, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:41, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:33, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:23, 3.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:07, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:55, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:40, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:40, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:45, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:34, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:24, 2.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:37, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:33, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:24, 2.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:28, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:34, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:18, 2.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:52, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:42, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:30, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:27, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:19, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:22, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:15, 2.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:11, 3.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:57, 724kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:45, 920kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:31, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:28, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:29, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:15, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:21, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:18, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:13, 2.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:09, 2.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:23, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:19, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:13, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:13, 1.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:02, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.84MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:49:38,  5.90it/s]  0%|          | 835/400000 [00:00<13:09:20,  8.43it/s]  0%|          | 1687/400000 [00:00<9:11:35, 12.04it/s]  1%|          | 2539/400000 [00:00<6:25:31, 17.18it/s]  1%|          | 3360/400000 [00:00<4:29:33, 24.52it/s]  1%|          | 4143/400000 [00:00<3:08:33, 34.99it/s]  1%|          | 4985/400000 [00:00<2:11:57, 49.89it/s]  1%|▏         | 5794/400000 [00:00<1:32:25, 71.09it/s]  2%|▏         | 6644/400000 [00:00<1:04:47, 101.19it/s]  2%|▏         | 7473/400000 [00:01<45:29, 143.81it/s]    2%|▏         | 8286/400000 [00:01<32:01, 203.90it/s]  2%|▏         | 9116/400000 [00:01<22:36, 288.24it/s]  2%|▏         | 9963/400000 [00:01<16:01, 405.85it/s]  3%|▎         | 10795/400000 [00:01<11:25, 567.91it/s]  3%|▎         | 11621/400000 [00:01<08:12, 788.06it/s]  3%|▎         | 12446/400000 [00:01<05:58, 1080.67it/s]  3%|▎         | 13300/400000 [00:01<04:24, 1464.34it/s]  4%|▎         | 14155/400000 [00:01<03:17, 1948.73it/s]  4%|▍         | 15013/400000 [00:01<02:31, 2536.82it/s]  4%|▍         | 15857/400000 [00:02<01:59, 3207.13it/s]  4%|▍         | 16699/400000 [00:02<01:37, 3917.62it/s]  4%|▍         | 17554/400000 [00:02<01:21, 4677.25it/s]  5%|▍         | 18406/400000 [00:02<01:10, 5408.99it/s]  5%|▍         | 19249/400000 [00:02<01:03, 6019.08it/s]  5%|▌         | 20084/400000 [00:02<00:57, 6564.96it/s]  5%|▌         | 20919/400000 [00:02<00:54, 6962.06it/s]  5%|▌         | 21770/400000 [00:02<00:51, 7361.77it/s]  6%|▌         | 22605/400000 [00:02<00:49, 7585.40it/s]  6%|▌         | 23434/400000 [00:02<00:48, 7764.36it/s]  6%|▌         | 24261/400000 [00:03<00:49, 7631.56it/s]  6%|▋         | 25111/400000 [00:03<00:47, 7870.77it/s]  6%|▋         | 25957/400000 [00:03<00:46, 8037.82it/s]  7%|▋         | 26810/400000 [00:03<00:45, 8176.83it/s]  7%|▋         | 27664/400000 [00:03<00:44, 8279.75it/s]  7%|▋         | 28520/400000 [00:03<00:44, 8361.84it/s]  7%|▋         | 29378/400000 [00:03<00:43, 8426.01it/s]  8%|▊         | 30226/400000 [00:03<00:43, 8418.20it/s]  8%|▊         | 31083/400000 [00:03<00:43, 8462.84it/s]  8%|▊         | 31934/400000 [00:04<00:43, 8476.44it/s]  8%|▊         | 32784/400000 [00:04<00:43, 8480.45it/s]  8%|▊         | 33634/400000 [00:04<00:43, 8478.98it/s]  9%|▊         | 34483/400000 [00:04<00:43, 8403.67it/s]  9%|▉         | 35327/400000 [00:04<00:43, 8412.57it/s]  9%|▉         | 36178/400000 [00:04<00:43, 8440.59it/s]  9%|▉         | 37037/400000 [00:04<00:42, 8483.74it/s]  9%|▉         | 37902/400000 [00:04<00:42, 8531.08it/s] 10%|▉         | 38756/400000 [00:04<00:42, 8470.32it/s] 10%|▉         | 39620/400000 [00:04<00:42, 8520.35it/s] 10%|█         | 40486/400000 [00:05<00:42, 8559.72it/s] 10%|█         | 41350/400000 [00:05<00:41, 8581.05it/s] 11%|█         | 42209/400000 [00:05<00:41, 8570.54it/s] 11%|█         | 43067/400000 [00:05<00:41, 8505.20it/s] 11%|█         | 43930/400000 [00:05<00:41, 8539.96it/s] 11%|█         | 44792/400000 [00:05<00:41, 8561.28it/s] 11%|█▏        | 45649/400000 [00:05<00:41, 8509.52it/s] 12%|█▏        | 46512/400000 [00:05<00:41, 8543.84it/s] 12%|█▏        | 47367/400000 [00:05<00:41, 8488.66it/s] 12%|█▏        | 48217/400000 [00:05<00:41, 8477.37it/s] 12%|█▏        | 49074/400000 [00:06<00:41, 8502.41it/s] 12%|█▏        | 49925/400000 [00:06<00:41, 8468.75it/s] 13%|█▎        | 50781/400000 [00:06<00:41, 8495.19it/s] 13%|█▎        | 51631/400000 [00:06<00:41, 8427.75it/s] 13%|█▎        | 52491/400000 [00:06<00:40, 8476.55it/s] 13%|█▎        | 53344/400000 [00:06<00:40, 8491.20it/s] 14%|█▎        | 54194/400000 [00:06<00:40, 8485.28it/s] 14%|█▍        | 55058/400000 [00:06<00:40, 8529.60it/s] 14%|█▍        | 55912/400000 [00:06<00:40, 8456.77it/s] 14%|█▍        | 56764/400000 [00:06<00:40, 8474.87it/s] 14%|█▍        | 57634/400000 [00:07<00:40, 8539.91it/s] 15%|█▍        | 58502/400000 [00:07<00:39, 8579.60it/s] 15%|█▍        | 59369/400000 [00:07<00:39, 8605.25it/s] 15%|█▌        | 60230/400000 [00:07<00:39, 8546.26it/s] 15%|█▌        | 61092/400000 [00:07<00:39, 8567.80it/s] 15%|█▌        | 61949/400000 [00:07<00:39, 8567.52it/s] 16%|█▌        | 62806/400000 [00:07<00:39, 8467.79it/s] 16%|█▌        | 63655/400000 [00:07<00:39, 8473.22it/s] 16%|█▌        | 64503/400000 [00:07<00:39, 8453.00it/s] 16%|█▋        | 65362/400000 [00:07<00:39, 8491.61it/s] 17%|█▋        | 66219/400000 [00:08<00:39, 8513.74it/s] 17%|█▋        | 67071/400000 [00:08<00:40, 8268.02it/s] 17%|█▋        | 67919/400000 [00:08<00:39, 8327.85it/s] 17%|█▋        | 68754/400000 [00:08<00:39, 8292.96it/s] 17%|█▋        | 69597/400000 [00:08<00:39, 8333.36it/s] 18%|█▊        | 70432/400000 [00:08<00:39, 8271.68it/s] 18%|█▊        | 71276/400000 [00:08<00:39, 8319.27it/s] 18%|█▊        | 72139/400000 [00:08<00:38, 8409.93it/s] 18%|█▊        | 72981/400000 [00:08<00:39, 8357.20it/s] 18%|█▊        | 73823/400000 [00:08<00:38, 8374.89it/s] 19%|█▊        | 74676/400000 [00:09<00:38, 8419.78it/s] 19%|█▉        | 75530/400000 [00:09<00:38, 8453.30it/s] 19%|█▉        | 76383/400000 [00:09<00:38, 8473.32it/s] 19%|█▉        | 77231/400000 [00:09<00:38, 8444.71it/s] 20%|█▉        | 78076/400000 [00:09<00:38, 8419.36it/s] 20%|█▉        | 78919/400000 [00:09<00:38, 8400.44it/s] 20%|█▉        | 79760/400000 [00:09<00:38, 8380.05it/s] 20%|██        | 80599/400000 [00:09<00:38, 8321.12it/s] 20%|██        | 81432/400000 [00:09<00:38, 8269.35it/s] 21%|██        | 82260/400000 [00:09<00:38, 8237.32it/s] 21%|██        | 83084/400000 [00:10<00:38, 8200.68it/s] 21%|██        | 83923/400000 [00:10<00:38, 8254.38it/s] 21%|██        | 84767/400000 [00:10<00:37, 8309.00it/s] 21%|██▏       | 85603/400000 [00:10<00:37, 8321.28it/s] 22%|██▏       | 86436/400000 [00:10<00:38, 8139.67it/s] 22%|██▏       | 87286/400000 [00:10<00:37, 8244.04it/s] 22%|██▏       | 88130/400000 [00:10<00:37, 8300.18it/s] 22%|██▏       | 88961/400000 [00:10<00:37, 8192.84it/s] 22%|██▏       | 89782/400000 [00:10<00:38, 8149.83it/s] 23%|██▎       | 90598/400000 [00:10<00:38, 8030.10it/s] 23%|██▎       | 91411/400000 [00:11<00:38, 8058.75it/s] 23%|██▎       | 92254/400000 [00:11<00:37, 8165.32it/s] 23%|██▎       | 93078/400000 [00:11<00:37, 8187.25it/s] 23%|██▎       | 93910/400000 [00:11<00:37, 8224.61it/s] 24%|██▎       | 94733/400000 [00:11<00:37, 8221.54it/s] 24%|██▍       | 95588/400000 [00:11<00:36, 8316.67it/s] 24%|██▍       | 96434/400000 [00:11<00:36, 8358.62it/s] 24%|██▍       | 97283/400000 [00:11<00:36, 8396.99it/s] 25%|██▍       | 98135/400000 [00:11<00:35, 8432.07it/s] 25%|██▍       | 98979/400000 [00:11<00:35, 8393.49it/s] 25%|██▍       | 99833/400000 [00:12<00:35, 8435.21it/s] 25%|██▌       | 100687/400000 [00:12<00:35, 8463.72it/s] 25%|██▌       | 101534/400000 [00:12<00:35, 8438.22it/s] 26%|██▌       | 102380/400000 [00:12<00:35, 8443.24it/s] 26%|██▌       | 103225/400000 [00:12<00:35, 8383.55it/s] 26%|██▌       | 104064/400000 [00:12<00:35, 8339.75it/s] 26%|██▌       | 104899/400000 [00:12<00:35, 8334.03it/s] 26%|██▋       | 105733/400000 [00:12<00:35, 8306.87it/s] 27%|██▋       | 106564/400000 [00:12<00:35, 8284.39it/s] 27%|██▋       | 107393/400000 [00:12<00:35, 8267.66it/s] 27%|██▋       | 108220/400000 [00:13<00:36, 8091.85it/s] 27%|██▋       | 109067/400000 [00:13<00:35, 8200.74it/s] 27%|██▋       | 109903/400000 [00:13<00:35, 8247.34it/s] 28%|██▊       | 110755/400000 [00:13<00:34, 8325.22it/s] 28%|██▊       | 111610/400000 [00:13<00:34, 8389.27it/s] 28%|██▊       | 112450/400000 [00:13<00:34, 8324.55it/s] 28%|██▊       | 113283/400000 [00:13<00:34, 8317.90it/s] 29%|██▊       | 114132/400000 [00:13<00:34, 8367.39it/s] 29%|██▊       | 114989/400000 [00:13<00:33, 8426.39it/s] 29%|██▉       | 115832/400000 [00:14<00:33, 8426.44it/s] 29%|██▉       | 116677/400000 [00:14<00:33, 8432.77it/s] 29%|██▉       | 117541/400000 [00:14<00:33, 8493.34it/s] 30%|██▉       | 118391/400000 [00:14<00:33, 8454.40it/s] 30%|██▉       | 119257/400000 [00:14<00:32, 8513.94it/s] 30%|███       | 120112/400000 [00:14<00:32, 8522.66it/s] 30%|███       | 120965/400000 [00:14<00:32, 8503.21it/s] 30%|███       | 121820/400000 [00:14<00:32, 8516.96it/s] 31%|███       | 122679/400000 [00:14<00:32, 8536.39it/s] 31%|███       | 123533/400000 [00:14<00:32, 8514.72it/s] 31%|███       | 124395/400000 [00:15<00:32, 8543.73it/s] 31%|███▏      | 125250/400000 [00:15<00:32, 8467.99it/s] 32%|███▏      | 126103/400000 [00:15<00:32, 8484.09it/s] 32%|███▏      | 126952/400000 [00:15<00:32, 8474.09it/s] 32%|███▏      | 127809/400000 [00:15<00:32, 8499.73it/s] 32%|███▏      | 128667/400000 [00:15<00:31, 8522.72it/s] 32%|███▏      | 129520/400000 [00:15<00:32, 8417.54it/s] 33%|███▎      | 130382/400000 [00:15<00:31, 8476.06it/s] 33%|███▎      | 131230/400000 [00:15<00:32, 8276.30it/s] 33%|███▎      | 132089/400000 [00:15<00:32, 8365.39it/s] 33%|███▎      | 132947/400000 [00:16<00:31, 8426.94it/s] 33%|███▎      | 133791/400000 [00:16<00:31, 8409.64it/s] 34%|███▎      | 134650/400000 [00:16<00:31, 8461.55it/s] 34%|███▍      | 135510/400000 [00:16<00:31, 8502.57it/s] 34%|███▍      | 136371/400000 [00:16<00:30, 8533.74it/s] 34%|███▍      | 137225/400000 [00:16<00:30, 8531.04it/s] 35%|███▍      | 138079/400000 [00:16<00:30, 8456.54it/s] 35%|███▍      | 138935/400000 [00:16<00:30, 8484.81it/s] 35%|███▍      | 139784/400000 [00:16<00:30, 8484.11it/s] 35%|███▌      | 140636/400000 [00:16<00:30, 8492.44it/s] 35%|███▌      | 141488/400000 [00:17<00:30, 8499.06it/s] 36%|███▌      | 142338/400000 [00:17<00:30, 8408.70it/s] 36%|███▌      | 143196/400000 [00:17<00:30, 8458.15it/s] 36%|███▌      | 144046/400000 [00:17<00:30, 8469.39it/s] 36%|███▌      | 144894/400000 [00:17<00:30, 8472.10it/s] 36%|███▋      | 145742/400000 [00:17<00:30, 8473.28it/s] 37%|███▋      | 146590/400000 [00:17<00:30, 8373.46it/s] 37%|███▋      | 147436/400000 [00:17<00:30, 8398.79it/s] 37%|███▋      | 148288/400000 [00:17<00:29, 8433.41it/s] 37%|███▋      | 149150/400000 [00:17<00:29, 8486.31it/s] 38%|███▊      | 150000/400000 [00:18<00:29, 8490.31it/s] 38%|███▊      | 150850/400000 [00:18<00:29, 8446.13it/s] 38%|███▊      | 151710/400000 [00:18<00:29, 8490.14it/s] 38%|███▊      | 152566/400000 [00:18<00:29, 8510.37it/s] 38%|███▊      | 153424/400000 [00:18<00:28, 8530.78it/s] 39%|███▊      | 154286/400000 [00:18<00:28, 8556.35it/s] 39%|███▉      | 155142/400000 [00:18<00:29, 8412.56it/s] 39%|███▉      | 156003/400000 [00:18<00:28, 8469.67it/s] 39%|███▉      | 156859/400000 [00:18<00:28, 8496.14it/s] 39%|███▉      | 157710/400000 [00:18<00:28, 8498.62it/s] 40%|███▉      | 158565/400000 [00:19<00:28, 8511.32it/s] 40%|███▉      | 159417/400000 [00:19<00:28, 8456.90it/s] 40%|████      | 160274/400000 [00:19<00:28, 8489.06it/s] 40%|████      | 161127/400000 [00:19<00:28, 8498.21it/s] 40%|████      | 161980/400000 [00:19<00:27, 8506.13it/s] 41%|████      | 162844/400000 [00:19<00:27, 8544.55it/s] 41%|████      | 163699/400000 [00:19<00:27, 8477.58it/s] 41%|████      | 164547/400000 [00:19<00:27, 8463.66it/s] 41%|████▏     | 165397/400000 [00:19<00:27, 8473.62it/s] 42%|████▏     | 166253/400000 [00:19<00:27, 8497.09it/s] 42%|████▏     | 167108/400000 [00:20<00:27, 8510.60it/s] 42%|████▏     | 167960/400000 [00:20<00:27, 8435.43it/s] 42%|████▏     | 168804/400000 [00:20<00:27, 8411.32it/s] 42%|████▏     | 169659/400000 [00:20<00:27, 8450.17it/s] 43%|████▎     | 170520/400000 [00:20<00:27, 8496.24it/s] 43%|████▎     | 171383/400000 [00:20<00:26, 8533.18it/s] 43%|████▎     | 172237/400000 [00:20<00:26, 8449.97it/s] 43%|████▎     | 173083/400000 [00:20<00:27, 8274.99it/s] 43%|████▎     | 173912/400000 [00:20<00:27, 8214.69it/s] 44%|████▎     | 174775/400000 [00:20<00:27, 8334.14it/s] 44%|████▍     | 175633/400000 [00:21<00:26, 8404.31it/s] 44%|████▍     | 176475/400000 [00:21<00:26, 8321.71it/s] 44%|████▍     | 177327/400000 [00:21<00:26, 8377.46it/s] 45%|████▍     | 178184/400000 [00:21<00:26, 8432.37it/s] 45%|████▍     | 179047/400000 [00:21<00:26, 8490.69it/s] 45%|████▍     | 179910/400000 [00:21<00:25, 8531.15it/s] 45%|████▌     | 180764/400000 [00:21<00:25, 8506.29it/s] 45%|████▌     | 181628/400000 [00:21<00:25, 8545.49it/s] 46%|████▌     | 182483/400000 [00:21<00:25, 8369.09it/s] 46%|████▌     | 183347/400000 [00:21<00:25, 8446.95it/s] 46%|████▌     | 184211/400000 [00:22<00:25, 8503.30it/s] 46%|████▋     | 185063/400000 [00:22<00:25, 8276.21it/s] 46%|████▋     | 185926/400000 [00:22<00:25, 8377.49it/s] 47%|████▋     | 186784/400000 [00:22<00:25, 8437.15it/s] 47%|████▋     | 187642/400000 [00:22<00:25, 8478.24it/s] 47%|████▋     | 188505/400000 [00:22<00:24, 8520.59it/s] 47%|████▋     | 189358/400000 [00:22<00:24, 8467.05it/s] 48%|████▊     | 190211/400000 [00:22<00:24, 8485.08it/s] 48%|████▊     | 191060/400000 [00:22<00:24, 8472.17it/s] 48%|████▊     | 191924/400000 [00:22<00:24, 8520.32it/s] 48%|████▊     | 192788/400000 [00:23<00:24, 8553.75it/s] 48%|████▊     | 193644/400000 [00:23<00:24, 8505.88it/s] 49%|████▊     | 194505/400000 [00:23<00:24, 8535.13it/s] 49%|████▉     | 195359/400000 [00:23<00:24, 8521.24it/s] 49%|████▉     | 196220/400000 [00:23<00:23, 8547.04it/s] 49%|████▉     | 197081/400000 [00:23<00:23, 8564.03it/s] 49%|████▉     | 197938/400000 [00:23<00:23, 8485.16it/s] 50%|████▉     | 198791/400000 [00:23<00:23, 8495.97it/s] 50%|████▉     | 199649/400000 [00:23<00:23, 8519.75it/s] 50%|█████     | 200511/400000 [00:23<00:23, 8548.93it/s] 50%|█████     | 201367/400000 [00:24<00:23, 8531.65it/s] 51%|█████     | 202221/400000 [00:24<00:23, 8487.06it/s] 51%|█████     | 203078/400000 [00:24<00:23, 8508.58it/s] 51%|█████     | 203935/400000 [00:24<00:22, 8524.96it/s] 51%|█████     | 204791/400000 [00:24<00:22, 8532.59it/s] 51%|█████▏    | 205659/400000 [00:24<00:22, 8575.39it/s] 52%|█████▏    | 206517/400000 [00:24<00:22, 8503.04it/s] 52%|█████▏    | 207376/400000 [00:24<00:22, 8527.33it/s] 52%|█████▏    | 208229/400000 [00:24<00:22, 8524.57it/s] 52%|█████▏    | 209082/400000 [00:25<00:22, 8489.73it/s] 52%|█████▏    | 209942/400000 [00:25<00:22, 8521.72it/s] 53%|█████▎    | 210795/400000 [00:25<00:22, 8359.55it/s] 53%|█████▎    | 211655/400000 [00:25<00:22, 8429.68it/s] 53%|█████▎    | 212511/400000 [00:25<00:22, 8466.72it/s] 53%|█████▎    | 213362/400000 [00:25<00:22, 8477.79it/s] 54%|█████▎    | 214218/400000 [00:25<00:21, 8500.78it/s] 54%|█████▍    | 215069/400000 [00:25<00:21, 8457.62it/s] 54%|█████▍    | 215922/400000 [00:25<00:21, 8476.30it/s] 54%|█████▍    | 216781/400000 [00:25<00:21, 8509.80it/s] 54%|█████▍    | 217633/400000 [00:26<00:21, 8508.63it/s] 55%|█████▍    | 218491/400000 [00:26<00:21, 8529.81it/s] 55%|█████▍    | 219345/400000 [00:26<00:21, 8452.25it/s] 55%|█████▌    | 220213/400000 [00:26<00:21, 8518.82it/s] 55%|█████▌    | 221066/400000 [00:26<00:21, 8480.68it/s] 55%|█████▌    | 221922/400000 [00:26<00:20, 8503.92it/s] 56%|█████▌    | 222781/400000 [00:26<00:20, 8529.04it/s] 56%|█████▌    | 223635/400000 [00:26<00:20, 8499.25it/s] 56%|█████▌    | 224497/400000 [00:26<00:20, 8535.04it/s] 56%|█████▋    | 225363/400000 [00:26<00:20, 8571.17it/s] 57%|█████▋    | 226221/400000 [00:27<00:20, 8531.18it/s] 57%|█████▋    | 227085/400000 [00:27<00:20, 8562.79it/s] 57%|█████▋    | 227942/400000 [00:27<00:20, 8497.17it/s] 57%|█████▋    | 228811/400000 [00:27<00:20, 8553.51it/s] 57%|█████▋    | 229671/400000 [00:27<00:19, 8566.68it/s] 58%|█████▊    | 230529/400000 [00:27<00:19, 8569.81it/s] 58%|█████▊    | 231387/400000 [00:27<00:20, 8298.88it/s] 58%|█████▊    | 232219/400000 [00:27<00:20, 8305.15it/s] 58%|█████▊    | 233071/400000 [00:27<00:19, 8366.70it/s] 58%|█████▊    | 233926/400000 [00:27<00:19, 8420.39it/s] 59%|█████▊    | 234784/400000 [00:28<00:19, 8466.19it/s] 59%|█████▉    | 235646/400000 [00:28<00:19, 8509.98it/s] 59%|█████▉    | 236498/400000 [00:28<00:19, 8450.57it/s] 59%|█████▉    | 237359/400000 [00:28<00:19, 8496.62it/s] 60%|█████▉    | 238220/400000 [00:28<00:18, 8527.62it/s] 60%|█████▉    | 239077/400000 [00:28<00:18, 8537.67it/s] 60%|█████▉    | 239938/400000 [00:28<00:18, 8558.95it/s] 60%|██████    | 240795/400000 [00:28<00:18, 8410.65it/s] 60%|██████    | 241658/400000 [00:28<00:18, 8474.70it/s] 61%|██████    | 242522/400000 [00:28<00:18, 8520.91it/s] 61%|██████    | 243375/400000 [00:29<00:18, 8509.42it/s] 61%|██████    | 244231/400000 [00:29<00:18, 8522.22it/s] 61%|██████▏   | 245084/400000 [00:29<00:18, 8410.83it/s] 61%|██████▏   | 245938/400000 [00:29<00:18, 8446.59it/s] 62%|██████▏   | 246794/400000 [00:29<00:18, 8480.28it/s] 62%|██████▏   | 247649/400000 [00:29<00:17, 8498.64it/s] 62%|██████▏   | 248511/400000 [00:29<00:17, 8534.01it/s] 62%|██████▏   | 249365/400000 [00:29<00:17, 8494.88it/s] 63%|██████▎   | 250215/400000 [00:29<00:17, 8487.72it/s] 63%|██████▎   | 251078/400000 [00:29<00:17, 8528.71it/s] 63%|██████▎   | 251935/400000 [00:30<00:17, 8537.93it/s] 63%|██████▎   | 252789/400000 [00:30<00:17, 8376.62it/s] 63%|██████▎   | 253628/400000 [00:30<00:17, 8367.13it/s] 64%|██████▎   | 254487/400000 [00:30<00:17, 8430.28it/s] 64%|██████▍   | 255348/400000 [00:30<00:17, 8481.08it/s] 64%|██████▍   | 256198/400000 [00:30<00:16, 8485.17it/s] 64%|██████▍   | 257059/400000 [00:30<00:16, 8521.98it/s] 64%|██████▍   | 257912/400000 [00:30<00:16, 8465.71it/s] 65%|██████▍   | 258773/400000 [00:30<00:16, 8507.77it/s] 65%|██████▍   | 259637/400000 [00:30<00:16, 8545.24it/s] 65%|██████▌   | 260492/400000 [00:31<00:16, 8524.08it/s] 65%|██████▌   | 261357/400000 [00:31<00:16, 8560.62it/s] 66%|██████▌   | 262214/400000 [00:31<00:16, 8487.87it/s] 66%|██████▌   | 263080/400000 [00:31<00:16, 8536.00it/s] 66%|██████▌   | 263942/400000 [00:31<00:15, 8559.84it/s] 66%|██████▌   | 264806/400000 [00:31<00:15, 8583.47it/s] 66%|██████▋   | 265665/400000 [00:31<00:15, 8511.30it/s] 67%|██████▋   | 266517/400000 [00:31<00:15, 8472.25it/s] 67%|██████▋   | 267383/400000 [00:31<00:15, 8525.47it/s] 67%|██████▋   | 268245/400000 [00:31<00:15, 8551.88it/s] 67%|██████▋   | 269101/400000 [00:32<00:15, 8407.10it/s] 67%|██████▋   | 269954/400000 [00:32<00:15, 8443.44it/s] 68%|██████▊   | 270799/400000 [00:32<00:15, 8421.70it/s] 68%|██████▊   | 271661/400000 [00:32<00:15, 8478.16it/s] 68%|██████▊   | 272526/400000 [00:32<00:14, 8527.46it/s] 68%|██████▊   | 273380/400000 [00:32<00:14, 8482.40it/s] 69%|██████▊   | 274229/400000 [00:32<00:14, 8428.01it/s] 69%|██████▉   | 275073/400000 [00:32<00:14, 8356.88it/s] 69%|██████▉   | 275925/400000 [00:32<00:14, 8402.63it/s] 69%|██████▉   | 276785/400000 [00:32<00:14, 8460.83it/s] 69%|██████▉   | 277632/400000 [00:33<00:14, 8445.30it/s] 70%|██████▉   | 278491/400000 [00:33<00:14, 8486.77it/s] 70%|██████▉   | 279340/400000 [00:33<00:14, 8349.88it/s] 70%|███████   | 280199/400000 [00:33<00:14, 8419.71it/s] 70%|███████   | 281048/400000 [00:33<00:14, 8439.85it/s] 70%|███████   | 281910/400000 [00:33<00:13, 8490.60it/s] 71%|███████   | 282770/400000 [00:33<00:13, 8521.65it/s] 71%|███████   | 283623/400000 [00:33<00:13, 8482.82it/s] 71%|███████   | 284483/400000 [00:33<00:13, 8517.63it/s] 71%|███████▏  | 285340/400000 [00:33<00:13, 8531.07it/s] 72%|███████▏  | 286194/400000 [00:34<00:13, 8501.60it/s] 72%|███████▏  | 287057/400000 [00:34<00:13, 8537.68it/s] 72%|███████▏  | 287911/400000 [00:34<00:13, 8508.36it/s] 72%|███████▏  | 288762/400000 [00:34<00:13, 8492.20it/s] 72%|███████▏  | 289612/400000 [00:34<00:13, 8401.67it/s] 73%|███████▎  | 290471/400000 [00:34<00:12, 8456.76it/s] 73%|███████▎  | 291332/400000 [00:34<00:12, 8501.69it/s] 73%|███████▎  | 292183/400000 [00:34<00:12, 8455.18it/s] 73%|███████▎  | 293038/400000 [00:34<00:12, 8482.12it/s] 73%|███████▎  | 293892/400000 [00:35<00:12, 8497.07it/s] 74%|███████▎  | 294742/400000 [00:35<00:12, 8468.99it/s] 74%|███████▍  | 295590/400000 [00:35<00:12, 8463.25it/s] 74%|███████▍  | 296437/400000 [00:35<00:12, 8433.71it/s] 74%|███████▍  | 297291/400000 [00:35<00:12, 8463.42it/s] 75%|███████▍  | 298148/400000 [00:35<00:11, 8493.35it/s] 75%|███████▍  | 298998/400000 [00:35<00:12, 8362.64it/s] 75%|███████▍  | 299861/400000 [00:35<00:11, 8439.24it/s] 75%|███████▌  | 300706/400000 [00:35<00:11, 8426.51it/s] 75%|███████▌  | 301566/400000 [00:35<00:11, 8476.40it/s] 76%|███████▌  | 302428/400000 [00:36<00:11, 8518.87it/s] 76%|███████▌  | 303281/400000 [00:36<00:11, 8503.77it/s] 76%|███████▌  | 304139/400000 [00:36<00:11, 8524.00it/s] 76%|███████▌  | 304992/400000 [00:36<00:11, 8496.38it/s] 76%|███████▋  | 305855/400000 [00:36<00:11, 8533.48it/s] 77%|███████▋  | 306717/400000 [00:36<00:10, 8557.94it/s] 77%|███████▋  | 307573/400000 [00:36<00:10, 8537.36it/s] 77%|███████▋  | 308430/400000 [00:36<00:10, 8546.87it/s] 77%|███████▋  | 309285/400000 [00:36<00:10, 8417.89it/s] 78%|███████▊  | 310142/400000 [00:36<00:10, 8461.23it/s] 78%|███████▊  | 311004/400000 [00:37<00:10, 8507.15it/s] 78%|███████▊  | 311856/400000 [00:37<00:10, 8491.58it/s] 78%|███████▊  | 312717/400000 [00:37<00:10, 8526.05it/s] 78%|███████▊  | 313570/400000 [00:37<00:10, 8346.31it/s] 79%|███████▊  | 314429/400000 [00:37<00:10, 8415.94it/s] 79%|███████▉  | 315293/400000 [00:37<00:09, 8481.39it/s] 79%|███████▉  | 316142/400000 [00:37<00:09, 8476.86it/s] 79%|███████▉  | 317005/400000 [00:37<00:09, 8521.53it/s] 79%|███████▉  | 317858/400000 [00:37<00:09, 8441.90it/s] 80%|███████▉  | 318715/400000 [00:37<00:09, 8478.98it/s] 80%|███████▉  | 319573/400000 [00:38<00:09, 8508.17it/s] 80%|████████  | 320430/400000 [00:38<00:09, 8523.87it/s] 80%|████████  | 321293/400000 [00:38<00:09, 8554.58it/s] 81%|████████  | 322149/400000 [00:38<00:09, 8448.89it/s] 81%|████████  | 323007/400000 [00:38<00:09, 8485.88it/s] 81%|████████  | 323863/400000 [00:38<00:08, 8505.93it/s] 81%|████████  | 324723/400000 [00:38<00:08, 8531.28it/s] 81%|████████▏ | 325583/400000 [00:38<00:08, 8549.07it/s] 82%|████████▏ | 326439/400000 [00:38<00:08, 8502.92it/s] 82%|████████▏ | 327301/400000 [00:38<00:08, 8535.75it/s] 82%|████████▏ | 328161/400000 [00:39<00:08, 8552.79it/s] 82%|████████▏ | 329023/400000 [00:39<00:08, 8571.41it/s] 82%|████████▏ | 329881/400000 [00:39<00:08, 8569.68it/s] 83%|████████▎ | 330739/400000 [00:39<00:08, 8529.24it/s] 83%|████████▎ | 331598/400000 [00:39<00:08, 8546.07it/s] 83%|████████▎ | 332453/400000 [00:39<00:08, 8437.40it/s] 83%|████████▎ | 333298/400000 [00:39<00:08, 8064.26it/s] 84%|████████▎ | 334151/400000 [00:39<00:08, 8196.36it/s] 84%|████████▎ | 334997/400000 [00:39<00:07, 8271.33it/s] 84%|████████▍ | 335853/400000 [00:39<00:07, 8355.71it/s] 84%|████████▍ | 336708/400000 [00:40<00:07, 8412.19it/s] 84%|████████▍ | 337560/400000 [00:40<00:07, 8441.81it/s] 85%|████████▍ | 338406/400000 [00:40<00:07, 8406.42it/s] 85%|████████▍ | 339255/400000 [00:40<00:07, 8430.97it/s] 85%|████████▌ | 340115/400000 [00:40<00:07, 8478.62it/s] 85%|████████▌ | 340970/400000 [00:40<00:06, 8499.12it/s] 85%|████████▌ | 341821/400000 [00:40<00:06, 8414.39it/s] 86%|████████▌ | 342663/400000 [00:40<00:06, 8332.53it/s] 86%|████████▌ | 343497/400000 [00:40<00:06, 8318.91it/s] 86%|████████▌ | 344331/400000 [00:40<00:06, 8322.39it/s] 86%|████████▋ | 345182/400000 [00:41<00:06, 8376.44it/s] 87%|████████▋ | 346036/400000 [00:41<00:06, 8423.00it/s] 87%|████████▋ | 346888/400000 [00:41<00:06, 8450.55it/s] 87%|████████▋ | 347734/400000 [00:41<00:06, 8402.63it/s] 87%|████████▋ | 348592/400000 [00:41<00:06, 8452.87it/s] 87%|████████▋ | 349438/400000 [00:41<00:06, 8303.57it/s] 88%|████████▊ | 350290/400000 [00:41<00:05, 8365.86it/s] 88%|████████▊ | 351149/400000 [00:41<00:05, 8429.76it/s] 88%|████████▊ | 351993/400000 [00:41<00:05, 8386.02it/s] 88%|████████▊ | 352848/400000 [00:41<00:05, 8433.97it/s] 88%|████████▊ | 353708/400000 [00:42<00:05, 8480.33it/s] 89%|████████▊ | 354557/400000 [00:42<00:05, 8192.57it/s] 89%|████████▉ | 355409/400000 [00:42<00:05, 8286.88it/s] 89%|████████▉ | 356249/400000 [00:42<00:05, 8319.09it/s] 89%|████████▉ | 357110/400000 [00:42<00:05, 8403.82it/s] 89%|████████▉ | 357968/400000 [00:42<00:04, 8454.16it/s] 90%|████████▉ | 358827/400000 [00:42<00:04, 8492.85it/s] 90%|████████▉ | 359689/400000 [00:42<00:04, 8527.88it/s] 90%|█████████ | 360543/400000 [00:42<00:04, 8480.89it/s] 90%|█████████ | 361404/400000 [00:42<00:04, 8518.01it/s] 91%|█████████ | 362265/400000 [00:43<00:04, 8543.65it/s] 91%|█████████ | 363124/400000 [00:43<00:04, 8556.82it/s] 91%|█████████ | 363980/400000 [00:43<00:04, 8550.87it/s] 91%|█████████ | 364836/400000 [00:43<00:04, 8407.69it/s] 91%|█████████▏| 365678/400000 [00:43<00:04, 8405.45it/s] 92%|█████████▏| 366526/400000 [00:43<00:03, 8425.83it/s] 92%|█████████▏| 367388/400000 [00:43<00:03, 8482.61it/s] 92%|█████████▏| 368237/400000 [00:43<00:03, 8323.69it/s] 92%|█████████▏| 369081/400000 [00:43<00:03, 8357.68it/s] 92%|█████████▏| 369945/400000 [00:44<00:03, 8439.19it/s] 93%|█████████▎| 370801/400000 [00:44<00:03, 8474.13it/s] 93%|█████████▎| 371656/400000 [00:44<00:03, 8495.34it/s] 93%|█████████▎| 372506/400000 [00:44<00:03, 8490.34it/s] 93%|█████████▎| 373356/400000 [00:44<00:03, 8492.99it/s] 94%|█████████▎| 374212/400000 [00:44<00:03, 8510.73it/s] 94%|█████████▍| 375064/400000 [00:44<00:02, 8387.82it/s] 94%|█████████▍| 375904/400000 [00:44<00:02, 8317.21it/s] 94%|█████████▍| 376737/400000 [00:44<00:02, 8298.00it/s] 94%|█████████▍| 377578/400000 [00:44<00:02, 8330.55it/s] 95%|█████████▍| 378438/400000 [00:45<00:02, 8408.46it/s] 95%|█████████▍| 379296/400000 [00:45<00:02, 8459.00it/s] 95%|█████████▌| 380152/400000 [00:45<00:02, 8488.79it/s] 95%|█████████▌| 381004/400000 [00:45<00:02, 8495.46it/s] 95%|█████████▌| 381854/400000 [00:45<00:02, 8447.73it/s] 96%|█████████▌| 382709/400000 [00:45<00:02, 8475.98it/s] 96%|█████████▌| 383557/400000 [00:45<00:02, 8152.54it/s] 96%|█████████▌| 384410/400000 [00:45<00:01, 8260.02it/s] 96%|█████████▋| 385263/400000 [00:45<00:01, 8338.33it/s] 97%|█████████▋| 386110/400000 [00:45<00:01, 8375.03it/s] 97%|█████████▋| 386970/400000 [00:46<00:01, 8441.24it/s] 97%|█████████▋| 387829/400000 [00:46<00:01, 8483.71it/s] 97%|█████████▋| 388687/400000 [00:46<00:01, 8510.96it/s] 97%|█████████▋| 389539/400000 [00:46<00:01, 8469.09it/s] 98%|█████████▊| 390387/400000 [00:46<00:01, 8465.83it/s] 98%|█████████▊| 391242/400000 [00:46<00:01, 8488.22it/s] 98%|█████████▊| 392092/400000 [00:46<00:00, 8435.07it/s] 98%|█████████▊| 392950/400000 [00:46<00:00, 8476.78it/s] 98%|█████████▊| 393799/400000 [00:46<00:00, 8479.85it/s] 99%|█████████▊| 394649/400000 [00:46<00:00, 8485.01it/s] 99%|█████████▉| 395506/400000 [00:47<00:00, 8507.53it/s] 99%|█████████▉| 396367/400000 [00:47<00:00, 8535.85it/s] 99%|█████████▉| 397226/400000 [00:47<00:00, 8551.80it/s]100%|█████████▉| 398082/400000 [00:47<00:00, 8525.42it/s]100%|█████████▉| 398943/400000 [00:47<00:00, 8550.27it/s]100%|█████████▉| 399799/400000 [00:47<00:00, 8545.21it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8410.12it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f27af5210f0>, <torchtext.data.dataset.TabularDataset object at 0x7f27af521240>, <torchtext.vocab.Vocab object at 0x7f27af521160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.83 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.83 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.83 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.15 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.81 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.81 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.59 file/s]2020-06-12 06:20:44.827265: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-12 06:20:44.832085: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-12 06:20:44.832783: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e756631c20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 06:20:44.832901: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 468446.07it/s] 34%|███▍      | 3391488/9912422 [00:00<00:09, 665209.68it/s]9920512it [00:00, 946166.76it/s]                             9920512it [00:00, 32488751.16it/s]
0it [00:00, ?it/s]32768it [00:00, 741086.58it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483342.04it/s]1654784it [00:00, 12344933.28it/s]                         
0it [00:00, ?it/s]8192it [00:00, 237128.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
