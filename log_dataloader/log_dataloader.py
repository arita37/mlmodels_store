
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe7d1ced6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe7d1ced6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe83d2b50d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe83d2b50d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe856fe1ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe856fe1ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7eab32950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7eab32950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7eab32950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3686400/9912422 [00:00<00:00, 36579720.99it/s]9920512it [00:00, 35040820.81it/s]                             
0it [00:00, ?it/s]32768it [00:00, 621148.09it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 453348.53it/s]1654784it [00:00, 11947140.11it/s]                         
0it [00:00, ?it/s]8192it [00:00, 209256.68it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7e807bb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7e8081d68>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe7eab32598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe7eab32598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe7eab32598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:35,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:34,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:33,  1.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:06,  2.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:06,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:04,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:44,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:44,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:44,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:30,  4.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:30,  4.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:01<00:21,  6.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:15,  8.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:15,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:15,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:15,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 39.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 45.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 45.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 48.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 51.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 51.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 53.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 53.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 56.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 58.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 59.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 60.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 60.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 60.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 60.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 60.94 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 61.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 61.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 60.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 60.93 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.77s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 60.93 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.77s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.77s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.09 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.77s/ url]
0 examples [00:00, ? examples/s]2020-06-11 06:09:30.852741: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-11 06:09:30.865854: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-11 06:09:30.866024: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x565020708a90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-11 06:09:30.866043: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
10 examples [00:00, 99.99 examples/s]110 examples [00:00, 136.95 examples/s]205 examples [00:00, 184.22 examples/s]304 examples [00:00, 243.56 examples/s]405 examples [00:00, 315.13 examples/s]497 examples [00:00, 392.51 examples/s]599 examples [00:00, 481.31 examples/s]697 examples [00:00, 567.99 examples/s]796 examples [00:00, 650.68 examples/s]889 examples [00:01, 701.63 examples/s]989 examples [00:01, 769.03 examples/s]1089 examples [00:01, 825.97 examples/s]1185 examples [00:01, 861.32 examples/s]1283 examples [00:01, 892.73 examples/s]1380 examples [00:01, 914.47 examples/s]1480 examples [00:01, 938.34 examples/s]1583 examples [00:01, 961.89 examples/s]1685 examples [00:01, 976.86 examples/s]1787 examples [00:01, 987.78 examples/s]1888 examples [00:02, 970.95 examples/s]1987 examples [00:02, 959.31 examples/s]2084 examples [00:02, 951.05 examples/s]2180 examples [00:02, 945.69 examples/s]2277 examples [00:02, 952.00 examples/s]2373 examples [00:02, 953.69 examples/s]2469 examples [00:02, 950.99 examples/s]2565 examples [00:02, 947.99 examples/s]2660 examples [00:02, 942.96 examples/s]2758 examples [00:02, 952.23 examples/s]2854 examples [00:03, 921.60 examples/s]2956 examples [00:03, 947.88 examples/s]3055 examples [00:03, 957.86 examples/s]3152 examples [00:03, 940.89 examples/s]3253 examples [00:03, 959.28 examples/s]3351 examples [00:03, 963.41 examples/s]3448 examples [00:03, 952.78 examples/s]3544 examples [00:03, 936.23 examples/s]3638 examples [00:03, 935.64 examples/s]3736 examples [00:03, 946.78 examples/s]3831 examples [00:04, 941.76 examples/s]3928 examples [00:04, 949.01 examples/s]4026 examples [00:04, 956.69 examples/s]4124 examples [00:04, 963.26 examples/s]4225 examples [00:04, 974.64 examples/s]4325 examples [00:04, 980.75 examples/s]4427 examples [00:04, 989.47 examples/s]4527 examples [00:04, 989.99 examples/s]4627 examples [00:04, 989.88 examples/s]4729 examples [00:04, 998.53 examples/s]4829 examples [00:05, 995.29 examples/s]4932 examples [00:05, 1000.42 examples/s]5033 examples [00:05, 962.48 examples/s] 5131 examples [00:05, 966.01 examples/s]5231 examples [00:05, 973.75 examples/s]5332 examples [00:05, 982.56 examples/s]5431 examples [00:05, 967.75 examples/s]5528 examples [00:05, 942.14 examples/s]5627 examples [00:05, 953.44 examples/s]5723 examples [00:06, 939.10 examples/s]5818 examples [00:06, 911.60 examples/s]5917 examples [00:06, 933.44 examples/s]6016 examples [00:06, 948.10 examples/s]6115 examples [00:06, 958.48 examples/s]6217 examples [00:06, 974.24 examples/s]6319 examples [00:06, 986.18 examples/s]6422 examples [00:06, 998.41 examples/s]6524 examples [00:06, 1004.21 examples/s]6626 examples [00:06, 1006.21 examples/s]6728 examples [00:07, 1008.54 examples/s]6830 examples [00:07, 1011.20 examples/s]6933 examples [00:07, 1016.03 examples/s]7035 examples [00:07, 1013.09 examples/s]7137 examples [00:07, 1013.95 examples/s]7240 examples [00:07, 1016.31 examples/s]7343 examples [00:07, 1017.88 examples/s]7446 examples [00:07, 1020.62 examples/s]7549 examples [00:07, 1016.28 examples/s]7651 examples [00:07, 1016.30 examples/s]7753 examples [00:08, 988.69 examples/s] 7856 examples [00:08, 998.91 examples/s]7957 examples [00:08, 1000.02 examples/s]8058 examples [00:08, 983.45 examples/s] 8157 examples [00:08, 961.60 examples/s]8254 examples [00:08, 953.53 examples/s]8354 examples [00:08, 966.50 examples/s]8454 examples [00:08, 975.30 examples/s]8555 examples [00:08, 982.83 examples/s]8654 examples [00:08, 970.24 examples/s]8752 examples [00:09, 971.83 examples/s]8850 examples [00:09, 968.69 examples/s]8953 examples [00:09, 985.22 examples/s]9055 examples [00:09, 994.96 examples/s]9155 examples [00:09, 995.23 examples/s]9255 examples [00:09, 995.56 examples/s]9355 examples [00:09, 994.70 examples/s]9457 examples [00:09, 999.45 examples/s]9557 examples [00:09, 998.58 examples/s]9658 examples [00:09, 1000.61 examples/s]9759 examples [00:10, 992.21 examples/s] 9861 examples [00:10, 997.76 examples/s]9961 examples [00:10, 995.26 examples/s]10061 examples [00:10, 908.75 examples/s]10156 examples [00:10, 919.55 examples/s]10255 examples [00:10, 937.32 examples/s]10355 examples [00:10, 954.20 examples/s]10455 examples [00:10, 965.01 examples/s]10557 examples [00:10, 979.96 examples/s]10659 examples [00:11, 990.32 examples/s]10760 examples [00:11, 994.17 examples/s]10862 examples [00:11, 1001.75 examples/s]10964 examples [00:11, 1006.33 examples/s]11066 examples [00:11, 1009.15 examples/s]11168 examples [00:11, 985.13 examples/s] 11267 examples [00:11, 960.81 examples/s]11364 examples [00:11, 951.07 examples/s]11460 examples [00:11, 944.14 examples/s]11561 examples [00:11, 961.77 examples/s]11661 examples [00:12, 971.09 examples/s]11759 examples [00:12, 907.35 examples/s]11860 examples [00:12, 935.34 examples/s]11960 examples [00:12, 953.05 examples/s]12061 examples [00:12, 968.31 examples/s]12161 examples [00:12, 975.40 examples/s]12261 examples [00:12, 979.24 examples/s]12360 examples [00:12, 931.07 examples/s]12454 examples [00:12, 930.68 examples/s]12548 examples [00:13, 933.16 examples/s]12642 examples [00:13, 904.35 examples/s]12741 examples [00:13, 926.24 examples/s]12843 examples [00:13, 951.83 examples/s]12945 examples [00:13, 970.01 examples/s]13044 examples [00:13, 972.96 examples/s]13142 examples [00:13, 953.11 examples/s]13242 examples [00:13, 964.72 examples/s]13343 examples [00:13, 976.77 examples/s]13444 examples [00:13, 985.93 examples/s]13546 examples [00:14, 993.65 examples/s]13646 examples [00:14, 993.16 examples/s]13747 examples [00:14, 997.69 examples/s]13849 examples [00:14, 1002.25 examples/s]13952 examples [00:14, 1007.56 examples/s]14054 examples [00:14, 1009.96 examples/s]14156 examples [00:14, 992.13 examples/s] 14256 examples [00:14, 987.09 examples/s]14355 examples [00:14, 971.49 examples/s]14453 examples [00:14, 958.79 examples/s]14549 examples [00:15, 905.04 examples/s]14641 examples [00:15, 900.95 examples/s]14732 examples [00:15, 863.76 examples/s]14827 examples [00:15, 887.49 examples/s]14929 examples [00:15, 922.05 examples/s]15030 examples [00:15, 945.56 examples/s]15127 examples [00:15, 952.15 examples/s]15229 examples [00:15, 967.40 examples/s]15331 examples [00:15, 981.47 examples/s]15434 examples [00:15, 995.28 examples/s]15536 examples [00:16, 1001.03 examples/s]15637 examples [00:16, 979.64 examples/s] 15736 examples [00:16, 979.50 examples/s]15837 examples [00:16, 987.64 examples/s]15938 examples [00:16, 992.55 examples/s]16041 examples [00:16, 1001.44 examples/s]16142 examples [00:16, 989.14 examples/s] 16243 examples [00:16, 994.59 examples/s]16346 examples [00:16, 1003.22 examples/s]16449 examples [00:17, 1009.36 examples/s]16552 examples [00:17, 1012.96 examples/s]16654 examples [00:17, 1002.08 examples/s]16755 examples [00:17, 976.04 examples/s] 16857 examples [00:17, 987.15 examples/s]16956 examples [00:17, 962.89 examples/s]17053 examples [00:17, 956.98 examples/s]17149 examples [00:17, 950.86 examples/s]17245 examples [00:17, 937.78 examples/s]17339 examples [00:17, 937.67 examples/s]17433 examples [00:18, 937.57 examples/s]17530 examples [00:18, 944.48 examples/s]17626 examples [00:18, 946.41 examples/s]17722 examples [00:18, 947.93 examples/s]17822 examples [00:18, 961.74 examples/s]17920 examples [00:18, 966.02 examples/s]18020 examples [00:18, 973.38 examples/s]18119 examples [00:18, 977.19 examples/s]18219 examples [00:18, 982.77 examples/s]18319 examples [00:18, 985.76 examples/s]18420 examples [00:19, 992.08 examples/s]18520 examples [00:19, 993.62 examples/s]18620 examples [00:19, 992.04 examples/s]18720 examples [00:19, 958.75 examples/s]18817 examples [00:19, 933.30 examples/s]18914 examples [00:19, 941.42 examples/s]19010 examples [00:19, 944.22 examples/s]19105 examples [00:19, 945.64 examples/s]19203 examples [00:19, 954.76 examples/s]19299 examples [00:19, 929.55 examples/s]19398 examples [00:20, 945.90 examples/s]19499 examples [00:20, 962.37 examples/s]19599 examples [00:20, 971.62 examples/s]19698 examples [00:20, 975.61 examples/s]19800 examples [00:20, 987.60 examples/s]19899 examples [00:20, 973.92 examples/s]19997 examples [00:20, 962.78 examples/s]20094 examples [00:20, 894.10 examples/s]20194 examples [00:20, 922.94 examples/s]20295 examples [00:21, 945.23 examples/s]20396 examples [00:21, 962.78 examples/s]20494 examples [00:21, 966.70 examples/s]20592 examples [00:21, 935.25 examples/s]20687 examples [00:21, 935.65 examples/s]20786 examples [00:21, 951.09 examples/s]20888 examples [00:21, 970.42 examples/s]20990 examples [00:21, 984.04 examples/s]21090 examples [00:21, 988.36 examples/s]21192 examples [00:21, 997.13 examples/s]21294 examples [00:22, 1002.18 examples/s]21396 examples [00:22, 1006.63 examples/s]21497 examples [00:22, 1006.20 examples/s]21598 examples [00:22, 954.07 examples/s] 21699 examples [00:22, 967.46 examples/s]21797 examples [00:22, 955.88 examples/s]21893 examples [00:22, 949.08 examples/s]21989 examples [00:22, 935.20 examples/s]22085 examples [00:22, 941.62 examples/s]22180 examples [00:22, 929.56 examples/s]22281 examples [00:23, 951.59 examples/s]22382 examples [00:23, 965.72 examples/s]22481 examples [00:23, 970.84 examples/s]22580 examples [00:23, 976.09 examples/s]22683 examples [00:23, 990.08 examples/s]22785 examples [00:23, 995.35 examples/s]22887 examples [00:23, 1000.19 examples/s]22988 examples [00:23, 994.45 examples/s] 23088 examples [00:23, 994.43 examples/s]23190 examples [00:23, 1000.46 examples/s]23291 examples [00:24, 982.45 examples/s] 23390 examples [00:24, 974.55 examples/s]23488 examples [00:24, 953.35 examples/s]23584 examples [00:24, 939.87 examples/s]23686 examples [00:24, 961.62 examples/s]23787 examples [00:24, 975.23 examples/s]23885 examples [00:24, 935.52 examples/s]23980 examples [00:24, 912.30 examples/s]24072 examples [00:24, 910.26 examples/s]24164 examples [00:25, 902.04 examples/s]24261 examples [00:25, 920.54 examples/s]24359 examples [00:25, 934.59 examples/s]24456 examples [00:25, 944.91 examples/s]24555 examples [00:25, 955.65 examples/s]24651 examples [00:25, 943.08 examples/s]24747 examples [00:25, 945.69 examples/s]24847 examples [00:25, 958.95 examples/s]24944 examples [00:25, 962.07 examples/s]25044 examples [00:25, 971.03 examples/s]25144 examples [00:26, 977.41 examples/s]25244 examples [00:26, 981.26 examples/s]25344 examples [00:26, 984.67 examples/s]25443 examples [00:26, 983.44 examples/s]25542 examples [00:26, 985.25 examples/s]25641 examples [00:26, 986.16 examples/s]25741 examples [00:26, 988.38 examples/s]25840 examples [00:26, 987.66 examples/s]25939 examples [00:26, 948.99 examples/s]26037 examples [00:26, 955.18 examples/s]26135 examples [00:27, 960.59 examples/s]26232 examples [00:27, 901.23 examples/s]26328 examples [00:27, 915.51 examples/s]26421 examples [00:27, 917.33 examples/s]26523 examples [00:27, 945.35 examples/s]26622 examples [00:27, 957.30 examples/s]26719 examples [00:27, 940.86 examples/s]26814 examples [00:27, 937.75 examples/s]26911 examples [00:27, 946.41 examples/s]27011 examples [00:28, 961.15 examples/s]27111 examples [00:28, 970.08 examples/s]27211 examples [00:28, 976.19 examples/s]27311 examples [00:28, 981.19 examples/s]27410 examples [00:28, 980.87 examples/s]27511 examples [00:28, 988.57 examples/s]27613 examples [00:28, 995.04 examples/s]27714 examples [00:28, 997.54 examples/s]27814 examples [00:28, 985.28 examples/s]27913 examples [00:28, 981.38 examples/s]28015 examples [00:29, 990.20 examples/s]28116 examples [00:29, 995.00 examples/s]28216 examples [00:29, 990.55 examples/s]28317 examples [00:29, 994.93 examples/s]28417 examples [00:29, 989.70 examples/s]28516 examples [00:29, 954.79 examples/s]28612 examples [00:29, 949.72 examples/s]28708 examples [00:29, 946.83 examples/s]28803 examples [00:29, 945.53 examples/s]28898 examples [00:29, 941.48 examples/s]29001 examples [00:30, 963.90 examples/s]29103 examples [00:30, 978.66 examples/s]29204 examples [00:30, 985.43 examples/s]29303 examples [00:30, 973.78 examples/s]29401 examples [00:30, 960.73 examples/s]29502 examples [00:30, 972.76 examples/s]29603 examples [00:30, 983.06 examples/s]29702 examples [00:30, 960.23 examples/s]29800 examples [00:30, 964.58 examples/s]29898 examples [00:30, 968.33 examples/s]29998 examples [00:31, 976.92 examples/s]30096 examples [00:31, 938.01 examples/s]30197 examples [00:31, 956.39 examples/s]30297 examples [00:31, 966.78 examples/s]30396 examples [00:31, 973.41 examples/s]30496 examples [00:31, 978.66 examples/s]30596 examples [00:31, 984.56 examples/s]30697 examples [00:31, 990.86 examples/s]30797 examples [00:31, 939.50 examples/s]30892 examples [00:32, 916.79 examples/s]30985 examples [00:32, 919.91 examples/s]31078 examples [00:32, 920.98 examples/s]31171 examples [00:32, 923.23 examples/s]31264 examples [00:32, 921.88 examples/s]31362 examples [00:32, 937.09 examples/s]31463 examples [00:32, 956.12 examples/s]31563 examples [00:32, 967.28 examples/s]31665 examples [00:32, 980.92 examples/s]31766 examples [00:32, 988.09 examples/s]31865 examples [00:33, 953.02 examples/s]31961 examples [00:33, 941.69 examples/s]32056 examples [00:33, 912.04 examples/s]32149 examples [00:33, 915.30 examples/s]32243 examples [00:33, 921.75 examples/s]32337 examples [00:33, 925.37 examples/s]32439 examples [00:33, 949.38 examples/s]32535 examples [00:33, 948.02 examples/s]32630 examples [00:33, 931.92 examples/s]32729 examples [00:33, 947.14 examples/s]32827 examples [00:34, 956.27 examples/s]32926 examples [00:34, 965.45 examples/s]33023 examples [00:34, 962.13 examples/s]33120 examples [00:34, 924.73 examples/s]33219 examples [00:34, 941.79 examples/s]33319 examples [00:34, 956.55 examples/s]33420 examples [00:34, 969.15 examples/s]33522 examples [00:34, 981.13 examples/s]33621 examples [00:34, 980.58 examples/s]33723 examples [00:34, 990.15 examples/s]33823 examples [00:35, 982.91 examples/s]33922 examples [00:35, 984.00 examples/s]34021 examples [00:35, 978.39 examples/s]34119 examples [00:35, 971.57 examples/s]34220 examples [00:35, 981.51 examples/s]34319 examples [00:35, 971.34 examples/s]34420 examples [00:35, 980.00 examples/s]34519 examples [00:35, 982.95 examples/s]34618 examples [00:35, 981.42 examples/s]34717 examples [00:36, 962.45 examples/s]34814 examples [00:36, 945.59 examples/s]34909 examples [00:36, 930.77 examples/s]35010 examples [00:36, 951.66 examples/s]35111 examples [00:36, 965.88 examples/s]35212 examples [00:36, 977.32 examples/s]35310 examples [00:36, 972.23 examples/s]35408 examples [00:36, 945.92 examples/s]35507 examples [00:36, 956.43 examples/s]35603 examples [00:36, 947.20 examples/s]35702 examples [00:37, 959.56 examples/s]35799 examples [00:37, 959.92 examples/s]35898 examples [00:37, 967.42 examples/s]35995 examples [00:37, 964.51 examples/s]36095 examples [00:37, 974.20 examples/s]36193 examples [00:37, 964.92 examples/s]36290 examples [00:37, 962.60 examples/s]36390 examples [00:37, 972.81 examples/s]36491 examples [00:37, 981.98 examples/s]36594 examples [00:37, 993.60 examples/s]36697 examples [00:38, 1002.41 examples/s]36798 examples [00:38, 974.91 examples/s] 36896 examples [00:38, 945.20 examples/s]36991 examples [00:38, 944.47 examples/s]37093 examples [00:38, 963.98 examples/s]37193 examples [00:38, 971.57 examples/s]37291 examples [00:38, 973.27 examples/s]37389 examples [00:38, 957.78 examples/s]37485 examples [00:38, 947.79 examples/s]37585 examples [00:38, 960.77 examples/s]37682 examples [00:39, 943.77 examples/s]37777 examples [00:39, 926.20 examples/s]37878 examples [00:39, 947.95 examples/s]37978 examples [00:39, 961.98 examples/s]38080 examples [00:39, 976.45 examples/s]38178 examples [00:39, 945.35 examples/s]38273 examples [00:39, 931.40 examples/s]38367 examples [00:39, 918.07 examples/s]38460 examples [00:39, 898.52 examples/s]38560 examples [00:40, 926.56 examples/s]38656 examples [00:40, 933.83 examples/s]38758 examples [00:40, 957.80 examples/s]38855 examples [00:40, 958.84 examples/s]38952 examples [00:40, 956.54 examples/s]39055 examples [00:40, 975.43 examples/s]39156 examples [00:40, 984.24 examples/s]39259 examples [00:40, 996.90 examples/s]39360 examples [00:40, 997.58 examples/s]39460 examples [00:40, 992.70 examples/s]39560 examples [00:41, 989.49 examples/s]39660 examples [00:41, 977.81 examples/s]39758 examples [00:41, 968.73 examples/s]39855 examples [00:41, 950.41 examples/s]39955 examples [00:41, 962.39 examples/s]40052 examples [00:41, 918.83 examples/s]40152 examples [00:41, 941.09 examples/s]40255 examples [00:41, 963.97 examples/s]40357 examples [00:41, 979.04 examples/s]40459 examples [00:41, 990.12 examples/s]40561 examples [00:42, 997.16 examples/s]40661 examples [00:42, 997.67 examples/s]40763 examples [00:42, 1002.04 examples/s]40865 examples [00:42, 1005.34 examples/s]40967 examples [00:42, 1008.47 examples/s]41068 examples [00:42, 1004.43 examples/s]41169 examples [00:42, 995.87 examples/s] 41269 examples [00:42, 991.87 examples/s]41369 examples [00:42, 987.13 examples/s]41468 examples [00:43, 952.35 examples/s]41564 examples [00:43, 936.51 examples/s]41661 examples [00:43, 944.38 examples/s]41761 examples [00:43, 958.60 examples/s]41859 examples [00:43, 964.69 examples/s]41958 examples [00:43, 970.45 examples/s]42059 examples [00:43, 980.36 examples/s]42158 examples [00:43, 979.02 examples/s]42259 examples [00:43, 987.73 examples/s]42358 examples [00:43, 936.50 examples/s]42457 examples [00:44, 951.30 examples/s]42558 examples [00:44, 965.79 examples/s]42655 examples [00:44, 966.34 examples/s]42757 examples [00:44, 979.27 examples/s]42857 examples [00:44, 983.65 examples/s]42956 examples [00:44, 959.88 examples/s]43055 examples [00:44, 966.70 examples/s]43152 examples [00:44, 957.06 examples/s]43248 examples [00:44, 947.07 examples/s]43349 examples [00:44, 962.83 examples/s]43450 examples [00:45, 973.82 examples/s]43551 examples [00:45, 981.87 examples/s]43650 examples [00:45, 978.60 examples/s]43750 examples [00:45, 982.44 examples/s]43849 examples [00:45, 969.85 examples/s]43947 examples [00:45, 937.92 examples/s]44042 examples [00:45, 922.06 examples/s]44140 examples [00:45, 936.27 examples/s]44234 examples [00:45, 919.95 examples/s]44332 examples [00:45, 933.16 examples/s]44426 examples [00:46, 892.33 examples/s]44521 examples [00:46, 906.61 examples/s]44613 examples [00:46, 890.43 examples/s]44705 examples [00:46, 897.15 examples/s]44799 examples [00:46, 908.31 examples/s]44893 examples [00:46, 917.20 examples/s]44985 examples [00:46, 892.93 examples/s]45077 examples [00:46, 898.53 examples/s]45170 examples [00:46, 905.60 examples/s]45263 examples [00:47, 911.47 examples/s]45356 examples [00:47, 914.28 examples/s]45453 examples [00:47, 930.15 examples/s]45553 examples [00:47, 947.73 examples/s]45653 examples [00:47, 962.63 examples/s]45754 examples [00:47, 974.26 examples/s]45852 examples [00:47, 953.96 examples/s]45949 examples [00:47, 958.60 examples/s]46048 examples [00:47, 966.90 examples/s]46148 examples [00:47, 974.00 examples/s]46248 examples [00:48, 980.45 examples/s]46349 examples [00:48, 986.48 examples/s]46448 examples [00:48, 985.44 examples/s]46548 examples [00:48, 988.04 examples/s]46648 examples [00:48, 989.09 examples/s]46748 examples [00:48, 991.67 examples/s]46849 examples [00:48, 995.57 examples/s]46949 examples [00:48, 947.05 examples/s]47047 examples [00:48, 954.68 examples/s]47143 examples [00:48, 947.19 examples/s]47243 examples [00:49, 960.13 examples/s]47340 examples [00:49, 901.95 examples/s]47432 examples [00:49, 862.24 examples/s]47521 examples [00:49, 868.88 examples/s]47616 examples [00:49, 890.05 examples/s]47713 examples [00:49, 911.18 examples/s]47806 examples [00:49, 914.01 examples/s]47898 examples [00:49, 913.03 examples/s]47992 examples [00:49, 919.16 examples/s]48085 examples [00:50, 920.63 examples/s]48186 examples [00:50, 944.50 examples/s]48288 examples [00:50, 964.30 examples/s]48385 examples [00:50, 960.48 examples/s]48483 examples [00:50, 965.21 examples/s]48580 examples [00:50, 952.86 examples/s]48676 examples [00:50, 928.46 examples/s]48770 examples [00:50, 900.24 examples/s]48868 examples [00:50, 920.27 examples/s]48967 examples [00:50, 938.74 examples/s]49068 examples [00:51, 957.24 examples/s]49167 examples [00:51, 965.96 examples/s]49264 examples [00:51, 930.77 examples/s]49364 examples [00:51, 949.14 examples/s]49465 examples [00:51, 964.51 examples/s]49565 examples [00:51, 973.28 examples/s]49666 examples [00:51, 981.84 examples/s]49766 examples [00:51, 984.72 examples/s]49865 examples [00:51, 978.56 examples/s]49963 examples [00:51, 978.62 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  9%|▊         | 4328/50000 [00:00<00:01, 43278.80 examples/s] 29%|██▉       | 14561/50000 [00:00<00:00, 52339.66 examples/s] 52%|█████▏    | 25762/50000 [00:00<00:00, 62281.28 examples/s] 75%|███████▌  | 37569/50000 [00:00<00:00, 72567.75 examples/s] 99%|█████████▊| 49275/50000 [00:00<00:00, 81905.53 examples/s]                                                               0 examples [00:00, ? examples/s]85 examples [00:00, 849.06 examples/s]188 examples [00:00, 894.62 examples/s]289 examples [00:00, 925.66 examples/s]390 examples [00:00, 947.19 examples/s]491 examples [00:00, 963.01 examples/s]593 examples [00:00, 977.51 examples/s]695 examples [00:00, 988.90 examples/s]795 examples [00:00, 991.14 examples/s]896 examples [00:00, 996.72 examples/s]997 examples [00:01, 997.98 examples/s]1098 examples [00:01, 1000.04 examples/s]1199 examples [00:01, 1001.79 examples/s]1301 examples [00:01, 1006.09 examples/s]1401 examples [00:01, 787.50 examples/s] 1502 examples [00:01, 842.91 examples/s]1604 examples [00:01, 887.32 examples/s]1704 examples [00:01, 916.76 examples/s]1804 examples [00:01, 938.30 examples/s]1905 examples [00:02, 956.36 examples/s]2007 examples [00:02, 972.64 examples/s]2108 examples [00:02, 982.24 examples/s]2208 examples [00:02, 974.64 examples/s]2308 examples [00:02, 980.00 examples/s]2408 examples [00:02, 984.77 examples/s]2508 examples [00:02, 986.44 examples/s]2607 examples [00:02, 945.13 examples/s]2703 examples [00:02, 907.67 examples/s]2795 examples [00:02, 902.25 examples/s]2894 examples [00:03, 924.57 examples/s]2993 examples [00:03, 942.39 examples/s]3096 examples [00:03, 965.03 examples/s]3193 examples [00:03, 949.57 examples/s]3291 examples [00:03, 957.63 examples/s]3394 examples [00:03, 976.44 examples/s]3494 examples [00:03, 982.74 examples/s]3593 examples [00:03, 959.89 examples/s]3690 examples [00:03, 957.81 examples/s]3786 examples [00:03, 953.98 examples/s]3882 examples [00:04, 942.97 examples/s]3980 examples [00:04, 953.48 examples/s]4083 examples [00:04, 974.44 examples/s]4181 examples [00:04, 955.05 examples/s]4280 examples [00:04, 961.90 examples/s]4377 examples [00:04, 945.16 examples/s]4476 examples [00:04, 955.48 examples/s]4574 examples [00:04, 960.18 examples/s]4671 examples [00:04, 956.14 examples/s]4770 examples [00:04, 965.17 examples/s]4868 examples [00:05, 968.24 examples/s]4967 examples [00:05, 973.99 examples/s]5068 examples [00:05, 982.48 examples/s]5167 examples [00:05, 983.93 examples/s]5266 examples [00:05, 970.83 examples/s]5365 examples [00:05, 974.48 examples/s]5463 examples [00:05, 954.91 examples/s]5559 examples [00:05, 923.68 examples/s]5652 examples [00:05, 919.22 examples/s]5748 examples [00:06, 928.20 examples/s]5844 examples [00:06, 936.51 examples/s]5945 examples [00:06, 956.26 examples/s]6048 examples [00:06, 974.53 examples/s]6146 examples [00:06, 975.48 examples/s]6246 examples [00:06, 982.60 examples/s]6348 examples [00:06, 991.07 examples/s]6448 examples [00:06, 980.54 examples/s]6547 examples [00:06, 964.96 examples/s]6645 examples [00:06, 969.20 examples/s]6745 examples [00:07, 976.00 examples/s]6845 examples [00:07, 980.34 examples/s]6944 examples [00:07, 962.14 examples/s]7046 examples [00:07, 977.98 examples/s]7146 examples [00:07, 982.58 examples/s]7245 examples [00:07, 981.12 examples/s]7346 examples [00:07, 988.39 examples/s]7446 examples [00:07, 989.93 examples/s]7546 examples [00:07, 987.47 examples/s]7645 examples [00:07, 980.17 examples/s]7744 examples [00:08, 981.51 examples/s]7846 examples [00:08, 990.33 examples/s]7948 examples [00:08, 997.72 examples/s]8048 examples [00:08, 988.26 examples/s]8147 examples [00:08, 987.27 examples/s]8248 examples [00:08, 992.90 examples/s]8350 examples [00:08, 1000.13 examples/s]8451 examples [00:08, 977.31 examples/s] 8549 examples [00:08, 939.43 examples/s]8651 examples [00:08, 960.56 examples/s]8753 examples [00:09, 976.79 examples/s]8852 examples [00:09, 972.56 examples/s]8950 examples [00:09, 966.86 examples/s]9049 examples [00:09, 972.06 examples/s]9150 examples [00:09, 980.72 examples/s]9249 examples [00:09, 982.31 examples/s]9348 examples [00:09, 980.17 examples/s]9447 examples [00:09, 979.52 examples/s]9547 examples [00:09, 985.06 examples/s]9647 examples [00:09, 986.72 examples/s]9749 examples [00:10, 995.95 examples/s]9850 examples [00:10, 1000.06 examples/s]9951 examples [00:10, 973.76 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF7XH7Z/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF7XH7Z/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7eab32950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7eab32950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7eab32950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe76ff1c0f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe83651d3c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe7eab32950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe7eab32950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe7eab32950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe76ff957f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe7e82ae0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe76c10b2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe76c10b2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fe76c10b2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe76c10b400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe76c10b400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe76c10b400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c696e60707fb1d8e966eb833640e9fb1b0580bcfd6f7a4ff50b869bb155a0cb1
  Stored in directory: /tmp/pip-ephem-wheel-cache-3zubejvw/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:03:39, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:16:31, 16.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:02:47, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:02:28, 34.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.44M/862M [00:01<4:55:01, 48.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.71M/862M [00:01<3:25:52, 69.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<2:23:23, 98.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:39:57, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.6M/862M [00:01<1:09:35, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.5M/862M [00:01<48:40, 287kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<33:56, 409kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.5M/862M [00:01<23:46, 581kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:02<16:37, 826kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.9M/862M [00:02<11:42, 1.17MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<08:13, 1.65MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<05:51, 2.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<04:57, 2.73MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<05:22, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<07:02, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<05:45, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:13, 3.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<11:22, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<09:35, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<07:41, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<08:37, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:43, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:09<05:35, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:09<05:16, 2.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:09<04:25, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<03:38, 3.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<14:32, 909kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<12:29, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:10<09:13, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:10<06:36, 1.99MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<12:34, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<13:04, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<10:04, 1.31MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:12<07:16, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<05:19, 2.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<21:42, 603kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<19:11, 682kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<14:27, 904kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:14<10:22, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<11:12, 1.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<11:53, 1.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<09:10, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:16<06:42, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<08:38, 1.50MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<10:00, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<07:49, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:18<05:44, 2.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<08:04, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<09:09, 1.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<07:19, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<05:21, 2.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<07:57, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<09:15, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<07:25, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:22<05:24, 2.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<07:51, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<09:09, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:18, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<05:20, 2.38MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:07, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:33, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:30, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:27, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:03, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<32:21, 390kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<26:03, 484kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<19:04, 660kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<13:31, 929kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:51, 904kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:55, 970kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<09:51, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<07:05, 1.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:28, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:12, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:03, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<05:50, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:02, 1.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<09:00, 1.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:09, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:50, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:01, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:04, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<05:11, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:35, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:38, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:47, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<04:58, 2.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:41, 1.58MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:31, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:43, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:55, 2.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:04, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:36, 1.41MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:41, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:54, 2.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:56, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:39, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:41, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:51, 2.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:31, 1.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:12, 1.46MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:30, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:46, 2.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:53, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:45, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:56, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:02, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:31, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:43, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:54, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:57, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:10, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<05:16, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:32, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:09, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:21, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:39, 2.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:41, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:57, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<07:01, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<05:05, 2.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<06:56, 1.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<07:50, 1.47MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:08, 1.88MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:27, 2.59MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<07:18, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:04, 1.42MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:22, 1.80MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:37, 2.47MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<07:50, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:25, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:36, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:49, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:44, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:19, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:32, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:44, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:42, 1.46MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:50, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:55, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:00, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:58, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:46, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:04, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:27, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:06, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:08, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:41, 1.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:10, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:14, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:56, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:07, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:29, 2.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:04, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:50, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:22, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:40, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:34, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:58, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:16, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:34, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:04, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:03, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:31, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:02, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:10, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:50, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:26, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:59, 2.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:26, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<07:41, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:55, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:18, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:51, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:24, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:44, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:11, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:58, 1.77MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:47, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:17, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:53, 2.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:46, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:37, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:14, 2.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<03:50, 2.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:37, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:53, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:07, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:27, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<08:19, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:22, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:36, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:48, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<07:35, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:59, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<04:22, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:03, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<07:09, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:29, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:59, 2.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<06:22, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:44, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:11, 1.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:45, 2.70MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<08:08, 1.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<08:11, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:16, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:31, 1.54MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:09, 1.40MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:32, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:03, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:32, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:50, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:34, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:19, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<28:00, 354kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<21:55, 451kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<15:49, 625kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<11:12, 880kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<11:06, 884kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<09:41, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:15, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:11, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<18:19, 533kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<15:07, 645kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<11:10, 873kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<07:54, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<11:20, 854kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<10:06, 959kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<07:31, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:25, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:48, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<07:01, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:22, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:57, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:09, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:45, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:29, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<03:16, 2.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:56, 1.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:21, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:54, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:36, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:09, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:12, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:58, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<02:53, 3.24MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<11:57, 782kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<10:28, 892kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<07:50, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<05:34, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<13:51, 669kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<11:36, 799kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<08:32, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<06:04, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:31, 1.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:55, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<06:02, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<04:18, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<22:33, 405kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<17:08, 533kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<12:18, 740kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<10:14, 885kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<10:34, 857kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<08:08, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<05:53, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:00, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:58, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:36, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:40, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:41, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:33, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:04, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:12, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:04, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:16, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:42, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:39, 2.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<02:40, 3.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:54, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:46, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:27, 1.95MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:32, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:51, 1.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:46, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<02:43, 3.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<13:59, 614kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:28, 748kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<08:26, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:16, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:45, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:08, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:57, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:08, 1.64MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:00, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:09, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:33, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:35, 2.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:52, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:20, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:26, 2.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:44, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:14, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:21, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:40, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:11, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:19, 2.45MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:38, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:04, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:14, 2.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:34, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:05, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:11, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:20, 3.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:19, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:17, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:05, 1.94MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<04:08, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:26, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:26, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:29, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:25, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:57, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:29, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:14, 2.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:15, 1.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:54, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:30, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:23, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:35, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:31, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:33, 2.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:49, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:34, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:12, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:01, 2.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:35, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:05, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:38, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:27, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:34, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:32, 2.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:33, 982kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:45, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<05:05, 1.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:44, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:45, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:41, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:45, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:03, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:11, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:24, 2.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:44, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:58, 2.42MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:14, 2.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:40, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:54, 2.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:11, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:12, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:26, 2.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:03, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:25, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:39, 2.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<01:56, 3.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:08, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:33, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:11, 1.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:04, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:06, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:10, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:17, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:58, 974kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:06, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:34, 1.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<03:15, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<16:51, 399kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<13:01, 516kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<09:24, 714kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<06:36, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<40:02, 166kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<29:13, 228kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<20:42, 321kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<14:27, 456kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<48:43, 135kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<35:14, 187kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<24:52, 264kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<17:21, 376kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<32:26, 201kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<23:52, 273kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<16:58, 384kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<12:53, 501kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<11:27, 564kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:35, 751kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<06:07, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:56, 1.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:16, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:58, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:49, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:49, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:06, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:18, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:35, 2.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:06, 1.99MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:26, 2.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:44, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:58, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:21, 2.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:39, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:57, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:20, 2.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:38, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:07, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:24, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:30, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:17, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:20, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:36, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:47, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:00, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:21, 2.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:36, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:52, 2.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:15, 2.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:32, 2.25MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:11, 2.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:35, 3.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:43, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:17, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:15, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:11, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:14, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:30, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:40, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:51, 1.92MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:14, 2.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:28, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:43, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:08, 2.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:23, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:36, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:03, 2.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:19, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:34, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:02, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:17, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:33, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:01, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:16, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:31, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:59, 2.57MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:14, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:28, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:57, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:12, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:26, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:56, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:10, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:25, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:52, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:22, 3.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:44, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:29, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:39, 1.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:39, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:41, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:05, 2.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:15, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:26, 1.95MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:52, 2.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:21, 3.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:39, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:06, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:04, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:54, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:52, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:12, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:17, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:23, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:52, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:03, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:15, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:46, 2.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:58, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:11, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:43, 2.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:55, 2.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:07, 2.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:40, 2.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:53, 2.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:04, 2.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:38, 2.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:50, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:02, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:36, 2.59MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:48, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:00, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:33, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:08, 3.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:35, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:56, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:07, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:52, 1.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:21, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:42, 2.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:15, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:15, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:44, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:51, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:57, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:30, 2.54MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:05, 3.47MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<03:45, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<03:17, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:27, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:19, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:17, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:45, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:50, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:36, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:11, 3.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:48, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:32, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:03, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:47, 1.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:51, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:27, 2.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:35, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:42, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:19, 2.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<00:57, 3.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:40, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<03:02, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:22, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:04, 1.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:01, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:33, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:37, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:42, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:19, 2.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:27, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:14, 2.54MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:22, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:29, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:10, 2.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:19, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:27, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:08, 2.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:49, 3.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<06:57, 424kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<05:22, 548kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<03:52, 756kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:09, 911kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:42, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<02:00, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:52, 1.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:22, 2.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:25, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:20, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:00, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:13, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:48, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:29, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:05, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:28, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:27, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:07, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:12, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:53, 2.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:06, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:38, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:21, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:59, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:20, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:20, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:02, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:06, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:10, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:54, 2.53MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:39, 3.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<05:21, 421kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:08, 546kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<02:57, 757kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<02:04, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:25, 904kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:03, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:31, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:24, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:21, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:01, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:03, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:05, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:50, 2.41MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:58, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:45, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:55, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:43, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:48, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:53, 2.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:41, 2.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:46, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:40, 2.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:38, 2.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:42, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:46, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:36, 2.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:40, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:31, 2.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:39, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:43, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:33, 2.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:37, 2.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:31, 2.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:35, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:38, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:30, 2.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:33, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:50, 1.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:41, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:29, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:31, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:33, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:34, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:26, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:29, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [05:58<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:24, 2.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:26, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:22, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:15, 3.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:55, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:48, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:36, 1.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:10, 741kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:58, 898kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:42, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:36, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:33, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:24, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:18, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:20, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:15, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:17, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:13, 2.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:11, 2.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:09, 2.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:08, 2.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:09, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 2.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.67MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 3.58MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<52:13:17,  2.13it/s]  0%|          | 736/400000 [00:00<36:29:32,  3.04it/s]  0%|          | 1505/400000 [00:00<25:29:59,  4.34it/s]  1%|          | 2269/400000 [00:00<17:49:12,  6.20it/s]  1%|          | 3008/400000 [00:00<12:27:19,  8.85it/s]  1%|          | 3604/400000 [00:00<8:42:40, 12.64it/s]   1%|          | 4279/400000 [00:01<6:05:32, 18.04it/s]  1%|          | 4961/400000 [00:01<4:15:43, 25.75it/s]  1%|▏         | 5733/400000 [00:01<2:58:54, 36.73it/s]  2%|▏         | 6412/400000 [00:01<2:05:18, 52.35it/s]  2%|▏         | 7096/400000 [00:01<1:27:51, 74.54it/s]  2%|▏         | 7773/400000 [00:01<1:01:42, 105.94it/s]  2%|▏         | 8436/400000 [00:01<43:25, 150.27it/s]    2%|▏         | 9187/400000 [00:01<30:36, 212.84it/s]  2%|▏         | 9960/400000 [00:01<21:37, 300.51it/s]  3%|▎         | 10707/400000 [00:01<15:22, 422.02it/s]  3%|▎         | 11428/400000 [00:02<11:01, 587.72it/s]  3%|▎         | 12162/400000 [00:02<07:57, 811.72it/s]  3%|▎         | 12932/400000 [00:02<05:48, 1109.45it/s]  3%|▎         | 13704/400000 [00:02<04:18, 1492.88it/s]  4%|▎         | 14450/400000 [00:02<03:16, 1960.82it/s]  4%|▍         | 15216/400000 [00:02<02:32, 2524.07it/s]  4%|▍         | 15965/400000 [00:02<02:01, 3150.15it/s]  4%|▍         | 16714/400000 [00:02<01:42, 3738.78it/s]  4%|▍         | 17486/400000 [00:02<01:26, 4422.38it/s]  5%|▍         | 18224/400000 [00:03<01:16, 5019.75it/s]  5%|▍         | 19001/400000 [00:03<01:07, 5615.06it/s]  5%|▍         | 19776/400000 [00:03<01:02, 6119.41it/s]  5%|▌         | 20532/400000 [00:03<00:58, 6462.94it/s]  5%|▌         | 21300/400000 [00:03<00:55, 6784.51it/s]  6%|▌         | 22057/400000 [00:03<00:54, 6890.87it/s]  6%|▌         | 22802/400000 [00:03<00:55, 6837.23it/s]  6%|▌         | 23525/400000 [00:03<00:54, 6846.56it/s]  6%|▌         | 24271/400000 [00:03<00:53, 7018.86it/s]  6%|▌         | 24993/400000 [00:03<00:54, 6847.41it/s]  6%|▋         | 25707/400000 [00:04<00:53, 6932.45it/s]  7%|▋         | 26462/400000 [00:04<00:52, 7104.35it/s]  7%|▋         | 27221/400000 [00:04<00:51, 7241.46it/s]  7%|▋         | 27978/400000 [00:04<00:50, 7335.15it/s]  7%|▋         | 28741/400000 [00:04<00:50, 7419.19it/s]  7%|▋         | 29487/400000 [00:04<00:51, 7204.68it/s]  8%|▊         | 30212/400000 [00:04<00:53, 6973.43it/s]  8%|▊         | 30982/400000 [00:04<00:51, 7173.79it/s]  8%|▊         | 31765/400000 [00:04<00:50, 7356.18it/s]  8%|▊         | 32553/400000 [00:04<00:48, 7503.28it/s]  8%|▊         | 33323/400000 [00:05<00:48, 7559.71it/s]  9%|▊         | 34082/400000 [00:05<00:48, 7515.23it/s]  9%|▊         | 34869/400000 [00:05<00:47, 7616.49it/s]  9%|▉         | 35634/400000 [00:05<00:47, 7626.32it/s]  9%|▉         | 36398/400000 [00:05<00:48, 7537.64it/s]  9%|▉         | 37164/400000 [00:05<00:47, 7573.22it/s]  9%|▉         | 37934/400000 [00:05<00:47, 7610.77it/s] 10%|▉         | 38696/400000 [00:05<00:47, 7603.33it/s] 10%|▉         | 39480/400000 [00:05<00:46, 7672.38it/s] 10%|█         | 40267/400000 [00:05<00:46, 7728.01it/s] 10%|█         | 41041/400000 [00:06<00:46, 7712.65it/s] 10%|█         | 41825/400000 [00:06<00:46, 7748.37it/s] 11%|█         | 42601/400000 [00:06<00:46, 7722.60it/s] 11%|█         | 43388/400000 [00:06<00:45, 7764.54it/s] 11%|█         | 44172/400000 [00:06<00:45, 7786.62it/s] 11%|█         | 44951/400000 [00:06<00:45, 7740.42it/s] 11%|█▏        | 45734/400000 [00:06<00:45, 7765.45it/s] 12%|█▏        | 46511/400000 [00:06<00:45, 7722.23it/s] 12%|█▏        | 47284/400000 [00:06<00:47, 7454.63it/s] 12%|█▏        | 48060/400000 [00:06<00:46, 7541.42it/s] 12%|█▏        | 48832/400000 [00:07<00:46, 7591.83it/s] 12%|█▏        | 49613/400000 [00:07<00:45, 7654.20it/s] 13%|█▎        | 50389/400000 [00:07<00:45, 7683.85it/s] 13%|█▎        | 51159/400000 [00:07<00:45, 7624.43it/s] 13%|█▎        | 51941/400000 [00:07<00:45, 7679.36it/s] 13%|█▎        | 52710/400000 [00:07<00:45, 7661.14it/s] 13%|█▎        | 53477/400000 [00:07<00:46, 7515.82it/s] 14%|█▎        | 54254/400000 [00:07<00:45, 7589.13it/s] 14%|█▍        | 55027/400000 [00:07<00:45, 7628.72it/s] 14%|█▍        | 55791/400000 [00:07<00:45, 7610.89it/s] 14%|█▍        | 56553/400000 [00:08<00:45, 7587.53it/s] 14%|█▍        | 57325/400000 [00:08<00:44, 7624.10it/s] 15%|█▍        | 58088/400000 [00:08<00:45, 7567.92it/s] 15%|█▍        | 58846/400000 [00:08<00:46, 7270.37it/s] 15%|█▍        | 59583/400000 [00:08<00:46, 7299.30it/s] 15%|█▌        | 60319/400000 [00:08<00:46, 7317.10it/s] 15%|█▌        | 61091/400000 [00:08<00:45, 7432.79it/s] 15%|█▌        | 61836/400000 [00:08<00:46, 7238.44it/s] 16%|█▌        | 62562/400000 [00:08<00:47, 7118.73it/s] 16%|█▌        | 63279/400000 [00:09<00:47, 7132.21it/s] 16%|█▌        | 64006/400000 [00:09<00:46, 7172.44it/s] 16%|█▌        | 64752/400000 [00:09<00:46, 7254.43it/s] 16%|█▋        | 65516/400000 [00:09<00:45, 7365.19it/s] 17%|█▋        | 66272/400000 [00:09<00:44, 7421.75it/s] 17%|█▋        | 67020/400000 [00:09<00:44, 7438.76it/s] 17%|█▋        | 67773/400000 [00:09<00:44, 7465.04it/s] 17%|█▋        | 68520/400000 [00:09<00:45, 7233.17it/s] 17%|█▋        | 69290/400000 [00:09<00:44, 7366.31it/s] 18%|█▊        | 70072/400000 [00:09<00:44, 7494.26it/s] 18%|█▊        | 70845/400000 [00:10<00:43, 7561.06it/s] 18%|█▊        | 71618/400000 [00:10<00:43, 7608.56it/s] 18%|█▊        | 72392/400000 [00:10<00:42, 7646.68it/s] 18%|█▊        | 73158/400000 [00:10<00:43, 7577.80it/s] 18%|█▊        | 73929/400000 [00:10<00:42, 7615.39it/s] 19%|█▊        | 74703/400000 [00:10<00:42, 7649.54it/s] 19%|█▉        | 75469/400000 [00:10<00:42, 7611.34it/s] 19%|█▉        | 76231/400000 [00:10<00:42, 7612.62it/s] 19%|█▉        | 76993/400000 [00:10<00:43, 7496.99it/s] 19%|█▉        | 77744/400000 [00:10<00:45, 7127.61it/s] 20%|█▉        | 78481/400000 [00:11<00:44, 7196.61it/s] 20%|█▉        | 79207/400000 [00:11<00:44, 7214.43it/s] 20%|█▉        | 79931/400000 [00:11<00:44, 7114.99it/s] 20%|██        | 80645/400000 [00:11<00:45, 7071.62it/s] 20%|██        | 81354/400000 [00:11<00:47, 6750.00it/s] 21%|██        | 82113/400000 [00:11<00:45, 6980.14it/s] 21%|██        | 82864/400000 [00:11<00:44, 7128.81it/s] 21%|██        | 83582/400000 [00:11<00:44, 7060.11it/s] 21%|██        | 84292/400000 [00:11<00:44, 7031.30it/s] 21%|██▏       | 85019/400000 [00:12<00:44, 7098.55it/s] 21%|██▏       | 85753/400000 [00:12<00:43, 7168.45it/s] 22%|██▏       | 86499/400000 [00:12<00:43, 7251.66it/s] 22%|██▏       | 87262/400000 [00:12<00:42, 7358.59it/s] 22%|██▏       | 88002/400000 [00:12<00:42, 7368.16it/s] 22%|██▏       | 88740/400000 [00:12<00:42, 7310.59it/s] 22%|██▏       | 89517/400000 [00:12<00:41, 7442.46it/s] 23%|██▎       | 90263/400000 [00:12<00:42, 7259.22it/s] 23%|██▎       | 91036/400000 [00:12<00:41, 7392.36it/s] 23%|██▎       | 91807/400000 [00:12<00:41, 7483.87it/s] 23%|██▎       | 92557/400000 [00:13<00:41, 7454.64it/s] 23%|██▎       | 93328/400000 [00:13<00:40, 7527.12it/s] 24%|██▎       | 94082/400000 [00:13<00:41, 7396.51it/s] 24%|██▎       | 94823/400000 [00:13<00:41, 7341.51it/s] 24%|██▍       | 95587/400000 [00:13<00:40, 7425.28it/s] 24%|██▍       | 96361/400000 [00:13<00:40, 7516.78it/s] 24%|██▍       | 97114/400000 [00:13<00:40, 7467.96it/s] 24%|██▍       | 97862/400000 [00:13<00:40, 7421.07it/s] 25%|██▍       | 98627/400000 [00:13<00:40, 7487.30it/s] 25%|██▍       | 99391/400000 [00:13<00:39, 7530.25it/s] 25%|██▌       | 100176/400000 [00:14<00:39, 7621.87it/s] 25%|██▌       | 100958/400000 [00:14<00:38, 7678.10it/s] 25%|██▌       | 101727/400000 [00:14<00:39, 7571.83it/s] 26%|██▌       | 102505/400000 [00:14<00:38, 7630.50it/s] 26%|██▌       | 103269/400000 [00:14<00:39, 7549.00it/s] 26%|██▌       | 104049/400000 [00:14<00:38, 7621.47it/s] 26%|██▌       | 104812/400000 [00:14<00:38, 7570.51it/s] 26%|██▋       | 105579/400000 [00:14<00:38, 7597.43it/s] 27%|██▋       | 106363/400000 [00:14<00:38, 7668.18it/s] 27%|██▋       | 107148/400000 [00:14<00:37, 7721.22it/s] 27%|██▋       | 107921/400000 [00:15<00:38, 7631.40it/s] 27%|██▋       | 108708/400000 [00:15<00:37, 7699.42it/s] 27%|██▋       | 109479/400000 [00:15<00:37, 7665.53it/s] 28%|██▊       | 110263/400000 [00:15<00:37, 7714.80it/s] 28%|██▊       | 111035/400000 [00:15<00:37, 7673.77it/s] 28%|██▊       | 111803/400000 [00:15<00:38, 7526.78it/s] 28%|██▊       | 112580/400000 [00:15<00:37, 7595.63it/s] 28%|██▊       | 113341/400000 [00:15<00:37, 7556.72it/s] 29%|██▊       | 114127/400000 [00:15<00:37, 7643.91it/s] 29%|██▊       | 114911/400000 [00:15<00:37, 7699.00it/s] 29%|██▉       | 115697/400000 [00:16<00:36, 7745.09it/s] 29%|██▉       | 116472/400000 [00:16<00:37, 7573.97it/s] 29%|██▉       | 117231/400000 [00:16<00:38, 7354.22it/s] 29%|██▉       | 117998/400000 [00:16<00:37, 7446.08it/s] 30%|██▉       | 118758/400000 [00:16<00:37, 7488.95it/s] 30%|██▉       | 119540/400000 [00:16<00:36, 7584.25it/s] 30%|███       | 120300/400000 [00:16<00:36, 7569.74it/s] 30%|███       | 121058/400000 [00:16<00:37, 7368.75it/s] 30%|███       | 121797/400000 [00:16<00:38, 7283.37it/s] 31%|███       | 122570/400000 [00:16<00:37, 7410.11it/s] 31%|███       | 123354/400000 [00:17<00:36, 7532.43it/s] 31%|███       | 124109/400000 [00:17<00:37, 7425.05it/s] 31%|███       | 124853/400000 [00:17<00:38, 7162.52it/s] 31%|███▏      | 125573/400000 [00:17<00:38, 7099.57it/s] 32%|███▏      | 126286/400000 [00:17<00:38, 7085.16it/s] 32%|███▏      | 127029/400000 [00:17<00:37, 7183.95it/s] 32%|███▏      | 127803/400000 [00:17<00:37, 7341.20it/s] 32%|███▏      | 128567/400000 [00:17<00:36, 7427.89it/s] 32%|███▏      | 129312/400000 [00:17<00:37, 7295.72it/s] 33%|███▎      | 130095/400000 [00:18<00:36, 7445.07it/s] 33%|███▎      | 130869/400000 [00:18<00:35, 7529.85it/s] 33%|███▎      | 131636/400000 [00:18<00:35, 7569.38it/s] 33%|███▎      | 132395/400000 [00:18<00:35, 7553.72it/s] 33%|███▎      | 133159/400000 [00:18<00:35, 7579.25it/s] 33%|███▎      | 133926/400000 [00:18<00:34, 7605.88it/s] 34%|███▎      | 134712/400000 [00:18<00:34, 7678.80it/s] 34%|███▍      | 135497/400000 [00:18<00:34, 7728.06it/s] 34%|███▍      | 136271/400000 [00:18<00:34, 7713.94it/s] 34%|███▍      | 137051/400000 [00:18<00:33, 7738.29it/s] 34%|███▍      | 137826/400000 [00:19<00:34, 7658.78it/s] 35%|███▍      | 138611/400000 [00:19<00:33, 7712.30it/s] 35%|███▍      | 139391/400000 [00:19<00:33, 7735.73it/s] 35%|███▌      | 140165/400000 [00:19<00:33, 7714.07it/s] 35%|███▌      | 140946/400000 [00:19<00:33, 7739.91it/s] 35%|███▌      | 141721/400000 [00:19<00:33, 7682.05it/s] 36%|███▌      | 142490/400000 [00:19<00:33, 7634.79it/s] 36%|███▌      | 143260/400000 [00:19<00:33, 7653.93it/s] 36%|███▌      | 144026/400000 [00:19<00:34, 7523.90it/s] 36%|███▌      | 144779/400000 [00:19<00:34, 7505.12it/s] 36%|███▋      | 145546/400000 [00:20<00:33, 7553.22it/s] 37%|███▋      | 146302/400000 [00:20<00:34, 7448.16it/s] 37%|███▋      | 147074/400000 [00:20<00:33, 7526.85it/s] 37%|███▋      | 147850/400000 [00:20<00:33, 7593.58it/s] 37%|███▋      | 148614/400000 [00:20<00:33, 7605.80it/s] 37%|███▋      | 149399/400000 [00:20<00:32, 7675.75it/s] 38%|███▊      | 150185/400000 [00:20<00:32, 7730.02it/s] 38%|███▊      | 150959/400000 [00:20<00:32, 7601.85it/s] 38%|███▊      | 151739/400000 [00:20<00:32, 7658.12it/s] 38%|███▊      | 152520/400000 [00:20<00:32, 7700.62it/s] 38%|███▊      | 153291/400000 [00:21<00:32, 7622.41it/s] 39%|███▊      | 154060/400000 [00:21<00:32, 7640.50it/s] 39%|███▊      | 154825/400000 [00:21<00:32, 7526.59it/s] 39%|███▉      | 155579/400000 [00:21<00:32, 7447.30it/s] 39%|███▉      | 156356/400000 [00:21<00:32, 7540.17it/s] 39%|███▉      | 157111/400000 [00:21<00:32, 7455.34it/s] 39%|███▉      | 157870/400000 [00:21<00:32, 7492.69it/s] 40%|███▉      | 158630/400000 [00:21<00:32, 7521.21it/s] 40%|███▉      | 159383/400000 [00:21<00:32, 7335.30it/s] 40%|████      | 160161/400000 [00:21<00:32, 7462.90it/s] 40%|████      | 160938/400000 [00:22<00:31, 7550.81it/s] 40%|████      | 161719/400000 [00:22<00:31, 7624.13it/s] 41%|████      | 162483/400000 [00:22<00:32, 7316.28it/s] 41%|████      | 163238/400000 [00:22<00:32, 7383.23it/s] 41%|████      | 164006/400000 [00:22<00:31, 7468.38it/s] 41%|████      | 164785/400000 [00:22<00:31, 7561.25it/s] 41%|████▏     | 165543/400000 [00:22<00:31, 7357.19it/s] 42%|████▏     | 166282/400000 [00:22<00:33, 7030.39it/s] 42%|████▏     | 166990/400000 [00:22<00:33, 7016.21it/s] 42%|████▏     | 167718/400000 [00:23<00:33, 7028.34it/s] 42%|████▏     | 168458/400000 [00:23<00:32, 7135.16it/s] 42%|████▏     | 169204/400000 [00:23<00:31, 7228.18it/s] 42%|████▏     | 169958/400000 [00:23<00:31, 7318.96it/s] 43%|████▎     | 170737/400000 [00:23<00:30, 7453.11it/s] 43%|████▎     | 171495/400000 [00:23<00:30, 7489.76it/s] 43%|████▎     | 172246/400000 [00:23<00:30, 7479.99it/s] 43%|████▎     | 173032/400000 [00:23<00:29, 7589.67it/s] 43%|████▎     | 173792/400000 [00:23<00:29, 7552.98it/s] 44%|████▎     | 174551/400000 [00:23<00:29, 7562.83it/s] 44%|████▍     | 175308/400000 [00:24<00:29, 7534.28it/s] 44%|████▍     | 176069/400000 [00:24<00:29, 7556.27it/s] 44%|████▍     | 176825/400000 [00:24<00:29, 7536.10it/s] 44%|████▍     | 177595/400000 [00:24<00:29, 7582.43it/s] 45%|████▍     | 178377/400000 [00:24<00:28, 7651.55it/s] 45%|████▍     | 179161/400000 [00:24<00:28, 7701.16it/s] 45%|████▍     | 179932/400000 [00:24<00:29, 7575.92it/s] 45%|████▌     | 180691/400000 [00:24<00:29, 7476.00it/s] 45%|████▌     | 181463/400000 [00:24<00:28, 7546.80it/s] 46%|████▌     | 182247/400000 [00:24<00:28, 7630.73it/s] 46%|████▌     | 183011/400000 [00:25<00:29, 7370.76it/s] 46%|████▌     | 183751/400000 [00:25<00:32, 6625.26it/s] 46%|████▌     | 184430/400000 [00:25<00:32, 6646.98it/s] 46%|████▋     | 185109/400000 [00:25<00:32, 6689.09it/s] 46%|████▋     | 185875/400000 [00:25<00:30, 6952.38it/s] 47%|████▋     | 186621/400000 [00:25<00:30, 7081.07it/s] 47%|████▋     | 187336/400000 [00:25<00:30, 7007.22it/s] 47%|████▋     | 188113/400000 [00:25<00:29, 7217.71it/s] 47%|████▋     | 188870/400000 [00:25<00:28, 7317.99it/s] 47%|████▋     | 189628/400000 [00:25<00:28, 7393.13it/s] 48%|████▊     | 190371/400000 [00:26<00:29, 7210.33it/s] 48%|████▊     | 191095/400000 [00:26<00:29, 7119.56it/s] 48%|████▊     | 191810/400000 [00:26<00:30, 6726.15it/s] 48%|████▊     | 192489/400000 [00:26<00:30, 6705.08it/s] 48%|████▊     | 193175/400000 [00:26<00:30, 6749.53it/s] 48%|████▊     | 193908/400000 [00:26<00:29, 6913.07it/s] 49%|████▊     | 194686/400000 [00:26<00:28, 7150.76it/s] 49%|████▉     | 195459/400000 [00:26<00:27, 7313.27it/s] 49%|████▉     | 196235/400000 [00:26<00:27, 7439.08it/s] 49%|████▉     | 197014/400000 [00:27<00:26, 7540.67it/s] 49%|████▉     | 197771/400000 [00:27<00:26, 7494.57it/s] 50%|████▉     | 198558/400000 [00:27<00:26, 7601.10it/s] 50%|████▉     | 199320/400000 [00:27<00:27, 7368.47it/s] 50%|█████     | 200098/400000 [00:27<00:26, 7486.08it/s] 50%|█████     | 200875/400000 [00:27<00:26, 7566.95it/s] 50%|█████     | 201654/400000 [00:27<00:25, 7632.38it/s] 51%|█████     | 202419/400000 [00:27<00:26, 7435.75it/s] 51%|█████     | 203165/400000 [00:27<00:27, 7119.23it/s] 51%|█████     | 203917/400000 [00:27<00:27, 7234.01it/s] 51%|█████     | 204680/400000 [00:28<00:26, 7346.09it/s] 51%|█████▏    | 205418/400000 [00:28<00:27, 7151.66it/s] 52%|█████▏    | 206137/400000 [00:28<00:27, 7019.84it/s] 52%|█████▏    | 206908/400000 [00:28<00:26, 7208.44it/s] 52%|█████▏    | 207633/400000 [00:28<00:27, 7086.30it/s] 52%|█████▏    | 208345/400000 [00:28<00:27, 7045.12it/s] 52%|█████▏    | 209052/400000 [00:28<00:27, 7031.62it/s] 52%|█████▏    | 209819/400000 [00:28<00:26, 7210.00it/s] 53%|█████▎    | 210543/400000 [00:28<00:26, 7214.81it/s] 53%|█████▎    | 211266/400000 [00:29<00:26, 6991.09it/s] 53%|█████▎    | 211968/400000 [00:29<00:26, 6977.14it/s] 53%|█████▎    | 212668/400000 [00:29<00:27, 6884.26it/s] 53%|█████▎    | 213358/400000 [00:29<00:27, 6710.10it/s] 54%|█████▎    | 214040/400000 [00:29<00:27, 6742.32it/s] 54%|█████▎    | 214716/400000 [00:29<00:27, 6639.80it/s] 54%|█████▍    | 215382/400000 [00:29<00:29, 6261.96it/s] 54%|█████▍    | 216031/400000 [00:29<00:29, 6328.48it/s] 54%|█████▍    | 216712/400000 [00:29<00:28, 6465.27it/s] 54%|█████▍    | 217363/400000 [00:29<00:28, 6472.38it/s] 55%|█████▍    | 218017/400000 [00:30<00:28, 6491.55it/s] 55%|█████▍    | 218703/400000 [00:30<00:27, 6596.77it/s] 55%|█████▍    | 219365/400000 [00:30<00:27, 6501.94it/s] 55%|█████▌    | 220059/400000 [00:30<00:27, 6625.39it/s] 55%|█████▌    | 220743/400000 [00:30<00:26, 6687.04it/s] 55%|█████▌    | 221436/400000 [00:30<00:26, 6757.35it/s] 56%|█████▌    | 222177/400000 [00:30<00:25, 6939.35it/s] 56%|█████▌    | 222956/400000 [00:30<00:24, 7172.50it/s] 56%|█████▌    | 223707/400000 [00:30<00:24, 7269.28it/s] 56%|█████▌    | 224451/400000 [00:30<00:23, 7317.77it/s] 56%|█████▋    | 225185/400000 [00:31<00:24, 7156.99it/s] 56%|█████▋    | 225903/400000 [00:31<00:24, 7091.21it/s] 57%|█████▋    | 226614/400000 [00:31<00:24, 7053.35it/s] 57%|█████▋    | 227360/400000 [00:31<00:24, 7162.12it/s] 57%|█████▋    | 228078/400000 [00:31<00:24, 7139.82it/s] 57%|█████▋    | 228862/400000 [00:31<00:23, 7334.41it/s] 57%|█████▋    | 229641/400000 [00:31<00:22, 7462.76it/s] 58%|█████▊    | 230420/400000 [00:31<00:22, 7557.25it/s] 58%|█████▊    | 231203/400000 [00:31<00:22, 7636.55it/s] 58%|█████▊    | 231968/400000 [00:31<00:22, 7542.56it/s] 58%|█████▊    | 232731/400000 [00:32<00:22, 7566.81it/s] 58%|█████▊    | 233508/400000 [00:32<00:21, 7625.45it/s] 59%|█████▊    | 234293/400000 [00:32<00:21, 7691.15it/s] 59%|█████▉    | 235076/400000 [00:32<00:21, 7731.39it/s] 59%|█████▉    | 235850/400000 [00:32<00:21, 7652.56it/s] 59%|█████▉    | 236616/400000 [00:32<00:21, 7640.60it/s] 59%|█████▉    | 237381/400000 [00:32<00:21, 7628.66it/s] 60%|█████▉    | 238145/400000 [00:32<00:21, 7471.92it/s] 60%|█████▉    | 238930/400000 [00:32<00:21, 7579.85it/s] 60%|█████▉    | 239696/400000 [00:32<00:21, 7602.47it/s] 60%|██████    | 240457/400000 [00:33<00:21, 7503.26it/s] 60%|██████    | 241240/400000 [00:33<00:20, 7596.36it/s] 61%|██████    | 242024/400000 [00:33<00:20, 7667.64it/s] 61%|██████    | 242792/400000 [00:33<00:20, 7582.86it/s] 61%|██████    | 243552/400000 [00:33<00:21, 7327.39it/s] 61%|██████    | 244327/400000 [00:33<00:20, 7448.32it/s] 61%|██████▏   | 245083/400000 [00:33<00:20, 7478.79it/s] 61%|██████▏   | 245860/400000 [00:33<00:20, 7562.78it/s] 62%|██████▏   | 246646/400000 [00:33<00:20, 7647.77it/s] 62%|██████▏   | 247412/400000 [00:34<00:20, 7430.26it/s] 62%|██████▏   | 248158/400000 [00:34<00:20, 7291.19it/s] 62%|██████▏   | 248890/400000 [00:34<00:20, 7257.30it/s] 62%|██████▏   | 249661/400000 [00:34<00:20, 7386.75it/s] 63%|██████▎   | 250437/400000 [00:34<00:19, 7492.32it/s] 63%|██████▎   | 251188/400000 [00:34<00:19, 7477.10it/s] 63%|██████▎   | 251964/400000 [00:34<00:19, 7559.25it/s] 63%|██████▎   | 252748/400000 [00:34<00:19, 7640.76it/s] 63%|██████▎   | 253513/400000 [00:34<00:19, 7589.00it/s] 64%|██████▎   | 254291/400000 [00:34<00:19, 7643.47it/s] 64%|██████▍   | 255056/400000 [00:35<00:19, 7333.20it/s] 64%|██████▍   | 255837/400000 [00:35<00:19, 7470.01it/s] 64%|██████▍   | 256622/400000 [00:35<00:18, 7579.97it/s] 64%|██████▍   | 257383/400000 [00:35<00:18, 7546.73it/s] 65%|██████▍   | 258140/400000 [00:35<00:18, 7524.64it/s] 65%|██████▍   | 258894/400000 [00:35<00:19, 7245.89it/s] 65%|██████▍   | 259622/400000 [00:35<00:19, 7157.52it/s] 65%|██████▌   | 260341/400000 [00:35<00:19, 7096.58it/s] 65%|██████▌   | 261131/400000 [00:35<00:18, 7317.35it/s] 65%|██████▌   | 261867/400000 [00:35<00:18, 7328.24it/s] 66%|██████▌   | 262602/400000 [00:36<00:18, 7309.12it/s] 66%|██████▌   | 263370/400000 [00:36<00:18, 7414.41it/s] 66%|██████▌   | 264153/400000 [00:36<00:18, 7533.92it/s] 66%|██████▌   | 264934/400000 [00:36<00:17, 7611.68it/s] 66%|██████▋   | 265713/400000 [00:36<00:17, 7662.63it/s] 67%|██████▋   | 266481/400000 [00:36<00:18, 7328.31it/s] 67%|██████▋   | 267232/400000 [00:36<00:17, 7379.61it/s] 67%|██████▋   | 267973/400000 [00:36<00:17, 7381.90it/s] 67%|██████▋   | 268751/400000 [00:36<00:17, 7495.27it/s] 67%|██████▋   | 269512/400000 [00:36<00:17, 7528.98it/s] 68%|██████▊   | 270267/400000 [00:37<00:17, 7508.84it/s] 68%|██████▊   | 271048/400000 [00:37<00:16, 7594.53it/s] 68%|██████▊   | 271832/400000 [00:37<00:16, 7665.92it/s] 68%|██████▊   | 272618/400000 [00:37<00:16, 7722.40it/s] 68%|██████▊   | 273403/400000 [00:37<00:16, 7758.32it/s] 69%|██████▊   | 274180/400000 [00:37<00:16, 7521.43it/s] 69%|██████▊   | 274935/400000 [00:37<00:17, 7236.83it/s] 69%|██████▉   | 275663/400000 [00:37<00:17, 7249.64it/s] 69%|██████▉   | 276441/400000 [00:37<00:16, 7398.88it/s] 69%|██████▉   | 277224/400000 [00:38<00:16, 7522.03it/s] 69%|██████▉   | 277979/400000 [00:38<00:16, 7184.66it/s] 70%|██████▉   | 278703/400000 [00:38<00:17, 6922.16it/s] 70%|██████▉   | 279401/400000 [00:38<00:17, 6879.95it/s] 70%|███████   | 280186/400000 [00:38<00:16, 7142.81it/s] 70%|███████   | 280940/400000 [00:38<00:16, 7257.15it/s] 70%|███████   | 281725/400000 [00:38<00:15, 7423.09it/s] 71%|███████   | 282505/400000 [00:38<00:15, 7531.41it/s] 71%|███████   | 283262/400000 [00:38<00:15, 7435.42it/s] 71%|███████   | 284043/400000 [00:38<00:15, 7541.26it/s] 71%|███████   | 284800/400000 [00:39<00:15, 7533.62it/s] 71%|███████▏  | 285555/400000 [00:39<00:15, 7367.12it/s] 72%|███████▏  | 286294/400000 [00:39<00:15, 7261.09it/s] 72%|███████▏  | 287022/400000 [00:39<00:15, 7178.64it/s] 72%|███████▏  | 287758/400000 [00:39<00:15, 7230.59it/s] 72%|███████▏  | 288520/400000 [00:39<00:15, 7342.31it/s] 72%|███████▏  | 289304/400000 [00:39<00:14, 7482.85it/s] 73%|███████▎  | 290054/400000 [00:39<00:15, 7251.27it/s] 73%|███████▎  | 290782/400000 [00:39<00:15, 7256.53it/s] 73%|███████▎  | 291510/400000 [00:39<00:14, 7245.77it/s] 73%|███████▎  | 292264/400000 [00:40<00:14, 7330.60it/s] 73%|███████▎  | 293040/400000 [00:40<00:14, 7452.16it/s] 73%|███████▎  | 293819/400000 [00:40<00:14, 7550.24it/s] 74%|███████▎  | 294604/400000 [00:40<00:13, 7635.68it/s] 74%|███████▍  | 295377/400000 [00:40<00:13, 7663.45it/s] 74%|███████▍  | 296145/400000 [00:40<00:13, 7544.03it/s] 74%|███████▍  | 296919/400000 [00:40<00:13, 7601.73it/s] 74%|███████▍  | 297697/400000 [00:40<00:13, 7652.47it/s] 75%|███████▍  | 298468/400000 [00:40<00:13, 7662.04it/s] 75%|███████▍  | 299235/400000 [00:40<00:13, 7407.34it/s] 75%|███████▍  | 299978/400000 [00:41<00:13, 7165.01it/s] 75%|███████▌  | 300698/400000 [00:41<00:14, 7068.52it/s] 75%|███████▌  | 301408/400000 [00:41<00:14, 7033.36it/s] 76%|███████▌  | 302114/400000 [00:41<00:13, 7025.95it/s] 76%|███████▌  | 302903/400000 [00:41<00:13, 7262.65it/s] 76%|███████▌  | 303660/400000 [00:41<00:13, 7352.16it/s] 76%|███████▌  | 304398/400000 [00:41<00:12, 7357.70it/s] 76%|███████▋  | 305176/400000 [00:41<00:12, 7477.40it/s] 76%|███████▋  | 305958/400000 [00:41<00:12, 7576.64it/s] 77%|███████▋  | 306750/400000 [00:42<00:12, 7674.56it/s] 77%|███████▋  | 307524/400000 [00:42<00:12, 7692.94it/s] 77%|███████▋  | 308306/400000 [00:42<00:11, 7729.31it/s] 77%|███████▋  | 309080/400000 [00:42<00:11, 7634.68it/s] 77%|███████▋  | 309862/400000 [00:42<00:11, 7687.07it/s] 78%|███████▊  | 310651/400000 [00:42<00:11, 7745.98it/s] 78%|███████▊  | 311427/400000 [00:42<00:11, 7714.48it/s] 78%|███████▊  | 312199/400000 [00:42<00:11, 7494.02it/s] 78%|███████▊  | 312951/400000 [00:42<00:12, 7152.19it/s] 78%|███████▊  | 313730/400000 [00:42<00:11, 7330.76it/s] 79%|███████▊  | 314512/400000 [00:43<00:11, 7470.87it/s] 79%|███████▉  | 315290/400000 [00:43<00:11, 7560.78it/s] 79%|███████▉  | 316078/400000 [00:43<00:10, 7651.35it/s] 79%|███████▉  | 316865/400000 [00:43<00:10, 7713.42it/s] 79%|███████▉  | 317639/400000 [00:43<00:10, 7602.74it/s] 80%|███████▉  | 318422/400000 [00:43<00:10, 7667.60it/s] 80%|███████▉  | 319190/400000 [00:43<00:10, 7609.08it/s] 80%|███████▉  | 319957/400000 [00:43<00:10, 7625.10it/s] 80%|████████  | 320721/400000 [00:43<00:10, 7400.17it/s] 80%|████████  | 321463/400000 [00:43<00:10, 7198.96it/s] 81%|████████  | 322186/400000 [00:44<00:10, 7143.09it/s] 81%|████████  | 322903/400000 [00:44<00:10, 7105.30it/s] 81%|████████  | 323679/400000 [00:44<00:10, 7288.26it/s] 81%|████████  | 324468/400000 [00:44<00:10, 7457.98it/s] 81%|████████▏ | 325247/400000 [00:44<00:09, 7554.23it/s] 82%|████████▏ | 326016/400000 [00:44<00:09, 7594.02it/s] 82%|████████▏ | 326779/400000 [00:44<00:09, 7603.83it/s] 82%|████████▏ | 327541/400000 [00:44<00:09, 7454.02it/s] 82%|████████▏ | 328320/400000 [00:44<00:09, 7550.77it/s] 82%|████████▏ | 329090/400000 [00:44<00:09, 7592.21it/s] 82%|████████▏ | 329851/400000 [00:45<00:09, 7594.19it/s] 83%|████████▎ | 330628/400000 [00:45<00:09, 7643.77it/s] 83%|████████▎ | 331413/400000 [00:45<00:08, 7702.48it/s] 83%|████████▎ | 332190/400000 [00:45<00:08, 7720.40it/s] 83%|████████▎ | 332975/400000 [00:45<00:08, 7756.57it/s] 83%|████████▎ | 333762/400000 [00:45<00:08, 7787.99it/s] 84%|████████▎ | 334542/400000 [00:45<00:08, 7737.60it/s] 84%|████████▍ | 335316/400000 [00:45<00:08, 7727.92it/s] 84%|████████▍ | 336098/400000 [00:45<00:08, 7753.25it/s] 84%|████████▍ | 336882/400000 [00:45<00:08, 7776.23it/s] 84%|████████▍ | 337667/400000 [00:46<00:07, 7796.12it/s] 85%|████████▍ | 338447/400000 [00:46<00:07, 7695.08it/s] 85%|████████▍ | 339229/400000 [00:46<00:07, 7731.57it/s] 85%|████████▌ | 340018/400000 [00:46<00:07, 7776.79it/s] 85%|████████▌ | 340804/400000 [00:46<00:07, 7799.36it/s] 85%|████████▌ | 341590/400000 [00:46<00:07, 7816.08it/s] 86%|████████▌ | 342372/400000 [00:46<00:07, 7748.24it/s] 86%|████████▌ | 343148/400000 [00:46<00:07, 7625.09it/s] 86%|████████▌ | 343928/400000 [00:46<00:07, 7675.86it/s] 86%|████████▌ | 344718/400000 [00:47<00:07, 7740.03it/s] 86%|████████▋ | 345503/400000 [00:47<00:07, 7772.18it/s] 87%|████████▋ | 346281/400000 [00:47<00:06, 7766.52it/s] 87%|████████▋ | 347058/400000 [00:47<00:06, 7686.43it/s] 87%|████████▋ | 347846/400000 [00:47<00:06, 7741.06it/s] 87%|████████▋ | 348634/400000 [00:47<00:06, 7779.45it/s] 87%|████████▋ | 349421/400000 [00:47<00:06, 7803.94it/s] 88%|████████▊ | 350202/400000 [00:47<00:06, 7688.17it/s] 88%|████████▊ | 350972/400000 [00:47<00:06, 7419.16it/s] 88%|████████▊ | 351717/400000 [00:47<00:06, 7223.94it/s] 88%|████████▊ | 352443/400000 [00:48<00:06, 7141.32it/s] 88%|████████▊ | 353218/400000 [00:48<00:06, 7312.71it/s] 88%|████████▊ | 353979/400000 [00:48<00:06, 7398.51it/s] 89%|████████▊ | 354763/400000 [00:48<00:06, 7525.41it/s] 89%|████████▉ | 355524/400000 [00:48<00:05, 7549.21it/s] 89%|████████▉ | 356310/400000 [00:48<00:05, 7637.06it/s] 89%|████████▉ | 357096/400000 [00:48<00:05, 7701.43it/s] 89%|████████▉ | 357868/400000 [00:48<00:05, 7572.94it/s] 90%|████████▉ | 358651/400000 [00:48<00:05, 7647.31it/s] 90%|████████▉ | 359432/400000 [00:48<00:05, 7695.09it/s] 90%|█████████ | 360203/400000 [00:49<00:05, 7399.65it/s] 90%|█████████ | 360946/400000 [00:49<00:05, 7380.22it/s] 90%|█████████ | 361687/400000 [00:49<00:05, 7149.23it/s] 91%|█████████ | 362465/400000 [00:49<00:05, 7325.43it/s] 91%|█████████ | 363238/400000 [00:49<00:04, 7440.97it/s] 91%|█████████ | 363988/400000 [00:49<00:04, 7456.05it/s] 91%|█████████ | 364775/400000 [00:49<00:04, 7575.28it/s] 91%|█████████▏| 365553/400000 [00:49<00:04, 7633.34it/s] 92%|█████████▏| 366318/400000 [00:49<00:04, 7590.79it/s] 92%|█████████▏| 367079/400000 [00:49<00:04, 7481.41it/s] 92%|█████████▏| 367851/400000 [00:50<00:04, 7549.39it/s] 92%|█████████▏| 368611/400000 [00:50<00:04, 7563.84it/s] 92%|█████████▏| 369380/400000 [00:50<00:04, 7600.95it/s] 93%|█████████▎| 370160/400000 [00:50<00:03, 7658.32it/s] 93%|█████████▎| 370937/400000 [00:50<00:03, 7690.76it/s] 93%|█████████▎| 371720/400000 [00:50<00:03, 7729.29it/s] 93%|█████████▎| 372494/400000 [00:50<00:03, 7710.96it/s] 93%|█████████▎| 373266/400000 [00:50<00:03, 7490.57it/s] 94%|█████████▎| 374017/400000 [00:50<00:03, 7293.51it/s] 94%|█████████▎| 374749/400000 [00:51<00:03, 7193.88it/s] 94%|█████████▍| 375499/400000 [00:51<00:03, 7282.30it/s] 94%|█████████▍| 376275/400000 [00:51<00:03, 7418.97it/s] 94%|█████████▍| 377030/400000 [00:51<00:03, 7455.48it/s] 94%|█████████▍| 377813/400000 [00:51<00:02, 7561.43it/s] 95%|█████████▍| 378601/400000 [00:51<00:02, 7652.55it/s] 95%|█████████▍| 379376/400000 [00:51<00:02, 7678.77it/s] 95%|█████████▌| 380165/400000 [00:51<00:02, 7739.53it/s] 95%|█████████▌| 380940/400000 [00:51<00:02, 7654.98it/s] 95%|█████████▌| 381725/400000 [00:51<00:02, 7711.07it/s] 96%|█████████▌| 382511/400000 [00:52<00:02, 7753.05it/s] 96%|█████████▌| 383297/400000 [00:52<00:02, 7784.71it/s] 96%|█████████▌| 384076/400000 [00:52<00:02, 7759.42it/s] 96%|█████████▌| 384853/400000 [00:52<00:01, 7720.76it/s] 96%|█████████▋| 385626/400000 [00:52<00:01, 7685.93it/s] 97%|█████████▋| 386410/400000 [00:52<00:01, 7730.55it/s] 97%|█████████▋| 387196/400000 [00:52<00:01, 7768.19it/s] 97%|█████████▋| 387981/400000 [00:52<00:01, 7790.16it/s] 97%|█████████▋| 388761/400000 [00:52<00:01, 7473.55it/s] 97%|█████████▋| 389521/400000 [00:52<00:01, 7509.52it/s] 98%|█████████▊| 390301/400000 [00:53<00:01, 7593.55it/s] 98%|█████████▊| 391063/400000 [00:53<00:01, 7444.23it/s] 98%|█████████▊| 391810/400000 [00:53<00:01, 7196.66it/s] 98%|█████████▊| 392558/400000 [00:53<00:01, 7279.04it/s] 98%|█████████▊| 393326/400000 [00:53<00:00, 7393.66it/s] 99%|█████████▊| 394078/400000 [00:53<00:00, 7429.72it/s] 99%|█████████▊| 394823/400000 [00:53<00:00, 7391.43it/s] 99%|█████████▉| 395564/400000 [00:53<00:00, 7228.88it/s] 99%|█████████▉| 396318/400000 [00:53<00:00, 7316.79it/s] 99%|█████████▉| 397103/400000 [00:53<00:00, 7466.59it/s] 99%|█████████▉| 397866/400000 [00:54<00:00, 7514.12it/s]100%|█████████▉| 398649/400000 [00:54<00:00, 7605.42it/s]100%|█████████▉| 399426/400000 [00:54<00:00, 7653.03it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7362.66it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe856d630f0>, <torchtext.data.dataset.TabularDataset object at 0x7fe856d63240>, <torchtext.vocab.Vocab object at 0x7fe856d63160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.98 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.98 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.98 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.17 file/s]2020-06-11 06:18:52.025797: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-11 06:18:52.031114: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-11 06:18:52.031593: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56331ae9a640 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-11 06:18:52.031899: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3457024/9912422 [00:00<00:00, 34475828.78it/s]9920512it [00:00, 34706690.64it/s]                             
0it [00:00, ?it/s]32768it [00:00, 586236.11it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478426.84it/s]1654784it [00:00, 12216774.36it/s]                         
0it [00:00, ?it/s]8192it [00:00, 206667.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
