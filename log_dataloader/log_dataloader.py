
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/fc0dfd7827ea882882c78d039135edfc41b55d70', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'fc0dfd7827ea882882c78d039135edfc41b55d70', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/fc0dfd7827ea882882c78d039135edfc41b55d70

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/fc0dfd7827ea882882c78d039135edfc41b55d70

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/fc0dfd7827ea882882c78d039135edfc41b55d70

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f906fd98620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f906fd98620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f90db35f0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f90db35f0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f90f50b0ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f90f50b0ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9088bdb950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9088bdb950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9088bdb950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 2990080/9912422 [00:00<00:00, 29892846.26it/s]9920512it [00:00, 34531626.60it/s]                             
0it [00:00, ?it/s]32768it [00:00, 610739.36it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154347.75it/s]1654784it [00:00, 11135266.28it/s]                         
0it [00:00, ?it/s]8192it [00:00, 199034.58it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9085ffad30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9085ff29e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9088bdb598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9088bdb598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9088bdb598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 14.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:00<00:05, 19.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.81 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.81 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.66 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.81s/ url]
0 examples [00:00, ? examples/s]2020-06-13 22:23:49.528682: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 22:23:49.547943: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-13 22:23:49.548170: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555e6d457250 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 22:23:49.548229: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.57 examples/s]84 examples [00:00, 13.60 examples/s]179 examples [00:00, 19.31 examples/s]273 examples [00:00, 27.35 examples/s]366 examples [00:00, 38.58 examples/s]462 examples [00:00, 54.17 examples/s]549 examples [00:00, 75.36 examples/s]642 examples [00:00, 104.01 examples/s]736 examples [00:00, 141.83 examples/s]824 examples [00:01, 189.46 examples/s]915 examples [00:01, 248.28 examples/s]1003 examples [00:01, 316.42 examples/s]1094 examples [00:01, 393.10 examples/s]1183 examples [00:01, 464.47 examples/s]1276 examples [00:01, 546.07 examples/s]1368 examples [00:01, 621.20 examples/s]1463 examples [00:01, 692.54 examples/s]1554 examples [00:01, 738.99 examples/s]1648 examples [00:01, 789.00 examples/s]1739 examples [00:02, 816.39 examples/s]1832 examples [00:02, 846.19 examples/s]1926 examples [00:02, 870.30 examples/s]2018 examples [00:02, 852.80 examples/s]2108 examples [00:02, 864.23 examples/s]2201 examples [00:02, 881.97 examples/s]2294 examples [00:02, 895.47 examples/s]2388 examples [00:02, 906.14 examples/s]2481 examples [00:02, 911.98 examples/s]2577 examples [00:02, 925.42 examples/s]2671 examples [00:03, 919.38 examples/s]2766 examples [00:03, 927.85 examples/s]2860 examples [00:03, 911.61 examples/s]2954 examples [00:03, 917.43 examples/s]3047 examples [00:03, 919.24 examples/s]3140 examples [00:03, 893.32 examples/s]3233 examples [00:03, 903.46 examples/s]3330 examples [00:03, 920.61 examples/s]3423 examples [00:03, 913.35 examples/s]3516 examples [00:03, 915.94 examples/s]3608 examples [00:04, 912.90 examples/s]3700 examples [00:04, 912.18 examples/s]3792 examples [00:04, 903.60 examples/s]3885 examples [00:04, 910.97 examples/s]3977 examples [00:04, 863.41 examples/s]4076 examples [00:04, 895.58 examples/s]4171 examples [00:04, 910.75 examples/s]4268 examples [00:04, 892.86 examples/s]4360 examples [00:04, 899.36 examples/s]4451 examples [00:05, 885.07 examples/s]4544 examples [00:05, 896.46 examples/s]4637 examples [00:05, 905.92 examples/s]4729 examples [00:05, 908.93 examples/s]4822 examples [00:05, 914.51 examples/s]4915 examples [00:05, 918.56 examples/s]5007 examples [00:05, 900.25 examples/s]5098 examples [00:05, 900.57 examples/s]5189 examples [00:05, 876.13 examples/s]5285 examples [00:05, 898.64 examples/s]5376 examples [00:06, 884.78 examples/s]5465 examples [00:06, 886.33 examples/s]5554 examples [00:06, 883.82 examples/s]5643 examples [00:06, 885.66 examples/s]5734 examples [00:06, 891.99 examples/s]5824 examples [00:06, 889.69 examples/s]5915 examples [00:06, 894.10 examples/s]6005 examples [00:06, 891.12 examples/s]6095 examples [00:06, 846.87 examples/s]6182 examples [00:06, 851.81 examples/s]6268 examples [00:07, 852.38 examples/s]6354 examples [00:07, 844.01 examples/s]6439 examples [00:07, 802.83 examples/s]6532 examples [00:07, 836.77 examples/s]6617 examples [00:07, 682.03 examples/s]6707 examples [00:07, 734.93 examples/s]6786 examples [00:07, 748.93 examples/s]6866 examples [00:07, 762.79 examples/s]6955 examples [00:07, 796.49 examples/s]7037 examples [00:08, 768.74 examples/s]7116 examples [00:08, 765.13 examples/s]7202 examples [00:08, 789.58 examples/s]7295 examples [00:08, 825.38 examples/s]7387 examples [00:08, 850.35 examples/s]7474 examples [00:08, 839.31 examples/s]7563 examples [00:08, 851.55 examples/s]7654 examples [00:08, 865.89 examples/s]7742 examples [00:08, 866.92 examples/s]7834 examples [00:09, 882.02 examples/s]7923 examples [00:09, 874.96 examples/s]8015 examples [00:09, 886.10 examples/s]8107 examples [00:09, 894.08 examples/s]8197 examples [00:09, 895.68 examples/s]8289 examples [00:09, 902.19 examples/s]8380 examples [00:09, 838.16 examples/s]8465 examples [00:09, 836.15 examples/s]8550 examples [00:09, 825.61 examples/s]8639 examples [00:09, 842.48 examples/s]8724 examples [00:10, 809.39 examples/s]8810 examples [00:10, 823.19 examples/s]8904 examples [00:10, 853.96 examples/s]8993 examples [00:10, 862.71 examples/s]9081 examples [00:10, 866.17 examples/s]9170 examples [00:10, 870.97 examples/s]9263 examples [00:10, 885.04 examples/s]9358 examples [00:10, 903.31 examples/s]9452 examples [00:10, 911.72 examples/s]9544 examples [00:10, 900.48 examples/s]9635 examples [00:11, 903.11 examples/s]9726 examples [00:11, 901.77 examples/s]9820 examples [00:11, 911.70 examples/s]9912 examples [00:11, 908.75 examples/s]10003 examples [00:11, 851.32 examples/s]10089 examples [00:11, 844.62 examples/s]10175 examples [00:11, 846.24 examples/s]10268 examples [00:11, 868.36 examples/s]10360 examples [00:11, 881.39 examples/s]10452 examples [00:12, 892.22 examples/s]10542 examples [00:12, 893.52 examples/s]10640 examples [00:12, 915.56 examples/s]10732 examples [00:12, 886.13 examples/s]10823 examples [00:12, 892.50 examples/s]10917 examples [00:12, 905.96 examples/s]11009 examples [00:12, 907.26 examples/s]11100 examples [00:12, 875.74 examples/s]11190 examples [00:12, 882.57 examples/s]11279 examples [00:12, 870.52 examples/s]11374 examples [00:13, 890.39 examples/s]11464 examples [00:13, 888.87 examples/s]11556 examples [00:13, 895.95 examples/s]11646 examples [00:13, 864.83 examples/s]11737 examples [00:13, 877.12 examples/s]11828 examples [00:13, 884.36 examples/s]11917 examples [00:13, 880.62 examples/s]12006 examples [00:13, 876.66 examples/s]12094 examples [00:13, 843.47 examples/s]12179 examples [00:13, 796.47 examples/s]12267 examples [00:14, 819.40 examples/s]12350 examples [00:14, 805.35 examples/s]12443 examples [00:14, 838.37 examples/s]12536 examples [00:14, 863.46 examples/s]12628 examples [00:14, 878.78 examples/s]12723 examples [00:14, 898.26 examples/s]12814 examples [00:14, 898.91 examples/s]12906 examples [00:14, 902.90 examples/s]12997 examples [00:14, 882.66 examples/s]13088 examples [00:15, 888.41 examples/s]13178 examples [00:15, 882.74 examples/s]13267 examples [00:15, 881.85 examples/s]13359 examples [00:15, 891.60 examples/s]13449 examples [00:15, 893.46 examples/s]13541 examples [00:15, 899.82 examples/s]13634 examples [00:15, 905.88 examples/s]13725 examples [00:15, 903.36 examples/s]13818 examples [00:15, 908.73 examples/s]13909 examples [00:15, 901.90 examples/s]14000 examples [00:16, 898.28 examples/s]14092 examples [00:16, 903.19 examples/s]14184 examples [00:16, 905.65 examples/s]14278 examples [00:16, 914.26 examples/s]14374 examples [00:16, 925.24 examples/s]14470 examples [00:16, 935.12 examples/s]14564 examples [00:16, 928.30 examples/s]14657 examples [00:16, 901.05 examples/s]14748 examples [00:16, 879.75 examples/s]14837 examples [00:16, 880.54 examples/s]14929 examples [00:17, 890.05 examples/s]15019 examples [00:17, 890.50 examples/s]15109 examples [00:17, 882.45 examples/s]15207 examples [00:17, 907.21 examples/s]15298 examples [00:17, 903.74 examples/s]15390 examples [00:17, 906.36 examples/s]15481 examples [00:17, 882.51 examples/s]15573 examples [00:17, 891.59 examples/s]15665 examples [00:17, 899.01 examples/s]15756 examples [00:17, 878.87 examples/s]15847 examples [00:18, 885.33 examples/s]15944 examples [00:18, 907.84 examples/s]16036 examples [00:18, 894.04 examples/s]16129 examples [00:18, 902.19 examples/s]16220 examples [00:18, 894.15 examples/s]16310 examples [00:18, 874.86 examples/s]16402 examples [00:18, 887.67 examples/s]16494 examples [00:18, 894.95 examples/s]16584 examples [00:18, 880.59 examples/s]16675 examples [00:19, 888.88 examples/s]16765 examples [00:19, 891.78 examples/s]16856 examples [00:19, 894.74 examples/s]16946 examples [00:19, 894.10 examples/s]17036 examples [00:19, 885.87 examples/s]17128 examples [00:19, 894.85 examples/s]17221 examples [00:19, 904.82 examples/s]17314 examples [00:19, 911.19 examples/s]17406 examples [00:19, 910.25 examples/s]17498 examples [00:19, 910.14 examples/s]17590 examples [00:20, 870.31 examples/s]17681 examples [00:20, 880.89 examples/s]17772 examples [00:20, 888.10 examples/s]17862 examples [00:20, 862.89 examples/s]17953 examples [00:20, 874.32 examples/s]18041 examples [00:20, 875.81 examples/s]18129 examples [00:20, 874.46 examples/s]18217 examples [00:20, 875.61 examples/s]18306 examples [00:20, 877.45 examples/s]18398 examples [00:20, 888.59 examples/s]18490 examples [00:21, 894.83 examples/s]18582 examples [00:21, 902.11 examples/s]18675 examples [00:21, 908.45 examples/s]18766 examples [00:21, 903.18 examples/s]18858 examples [00:21, 905.31 examples/s]18949 examples [00:21, 906.09 examples/s]19040 examples [00:21, 897.40 examples/s]19130 examples [00:21, 893.55 examples/s]19220 examples [00:21, 878.95 examples/s]19308 examples [00:21, 871.47 examples/s]19399 examples [00:22, 881.74 examples/s]19492 examples [00:22, 893.51 examples/s]19584 examples [00:22, 899.87 examples/s]19675 examples [00:22, 896.45 examples/s]19766 examples [00:22, 898.35 examples/s]19856 examples [00:22, 894.85 examples/s]19947 examples [00:22, 899.04 examples/s]20037 examples [00:22, 815.98 examples/s]20130 examples [00:22, 845.41 examples/s]20216 examples [00:23, 846.04 examples/s]20303 examples [00:23, 851.16 examples/s]20389 examples [00:23, 840.05 examples/s]20476 examples [00:23, 847.24 examples/s]20562 examples [00:23, 844.08 examples/s]20650 examples [00:23, 854.39 examples/s]20738 examples [00:23, 859.08 examples/s]20829 examples [00:23, 870.79 examples/s]20917 examples [00:23, 871.92 examples/s]21005 examples [00:23, 869.32 examples/s]21096 examples [00:24, 878.11 examples/s]21187 examples [00:24, 886.85 examples/s]21276 examples [00:24, 886.21 examples/s]21366 examples [00:24, 888.21 examples/s]21455 examples [00:24, 884.94 examples/s]21545 examples [00:24, 887.28 examples/s]21634 examples [00:24, 880.93 examples/s]21728 examples [00:24, 896.99 examples/s]21818 examples [00:24, 891.90 examples/s]21908 examples [00:24, 889.22 examples/s]21998 examples [00:25, 890.41 examples/s]22088 examples [00:25, 878.05 examples/s]22176 examples [00:25, 872.62 examples/s]22266 examples [00:25, 878.91 examples/s]22354 examples [00:25, 871.14 examples/s]22444 examples [00:25, 878.56 examples/s]22534 examples [00:25, 882.19 examples/s]22624 examples [00:25, 885.14 examples/s]22715 examples [00:25, 891.79 examples/s]22805 examples [00:25, 859.11 examples/s]22893 examples [00:26, 863.44 examples/s]22984 examples [00:26, 876.52 examples/s]23074 examples [00:26, 882.51 examples/s]23163 examples [00:26, 878.76 examples/s]23251 examples [00:26, 877.36 examples/s]23342 examples [00:26, 886.07 examples/s]23432 examples [00:26, 887.91 examples/s]23526 examples [00:26, 900.99 examples/s]23617 examples [00:26, 899.29 examples/s]23707 examples [00:26, 897.60 examples/s]23798 examples [00:27, 900.54 examples/s]23889 examples [00:27, 882.52 examples/s]23978 examples [00:27, 882.46 examples/s]24067 examples [00:27, 879.36 examples/s]24162 examples [00:27, 897.36 examples/s]24253 examples [00:27, 900.37 examples/s]24344 examples [00:27, 902.49 examples/s]24435 examples [00:27, 901.42 examples/s]24528 examples [00:27, 909.78 examples/s]24620 examples [00:27, 909.17 examples/s]24711 examples [00:28, 908.51 examples/s]24804 examples [00:28, 914.57 examples/s]24899 examples [00:28, 923.56 examples/s]24993 examples [00:28, 926.69 examples/s]25086 examples [00:28, 888.48 examples/s]25184 examples [00:28, 911.92 examples/s]25277 examples [00:28, 916.52 examples/s]25369 examples [00:28, 915.54 examples/s]25462 examples [00:28, 918.98 examples/s]25555 examples [00:29, 902.79 examples/s]25647 examples [00:29, 907.82 examples/s]25745 examples [00:29, 927.30 examples/s]25839 examples [00:29, 930.27 examples/s]25933 examples [00:29, 922.44 examples/s]26026 examples [00:29, 908.62 examples/s]26119 examples [00:29, 914.16 examples/s]26211 examples [00:29, 899.59 examples/s]26303 examples [00:29, 905.03 examples/s]26395 examples [00:29, 909.13 examples/s]26487 examples [00:30, 910.75 examples/s]26579 examples [00:30, 909.51 examples/s]26670 examples [00:30, 909.46 examples/s]26761 examples [00:30, 890.19 examples/s]26852 examples [00:30, 895.36 examples/s]26944 examples [00:30, 901.12 examples/s]27036 examples [00:30, 904.73 examples/s]27127 examples [00:30, 906.13 examples/s]27218 examples [00:30, 906.26 examples/s]27309 examples [00:30, 893.97 examples/s]27404 examples [00:31, 909.53 examples/s]27496 examples [00:31, 898.89 examples/s]27587 examples [00:31, 901.16 examples/s]27682 examples [00:31, 914.76 examples/s]27774 examples [00:31, 899.86 examples/s]27865 examples [00:31, 900.79 examples/s]27956 examples [00:31, 899.25 examples/s]28048 examples [00:31, 902.94 examples/s]28139 examples [00:31, 870.78 examples/s]28227 examples [00:31, 854.50 examples/s]28313 examples [00:32, 837.48 examples/s]28403 examples [00:32, 854.22 examples/s]28491 examples [00:32, 861.48 examples/s]28580 examples [00:32, 869.15 examples/s]28668 examples [00:32, 867.30 examples/s]28755 examples [00:32, 867.78 examples/s]28846 examples [00:32, 877.75 examples/s]28938 examples [00:32, 888.95 examples/s]29031 examples [00:32, 898.06 examples/s]29121 examples [00:33, 889.76 examples/s]29211 examples [00:33, 881.54 examples/s]29301 examples [00:33, 886.57 examples/s]29390 examples [00:33, 881.00 examples/s]29486 examples [00:33, 901.98 examples/s]29577 examples [00:33, 888.82 examples/s]29667 examples [00:33, 843.89 examples/s]29753 examples [00:33, 847.43 examples/s]29844 examples [00:33, 865.28 examples/s]29933 examples [00:33, 870.18 examples/s]30021 examples [00:34, 814.20 examples/s]30113 examples [00:34, 841.87 examples/s]30203 examples [00:34, 855.62 examples/s]30290 examples [00:34, 853.29 examples/s]30382 examples [00:34, 870.58 examples/s]30471 examples [00:34, 876.04 examples/s]30559 examples [00:34, 847.42 examples/s]30653 examples [00:34, 871.43 examples/s]30745 examples [00:34, 885.08 examples/s]30834 examples [00:34, 868.41 examples/s]30922 examples [00:35, 852.17 examples/s]31010 examples [00:35, 859.18 examples/s]31100 examples [00:35, 870.13 examples/s]31189 examples [00:35, 874.81 examples/s]31277 examples [00:35, 876.33 examples/s]31365 examples [00:35, 872.34 examples/s]31459 examples [00:35, 889.42 examples/s]31551 examples [00:35, 896.45 examples/s]31646 examples [00:35, 909.96 examples/s]31738 examples [00:36, 909.37 examples/s]31830 examples [00:36, 895.47 examples/s]31920 examples [00:36, 895.64 examples/s]32011 examples [00:36, 897.84 examples/s]32107 examples [00:36, 915.19 examples/s]32200 examples [00:36, 917.13 examples/s]32292 examples [00:36, 905.83 examples/s]32383 examples [00:36, 893.18 examples/s]32476 examples [00:36, 901.24 examples/s]32568 examples [00:36, 905.10 examples/s]32659 examples [00:37, 903.49 examples/s]32750 examples [00:37, 895.57 examples/s]32840 examples [00:37, 890.59 examples/s]32930 examples [00:37, 885.31 examples/s]33019 examples [00:37, 871.89 examples/s]33107 examples [00:37, 867.47 examples/s]33195 examples [00:37, 869.08 examples/s]33288 examples [00:37, 884.25 examples/s]33379 examples [00:37, 890.39 examples/s]33475 examples [00:37, 908.22 examples/s]33566 examples [00:38, 875.11 examples/s]33655 examples [00:38, 877.10 examples/s]33743 examples [00:38, 873.75 examples/s]33833 examples [00:38, 879.68 examples/s]33926 examples [00:38, 891.27 examples/s]34016 examples [00:38, 892.91 examples/s]34106 examples [00:38, 854.16 examples/s]34195 examples [00:38, 862.05 examples/s]34282 examples [00:38, 862.75 examples/s]34372 examples [00:38, 872.83 examples/s]34463 examples [00:39, 881.92 examples/s]34552 examples [00:39, 882.30 examples/s]34643 examples [00:39, 889.06 examples/s]34735 examples [00:39, 896.62 examples/s]34827 examples [00:39, 898.46 examples/s]34925 examples [00:39, 919.22 examples/s]35019 examples [00:39, 924.09 examples/s]35112 examples [00:39, 890.58 examples/s]35202 examples [00:39, 891.78 examples/s]35293 examples [00:40, 896.96 examples/s]35383 examples [00:40, 886.74 examples/s]35474 examples [00:40, 893.53 examples/s]35567 examples [00:40, 902.38 examples/s]35660 examples [00:40, 908.58 examples/s]35751 examples [00:40, 894.94 examples/s]35843 examples [00:40, 899.72 examples/s]35934 examples [00:40, 889.42 examples/s]36024 examples [00:40, 878.68 examples/s]36113 examples [00:40, 881.77 examples/s]36202 examples [00:41, 840.33 examples/s]36294 examples [00:41, 860.59 examples/s]36383 examples [00:41, 868.82 examples/s]36474 examples [00:41, 879.21 examples/s]36566 examples [00:41, 888.28 examples/s]36656 examples [00:41, 887.41 examples/s]36745 examples [00:41, 880.65 examples/s]36834 examples [00:41, 862.11 examples/s]36922 examples [00:41, 867.29 examples/s]37012 examples [00:41, 874.94 examples/s]37100 examples [00:42, 852.84 examples/s]37187 examples [00:42, 857.68 examples/s]37273 examples [00:42, 857.51 examples/s]37361 examples [00:42, 863.98 examples/s]37448 examples [00:42, 861.48 examples/s]37537 examples [00:42, 867.78 examples/s]37629 examples [00:42, 880.70 examples/s]37718 examples [00:42, 864.46 examples/s]37812 examples [00:42, 883.40 examples/s]37904 examples [00:42, 893.00 examples/s]37995 examples [00:43, 895.98 examples/s]38085 examples [00:43, 893.48 examples/s]38176 examples [00:43, 896.69 examples/s]38269 examples [00:43, 904.15 examples/s]38362 examples [00:43, 910.15 examples/s]38454 examples [00:43, 910.63 examples/s]38547 examples [00:43, 914.70 examples/s]38639 examples [00:43, 912.37 examples/s]38731 examples [00:43, 911.61 examples/s]38823 examples [00:43, 912.02 examples/s]38915 examples [00:44, 900.98 examples/s]39006 examples [00:44, 884.87 examples/s]39097 examples [00:44, 890.91 examples/s]39187 examples [00:44, 882.08 examples/s]39276 examples [00:44, 872.67 examples/s]39368 examples [00:44, 885.48 examples/s]39457 examples [00:44, 878.32 examples/s]39545 examples [00:44, 860.43 examples/s]39635 examples [00:44, 871.28 examples/s]39727 examples [00:45, 883.39 examples/s]39817 examples [00:45, 888.23 examples/s]39907 examples [00:45, 891.63 examples/s]40000 examples [00:45, 901.57 examples/s]40091 examples [00:45, 834.85 examples/s]40176 examples [00:45, 813.32 examples/s]40260 examples [00:45, 819.68 examples/s]40345 examples [00:45, 826.69 examples/s]40439 examples [00:45, 855.38 examples/s]40528 examples [00:45, 863.67 examples/s]40621 examples [00:46, 880.72 examples/s]40713 examples [00:46, 890.47 examples/s]40805 examples [00:46, 897.24 examples/s]40897 examples [00:46, 901.80 examples/s]40995 examples [00:46, 922.68 examples/s]41091 examples [00:46, 933.36 examples/s]41186 examples [00:46, 936.11 examples/s]41280 examples [00:46, 929.23 examples/s]41374 examples [00:46, 924.51 examples/s]41469 examples [00:46, 929.76 examples/s]41563 examples [00:47, 925.41 examples/s]41656 examples [00:47, 918.46 examples/s]41749 examples [00:47, 921.22 examples/s]41847 examples [00:47, 936.06 examples/s]41941 examples [00:47, 929.88 examples/s]42035 examples [00:47, 927.82 examples/s]42128 examples [00:47, 921.11 examples/s]42221 examples [00:47, 897.10 examples/s]42311 examples [00:47, 890.66 examples/s]42401 examples [00:48, 890.20 examples/s]42492 examples [00:48, 894.19 examples/s]42584 examples [00:48, 899.72 examples/s]42675 examples [00:48, 893.98 examples/s]42769 examples [00:48, 906.60 examples/s]42863 examples [00:48, 915.52 examples/s]42955 examples [00:48, 914.11 examples/s]43049 examples [00:48, 919.75 examples/s]43142 examples [00:48, 910.41 examples/s]43234 examples [00:48, 896.58 examples/s]43325 examples [00:49, 899.19 examples/s]43416 examples [00:49, 900.74 examples/s]43507 examples [00:49, 892.68 examples/s]43597 examples [00:49, 883.22 examples/s]43689 examples [00:49, 893.42 examples/s]43781 examples [00:49, 899.87 examples/s]43872 examples [00:49, 899.89 examples/s]43963 examples [00:49, 891.98 examples/s]44053 examples [00:49, 889.98 examples/s]44148 examples [00:49, 905.61 examples/s]44239 examples [00:50, 876.48 examples/s]44328 examples [00:50, 877.85 examples/s]44419 examples [00:50, 886.37 examples/s]44508 examples [00:50, 884.99 examples/s]44598 examples [00:50, 886.37 examples/s]44689 examples [00:50, 891.88 examples/s]44779 examples [00:50, 893.53 examples/s]44869 examples [00:50, 892.60 examples/s]44959 examples [00:50, 863.06 examples/s]45051 examples [00:50, 877.19 examples/s]45142 examples [00:51, 885.00 examples/s]45233 examples [00:51, 889.69 examples/s]45326 examples [00:51, 899.06 examples/s]45417 examples [00:51, 891.90 examples/s]45508 examples [00:51, 895.86 examples/s]45601 examples [00:51, 904.33 examples/s]45696 examples [00:51, 916.04 examples/s]45794 examples [00:51, 932.01 examples/s]45888 examples [00:51, 927.48 examples/s]45981 examples [00:51, 925.68 examples/s]46074 examples [00:52, 926.88 examples/s]46167 examples [00:52, 919.94 examples/s]46260 examples [00:52, 921.11 examples/s]46356 examples [00:52, 929.77 examples/s]46457 examples [00:52, 951.94 examples/s]46553 examples [00:52, 946.46 examples/s]46648 examples [00:52, 930.27 examples/s]46742 examples [00:52, 925.49 examples/s]46835 examples [00:52, 918.96 examples/s]46927 examples [00:53, 907.61 examples/s]47018 examples [00:53, 867.56 examples/s]47109 examples [00:53, 879.37 examples/s]47198 examples [00:53, 880.85 examples/s]47289 examples [00:53, 888.77 examples/s]47383 examples [00:53, 902.46 examples/s]47474 examples [00:53, 902.41 examples/s]47568 examples [00:53, 911.73 examples/s]47660 examples [00:53, 898.13 examples/s]47750 examples [00:53, 851.00 examples/s]47841 examples [00:54, 866.69 examples/s]47932 examples [00:54, 877.09 examples/s]48024 examples [00:54, 887.61 examples/s]48115 examples [00:54, 892.10 examples/s]48205 examples [00:54, 878.48 examples/s]48295 examples [00:54, 884.78 examples/s]48385 examples [00:54, 887.12 examples/s]48484 examples [00:54, 915.48 examples/s]48576 examples [00:54, 909.63 examples/s]48668 examples [00:54, 911.82 examples/s]48760 examples [00:55, 911.98 examples/s]48852 examples [00:55, 903.24 examples/s]48943 examples [00:55, 875.96 examples/s]49031 examples [00:55, 867.31 examples/s]49118 examples [00:55, 861.57 examples/s]49205 examples [00:55, 841.53 examples/s]49290 examples [00:55, 842.69 examples/s]49380 examples [00:55, 858.18 examples/s]49470 examples [00:55, 869.76 examples/s]49560 examples [00:55, 876.87 examples/s]49651 examples [00:56, 886.09 examples/s]49740 examples [00:56, 881.56 examples/s]49829 examples [00:56, 883.52 examples/s]49918 examples [00:56, 882.94 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6114/50000 [00:00<00:00, 61138.60 examples/s] 34%|███▍      | 16934/50000 [00:00<00:00, 70311.98 examples/s] 54%|█████▍    | 27185/50000 [00:00<00:00, 77625.67 examples/s] 76%|███████▌  | 38078/50000 [00:00<00:00, 84949.20 examples/s] 97%|█████████▋| 48748/50000 [00:00<00:00, 90481.87 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 708.74 examples/s]165 examples [00:00, 765.18 examples/s]251 examples [00:00, 789.39 examples/s]341 examples [00:00, 818.67 examples/s]435 examples [00:00, 848.65 examples/s]528 examples [00:00, 869.83 examples/s]622 examples [00:00, 887.41 examples/s]719 examples [00:00, 909.01 examples/s]809 examples [00:00, 904.75 examples/s]902 examples [00:01, 910.52 examples/s]992 examples [00:01, 856.92 examples/s]1084 examples [00:01, 872.86 examples/s]1181 examples [00:01, 897.74 examples/s]1272 examples [00:01, 899.41 examples/s]1370 examples [00:01, 921.09 examples/s]1463 examples [00:01, 710.36 examples/s]1550 examples [00:01, 750.22 examples/s]1643 examples [00:01, 793.61 examples/s]1733 examples [00:02, 822.37 examples/s]1819 examples [00:02, 832.41 examples/s]1914 examples [00:02, 864.00 examples/s]2017 examples [00:02, 906.05 examples/s]2115 examples [00:02, 925.68 examples/s]2210 examples [00:02, 927.34 examples/s]2304 examples [00:02, 909.33 examples/s]2397 examples [00:02, 914.04 examples/s]2489 examples [00:02, 912.23 examples/s]2581 examples [00:02, 911.33 examples/s]2673 examples [00:03, 895.60 examples/s]2769 examples [00:03, 913.32 examples/s]2862 examples [00:03, 917.42 examples/s]2956 examples [00:03, 923.36 examples/s]3051 examples [00:03, 928.25 examples/s]3144 examples [00:03, 917.52 examples/s]3236 examples [00:03, 900.78 examples/s]3333 examples [00:03, 920.14 examples/s]3427 examples [00:03, 924.30 examples/s]3521 examples [00:03, 927.14 examples/s]3615 examples [00:04, 928.50 examples/s]3709 examples [00:04, 929.05 examples/s]3802 examples [00:04, 921.44 examples/s]3900 examples [00:04, 938.24 examples/s]3994 examples [00:04, 938.31 examples/s]4088 examples [00:04, 937.50 examples/s]4182 examples [00:04, 932.60 examples/s]4276 examples [00:04, 930.32 examples/s]4370 examples [00:04, 928.24 examples/s]4464 examples [00:04, 929.18 examples/s]4557 examples [00:05, 897.10 examples/s]4647 examples [00:05, 888.44 examples/s]4737 examples [00:05, 891.13 examples/s]4827 examples [00:05, 891.00 examples/s]4926 examples [00:05, 916.65 examples/s]5018 examples [00:05, 915.50 examples/s]5111 examples [00:05, 917.04 examples/s]5207 examples [00:05, 926.50 examples/s]5304 examples [00:05, 938.83 examples/s]5398 examples [00:06, 926.26 examples/s]5491 examples [00:06, 893.05 examples/s]5587 examples [00:06, 910.42 examples/s]5679 examples [00:06, 911.19 examples/s]5771 examples [00:06, 913.34 examples/s]5864 examples [00:06, 914.62 examples/s]5962 examples [00:06, 932.16 examples/s]6056 examples [00:06, 928.09 examples/s]6152 examples [00:06, 937.10 examples/s]6246 examples [00:06, 909.15 examples/s]6339 examples [00:07, 914.60 examples/s]6431 examples [00:07, 902.86 examples/s]6525 examples [00:07, 911.68 examples/s]6618 examples [00:07, 915.54 examples/s]6712 examples [00:07, 920.67 examples/s]6805 examples [00:07, 922.19 examples/s]6899 examples [00:07, 927.35 examples/s]6993 examples [00:07, 931.05 examples/s]7087 examples [00:07, 901.28 examples/s]7178 examples [00:07, 900.62 examples/s]7269 examples [00:08, 877.04 examples/s]7357 examples [00:08, 859.54 examples/s]7452 examples [00:08, 883.98 examples/s]7548 examples [00:08, 904.75 examples/s]7641 examples [00:08, 911.54 examples/s]7733 examples [00:08, 894.04 examples/s]7828 examples [00:08, 907.60 examples/s]7919 examples [00:08, 908.00 examples/s]8011 examples [00:08, 910.02 examples/s]8103 examples [00:08, 886.28 examples/s]8196 examples [00:09, 898.93 examples/s]8290 examples [00:09, 910.51 examples/s]8384 examples [00:09, 918.78 examples/s]8477 examples [00:09, 916.90 examples/s]8570 examples [00:09, 920.46 examples/s]8663 examples [00:09, 919.33 examples/s]8756 examples [00:09, 922.09 examples/s]8849 examples [00:09, 907.66 examples/s]8941 examples [00:09, 909.96 examples/s]9033 examples [00:10, 910.91 examples/s]9125 examples [00:10, 900.73 examples/s]9218 examples [00:10, 907.03 examples/s]9310 examples [00:10, 908.27 examples/s]9401 examples [00:10, 905.57 examples/s]9493 examples [00:10, 909.20 examples/s]9584 examples [00:10, 898.52 examples/s]9674 examples [00:10, 893.91 examples/s]9764 examples [00:10, 894.29 examples/s]9854 examples [00:10, 894.04 examples/s]9944 examples [00:11, 894.73 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 96%|█████████▌| 9622/10000 [00:00<00:00, 96213.90 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY9E41V/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteY9E41V/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9088bdb950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9088bdb950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9088bdb950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9012053a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9013154a20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9088bdb950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9088bdb950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9088bdb950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f90d45c7400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f906fc8b080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f90100dc268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f90100dc268> 

  function with postional parmater data_info <function split_train_valid at 0x7f90100dc268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f90100dc378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f90100dc378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f90100dc378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:34:54, 10.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:02:17, 14.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:16:48, 21.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:54:14, 30.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:31:07, 43.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.40M/862M [00:01<3:50:19, 61.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:40:14, 88.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:51:31, 126kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<1:17:38, 179kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:01<54:04, 256kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.7M/862M [00:02<37:40, 365kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<26:25, 518kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<18:26, 737kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.8M/862M [00:02<12:58, 1.04MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<10:02, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<08:54, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<08:29, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<06:24, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:05<04:40, 2.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<09:44, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<08:31, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<06:19, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<07:03, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<08:10, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<06:31, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<04:45, 2.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<11:46, 1.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<09:34, 1.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<07:01, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<07:59, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<08:16, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<04:36, 2.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<11:48, 1.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<09:37, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<07:03, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:58, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<08:13, 1.58MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<06:18, 2.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<04:34, 2.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<09:59, 1.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<08:17, 1.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:19<06:07, 2.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:18, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<06:11, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:21<04:42, 2.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:21<03:27, 3.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<1:38:03, 131kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<1:09:53, 183kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<49:09, 260kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<37:19, 341kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<28:42, 444kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:25<20:42, 614kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<16:30, 768kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<12:49, 987kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<09:17, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:25, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:16, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:03, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<05:03, 2.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<31:57, 392kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<23:40, 529kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<16:51, 742kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<14:41, 848kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:56, 963kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:35, 1.30MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<06:52, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<10:35, 1.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:59, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<05:45, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:29, 1.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:15, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:06, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<05:06, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:31:07, 134kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:04:59, 188kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<45:39, 268kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<34:44, 351kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<26:46, 455kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<19:20, 629kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:26, 784kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:02, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<08:42, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:54, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:42, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:36, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:45, 2.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:33, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:19, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:49, 1.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:33, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:29, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:50, 2.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:09, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:42, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:13, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:46, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<15:16, 771kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:53, 990kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<08:33, 1.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:43, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:05, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:16, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:49, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<46:43, 249kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<33:52, 343kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<23:57, 484kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<19:28, 594kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<15:58, 724kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<11:45, 983kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:03, 1.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<08:13, 1.40MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:00, 1.91MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:51, 1.67MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:06, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:33, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:42, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:10, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:52, 2.93MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:02, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<04:41, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:31, 3.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:29, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:00, 2.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:44, 2.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:13, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:55, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:38, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:23, 3.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:50, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:33, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:40, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<21:56, 502kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:26, 670kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<11:45, 934kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:48, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<09:46, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<07:18, 1.50MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:16, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:43, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:30, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:49, 2.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:54, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:19, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:54, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:31, 3.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<19:09, 561kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<14:30, 741kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<10:21, 1.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:44, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:54, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:44, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<04:15, 2.49MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:50, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<08:05, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:05, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:09, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:41, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:54, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:02, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:10, 1.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:47, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:05, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:28, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:12, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:45, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:42, 1.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:21, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:53, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:52, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:07, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:15, 1.64MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:30, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:13, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:40, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:00, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:27, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:23, 1.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:29, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:56, 1.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:45, 1.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:44, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<04:51, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:07, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:41, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:41, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:47, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:04, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:45, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:39, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:46, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:39, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:38, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:54, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<04:15, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<09:27, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:00, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:49, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:54, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:09, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:09, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:32, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<04:00, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:24, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:48, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:37, 1.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:45, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:24, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:23, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:38, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<04:03, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:31, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:22, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:40, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:30, 992kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:43, 1.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:34, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<04:41, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<09:04, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:26, 1.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:26, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<04:37, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:59, 931kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:22, 992kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:06, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<05:06, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:49, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:12, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:14, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:28, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:36, 953kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:03, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:49, 1.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<04:52, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<08:27, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:55, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:58, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<04:18, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:23, 959kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:53, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:46, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<04:52, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<08:15, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:59, 1.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:08, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<04:25, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:08, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:39, 1.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:50, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<04:11, 2.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:10, 960kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<08:40, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:31, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:42, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:41, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:07, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:43, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:25, 2.54MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:29, 1.58MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:47, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:26, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:13, 2.67MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:37, 1.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:50, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:29, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:14, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:17, 1.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:10, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:28, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:55, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:52, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:40, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<05:08, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<03:41, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:31, 985kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:00, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:06, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<04:23, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:48, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:18, 1.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:29, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:58, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:38, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:47, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:30, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:14, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<08:08, 1.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:42, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:54, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<04:13, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:17, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:04, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:25, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:54, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:33, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:05, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:19, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:49, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:57, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:52, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:28, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:12, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:49, 1.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:32, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:56, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:33, 2.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:37, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:45, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:11, 2.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:49, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:43, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:24, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:10, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:19, 826kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<08:24, 915kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:21, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:31, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<07:21, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<07:07, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:27, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:54, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:45, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<06:30, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:59, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:35, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:56, 943kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<07:17, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:27, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:20, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<05:12, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:56, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:51, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:54, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:19, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:06, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:57, 2.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<05:40, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<05:40, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:20, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:08, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:44, 1.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:18, 1.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:48, 1.50MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:26, 2.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<07:27, 957kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<07:00, 1.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:19, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:48, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:56, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:29, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:53, 1.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:33, 1.98MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:21, 1.61MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:18, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:16, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:22, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:44, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:49, 886kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:15, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:31, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:35, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:21, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:20, 2.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:29, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:02, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:11, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:18, 2.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:14, 928kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<06:26, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<04:47, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:27, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:47, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:46, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:41, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:39, 2.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<10:23, 634kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<08:39, 760kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<06:24, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<04:32, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<14:48, 440kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<11:19, 575kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<08:08, 798kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:50, 942kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:05, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:34, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:12, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:17, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:35, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:49, 2.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:58, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:21, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:39, 2.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:11, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:32, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:45, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:11, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:31, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:44, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:05, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:27, 2.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:41, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:02, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:25, 2.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:02, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:24, 2.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:47, 3.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:03, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:30, 1.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:29, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:32, 2.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:26, 1.67MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:03, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:14, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:21, 2.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:54, 1.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:17, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:22, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:26, 2.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:10, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:21, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:21, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:25, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:40, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:55, 1.42MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:02, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:11, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:44, 3.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:48, 623kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:12, 667kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:13, 879kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:26, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:31, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:02, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:00, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:52, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:28, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:10, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:23, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<01:47, 2.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<07:13, 730kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:46, 779kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<05:09, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<03:40, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:03, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:31, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:31, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:32, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:52, 2.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<08:22, 614kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<07:25, 692kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:35, 918kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<03:58, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:16, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:33, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:34, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:35, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:16, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:49, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:00, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<02:11, 2.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:01, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:38, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:51, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:04, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:32, 3.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<12:00, 405kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<09:47, 496kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<07:09, 677kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<05:03, 952kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<03:37, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<15:22, 312kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<12:14, 392kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<08:54, 537kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<06:16, 756kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<05:48, 813kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:25, 870kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:05, 1.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:55, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<02:08, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<12:53, 361kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<10:26, 445kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<07:38, 607kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<05:24, 852kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<05:12, 882kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<05:02, 908kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:52, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:47, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:20, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:43, 1.21MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:56, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:07, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:50, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:21, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:37, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:54, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:22<01:25, 3.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<13:08, 333kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<10:27, 418kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<07:35, 576kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:20, 811kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<03:48, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<12:45, 338kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<10:14, 421kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<07:28, 575kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<05:16, 810kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<04:57, 856kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<04:40, 905kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:35, 1.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:34, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:04, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:26, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:42, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:57, 2.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:38, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:01, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:21, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:43, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:27, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:56, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:21, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:42, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:51, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:13, 1.77MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:37, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:11, 3.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<12:59, 300kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<10:13, 381kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<07:25, 523kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<05:13, 736kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:52, 785kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<04:30, 847kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<03:25, 1.11MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<02:26, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:55, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:08, 1.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:25, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:47, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:17, 2.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<26:50, 137kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<19:54, 185kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<14:07, 260kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<09:52, 369kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<08:01, 451kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:35, 548kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<04:49, 748kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:24, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:30, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<03:28, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:38, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:54, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:22, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<25:25, 137kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<18:43, 185kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<13:17, 261kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<09:16, 370kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<07:34, 450kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<06:16, 542kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:35, 739kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<03:15, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<02:18, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<48:38, 68.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<34:53, 95.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<24:32, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<17:06, 193kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<12:43, 257kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<09:46, 334kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<07:02, 463kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:55, 653kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:35, 696kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:08, 772kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:04, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<02:11, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:24, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:32, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:58, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:12, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:18, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:47, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:07, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:18, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:47, 1.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:17, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:53, 1.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:10, 1.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:43, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:48, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:56, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:30, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<01:05, 2.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:35, 1.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:25, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:50, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:35, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:30, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:55, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:11<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:09, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:04, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:34, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:45, 932kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:27, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:50, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<01:17, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:43, 922kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:23, 1.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:46, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<01:15, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<04:09, 587kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<03:21, 724kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:26, 993kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<01:42, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:52, 822kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:33, 923kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:53, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:21, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:31, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:36, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:13, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:53, 2.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:15, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:23, 1.59MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:04, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:46, 2.81MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:27, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:31, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:10, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:49, 2.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:49, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:45, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:21, 1.54MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:57, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:49, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:43, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:18, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:55, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:04, 944kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:55, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:26, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<01:01, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:21, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:14, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:55, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:57, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:11, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:56, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:40, 2.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:08, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:08, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:52, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:17, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:02, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:53, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:55, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:43, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:30, 3.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:42, 896kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:29, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:06, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:59, 1.48MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:58, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:43, 1.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:46, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:36, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:37, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:41, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:31, 2.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:22, 3.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<08:32, 148kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<06:27, 196kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<04:35, 273kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<03:10, 387kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<02:27, 486kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:56, 615kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:23, 843kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:07, 1.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:59, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:44, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:40, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:39, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:30, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:30, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:36, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:26, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:29, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:37, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:21, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:25, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:26, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:20, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:20, 2.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:22, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:17, 2.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:17, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:23, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:19, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:13, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:16, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:12, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:08, 3.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:22, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:25, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:19, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:13, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:07, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:05, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:03, 3.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:10, 914kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:10, 876kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:04, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.97MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 683kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 713kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 940kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 688/400000 [00:00<00:58, 6876.99it/s]  0%|          | 1463/400000 [00:00<00:56, 7115.17it/s]  1%|          | 2248/400000 [00:00<00:54, 7318.28it/s]  1%|          | 3006/400000 [00:00<00:53, 7393.67it/s]  1%|          | 3726/400000 [00:00<00:54, 7331.80it/s]  1%|          | 4441/400000 [00:00<00:54, 7274.21it/s]  1%|▏         | 5187/400000 [00:00<00:53, 7327.26it/s]  1%|▏         | 5975/400000 [00:00<00:52, 7484.32it/s]  2%|▏         | 6746/400000 [00:00<00:52, 7548.44it/s]  2%|▏         | 7526/400000 [00:01<00:51, 7619.74it/s]  2%|▏         | 8270/400000 [00:01<00:51, 7547.77it/s]  2%|▏         | 9030/400000 [00:01<00:51, 7563.29it/s]  2%|▏         | 9783/400000 [00:01<00:51, 7550.52it/s]  3%|▎         | 10596/400000 [00:01<00:50, 7714.94it/s]  3%|▎         | 11404/400000 [00:01<00:49, 7819.01it/s]  3%|▎         | 12184/400000 [00:01<00:50, 7664.27it/s]  3%|▎         | 12987/400000 [00:01<00:49, 7767.71it/s]  3%|▎         | 13818/400000 [00:01<00:48, 7919.58it/s]  4%|▎         | 14611/400000 [00:01<00:49, 7806.02it/s]  4%|▍         | 15401/400000 [00:02<00:49, 7832.11it/s]  4%|▍         | 16185/400000 [00:02<00:49, 7719.49it/s]  4%|▍         | 16958/400000 [00:02<00:50, 7641.15it/s]  4%|▍         | 17745/400000 [00:02<00:49, 7705.93it/s]  5%|▍         | 18540/400000 [00:02<00:49, 7775.49it/s]  5%|▍         | 19319/400000 [00:02<00:49, 7758.82it/s]  5%|▌         | 20096/400000 [00:02<00:49, 7747.39it/s]  5%|▌         | 20891/400000 [00:02<00:48, 7806.72it/s]  5%|▌         | 21672/400000 [00:02<00:49, 7715.18it/s]  6%|▌         | 22444/400000 [00:02<00:49, 7674.35it/s]  6%|▌         | 23240/400000 [00:03<00:48, 7757.67it/s]  6%|▌         | 24021/400000 [00:03<00:48, 7771.25it/s]  6%|▌         | 24803/400000 [00:03<00:48, 7784.47it/s]  6%|▋         | 25582/400000 [00:03<00:48, 7725.06it/s]  7%|▋         | 26366/400000 [00:03<00:48, 7758.59it/s]  7%|▋         | 27143/400000 [00:03<00:48, 7668.34it/s]  7%|▋         | 27911/400000 [00:03<00:48, 7669.61it/s]  7%|▋         | 28702/400000 [00:03<00:47, 7738.53it/s]  7%|▋         | 29499/400000 [00:03<00:47, 7805.78it/s]  8%|▊         | 30280/400000 [00:03<00:48, 7702.46it/s]  8%|▊         | 31055/400000 [00:04<00:47, 7714.55it/s]  8%|▊         | 31827/400000 [00:04<00:48, 7597.06it/s]  8%|▊         | 32610/400000 [00:04<00:47, 7664.70it/s]  8%|▊         | 33388/400000 [00:04<00:47, 7696.56it/s]  9%|▊         | 34159/400000 [00:04<00:47, 7665.96it/s]  9%|▊         | 34926/400000 [00:04<00:48, 7575.15it/s]  9%|▉         | 35685/400000 [00:04<00:48, 7552.23it/s]  9%|▉         | 36544/400000 [00:04<00:46, 7835.50it/s]  9%|▉         | 37377/400000 [00:04<00:45, 7975.42it/s] 10%|▉         | 38178/400000 [00:04<00:45, 7924.85it/s] 10%|▉         | 38973/400000 [00:05<00:45, 7931.13it/s] 10%|▉         | 39772/400000 [00:05<00:45, 7947.53it/s] 10%|█         | 40604/400000 [00:05<00:44, 8054.20it/s] 10%|█         | 41425/400000 [00:05<00:44, 8097.80it/s] 11%|█         | 42292/400000 [00:05<00:43, 8259.78it/s] 11%|█         | 43120/400000 [00:05<00:43, 8196.20it/s] 11%|█         | 43941/400000 [00:05<00:45, 7867.44it/s] 11%|█         | 44755/400000 [00:05<00:44, 7945.87it/s] 11%|█▏        | 45582/400000 [00:05<00:44, 8037.77it/s] 12%|█▏        | 46406/400000 [00:05<00:43, 8096.23it/s] 12%|█▏        | 47218/400000 [00:06<00:44, 7960.01it/s] 12%|█▏        | 48016/400000 [00:06<00:45, 7787.82it/s] 12%|█▏        | 48818/400000 [00:06<00:44, 7855.43it/s] 12%|█▏        | 49614/400000 [00:06<00:44, 7886.44it/s] 13%|█▎        | 50488/400000 [00:06<00:43, 8121.90it/s] 13%|█▎        | 51312/400000 [00:06<00:42, 8156.83it/s] 13%|█▎        | 52130/400000 [00:06<00:43, 7924.20it/s] 13%|█▎        | 53006/400000 [00:06<00:42, 8157.48it/s] 13%|█▎        | 53829/400000 [00:06<00:42, 8177.22it/s] 14%|█▎        | 54650/400000 [00:06<00:42, 8154.15it/s] 14%|█▍        | 55468/400000 [00:07<00:43, 7953.70it/s] 14%|█▍        | 56266/400000 [00:07<00:44, 7717.52it/s] 14%|█▍        | 57041/400000 [00:07<00:44, 7701.78it/s] 14%|█▍        | 57815/400000 [00:07<00:44, 7713.07it/s] 15%|█▍        | 58630/400000 [00:07<00:43, 7831.95it/s] 15%|█▍        | 59433/400000 [00:07<00:43, 7889.96it/s] 15%|█▌        | 60224/400000 [00:07<00:43, 7795.01it/s] 15%|█▌        | 61022/400000 [00:07<00:43, 7847.92it/s] 15%|█▌        | 61842/400000 [00:07<00:42, 7948.60it/s] 16%|█▌        | 62670/400000 [00:08<00:41, 8044.14it/s] 16%|█▌        | 63515/400000 [00:08<00:41, 8159.52it/s] 16%|█▌        | 64332/400000 [00:08<00:42, 7881.65it/s] 16%|█▋        | 65158/400000 [00:08<00:41, 7990.47it/s] 16%|█▋        | 65960/400000 [00:08<00:41, 7998.50it/s] 17%|█▋        | 66762/400000 [00:08<00:41, 7934.76it/s] 17%|█▋        | 67557/400000 [00:08<00:41, 7937.24it/s] 17%|█▋        | 68352/400000 [00:08<00:41, 7914.14it/s] 17%|█▋        | 69145/400000 [00:08<00:41, 7897.35it/s] 17%|█▋        | 69936/400000 [00:08<00:42, 7769.21it/s] 18%|█▊        | 70714/400000 [00:09<00:43, 7653.44it/s] 18%|█▊        | 71481/400000 [00:09<00:43, 7496.42it/s] 18%|█▊        | 72258/400000 [00:09<00:43, 7576.23it/s] 18%|█▊        | 73070/400000 [00:09<00:42, 7729.63it/s] 18%|█▊        | 73882/400000 [00:09<00:41, 7841.07it/s] 19%|█▊        | 74668/400000 [00:09<00:41, 7830.45it/s] 19%|█▉        | 75453/400000 [00:09<00:42, 7696.40it/s] 19%|█▉        | 76224/400000 [00:09<00:42, 7531.11it/s] 19%|█▉        | 76995/400000 [00:09<00:42, 7581.95it/s] 19%|█▉        | 77793/400000 [00:09<00:41, 7695.46it/s] 20%|█▉        | 78589/400000 [00:10<00:41, 7772.93it/s] 20%|█▉        | 79368/400000 [00:10<00:41, 7747.40it/s] 20%|██        | 80144/400000 [00:10<00:41, 7729.49it/s] 20%|██        | 80932/400000 [00:10<00:41, 7772.63it/s] 20%|██        | 81710/400000 [00:10<00:41, 7676.53it/s] 21%|██        | 82479/400000 [00:10<00:41, 7634.51it/s] 21%|██        | 83262/400000 [00:10<00:41, 7689.73it/s] 21%|██        | 84032/400000 [00:10<00:41, 7678.80it/s] 21%|██        | 84801/400000 [00:10<00:41, 7664.02it/s] 21%|██▏       | 85568/400000 [00:10<00:41, 7654.73it/s] 22%|██▏       | 86334/400000 [00:11<00:41, 7617.42it/s] 22%|██▏       | 87096/400000 [00:11<00:41, 7581.51it/s] 22%|██▏       | 87855/400000 [00:11<00:41, 7527.93it/s] 22%|██▏       | 88612/400000 [00:11<00:41, 7538.60it/s] 22%|██▏       | 89366/400000 [00:11<00:41, 7498.27it/s] 23%|██▎       | 90140/400000 [00:11<00:40, 7569.04it/s] 23%|██▎       | 90947/400000 [00:11<00:40, 7711.10it/s] 23%|██▎       | 91719/400000 [00:11<00:40, 7671.70it/s] 23%|██▎       | 92487/400000 [00:11<00:40, 7636.73it/s] 23%|██▎       | 93301/400000 [00:11<00:39, 7779.63it/s] 24%|██▎       | 94158/400000 [00:12<00:38, 7999.65it/s] 24%|██▎       | 94961/400000 [00:12<00:38, 7909.59it/s] 24%|██▍       | 95758/400000 [00:12<00:38, 7924.06it/s] 24%|██▍       | 96553/400000 [00:12<00:38, 7929.55it/s] 24%|██▍       | 97362/400000 [00:12<00:37, 7975.26it/s] 25%|██▍       | 98161/400000 [00:12<00:37, 7949.10it/s] 25%|██▍       | 98957/400000 [00:12<00:38, 7876.00it/s] 25%|██▍       | 99746/400000 [00:12<00:38, 7842.77it/s] 25%|██▌       | 100531/400000 [00:12<00:39, 7581.23it/s] 25%|██▌       | 101295/400000 [00:13<00:39, 7596.89it/s] 26%|██▌       | 102137/400000 [00:13<00:38, 7825.60it/s] 26%|██▌       | 102923/400000 [00:13<00:38, 7730.07it/s] 26%|██▌       | 103699/400000 [00:13<00:38, 7629.39it/s] 26%|██▌       | 104464/400000 [00:13<00:38, 7614.63it/s] 26%|██▋       | 105264/400000 [00:13<00:38, 7725.21it/s] 27%|██▋       | 106091/400000 [00:13<00:37, 7879.71it/s] 27%|██▋       | 106891/400000 [00:13<00:37, 7911.72it/s] 27%|██▋       | 107684/400000 [00:13<00:37, 7894.92it/s] 27%|██▋       | 108475/400000 [00:13<00:37, 7847.97it/s] 27%|██▋       | 109261/400000 [00:14<00:37, 7850.30it/s] 28%|██▊       | 110099/400000 [00:14<00:36, 8000.23it/s] 28%|██▊       | 110926/400000 [00:14<00:35, 8077.13it/s] 28%|██▊       | 111735/400000 [00:14<00:35, 8032.57it/s] 28%|██▊       | 112539/400000 [00:14<00:36, 7942.64it/s] 28%|██▊       | 113349/400000 [00:14<00:35, 7987.40it/s] 29%|██▊       | 114149/400000 [00:14<00:36, 7726.04it/s] 29%|██▊       | 114971/400000 [00:14<00:36, 7865.84it/s] 29%|██▉       | 115760/400000 [00:14<00:36, 7763.72it/s] 29%|██▉       | 116539/400000 [00:14<00:37, 7660.66it/s] 29%|██▉       | 117307/400000 [00:15<00:37, 7597.91it/s] 30%|██▉       | 118081/400000 [00:15<00:36, 7638.37it/s] 30%|██▉       | 118846/400000 [00:15<00:37, 7593.85it/s] 30%|██▉       | 119607/400000 [00:15<00:37, 7560.04it/s] 30%|███       | 120394/400000 [00:15<00:36, 7647.13it/s] 30%|███       | 121223/400000 [00:15<00:35, 7826.70it/s] 31%|███       | 122008/400000 [00:15<00:35, 7823.21it/s] 31%|███       | 122816/400000 [00:15<00:35, 7898.07it/s] 31%|███       | 123607/400000 [00:15<00:35, 7881.85it/s] 31%|███       | 124396/400000 [00:15<00:35, 7818.64it/s] 31%|███▏      | 125179/400000 [00:16<00:35, 7702.85it/s] 31%|███▏      | 125969/400000 [00:16<00:35, 7758.76it/s] 32%|███▏      | 126746/400000 [00:16<00:36, 7536.04it/s] 32%|███▏      | 127502/400000 [00:16<00:36, 7533.12it/s] 32%|███▏      | 128295/400000 [00:16<00:35, 7646.40it/s] 32%|███▏      | 129071/400000 [00:16<00:35, 7677.90it/s] 32%|███▏      | 129874/400000 [00:16<00:34, 7778.30it/s] 33%|███▎      | 130691/400000 [00:16<00:34, 7885.53it/s] 33%|███▎      | 131481/400000 [00:16<00:34, 7762.98it/s] 33%|███▎      | 132293/400000 [00:16<00:34, 7864.98it/s] 33%|███▎      | 133112/400000 [00:17<00:33, 7959.30it/s] 33%|███▎      | 133948/400000 [00:17<00:32, 8073.07it/s] 34%|███▎      | 134757/400000 [00:17<00:33, 8018.10it/s] 34%|███▍      | 135569/400000 [00:17<00:32, 8048.01it/s] 34%|███▍      | 136378/400000 [00:17<00:32, 8060.03it/s] 34%|███▍      | 137220/400000 [00:17<00:32, 8162.31it/s] 35%|███▍      | 138043/400000 [00:17<00:32, 8180.33it/s] 35%|███▍      | 138862/400000 [00:17<00:32, 8117.48it/s] 35%|███▍      | 139681/400000 [00:17<00:31, 8138.00it/s] 35%|███▌      | 140513/400000 [00:17<00:31, 8189.66it/s] 35%|███▌      | 141354/400000 [00:18<00:31, 8254.44it/s] 36%|███▌      | 142180/400000 [00:18<00:31, 8154.31it/s] 36%|███▌      | 142996/400000 [00:18<00:31, 8068.88it/s] 36%|███▌      | 143812/400000 [00:18<00:31, 8093.95it/s] 36%|███▌      | 144655/400000 [00:18<00:31, 8190.96it/s] 36%|███▋      | 145518/400000 [00:18<00:30, 8316.85it/s] 37%|███▋      | 146351/400000 [00:18<00:30, 8286.77it/s] 37%|███▋      | 147219/400000 [00:18<00:30, 8399.54it/s] 37%|███▋      | 148060/400000 [00:18<00:30, 8195.85it/s] 37%|███▋      | 148887/400000 [00:19<00:30, 8216.29it/s] 37%|███▋      | 149710/400000 [00:19<00:30, 8213.64it/s] 38%|███▊      | 150534/400000 [00:19<00:30, 8217.91it/s] 38%|███▊      | 151357/400000 [00:19<00:31, 7978.18it/s] 38%|███▊      | 152157/400000 [00:19<00:31, 7963.91it/s] 38%|███▊      | 153003/400000 [00:19<00:30, 8104.53it/s] 38%|███▊      | 153816/400000 [00:19<00:30, 8069.61it/s] 39%|███▊      | 154625/400000 [00:19<00:30, 8013.83it/s] 39%|███▉      | 155450/400000 [00:19<00:30, 8080.35it/s] 39%|███▉      | 156259/400000 [00:19<00:30, 7903.50it/s] 39%|███▉      | 157089/400000 [00:20<00:30, 8017.11it/s] 39%|███▉      | 157893/400000 [00:20<00:30, 7946.48it/s] 40%|███▉      | 158689/400000 [00:20<00:30, 7861.10it/s] 40%|███▉      | 159477/400000 [00:20<00:31, 7749.04it/s] 40%|████      | 160276/400000 [00:20<00:30, 7818.92it/s] 40%|████      | 161084/400000 [00:20<00:30, 7895.39it/s] 40%|████      | 161914/400000 [00:20<00:29, 8011.27it/s] 41%|████      | 162815/400000 [00:20<00:28, 8286.11it/s] 41%|████      | 163713/400000 [00:20<00:27, 8481.76it/s] 41%|████      | 164571/400000 [00:20<00:27, 8509.02it/s] 41%|████▏     | 165425/400000 [00:21<00:27, 8475.79it/s] 42%|████▏     | 166283/400000 [00:21<00:27, 8504.49it/s] 42%|████▏     | 167135/400000 [00:21<00:28, 8157.19it/s] 42%|████▏     | 167955/400000 [00:21<00:29, 7982.59it/s] 42%|████▏     | 168757/400000 [00:21<00:28, 7991.28it/s] 42%|████▏     | 169617/400000 [00:21<00:28, 8163.58it/s] 43%|████▎     | 170463/400000 [00:21<00:27, 8248.80it/s] 43%|████▎     | 171342/400000 [00:21<00:27, 8401.53it/s] 43%|████▎     | 172185/400000 [00:21<00:27, 8243.55it/s] 43%|████▎     | 173012/400000 [00:21<00:28, 8031.34it/s] 43%|████▎     | 173855/400000 [00:22<00:27, 8145.15it/s] 44%|████▎     | 174673/400000 [00:22<00:27, 8152.64it/s] 44%|████▍     | 175490/400000 [00:22<00:27, 8078.47it/s] 44%|████▍     | 176300/400000 [00:22<00:28, 7928.00it/s] 44%|████▍     | 177095/400000 [00:22<00:28, 7871.36it/s] 44%|████▍     | 177889/400000 [00:22<00:28, 7889.75it/s] 45%|████▍     | 178737/400000 [00:22<00:27, 8056.44it/s] 45%|████▍     | 179604/400000 [00:22<00:26, 8229.20it/s] 45%|████▌     | 180449/400000 [00:22<00:26, 8292.74it/s] 45%|████▌     | 181280/400000 [00:23<00:26, 8203.07it/s] 46%|████▌     | 182133/400000 [00:23<00:26, 8297.04it/s] 46%|████▌     | 183030/400000 [00:23<00:25, 8486.48it/s] 46%|████▌     | 183910/400000 [00:23<00:25, 8576.32it/s] 46%|████▌     | 184770/400000 [00:23<00:25, 8502.09it/s] 46%|████▋     | 185622/400000 [00:23<00:26, 8237.84it/s] 47%|████▋     | 186449/400000 [00:23<00:26, 8192.45it/s] 47%|████▋     | 187271/400000 [00:23<00:26, 8040.92it/s] 47%|████▋     | 188107/400000 [00:23<00:26, 8131.97it/s] 47%|████▋     | 188922/400000 [00:23<00:26, 8094.99it/s] 47%|████▋     | 189733/400000 [00:24<00:26, 8055.39it/s] 48%|████▊     | 190540/400000 [00:24<00:26, 7964.21it/s] 48%|████▊     | 191338/400000 [00:24<00:26, 7926.48it/s] 48%|████▊     | 192143/400000 [00:24<00:26, 7962.66it/s] 48%|████▊     | 192940/400000 [00:24<00:26, 7819.59it/s] 48%|████▊     | 193723/400000 [00:24<00:26, 7730.62it/s] 49%|████▊     | 194497/400000 [00:24<00:26, 7674.01it/s] 49%|████▉     | 195285/400000 [00:24<00:26, 7732.64it/s] 49%|████▉     | 196135/400000 [00:24<00:25, 7947.27it/s] 49%|████▉     | 196932/400000 [00:24<00:25, 7840.78it/s] 49%|████▉     | 197718/400000 [00:25<00:25, 7817.45it/s] 50%|████▉     | 198567/400000 [00:25<00:25, 8006.27it/s] 50%|████▉     | 199387/400000 [00:25<00:24, 8061.81it/s] 50%|█████     | 200235/400000 [00:25<00:24, 8180.55it/s] 50%|█████     | 201067/400000 [00:25<00:24, 8221.89it/s] 50%|█████     | 201892/400000 [00:25<00:24, 8229.05it/s] 51%|█████     | 202716/400000 [00:25<00:24, 8217.32it/s] 51%|█████     | 203572/400000 [00:25<00:23, 8316.57it/s] 51%|█████     | 204462/400000 [00:25<00:23, 8481.16it/s] 51%|█████▏    | 205312/400000 [00:25<00:23, 8450.03it/s] 52%|█████▏    | 206176/400000 [00:26<00:22, 8504.11it/s] 52%|█████▏    | 207103/400000 [00:26<00:22, 8719.12it/s] 52%|█████▏    | 207979/400000 [00:26<00:21, 8728.86it/s] 52%|█████▏    | 208854/400000 [00:26<00:22, 8470.63it/s] 52%|█████▏    | 209704/400000 [00:26<00:23, 8197.78it/s] 53%|█████▎    | 210528/400000 [00:26<00:23, 7938.98it/s] 53%|█████▎    | 211327/400000 [00:26<00:23, 7905.03it/s] 53%|█████▎    | 212134/400000 [00:26<00:23, 7953.61it/s] 53%|█████▎    | 212950/400000 [00:26<00:23, 8014.16it/s] 53%|█████▎    | 213765/400000 [00:26<00:23, 8053.46it/s] 54%|█████▎    | 214572/400000 [00:27<00:23, 8022.70it/s] 54%|█████▍    | 215397/400000 [00:27<00:22, 8087.65it/s] 54%|█████▍    | 216224/400000 [00:27<00:22, 8140.32it/s] 54%|█████▍    | 217079/400000 [00:27<00:22, 8256.09it/s] 54%|█████▍    | 217930/400000 [00:27<00:21, 8329.52it/s] 55%|█████▍    | 218780/400000 [00:27<00:21, 8379.93it/s] 55%|█████▍    | 219619/400000 [00:27<00:21, 8203.29it/s] 55%|█████▌    | 220498/400000 [00:27<00:21, 8370.57it/s] 55%|█████▌    | 221337/400000 [00:27<00:21, 8360.28it/s] 56%|█████▌    | 222175/400000 [00:28<00:21, 8151.56it/s] 56%|█████▌    | 223022/400000 [00:28<00:21, 8244.16it/s] 56%|█████▌    | 223882/400000 [00:28<00:21, 8345.94it/s] 56%|█████▌    | 224754/400000 [00:28<00:20, 8453.88it/s] 56%|█████▋    | 225601/400000 [00:28<00:21, 8240.46it/s] 57%|█████▋    | 226428/400000 [00:28<00:21, 7994.81it/s] 57%|█████▋    | 227263/400000 [00:28<00:21, 8097.36it/s] 57%|█████▋    | 228076/400000 [00:28<00:21, 8050.93it/s] 57%|█████▋    | 228947/400000 [00:28<00:20, 8236.73it/s] 57%|█████▋    | 229773/400000 [00:28<00:20, 8186.31it/s] 58%|█████▊    | 230594/400000 [00:29<00:21, 8025.57it/s] 58%|█████▊    | 231413/400000 [00:29<00:20, 8073.60it/s] 58%|█████▊    | 232222/400000 [00:29<00:22, 7578.94it/s] 58%|█████▊    | 232988/400000 [00:29<00:22, 7546.84it/s] 58%|█████▊    | 233748/400000 [00:29<00:22, 7517.36it/s] 59%|█████▊    | 234519/400000 [00:29<00:21, 7573.26it/s] 59%|█████▉    | 235324/400000 [00:29<00:21, 7705.96it/s] 59%|█████▉    | 236097/400000 [00:29<00:21, 7692.40it/s] 59%|█████▉    | 236921/400000 [00:29<00:20, 7847.12it/s] 59%|█████▉    | 237810/400000 [00:29<00:19, 8133.15it/s] 60%|█████▉    | 238628/400000 [00:30<00:20, 8051.73it/s] 60%|█████▉    | 239509/400000 [00:30<00:19, 8262.87it/s] 60%|██████    | 240339/400000 [00:30<00:19, 8251.27it/s] 60%|██████    | 241167/400000 [00:30<00:19, 7966.66it/s] 60%|██████    | 241968/400000 [00:30<00:20, 7781.31it/s] 61%|██████    | 242755/400000 [00:30<00:20, 7806.85it/s] 61%|██████    | 243571/400000 [00:30<00:19, 7907.25it/s] 61%|██████    | 244381/400000 [00:30<00:19, 7962.29it/s] 61%|██████▏   | 245190/400000 [00:30<00:19, 7998.99it/s] 61%|██████▏   | 245992/400000 [00:31<00:19, 7833.55it/s] 62%|██████▏   | 246777/400000 [00:31<00:19, 7816.60it/s] 62%|██████▏   | 247623/400000 [00:31<00:19, 7996.55it/s] 62%|██████▏   | 248453/400000 [00:31<00:18, 8084.17it/s] 62%|██████▏   | 249263/400000 [00:31<00:19, 7790.61it/s] 63%|██████▎   | 250046/400000 [00:31<00:19, 7757.25it/s] 63%|██████▎   | 250839/400000 [00:31<00:19, 7807.83it/s] 63%|██████▎   | 251638/400000 [00:31<00:18, 7859.06it/s] 63%|██████▎   | 252468/400000 [00:31<00:18, 7984.87it/s] 63%|██████▎   | 253298/400000 [00:31<00:18, 8075.96it/s] 64%|██████▎   | 254164/400000 [00:32<00:17, 8242.19it/s] 64%|██████▎   | 254990/400000 [00:32<00:17, 8122.49it/s] 64%|██████▍   | 255861/400000 [00:32<00:17, 8289.32it/s] 64%|██████▍   | 256692/400000 [00:32<00:17, 8288.28it/s] 64%|██████▍   | 257535/400000 [00:32<00:17, 8327.73it/s] 65%|██████▍   | 258369/400000 [00:32<00:17, 8163.39it/s] 65%|██████▍   | 259187/400000 [00:32<00:17, 8073.04it/s] 65%|██████▍   | 259996/400000 [00:32<00:17, 8036.89it/s] 65%|██████▌   | 260801/400000 [00:32<00:17, 7984.79it/s] 65%|██████▌   | 261673/400000 [00:32<00:16, 8191.07it/s] 66%|██████▌   | 262494/400000 [00:33<00:16, 8176.47it/s] 66%|██████▌   | 263313/400000 [00:33<00:17, 7991.16it/s] 66%|██████▌   | 264118/400000 [00:33<00:16, 8005.99it/s] 66%|██████▌   | 264925/400000 [00:33<00:16, 8022.49it/s] 66%|██████▋   | 265729/400000 [00:33<00:16, 8003.91it/s] 67%|██████▋   | 266531/400000 [00:33<00:16, 7864.09it/s] 67%|██████▋   | 267319/400000 [00:33<00:16, 7828.95it/s] 67%|██████▋   | 268137/400000 [00:33<00:16, 7930.09it/s] 67%|██████▋   | 268982/400000 [00:33<00:16, 8076.14it/s] 67%|██████▋   | 269793/400000 [00:33<00:16, 8085.88it/s] 68%|██████▊   | 270623/400000 [00:34<00:15, 8147.47it/s] 68%|██████▊   | 271463/400000 [00:34<00:15, 8220.42it/s] 68%|██████▊   | 272376/400000 [00:34<00:15, 8471.41it/s] 68%|██████▊   | 273246/400000 [00:34<00:14, 8536.33it/s] 69%|██████▊   | 274102/400000 [00:34<00:14, 8423.88it/s] 69%|██████▊   | 274946/400000 [00:34<00:15, 8227.39it/s] 69%|██████▉   | 275771/400000 [00:34<00:15, 8024.46it/s] 69%|██████▉   | 276577/400000 [00:34<00:15, 7876.77it/s] 69%|██████▉   | 277386/400000 [00:34<00:15, 7937.12it/s] 70%|██████▉   | 278188/400000 [00:34<00:15, 7959.81it/s] 70%|██████▉   | 278986/400000 [00:35<00:15, 7867.87it/s] 70%|██████▉   | 279774/400000 [00:35<00:15, 7792.34it/s] 70%|███████   | 280558/400000 [00:35<00:15, 7804.56it/s] 70%|███████   | 281340/400000 [00:35<00:15, 7678.63it/s] 71%|███████   | 282127/400000 [00:35<00:15, 7734.51it/s] 71%|███████   | 282903/400000 [00:35<00:15, 7736.38it/s] 71%|███████   | 283678/400000 [00:35<00:15, 7627.48it/s] 71%|███████   | 284464/400000 [00:35<00:15, 7693.42it/s] 71%|███████▏  | 285234/400000 [00:35<00:14, 7672.80it/s] 72%|███████▏  | 286049/400000 [00:36<00:14, 7809.93it/s] 72%|███████▏  | 286831/400000 [00:36<00:14, 7799.02it/s] 72%|███████▏  | 287612/400000 [00:36<00:14, 7654.83it/s] 72%|███████▏  | 288379/400000 [00:36<00:14, 7586.48it/s] 72%|███████▏  | 289148/400000 [00:36<00:14, 7616.26it/s] 72%|███████▏  | 289925/400000 [00:36<00:14, 7661.47it/s] 73%|███████▎  | 290709/400000 [00:36<00:14, 7711.74it/s] 73%|███████▎  | 291519/400000 [00:36<00:13, 7822.49it/s] 73%|███████▎  | 292302/400000 [00:36<00:13, 7786.81it/s] 73%|███████▎  | 293082/400000 [00:36<00:14, 7625.78it/s] 73%|███████▎  | 293861/400000 [00:37<00:13, 7673.69it/s] 74%|███████▎  | 294680/400000 [00:37<00:13, 7821.30it/s] 74%|███████▍  | 295516/400000 [00:37<00:13, 7975.47it/s] 74%|███████▍  | 296371/400000 [00:37<00:12, 8137.36it/s] 74%|███████▍  | 297208/400000 [00:37<00:12, 8203.37it/s] 75%|███████▍  | 298030/400000 [00:37<00:12, 8051.11it/s] 75%|███████▍  | 298837/400000 [00:37<00:12, 7910.84it/s] 75%|███████▍  | 299630/400000 [00:37<00:12, 7910.92it/s] 75%|███████▌  | 300423/400000 [00:37<00:12, 7781.72it/s] 75%|███████▌  | 301264/400000 [00:37<00:12, 7960.10it/s] 76%|███████▌  | 302125/400000 [00:38<00:12, 8142.60it/s] 76%|███████▌  | 302957/400000 [00:38<00:11, 8194.45it/s] 76%|███████▌  | 303779/400000 [00:38<00:11, 8186.81it/s] 76%|███████▌  | 304599/400000 [00:38<00:11, 8180.41it/s] 76%|███████▋  | 305428/400000 [00:38<00:11, 8212.89it/s] 77%|███████▋  | 306250/400000 [00:38<00:11, 8067.49it/s] 77%|███████▋  | 307058/400000 [00:38<00:11, 7963.76it/s] 77%|███████▋  | 307856/400000 [00:38<00:11, 7896.37it/s] 77%|███████▋  | 308647/400000 [00:38<00:11, 7896.11it/s] 77%|███████▋  | 309462/400000 [00:38<00:11, 7968.00it/s] 78%|███████▊  | 310284/400000 [00:39<00:11, 8039.07it/s] 78%|███████▊  | 311089/400000 [00:39<00:11, 7984.34it/s] 78%|███████▊  | 311888/400000 [00:39<00:11, 7875.97it/s] 78%|███████▊  | 312735/400000 [00:39<00:10, 8043.90it/s] 78%|███████▊  | 313542/400000 [00:39<00:10, 8050.71it/s] 79%|███████▊  | 314368/400000 [00:39<00:10, 8111.17it/s] 79%|███████▉  | 315180/400000 [00:39<00:10, 8066.70it/s] 79%|███████▉  | 316000/400000 [00:39<00:10, 8105.23it/s] 79%|███████▉  | 316811/400000 [00:39<00:10, 8092.74it/s] 79%|███████▉  | 317621/400000 [00:39<00:10, 7758.22it/s] 80%|███████▉  | 318400/400000 [00:40<00:10, 7731.58it/s] 80%|███████▉  | 319232/400000 [00:40<00:10, 7898.22it/s] 80%|████████  | 320091/400000 [00:40<00:09, 8092.07it/s] 80%|████████  | 320960/400000 [00:40<00:09, 8260.53it/s] 80%|████████  | 321789/400000 [00:40<00:09, 8115.50it/s] 81%|████████  | 322604/400000 [00:40<00:09, 7929.25it/s] 81%|████████  | 323400/400000 [00:40<00:09, 7766.61it/s] 81%|████████  | 324180/400000 [00:40<00:09, 7732.90it/s] 81%|████████  | 324978/400000 [00:40<00:09, 7800.62it/s] 81%|████████▏ | 325760/400000 [00:41<00:09, 7793.82it/s] 82%|████████▏ | 326557/400000 [00:41<00:09, 7844.99it/s] 82%|████████▏ | 327353/400000 [00:41<00:09, 7878.96it/s] 82%|████████▏ | 328195/400000 [00:41<00:08, 8031.47it/s] 82%|████████▏ | 329034/400000 [00:41<00:08, 8135.31it/s] 82%|████████▏ | 329849/400000 [00:41<00:08, 8071.06it/s] 83%|████████▎ | 330658/400000 [00:41<00:08, 7747.32it/s] 83%|████████▎ | 331437/400000 [00:41<00:08, 7722.02it/s] 83%|████████▎ | 332212/400000 [00:41<00:08, 7695.83it/s] 83%|████████▎ | 333008/400000 [00:41<00:08, 7769.81it/s] 83%|████████▎ | 333841/400000 [00:42<00:08, 7928.97it/s] 84%|████████▎ | 334636/400000 [00:42<00:08, 7879.04it/s] 84%|████████▍ | 335444/400000 [00:42<00:08, 7938.12it/s] 84%|████████▍ | 336239/400000 [00:42<00:08, 7886.10it/s] 84%|████████▍ | 337029/400000 [00:42<00:08, 7862.60it/s] 84%|████████▍ | 337816/400000 [00:42<00:08, 7689.15it/s] 85%|████████▍ | 338587/400000 [00:42<00:08, 7585.78it/s] 85%|████████▍ | 339360/400000 [00:42<00:07, 7628.43it/s] 85%|████████▌ | 340144/400000 [00:42<00:07, 7688.20it/s] 85%|████████▌ | 340914/400000 [00:42<00:07, 7438.49it/s] 85%|████████▌ | 341706/400000 [00:43<00:07, 7576.02it/s] 86%|████████▌ | 342528/400000 [00:43<00:07, 7755.67it/s] 86%|████████▌ | 343307/400000 [00:43<00:07, 7684.36it/s] 86%|████████▌ | 344144/400000 [00:43<00:07, 7875.54it/s] 86%|████████▌ | 344940/400000 [00:43<00:06, 7898.68it/s] 86%|████████▋ | 345777/400000 [00:43<00:06, 8032.25it/s] 87%|████████▋ | 346583/400000 [00:43<00:06, 7725.40it/s] 87%|████████▋ | 347360/400000 [00:43<00:07, 7020.05it/s] 87%|████████▋ | 348112/400000 [00:43<00:07, 7160.63it/s] 87%|████████▋ | 348867/400000 [00:44<00:07, 7272.82it/s] 87%|████████▋ | 349687/400000 [00:44<00:06, 7525.73it/s] 88%|████████▊ | 350552/400000 [00:44<00:06, 7830.16it/s] 88%|████████▊ | 351378/400000 [00:44<00:06, 7952.00it/s] 88%|████████▊ | 352284/400000 [00:44<00:05, 8254.46it/s] 88%|████████▊ | 353142/400000 [00:44<00:05, 8349.41it/s] 88%|████████▊ | 353990/400000 [00:44<00:05, 8387.93it/s] 89%|████████▊ | 354833/400000 [00:44<00:05, 8133.97it/s] 89%|████████▉ | 355651/400000 [00:44<00:05, 7943.84it/s] 89%|████████▉ | 356450/400000 [00:44<00:05, 7946.72it/s] 89%|████████▉ | 357298/400000 [00:45<00:05, 8098.31it/s] 90%|████████▉ | 358130/400000 [00:45<00:05, 8161.82it/s] 90%|████████▉ | 358984/400000 [00:45<00:04, 8268.61it/s] 90%|████████▉ | 359813/400000 [00:45<00:04, 8090.35it/s] 90%|█████████ | 360642/400000 [00:45<00:04, 8147.69it/s] 90%|█████████ | 361459/400000 [00:45<00:04, 8020.83it/s] 91%|█████████ | 362275/400000 [00:45<00:04, 8058.86it/s] 91%|█████████ | 363099/400000 [00:45<00:04, 8112.00it/s] 91%|█████████ | 363912/400000 [00:45<00:04, 8060.12it/s] 91%|█████████ | 364719/400000 [00:45<00:04, 7907.65it/s] 91%|█████████▏| 365511/400000 [00:46<00:04, 7901.86it/s] 92%|█████████▏| 366375/400000 [00:46<00:04, 8107.38it/s] 92%|█████████▏| 367214/400000 [00:46<00:04, 8188.44it/s] 92%|█████████▏| 368038/400000 [00:46<00:03, 8203.69it/s] 92%|█████████▏| 368860/400000 [00:46<00:03, 8202.91it/s] 92%|█████████▏| 369696/400000 [00:46<00:03, 8247.72it/s] 93%|█████████▎| 370545/400000 [00:46<00:03, 8317.71it/s] 93%|█████████▎| 371378/400000 [00:46<00:03, 8269.07it/s] 93%|█████████▎| 372211/400000 [00:46<00:03, 8286.25it/s] 93%|█████████▎| 373040/400000 [00:46<00:03, 8247.96it/s] 93%|█████████▎| 373926/400000 [00:47<00:03, 8420.11it/s] 94%|█████████▎| 374770/400000 [00:47<00:03, 8363.87it/s] 94%|█████████▍| 375608/400000 [00:47<00:02, 8179.78it/s] 94%|█████████▍| 376428/400000 [00:47<00:02, 8077.41it/s] 94%|█████████▍| 377264/400000 [00:47<00:02, 8159.74it/s] 95%|█████████▍| 378088/400000 [00:47<00:02, 8182.28it/s] 95%|█████████▍| 378924/400000 [00:47<00:02, 8231.67it/s] 95%|█████████▍| 379748/400000 [00:47<00:02, 8039.98it/s] 95%|█████████▌| 380554/400000 [00:47<00:02, 7884.90it/s] 95%|█████████▌| 381345/400000 [00:48<00:02, 7774.84it/s] 96%|█████████▌| 382179/400000 [00:48<00:02, 7935.60it/s] 96%|█████████▌| 382975/400000 [00:48<00:02, 7768.85it/s] 96%|█████████▌| 383775/400000 [00:48<00:02, 7835.31it/s] 96%|█████████▌| 384561/400000 [00:48<00:02, 7651.25it/s] 96%|█████████▋| 385332/400000 [00:48<00:01, 7668.74it/s] 97%|█████████▋| 386121/400000 [00:48<00:01, 7732.15it/s] 97%|█████████▋| 386942/400000 [00:48<00:01, 7869.43it/s] 97%|█████████▋| 387757/400000 [00:48<00:01, 7951.43it/s] 97%|█████████▋| 388554/400000 [00:48<00:01, 7952.31it/s] 97%|█████████▋| 389351/400000 [00:49<00:01, 7780.25it/s] 98%|█████████▊| 390169/400000 [00:49<00:01, 7893.97it/s] 98%|█████████▊| 391053/400000 [00:49<00:01, 8154.35it/s] 98%|█████████▊| 391883/400000 [00:49<00:00, 8196.58it/s] 98%|█████████▊| 392720/400000 [00:49<00:00, 8245.48it/s] 98%|█████████▊| 393566/400000 [00:49<00:00, 8306.21it/s] 99%|█████████▊| 394398/400000 [00:49<00:00, 8221.64it/s] 99%|█████████▉| 395235/400000 [00:49<00:00, 8265.46it/s] 99%|█████████▉| 396072/400000 [00:49<00:00, 8296.39it/s] 99%|█████████▉| 396903/400000 [00:49<00:00, 7960.31it/s] 99%|█████████▉| 397770/400000 [00:50<00:00, 8159.08it/s]100%|█████████▉| 398596/400000 [00:50<00:00, 8186.93it/s]100%|█████████▉| 399481/400000 [00:50<00:00, 8374.94it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7951.39it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f90003de828>, <torchtext.data.dataset.TabularDataset object at 0x7f90003de978>, <torchtext.vocab.Vocab object at 0x7f90003de898>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 11.58 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 11.58 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.56 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.56 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.02 file/s]2020-06-13 22:33:05.842442: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 22:33:05.846783: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-13 22:33:05.847016: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5618d0a81c20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 22:33:05.847033: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2564096/9912422 [00:00<00:00, 25145014.44it/s]9920512it [00:00, 31885028.36it/s]                             
0it [00:00, ?it/s]32768it [00:00, 514876.05it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160419.72it/s]1654784it [00:00, 11585620.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 189163.94it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
