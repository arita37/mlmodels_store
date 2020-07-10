
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fbc54bcc0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fbc54bcc0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fbcbff151e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fbcbff151e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fbcde25de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fbcde25de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbc6d243620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbc6d243620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbc6d243620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148401.36it/s] 76%|███████▋  | 7561216/9912422 [00:00<00:11, 211811.17it/s]9920512it [00:00, 41082797.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 489352.14it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 163198.15it/s]1654784it [00:00, 11837960.30it/s]                         
0it [00:00, ?it/s]8192it [00:00, 139370.63it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc6a79b940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc54276be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbc6d243268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbc6d243268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fbc6d243268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:08,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:08,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:07,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:07,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:07,  2.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 11.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 20.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 32.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 38.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 59.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 59.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 62.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 66.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 67.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.39 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.39 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.40 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ url]
0 examples [00:00, ? examples/s]2020-07-10 00:10:31.788949: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-10 00:10:31.895266: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-10 00:10:31.895533: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5611bbf6d2f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-10 00:10:31.895550: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.06 examples/s]93 examples [00:00,  2.94 examples/s]187 examples [00:00,  4.19 examples/s]281 examples [00:00,  5.98 examples/s]380 examples [00:00,  8.52 examples/s]488 examples [00:00, 12.13 examples/s]593 examples [00:01, 17.24 examples/s]696 examples [00:01, 24.45 examples/s]795 examples [00:01, 34.56 examples/s]891 examples [00:01, 48.63 examples/s]991 examples [00:01, 68.04 examples/s]1095 examples [00:01, 94.54 examples/s]1197 examples [00:01, 129.89 examples/s]1299 examples [00:01, 175.94 examples/s]1400 examples [00:01, 233.10 examples/s]1500 examples [00:02, 301.68 examples/s]1599 examples [00:02, 380.56 examples/s]1698 examples [00:02, 463.48 examples/s]1795 examples [00:02, 547.60 examples/s]1893 examples [00:02, 631.11 examples/s]1990 examples [00:02, 700.52 examples/s]2087 examples [00:02, 761.24 examples/s]2183 examples [00:02, 807.21 examples/s]2279 examples [00:02, 846.94 examples/s]2375 examples [00:02, 877.86 examples/s]2471 examples [00:03, 899.62 examples/s]2568 examples [00:03, 917.78 examples/s]2664 examples [00:03, 899.58 examples/s]2766 examples [00:03, 932.04 examples/s]2864 examples [00:03, 942.99 examples/s]2960 examples [00:03, 925.24 examples/s]3054 examples [00:03, 927.31 examples/s]3148 examples [00:03, 915.34 examples/s]3241 examples [00:03, 901.32 examples/s]3340 examples [00:03, 924.99 examples/s]3434 examples [00:04, 914.92 examples/s]3530 examples [00:04, 927.38 examples/s]3624 examples [00:04, 923.48 examples/s]3719 examples [00:04, 930.28 examples/s]3816 examples [00:04, 939.45 examples/s]3911 examples [00:04, 941.41 examples/s]4006 examples [00:04, 939.66 examples/s]4103 examples [00:04, 946.23 examples/s]4209 examples [00:04, 976.96 examples/s]4316 examples [00:04, 1002.49 examples/s]4425 examples [00:05, 1025.60 examples/s]4534 examples [00:05, 1042.67 examples/s]4639 examples [00:05, 1017.45 examples/s]4742 examples [00:05, 1009.80 examples/s]4849 examples [00:05, 1025.75 examples/s]4955 examples [00:05, 1034.59 examples/s]5059 examples [00:05, 1030.07 examples/s]5163 examples [00:05, 1031.07 examples/s]5268 examples [00:05, 1035.86 examples/s]5380 examples [00:05, 1057.84 examples/s]5486 examples [00:06, 1057.26 examples/s]5592 examples [00:06, 1046.52 examples/s]5697 examples [00:06, 1024.92 examples/s]5806 examples [00:06, 1042.24 examples/s]5911 examples [00:06, 1037.33 examples/s]6015 examples [00:06, 1037.15 examples/s]6119 examples [00:06, 1037.28 examples/s]6223 examples [00:06, 985.73 examples/s] 6323 examples [00:06, 981.06 examples/s]6423 examples [00:07, 986.28 examples/s]6522 examples [00:07, 965.56 examples/s]6621 examples [00:07, 967.47 examples/s]6718 examples [00:07, 944.88 examples/s]6819 examples [00:07, 963.20 examples/s]6919 examples [00:07, 971.98 examples/s]7017 examples [00:07, 972.13 examples/s]7115 examples [00:07, 971.58 examples/s]7213 examples [00:07, 961.50 examples/s]7314 examples [00:07, 974.90 examples/s]7417 examples [00:08, 987.60 examples/s]7519 examples [00:08, 995.42 examples/s]7619 examples [00:08, 976.13 examples/s]7720 examples [00:08, 983.65 examples/s]7819 examples [00:08, 965.25 examples/s]7916 examples [00:08, 964.89 examples/s]8017 examples [00:08, 975.40 examples/s]8120 examples [00:08, 986.98 examples/s]8219 examples [00:08, 978.04 examples/s]8325 examples [00:08, 998.74 examples/s]8426 examples [00:09, 993.62 examples/s]8527 examples [00:09, 996.32 examples/s]8630 examples [00:09, 1006.18 examples/s]8731 examples [00:09, 991.96 examples/s] 8832 examples [00:09, 995.33 examples/s]8934 examples [00:09, 1001.28 examples/s]9037 examples [00:09, 1006.78 examples/s]9138 examples [00:09, 1000.40 examples/s]9239 examples [00:09, 979.64 examples/s] 9338 examples [00:09, 977.18 examples/s]9436 examples [00:10, 977.45 examples/s]9534 examples [00:10, 963.54 examples/s]9634 examples [00:10, 972.94 examples/s]9737 examples [00:10, 985.89 examples/s]9844 examples [00:10, 1008.95 examples/s]9950 examples [00:10, 1023.39 examples/s]10053 examples [00:10, 955.75 examples/s]10150 examples [00:10, 946.19 examples/s]10246 examples [00:10, 945.21 examples/s]10348 examples [00:11, 963.32 examples/s]10451 examples [00:11, 980.39 examples/s]10550 examples [00:11, 965.22 examples/s]10647 examples [00:11, 937.62 examples/s]10743 examples [00:11, 943.45 examples/s]10840 examples [00:11, 949.01 examples/s]10943 examples [00:11, 971.39 examples/s]11042 examples [00:11, 974.26 examples/s]11140 examples [00:11, 959.11 examples/s]11237 examples [00:11, 955.60 examples/s]11338 examples [00:12, 970.10 examples/s]11440 examples [00:12, 984.11 examples/s]11539 examples [00:12, 959.65 examples/s]11636 examples [00:12, 918.77 examples/s]11729 examples [00:12, 901.54 examples/s]11820 examples [00:12, 901.30 examples/s]11911 examples [00:12, 882.65 examples/s]12000 examples [00:12, 870.81 examples/s]12088 examples [00:12, 862.38 examples/s]12175 examples [00:13, 863.33 examples/s]12267 examples [00:13, 877.11 examples/s]12360 examples [00:13, 891.73 examples/s]12450 examples [00:13, 858.36 examples/s]12543 examples [00:13, 877.82 examples/s]12637 examples [00:13, 894.15 examples/s]12728 examples [00:13, 898.33 examples/s]12822 examples [00:13, 907.13 examples/s]12918 examples [00:13, 921.25 examples/s]13013 examples [00:13, 929.62 examples/s]13107 examples [00:14, 929.18 examples/s]13204 examples [00:14, 939.04 examples/s]13299 examples [00:14, 937.65 examples/s]13394 examples [00:14, 939.32 examples/s]13488 examples [00:14, 936.59 examples/s]13583 examples [00:14, 938.12 examples/s]13684 examples [00:14, 958.17 examples/s]13785 examples [00:14, 970.53 examples/s]13886 examples [00:14, 979.81 examples/s]13985 examples [00:14, 971.02 examples/s]14088 examples [00:15, 987.08 examples/s]14187 examples [00:15, 985.39 examples/s]14286 examples [00:15, 974.10 examples/s]14385 examples [00:15, 977.91 examples/s]14483 examples [00:15, 970.81 examples/s]14585 examples [00:15, 984.36 examples/s]14684 examples [00:15, 983.70 examples/s]14784 examples [00:15, 986.70 examples/s]14883 examples [00:15, 963.54 examples/s]14980 examples [00:15, 951.19 examples/s]15091 examples [00:16, 993.07 examples/s]15199 examples [00:16, 1016.28 examples/s]15303 examples [00:16, 1021.79 examples/s]15406 examples [00:16, 982.08 examples/s] 15505 examples [00:16, 960.73 examples/s]15602 examples [00:16, 962.01 examples/s]15699 examples [00:16, 954.60 examples/s]15800 examples [00:16, 968.34 examples/s]15898 examples [00:16, 956.33 examples/s]15994 examples [00:17, 941.01 examples/s]16097 examples [00:17, 965.75 examples/s]16196 examples [00:17, 969.71 examples/s]16300 examples [00:17, 989.34 examples/s]16400 examples [00:17, 978.73 examples/s]16499 examples [00:17, 963.18 examples/s]16598 examples [00:17, 968.17 examples/s]16698 examples [00:17, 976.06 examples/s]16796 examples [00:17, 962.24 examples/s]16896 examples [00:17, 972.36 examples/s]17001 examples [00:18, 993.51 examples/s]17107 examples [00:18, 1012.56 examples/s]17211 examples [00:18, 1018.29 examples/s]17318 examples [00:18, 1031.35 examples/s]17422 examples [00:18, 1033.23 examples/s]17527 examples [00:18, 1036.57 examples/s]17634 examples [00:18, 1045.59 examples/s]17741 examples [00:18, 1050.92 examples/s]17847 examples [00:18, 1039.32 examples/s]17952 examples [00:18, 1033.97 examples/s]18056 examples [00:19, 1017.53 examples/s]18161 examples [00:19, 1025.58 examples/s]18266 examples [00:19, 1031.89 examples/s]18370 examples [00:19, 1027.94 examples/s]18473 examples [00:19, 996.66 examples/s] 18581 examples [00:19, 1018.98 examples/s]18687 examples [00:19, 1030.70 examples/s]18795 examples [00:19, 1043.56 examples/s]18902 examples [00:19, 1049.30 examples/s]19008 examples [00:19, 1035.55 examples/s]19113 examples [00:20, 1039.20 examples/s]19220 examples [00:20, 1047.74 examples/s]19326 examples [00:20, 1049.52 examples/s]19432 examples [00:20, 1033.49 examples/s]19536 examples [00:20, 1023.08 examples/s]19641 examples [00:20, 1029.67 examples/s]19748 examples [00:20, 1041.16 examples/s]19854 examples [00:20, 1045.94 examples/s]19960 examples [00:20, 1048.67 examples/s]20065 examples [00:21, 962.93 examples/s] 20169 examples [00:21, 982.89 examples/s]20272 examples [00:21, 994.38 examples/s]20373 examples [00:21, 987.12 examples/s]20474 examples [00:21, 993.06 examples/s]20577 examples [00:21, 1003.49 examples/s]20679 examples [00:21, 1006.97 examples/s]20780 examples [00:21, 998.27 examples/s] 20884 examples [00:21, 1009.91 examples/s]20992 examples [00:21, 1029.54 examples/s]21096 examples [00:22, 1001.93 examples/s]21197 examples [00:22, 978.20 examples/s] 21296 examples [00:22, 979.49 examples/s]21396 examples [00:22, 983.54 examples/s]21495 examples [00:22, 974.99 examples/s]21593 examples [00:22, 944.96 examples/s]21698 examples [00:22, 972.48 examples/s]21802 examples [00:22, 990.01 examples/s]21907 examples [00:22, 1006.93 examples/s]22012 examples [00:22, 1016.84 examples/s]22114 examples [00:23, 1015.22 examples/s]22220 examples [00:23, 1025.55 examples/s]22325 examples [00:23, 1030.18 examples/s]22429 examples [00:23, 1031.71 examples/s]22533 examples [00:23, 1032.19 examples/s]22637 examples [00:23, 1013.58 examples/s]22739 examples [00:23, 1011.03 examples/s]22841 examples [00:23, 1006.03 examples/s]22942 examples [00:23, 992.01 examples/s] 23042 examples [00:23, 988.84 examples/s]23141 examples [00:24, 972.55 examples/s]23239 examples [00:24, 965.94 examples/s]23337 examples [00:24, 969.16 examples/s]23438 examples [00:24, 978.73 examples/s]23536 examples [00:24, 972.91 examples/s]23634 examples [00:24, 951.63 examples/s]23730 examples [00:24, 943.54 examples/s]23830 examples [00:24, 957.60 examples/s]23931 examples [00:24, 970.41 examples/s]24032 examples [00:25, 981.92 examples/s]24131 examples [00:25, 965.24 examples/s]24228 examples [00:25, 963.98 examples/s]24330 examples [00:25, 977.68 examples/s]24428 examples [00:25, 970.60 examples/s]24526 examples [00:25, 946.20 examples/s]24624 examples [00:25, 955.34 examples/s]24723 examples [00:25, 964.93 examples/s]24824 examples [00:25, 975.39 examples/s]24923 examples [00:25, 978.74 examples/s]25026 examples [00:26, 991.71 examples/s]25126 examples [00:26, 980.64 examples/s]25225 examples [00:26, 979.43 examples/s]25324 examples [00:26, 971.22 examples/s]25425 examples [00:26, 980.96 examples/s]25529 examples [00:26, 997.74 examples/s]25632 examples [00:26, 1005.50 examples/s]25734 examples [00:26, 1009.29 examples/s]25836 examples [00:26, 1009.82 examples/s]25938 examples [00:26, 991.63 examples/s] 26038 examples [00:27, 991.17 examples/s]26138 examples [00:27, 988.95 examples/s]26237 examples [00:27, 988.76 examples/s]26338 examples [00:27, 993.24 examples/s]26442 examples [00:27, 1006.68 examples/s]26549 examples [00:27, 1021.90 examples/s]26652 examples [00:27, 1000.74 examples/s]26753 examples [00:27, 1001.49 examples/s]26854 examples [00:27, 992.21 examples/s] 26954 examples [00:27, 987.91 examples/s]27053 examples [00:28, 980.22 examples/s]27152 examples [00:28, 976.12 examples/s]27252 examples [00:28, 982.56 examples/s]27353 examples [00:28, 988.53 examples/s]27454 examples [00:28, 993.96 examples/s]27554 examples [00:28, 974.76 examples/s]27652 examples [00:28, 970.57 examples/s]27752 examples [00:28, 977.91 examples/s]27857 examples [00:28, 996.23 examples/s]27957 examples [00:28, 991.09 examples/s]28064 examples [00:29, 1010.71 examples/s]28166 examples [00:29, 1004.40 examples/s]28267 examples [00:29, 999.67 examples/s] 28368 examples [00:29, 1002.05 examples/s]28469 examples [00:29, 999.27 examples/s] 28570 examples [00:29, 1001.36 examples/s]28671 examples [00:29, 1002.83 examples/s]28772 examples [00:29, 999.27 examples/s] 28874 examples [00:29, 1004.25 examples/s]28975 examples [00:29, 992.89 examples/s] 29075 examples [00:30, 982.41 examples/s]29174 examples [00:30, 973.01 examples/s]29273 examples [00:30, 976.68 examples/s]29371 examples [00:30, 970.61 examples/s]29473 examples [00:30, 982.69 examples/s]29574 examples [00:30, 989.48 examples/s]29674 examples [00:30, 979.44 examples/s]29775 examples [00:30, 986.69 examples/s]29880 examples [00:30, 1003.53 examples/s]29981 examples [00:31, 1001.29 examples/s]30082 examples [00:31, 938.49 examples/s] 30183 examples [00:31, 956.65 examples/s]30281 examples [00:31, 962.78 examples/s]30379 examples [00:31, 967.49 examples/s]30477 examples [00:31, 963.94 examples/s]30574 examples [00:31, 946.07 examples/s]30669 examples [00:31, 930.30 examples/s]30764 examples [00:31, 934.40 examples/s]30860 examples [00:31, 938.93 examples/s]30955 examples [00:32, 942.15 examples/s]31053 examples [00:32, 951.07 examples/s]31149 examples [00:32, 947.20 examples/s]31249 examples [00:32, 962.29 examples/s]31349 examples [00:32, 972.35 examples/s]31447 examples [00:32, 972.90 examples/s]31551 examples [00:32, 989.38 examples/s]31651 examples [00:32, 977.46 examples/s]31749 examples [00:32, 977.31 examples/s]31851 examples [00:32, 987.62 examples/s]31950 examples [00:33, 985.91 examples/s]32049 examples [00:33, 976.87 examples/s]32147 examples [00:33, 957.59 examples/s]32244 examples [00:33, 957.90 examples/s]32340 examples [00:33, 952.49 examples/s]32437 examples [00:33, 954.12 examples/s]32533 examples [00:33, 955.64 examples/s]32629 examples [00:33, 939.61 examples/s]32724 examples [00:33, 941.50 examples/s]32819 examples [00:33, 936.60 examples/s]32920 examples [00:34, 956.00 examples/s]33017 examples [00:34, 958.84 examples/s]33113 examples [00:34, 952.99 examples/s]33213 examples [00:34, 965.50 examples/s]33313 examples [00:34, 973.91 examples/s]33411 examples [00:34, 970.66 examples/s]33509 examples [00:34, 946.71 examples/s]33604 examples [00:34, 947.49 examples/s]33699 examples [00:34, 945.61 examples/s]33801 examples [00:35, 965.09 examples/s]33905 examples [00:35, 984.87 examples/s]34005 examples [00:35, 987.64 examples/s]34104 examples [00:35, 982.84 examples/s]34203 examples [00:35, 983.22 examples/s]34302 examples [00:35, 982.28 examples/s]34403 examples [00:35, 990.22 examples/s]34506 examples [00:35, 999.47 examples/s]34607 examples [00:35, 988.15 examples/s]34706 examples [00:35, 983.65 examples/s]34805 examples [00:36, 964.90 examples/s]34903 examples [00:36, 968.39 examples/s]35004 examples [00:36, 977.89 examples/s]35102 examples [00:36, 958.18 examples/s]35198 examples [00:36, 958.49 examples/s]35294 examples [00:36, 958.24 examples/s]35390 examples [00:36, 954.17 examples/s]35487 examples [00:36, 958.84 examples/s]35589 examples [00:36, 973.75 examples/s]35687 examples [00:36, 969.83 examples/s]35785 examples [00:37, 967.16 examples/s]35886 examples [00:37, 978.02 examples/s]35984 examples [00:37, 978.50 examples/s]36082 examples [00:37, 940.73 examples/s]36182 examples [00:37, 956.95 examples/s]36279 examples [00:37, 956.96 examples/s]36378 examples [00:37, 962.93 examples/s]36475 examples [00:37, 934.99 examples/s]36573 examples [00:37, 945.75 examples/s]36673 examples [00:37, 959.28 examples/s]36770 examples [00:38, 960.39 examples/s]36867 examples [00:38, 955.94 examples/s]36963 examples [00:38, 954.31 examples/s]37062 examples [00:38, 962.35 examples/s]37159 examples [00:38, 955.82 examples/s]37257 examples [00:38, 958.27 examples/s]37355 examples [00:38, 962.68 examples/s]37452 examples [00:38, 937.60 examples/s]37547 examples [00:38, 939.24 examples/s]37642 examples [00:39, 935.07 examples/s]37736 examples [00:39, 928.42 examples/s]37830 examples [00:39, 929.77 examples/s]37924 examples [00:39, 911.91 examples/s]38020 examples [00:39, 923.36 examples/s]38115 examples [00:39, 929.13 examples/s]38211 examples [00:39, 938.01 examples/s]38309 examples [00:39, 950.10 examples/s]38405 examples [00:39, 937.80 examples/s]38505 examples [00:39, 953.87 examples/s]38605 examples [00:40, 964.71 examples/s]38702 examples [00:40, 954.18 examples/s]38801 examples [00:40, 963.21 examples/s]38898 examples [00:40, 952.61 examples/s]38996 examples [00:40, 960.29 examples/s]39094 examples [00:40, 963.85 examples/s]39191 examples [00:40, 955.88 examples/s]39287 examples [00:40, 934.37 examples/s]39383 examples [00:40, 941.60 examples/s]39482 examples [00:40, 954.63 examples/s]39585 examples [00:41, 975.59 examples/s]39683 examples [00:41, 970.54 examples/s]39781 examples [00:41, 969.68 examples/s]39879 examples [00:41, 949.72 examples/s]39975 examples [00:41, 950.07 examples/s]40071 examples [00:41, 897.03 examples/s]40171 examples [00:41, 923.83 examples/s]40267 examples [00:41, 933.18 examples/s]40361 examples [00:41, 933.18 examples/s]40455 examples [00:41, 913.55 examples/s]40551 examples [00:42, 926.47 examples/s]40650 examples [00:42, 944.14 examples/s]40747 examples [00:42, 950.22 examples/s]40843 examples [00:42, 898.34 examples/s]40940 examples [00:42, 917.04 examples/s]41033 examples [00:42, 910.25 examples/s]41129 examples [00:42, 921.44 examples/s]41222 examples [00:42, 923.48 examples/s]41315 examples [00:42, 911.50 examples/s]41412 examples [00:43, 927.53 examples/s]41508 examples [00:43, 934.86 examples/s]41610 examples [00:43, 957.12 examples/s]41707 examples [00:43, 958.91 examples/s]41804 examples [00:43, 952.75 examples/s]41902 examples [00:43, 958.46 examples/s]41998 examples [00:43, 955.89 examples/s]42098 examples [00:43, 967.61 examples/s]42195 examples [00:43, 931.37 examples/s]42289 examples [00:43, 924.40 examples/s]42386 examples [00:44, 936.36 examples/s]42484 examples [00:44, 946.98 examples/s]42581 examples [00:44, 952.67 examples/s]42678 examples [00:44, 956.71 examples/s]42774 examples [00:44, 945.15 examples/s]42876 examples [00:44, 964.93 examples/s]42974 examples [00:44, 966.96 examples/s]43072 examples [00:44, 968.75 examples/s]43172 examples [00:44, 975.77 examples/s]43270 examples [00:44, 971.63 examples/s]43368 examples [00:45, 973.94 examples/s]43466 examples [00:45, 964.62 examples/s]43563 examples [00:45, 956.06 examples/s]43659 examples [00:45, 952.34 examples/s]43755 examples [00:45, 947.67 examples/s]43851 examples [00:45, 949.45 examples/s]43946 examples [00:45, 943.58 examples/s]44046 examples [00:45, 959.61 examples/s]44143 examples [00:45, 946.96 examples/s]44238 examples [00:45, 941.55 examples/s]44334 examples [00:46, 944.82 examples/s]44431 examples [00:46, 951.09 examples/s]44527 examples [00:46, 946.02 examples/s]44626 examples [00:46, 956.21 examples/s]44722 examples [00:46, 954.02 examples/s]44818 examples [00:46, 950.81 examples/s]44914 examples [00:46, 950.98 examples/s]45010 examples [00:46, 936.67 examples/s]45104 examples [00:46, 917.45 examples/s]45196 examples [00:46, 916.10 examples/s]45292 examples [00:47, 927.33 examples/s]45390 examples [00:47, 940.77 examples/s]45488 examples [00:47, 952.11 examples/s]45584 examples [00:47, 945.33 examples/s]45679 examples [00:47, 937.03 examples/s]45773 examples [00:47, 936.46 examples/s]45869 examples [00:47, 941.00 examples/s]45964 examples [00:47, 934.30 examples/s]46059 examples [00:47, 936.32 examples/s]46153 examples [00:48, 927.17 examples/s]46246 examples [00:48, 922.80 examples/s]46339 examples [00:48, 919.85 examples/s]46432 examples [00:48, 901.63 examples/s]46526 examples [00:48, 909.78 examples/s]46621 examples [00:48, 918.99 examples/s]46721 examples [00:48, 939.91 examples/s]46817 examples [00:48, 945.40 examples/s]46912 examples [00:48, 939.64 examples/s]47007 examples [00:48, 941.18 examples/s]47102 examples [00:49, 934.37 examples/s]47196 examples [00:49, 935.88 examples/s]47290 examples [00:49, 931.14 examples/s]47384 examples [00:49, 933.58 examples/s]47479 examples [00:49, 937.44 examples/s]47573 examples [00:49, 929.46 examples/s]47666 examples [00:49, 921.94 examples/s]47762 examples [00:49, 930.37 examples/s]47856 examples [00:49, 920.04 examples/s]47949 examples [00:49, 909.79 examples/s]48041 examples [00:50, 907.46 examples/s]48133 examples [00:50, 911.01 examples/s]48230 examples [00:50, 927.22 examples/s]48328 examples [00:50, 939.70 examples/s]48430 examples [00:50, 961.02 examples/s]48528 examples [00:50, 964.16 examples/s]48626 examples [00:50, 968.11 examples/s]48726 examples [00:50, 975.70 examples/s]48826 examples [00:50, 979.91 examples/s]48925 examples [00:50, 971.20 examples/s]49023 examples [00:51, 966.61 examples/s]49121 examples [00:51, 967.97 examples/s]49218 examples [00:51, 959.41 examples/s]49315 examples [00:51, 960.31 examples/s]49412 examples [00:51, 953.39 examples/s]49508 examples [00:51, 945.55 examples/s]49605 examples [00:51, 950.30 examples/s]49701 examples [00:51, 945.04 examples/s]49796 examples [00:51, 945.54 examples/s]49892 examples [00:51, 949.00 examples/s]49987 examples [00:52, 933.13 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7598/50000 [00:00<00:00, 75979.17 examples/s] 40%|████      | 20019/50000 [00:00<00:00, 85996.61 examples/s] 63%|██████▎   | 31605/50000 [00:00<00:00, 93200.02 examples/s] 89%|████████▉ | 44464/50000 [00:00<00:00, 101585.75 examples/s]                                                                0 examples [00:00, ? examples/s]76 examples [00:00, 753.63 examples/s]172 examples [00:00, 805.17 examples/s]261 examples [00:00, 828.64 examples/s]349 examples [00:00, 841.82 examples/s]440 examples [00:00, 860.84 examples/s]538 examples [00:00, 892.96 examples/s]636 examples [00:00, 915.46 examples/s]732 examples [00:00, 926.19 examples/s]830 examples [00:00, 941.42 examples/s]922 examples [00:01, 934.80 examples/s]1016 examples [00:01, 936.35 examples/s]1114 examples [00:01, 947.75 examples/s]1211 examples [00:01, 954.01 examples/s]1306 examples [00:01, 942.10 examples/s]1402 examples [00:01, 946.14 examples/s]1500 examples [00:01, 954.57 examples/s]1598 examples [00:01, 961.32 examples/s]1695 examples [00:01, 963.30 examples/s]1792 examples [00:01, 959.07 examples/s]1888 examples [00:02, 957.47 examples/s]1984 examples [00:02, 957.69 examples/s]2086 examples [00:02, 972.98 examples/s]2187 examples [00:02, 981.19 examples/s]2287 examples [00:02, 984.44 examples/s]2386 examples [00:02, 974.87 examples/s]2484 examples [00:02, 973.02 examples/s]2582 examples [00:02, 967.94 examples/s]2679 examples [00:02, 965.21 examples/s]2777 examples [00:02, 967.03 examples/s]2874 examples [00:03, 952.89 examples/s]2970 examples [00:03, 953.44 examples/s]3068 examples [00:03, 960.42 examples/s]3165 examples [00:03, 960.11 examples/s]3262 examples [00:03, 946.13 examples/s]3363 examples [00:03, 964.41 examples/s]3464 examples [00:03, 975.14 examples/s]3562 examples [00:03, 975.47 examples/s]3662 examples [00:03, 980.30 examples/s]3762 examples [00:03, 985.01 examples/s]3861 examples [00:04, 978.13 examples/s]3959 examples [00:04, 977.32 examples/s]4057 examples [00:04, 967.03 examples/s]4158 examples [00:04, 978.81 examples/s]4257 examples [00:04, 980.55 examples/s]4356 examples [00:04, 967.51 examples/s]4453 examples [00:04, 967.55 examples/s]4550 examples [00:04, 963.04 examples/s]4650 examples [00:04, 972.20 examples/s]4748 examples [00:04, 964.79 examples/s]4845 examples [00:05, 956.72 examples/s]4946 examples [00:05, 970.90 examples/s]5046 examples [00:05, 978.60 examples/s]5146 examples [00:05, 982.30 examples/s]5245 examples [00:05, 968.55 examples/s]5342 examples [00:05, 968.93 examples/s]5442 examples [00:05, 976.91 examples/s]5544 examples [00:05, 987.45 examples/s]5643 examples [00:05, 979.09 examples/s]5745 examples [00:05, 988.26 examples/s]5844 examples [00:06, 983.56 examples/s]5943 examples [00:06, 970.59 examples/s]6041 examples [00:06, 971.60 examples/s]6139 examples [00:06, 970.07 examples/s]6237 examples [00:06, 948.41 examples/s]6332 examples [00:06, 948.03 examples/s]6435 examples [00:06, 969.26 examples/s]6535 examples [00:06, 975.77 examples/s]6634 examples [00:06, 979.72 examples/s]6733 examples [00:06, 975.78 examples/s]6831 examples [00:07, 969.49 examples/s]6931 examples [00:07, 975.89 examples/s]7029 examples [00:07, 976.47 examples/s]7130 examples [00:07, 984.45 examples/s]7230 examples [00:07, 987.56 examples/s]7329 examples [00:07, 972.59 examples/s]7427 examples [00:07, 960.15 examples/s]7524 examples [00:07, 962.20 examples/s]7622 examples [00:07, 965.55 examples/s]7719 examples [00:08, 963.21 examples/s]7816 examples [00:08, 954.61 examples/s]7913 examples [00:08, 956.68 examples/s]8013 examples [00:08, 967.25 examples/s]8110 examples [00:08, 963.35 examples/s]8209 examples [00:08, 969.27 examples/s]8306 examples [00:08, 968.81 examples/s]8403 examples [00:08, 958.71 examples/s]8502 examples [00:08, 966.03 examples/s]8599 examples [00:08, 964.46 examples/s]8697 examples [00:09, 967.69 examples/s]8794 examples [00:09, 946.36 examples/s]8889 examples [00:09, 937.92 examples/s]8994 examples [00:09, 966.82 examples/s]9093 examples [00:09, 972.73 examples/s]9191 examples [00:09, 942.63 examples/s]9286 examples [00:09, 940.02 examples/s]9385 examples [00:09, 953.24 examples/s]9481 examples [00:09, 953.24 examples/s]9577 examples [00:09, 954.71 examples/s]9673 examples [00:10, 952.32 examples/s]9769 examples [00:10, 945.87 examples/s]9867 examples [00:10, 955.30 examples/s]9968 examples [00:10, 968.46 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZYSS21/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZYSS21/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbc6d243620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbc6d243620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbc6d243620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbbf66c7550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbbf6705550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbc6d243620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbc6d243620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbc6d243620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc542765c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc54276358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fbbe4de3378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fbbe4de3378> 

  function with postional parmater data_info <function split_train_valid at 0x7fbbe4de3378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fbbe4de3488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fbbe4de3488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fbbe4de3488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=0e38541219c2fd0279f560628f4a14f07217434b1dfc82a78c1076c5fc92d22a
  Stored in directory: /tmp/pip-ephem-wheel-cache-urdmo_i9/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:13:42, 13.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:59:29, 18.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:08:53, 26.2kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:24:45, 37.3kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:28:40, 53.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.19M/862M [00:01<3:07:10, 76.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:10:29, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.0M/862M [00:01<1:30:55, 155kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:03:27, 221kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:01<44:14, 315kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<30:57, 448kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:01<21:37, 638kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:01<15:11, 904kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<10:39, 1.28MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<07:32, 1.80MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:19, 2.54MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<06:20, 2.13MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:02<04:31, 2.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<17:41, 759kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<14:28, 928kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<10:37, 1.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<09:50, 1.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<08:16, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:23, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<07:51, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:08<06:04, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<04:23, 3.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<15:18, 864kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<12:02, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<08:41, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<09:10, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<09:05, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<07:00, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:01, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:14, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<04:38, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:18, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<07:02, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<05:28, 2.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:16<04:00, 3.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<08:37, 1.50MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:18<05:28, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<06:51, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:23, 1.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<05:49, 2.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<06:08, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<05:35, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:22<04:14, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<05:27, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<04:07, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:53, 2.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:24, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:50, 2.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:38, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:16, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:43, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:15, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<03:57, 3.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:41, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:14, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<03:58, 3.13MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:43, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:30, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:11, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<03:44, 3.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<37:38, 328kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<3:26:40, 59.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:25:52, 84.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<1:42:10, 120kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<1:14:10, 165kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<54:23, 225kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<38:39, 317kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<28:55, 422kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<21:29, 567kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<15:17, 796kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<13:31, 897kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:54, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:51, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<06:18, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:55, 757kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<12:22, 974kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<08:54, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<09:01, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:44, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:43, 1.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:37, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:50, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:22, 2.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:50, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:28, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:07, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:19, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:01, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:38, 3.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<44:54, 261kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<32:37, 359kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<23:04, 506kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:50, 618kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<15:33, 749kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<11:27, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:52, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<08:04, 1.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:51, 1.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:07, 1.62MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<05:29, 2.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:57, 2.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:40, 981kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:20, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<06:46, 1.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:24, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:28, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:42, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<04:09, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:22, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:18, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<04:39, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:53, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:23, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:02, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:19, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:38, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:08, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:50, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:39, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:01, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:39, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:32, 3.10MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:02, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:45, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:30, 2.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:17, 3.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:33, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:06, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<05:14, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:10, 1.75MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:24, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:03, 2.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:23, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:50, 2.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:36, 2.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:04, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:42, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:27, 2.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<03:13, 3.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<12:54, 823kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<10:06, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:16, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:34, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:25, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:39, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<04:03, 2.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<11:50, 885kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<09:21, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:45, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:10, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:04, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:30, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:35, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:59, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:38, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:21, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<12:09, 845kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<09:33, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<07:14, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:04, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:32, 1.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:54, 2.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:40, 2.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:57, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:34, 1.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:21, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:14, 3.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:22, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:46, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:33, 2.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:48, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:17, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:07, 2.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:00, 3.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:21, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<06:09, 1.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:31, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:27, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:48, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:36, 2.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:49, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:19, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:09, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:01, 3.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:33, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:16, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<04:37, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:28, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:45, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:30, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:41, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:15, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:13, 2.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:26, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:00, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:55, 2.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<02:50, 3.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<11:32, 813kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<09:01, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<06:30, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:44, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:35, 1.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:01, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<03:35, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<14:08, 654kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:50, 852kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<07:48, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:35, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:14, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:06<05:28, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:55, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<09:18, 978kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:19, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<05:20, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:49, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:00, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:43, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:42, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:06, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:01, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:53, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:56, 3.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:07, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:40, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:42, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:00, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:43, 2.35MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:47, 3.13MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:57, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<04:31, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:33, 2.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:35, 3.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:38, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:42, 1.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:27, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:17, 2.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:32, 1.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:06, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:05, 2.76MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:16, 3.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<10:55, 777kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<09:23, 903kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<07:00, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:14, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:13, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:50, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:38, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:00, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:54, 2.14MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:56, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:58, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:38, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:45, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:49, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:21, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:24, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:32, 3.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:13, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:47, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:51, 2.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<02:05, 3.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<33:35, 240kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<25:09, 321kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<17:56, 449kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<12:40, 634kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<11:17, 709kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<08:43, 917kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:17, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:14, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:58, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:35, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:17, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<59:09, 133kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<42:11, 186kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<29:38, 265kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<20:51, 375kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<18:02, 433kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<13:39, 571kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<09:46, 796kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<08:18, 931kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:36, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<04:48, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:07, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:55, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:52, 2.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:25, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:44, 2.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:48, 2.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<02:02, 3.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<31:48, 237kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<23:46, 316kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<16:57, 443kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<11:54, 628kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<12:05, 618kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<09:12, 810kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<06:36, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<06:20, 1.17MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:55, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:27, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<03:14, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:38, 1.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:00, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:58, 2.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:47, 1.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:06, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:14, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<02:19, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<54:03, 133kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<38:32, 186kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<27:03, 264kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<20:31, 347kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<15:44, 452kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<11:21, 626kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<09:02, 780kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:56, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<05:00, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<03:36, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<19:56, 350kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<15:22, 454kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<11:02, 631kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<07:46, 891kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<10:01, 690kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:43, 893kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:32, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<03:57, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<28:12, 243kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<20:44, 330kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<14:50, 460kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<10:28, 649kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<09:13, 735kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<07:08, 948kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<05:09, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<05:21, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<03:52, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:12, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:37, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:42, 2.44MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<01:58, 3.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<10:54, 602kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<08:18, 790kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<05:57, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:39, 1.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:37, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:23, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:53, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<04:01, 1.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:06, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:16, 2.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:37, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:10, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:21, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:07, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:49, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:07, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:21, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:37, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:54, 3.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:33, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:18, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:44, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:51, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:08, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<45:27, 132kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<32:23, 186kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<22:43, 263kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<17:12, 346kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<13:17, 448kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<09:35, 619kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<06:46, 872kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<07:01, 838kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:31, 1.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:59, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:09, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:29, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:34, 2.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:09, 1.82MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:48, 2.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:05, 2.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:47, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:32, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:55, 2.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:39, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:50, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:36, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<01:57, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:25, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:49, 3.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:32, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:53, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:18, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<01:40, 3.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:03:00, 85.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<44:36, 121kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<31:13, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<22:56, 233kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<17:09, 311kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<12:15, 434kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<09:20, 563kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<07:05, 742kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<05:04, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:44, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:22, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:19, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:21, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<14:42, 349kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<10:48, 474kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<07:39, 666kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:30, 777kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:35, 905kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:08, 1.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:55, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<08:09, 612kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<06:13, 800kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<04:26, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:14, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:58, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:07, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<05:12, 931kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<04:08, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:12, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:13, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:28, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:46, 2.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<05:27, 866kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:17, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:15, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:13, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:27, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:47, 2.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:03, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:37, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:55, 2.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:23, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:35, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:00, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:26, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<05:06, 870kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<04:01, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:53, 1.52MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:02, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:00, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:18, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:38, 2.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:27, 790kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:15, 1.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:03, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:06, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:03, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:19, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:39, 2.53MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<04:46, 872kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:47, 1.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:45, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:57, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:33, 190kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:31<15:29, 264kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<10:52, 374kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:30, 474kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<06:46, 595kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<04:55, 815kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:27, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<3:50:13, 17.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<2:41:18, 24.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:52:16, 35.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:18:47, 49.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<55:27, 70.1kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<38:39, 99.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<27:42, 138kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<19:45, 193kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<13:50, 274kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<10:28, 359kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<07:44, 485kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<05:28, 680kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:37, 798kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:58, 927kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:57, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:37, 1.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:13, 1.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:38, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:57, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:43, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:17, 2.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:43, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:33, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:10, 2.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:37, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:50, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:27, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:02, 3.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<3:12:46, 17.4kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<2:15:01, 24.7kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<1:33:52, 35.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<1:05:45, 49.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<46:16, 70.7kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<32:12, 101kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<23:03, 139kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<16:23, 195kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<11:36, 275kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<08:04, 391kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<07:33, 416kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<05:56, 528kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<04:27, 703kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:11, 978kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<02:14, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<23:32, 131kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<16:52, 182kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<11:56, 256kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<08:21, 363kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<06:33, 459kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:57, 604kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<03:32, 839kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:01, 968kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:43, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:01, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:25, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:10, 905kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:30, 1.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:54, 1.46MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:34, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:12, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:51, 3.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<11:35, 235kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<08:40, 314kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<06:10, 439kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<04:16, 623kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<15:55, 167kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<11:23, 233kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<07:57, 330kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<06:06, 425kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<04:46, 542kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:26, 748kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:24, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:55, 864kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:18, 1.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:39, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:42, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:35, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:24, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:15, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:12, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<00:57, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:42, 3.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:15, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:13, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:00, 2.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:44, 3.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:04, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:06, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:57, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:43, 3.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:57, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:06, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:14, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:09, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:53, 2.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:38, 3.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:37, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:31, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:23, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:04, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:46, 2.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:13, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:02, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:48, 2.49MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:35, 3.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:24, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:17, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:05, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:52, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:37, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:06, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:04, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:56, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:44, 2.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:32, 3.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:08, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:05, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:57, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:44, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:32, 3.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:02, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:03, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:03, 1.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:53, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:39, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:51, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:59, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:03, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:02, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:40, 2.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:30, 3.22MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:52, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:37, 2.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:29, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:22, 4.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:57, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:53, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:50, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:38, 2.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:27, 3.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:45, 847kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:27, 1.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:11, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:59, 1.49MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<00:31, 2.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:22, 421kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<02:43, 522kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<02:09, 658kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:51, 764kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:24, 1.00MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:03, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:50, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:42, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:29, 2.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:21, 3.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:47, 1.62MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:52, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:50, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:44, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:33, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:35, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:35, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:48, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:50, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:51, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:47, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:30, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:23, 3.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:24, 2.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:22, 3.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:17, 3.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:28, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:26, 2.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:19, 3.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:26, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:19, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:24, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:27, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:16, 3.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:45, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:35, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:31, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:16, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:29, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:27, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:15, 2.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:26, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:28, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:30, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:26, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:19, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:13, 3.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:28, 451kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:09, 568kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:56, 697kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:42, 923kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:29, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:23, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:11, 2.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:44, 704kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:35, 872kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:28, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:20, 1.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:14, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:20, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:20, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:18, 1.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:09, 2.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:11, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:15, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:05, 3.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:15, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:13, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:11, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:05, 2.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:04, 3.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.69MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:04, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:03, 3.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:07, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:07, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 3.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.59MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.58MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<12:59:09,  8.56it/s]  0%|          | 918/400000 [00:00<9:04:22, 12.22it/s]  0%|          | 1782/400000 [00:00<6:20:28, 17.44it/s]  1%|          | 2632/400000 [00:00<4:25:59, 24.90it/s]  1%|          | 3481/400000 [00:00<3:06:01, 35.52it/s]  1%|          | 4383/400000 [00:00<2:10:08, 50.66it/s]  1%|▏         | 5308/400000 [00:00<1:31:06, 72.21it/s]  2%|▏         | 6157/400000 [00:00<1:03:51, 102.78it/s]  2%|▏         | 6963/400000 [00:00<44:51, 146.01it/s]    2%|▏         | 7845/400000 [00:01<31:33, 207.12it/s]  2%|▏         | 8753/400000 [00:01<22:15, 293.02it/s]  2%|▏         | 9664/400000 [00:01<15:45, 412.91it/s]  3%|▎         | 10570/400000 [00:01<11:13, 578.57it/s]  3%|▎         | 11450/400000 [00:01<08:05, 801.02it/s]  3%|▎         | 12301/400000 [00:01<05:52, 1099.92it/s]  3%|▎         | 13151/400000 [00:01<04:20, 1485.69it/s]  3%|▎         | 13992/400000 [00:01<03:15, 1971.64it/s]  4%|▎         | 14867/400000 [00:01<02:29, 2568.41it/s]  4%|▍         | 15774/400000 [00:01<01:57, 3271.75it/s]  4%|▍         | 16675/400000 [00:02<01:34, 4044.31it/s]  4%|▍         | 17552/400000 [00:02<01:19, 4811.87it/s]  5%|▍         | 18464/400000 [00:02<01:08, 5605.66it/s]  5%|▍         | 19349/400000 [00:02<01:01, 6229.16it/s]  5%|▌         | 20239/400000 [00:02<00:55, 6844.04it/s]  5%|▌         | 21164/400000 [00:02<00:51, 7421.75it/s]  6%|▌         | 22069/400000 [00:02<00:48, 7844.88it/s]  6%|▌         | 22965/400000 [00:02<00:46, 8093.62it/s]  6%|▌         | 23912/400000 [00:02<00:44, 8460.18it/s]  6%|▌         | 24840/400000 [00:02<00:43, 8690.15it/s]  6%|▋         | 25753/400000 [00:03<00:42, 8770.81it/s]  7%|▋         | 26661/400000 [00:03<00:43, 8520.23it/s]  7%|▋         | 27571/400000 [00:03<00:42, 8684.72it/s]  7%|▋         | 28488/400000 [00:03<00:42, 8823.16it/s]  7%|▋         | 29383/400000 [00:03<00:42, 8817.79it/s]  8%|▊         | 30274/400000 [00:03<00:42, 8667.76it/s]  8%|▊         | 31154/400000 [00:03<00:42, 8705.59it/s]  8%|▊         | 32044/400000 [00:03<00:41, 8761.04it/s]  8%|▊         | 32952/400000 [00:03<00:41, 8852.92it/s]  8%|▊         | 33840/400000 [00:03<00:41, 8801.23it/s]  9%|▊         | 34744/400000 [00:04<00:41, 8870.27it/s]  9%|▉         | 35698/400000 [00:04<00:40, 9059.00it/s]  9%|▉         | 36606/400000 [00:04<00:40, 8991.28it/s]  9%|▉         | 37507/400000 [00:04<00:40, 8992.77it/s] 10%|▉         | 38408/400000 [00:04<00:40, 8908.82it/s] 10%|▉         | 39300/400000 [00:04<00:40, 8861.69it/s] 10%|█         | 40259/400000 [00:04<00:39, 9068.25it/s] 10%|█         | 41168/400000 [00:04<00:40, 8816.60it/s] 11%|█         | 42053/400000 [00:04<00:41, 8710.90it/s] 11%|█         | 42930/400000 [00:04<00:40, 8727.48it/s] 11%|█         | 43812/400000 [00:05<00:40, 8752.75it/s] 11%|█         | 44749/400000 [00:05<00:39, 8926.09it/s] 11%|█▏        | 45644/400000 [00:05<00:39, 8923.00it/s] 12%|█▏        | 46558/400000 [00:05<00:39, 8986.59it/s] 12%|█▏        | 47458/400000 [00:05<00:39, 8923.22it/s] 12%|█▏        | 48405/400000 [00:05<00:38, 9078.73it/s] 12%|█▏        | 49362/400000 [00:05<00:38, 9220.69it/s] 13%|█▎        | 50286/400000 [00:05<00:38, 9097.42it/s] 13%|█▎        | 51198/400000 [00:05<00:38, 9065.14it/s] 13%|█▎        | 52106/400000 [00:06<00:39, 8907.67it/s] 13%|█▎        | 53036/400000 [00:06<00:38, 9021.85it/s] 14%|█▎        | 54021/400000 [00:06<00:37, 9254.49it/s] 14%|█▎        | 54949/400000 [00:06<00:37, 9191.12it/s] 14%|█▍        | 55870/400000 [00:06<00:37, 9171.94it/s] 14%|█▍        | 56789/400000 [00:06<00:37, 9141.66it/s] 14%|█▍        | 57705/400000 [00:06<00:37, 9018.67it/s] 15%|█▍        | 58608/400000 [00:06<00:37, 8995.05it/s] 15%|█▍        | 59509/400000 [00:06<00:38, 8895.55it/s] 15%|█▌        | 60434/400000 [00:06<00:37, 8996.98it/s] 15%|█▌        | 61335/400000 [00:07<00:38, 8907.94it/s] 16%|█▌        | 62230/400000 [00:07<00:37, 8920.40it/s] 16%|█▌        | 63190/400000 [00:07<00:36, 9113.74it/s] 16%|█▌        | 64103/400000 [00:07<00:37, 8992.29it/s] 16%|█▋        | 65004/400000 [00:07<00:37, 8899.13it/s] 16%|█▋        | 65896/400000 [00:07<00:38, 8763.93it/s] 17%|█▋        | 66816/400000 [00:07<00:37, 8888.85it/s] 17%|█▋        | 67743/400000 [00:07<00:36, 8998.00it/s] 17%|█▋        | 68645/400000 [00:07<00:37, 8954.59it/s] 17%|█▋        | 69542/400000 [00:07<00:37, 8885.19it/s] 18%|█▊        | 70432/400000 [00:08<00:37, 8878.46it/s] 18%|█▊        | 71321/400000 [00:08<00:37, 8758.41it/s] 18%|█▊        | 72255/400000 [00:08<00:36, 8922.93it/s] 18%|█▊        | 73149/400000 [00:08<00:36, 8916.94it/s] 19%|█▊        | 74042/400000 [00:08<00:36, 8867.71it/s] 19%|█▊        | 74930/400000 [00:08<00:36, 8862.99it/s] 19%|█▉        | 75837/400000 [00:08<00:36, 8921.78it/s] 19%|█▉        | 76730/400000 [00:08<00:36, 8862.99it/s] 19%|█▉        | 77642/400000 [00:08<00:36, 8936.94it/s] 20%|█▉        | 78553/400000 [00:08<00:35, 8988.14it/s] 20%|█▉        | 79455/400000 [00:09<00:35, 8996.84it/s] 20%|██        | 80366/400000 [00:09<00:35, 9028.48it/s] 20%|██        | 81270/400000 [00:09<00:35, 9024.95it/s] 21%|██        | 82173/400000 [00:09<00:35, 8966.20it/s] 21%|██        | 83074/400000 [00:09<00:35, 8978.25it/s] 21%|██        | 83972/400000 [00:09<00:35, 8956.59it/s] 21%|██        | 84869/400000 [00:09<00:35, 8957.97it/s] 21%|██▏       | 85812/400000 [00:09<00:34, 9092.76it/s] 22%|██▏       | 86753/400000 [00:09<00:34, 9183.47it/s] 22%|██▏       | 87673/400000 [00:09<00:33, 9187.96it/s] 22%|██▏       | 88593/400000 [00:10<00:34, 9002.21it/s] 22%|██▏       | 89495/400000 [00:10<00:34, 8999.79it/s] 23%|██▎       | 90444/400000 [00:10<00:33, 9139.72it/s] 23%|██▎       | 91360/400000 [00:10<00:33, 9085.67it/s] 23%|██▎       | 92270/400000 [00:10<00:33, 9076.23it/s] 23%|██▎       | 93179/400000 [00:10<00:34, 8998.63it/s] 24%|██▎       | 94102/400000 [00:10<00:33, 9065.28it/s] 24%|██▍       | 95014/400000 [00:10<00:33, 9079.68it/s] 24%|██▍       | 95977/400000 [00:10<00:32, 9237.50it/s] 24%|██▍       | 96929/400000 [00:10<00:32, 9319.10it/s] 24%|██▍       | 97882/400000 [00:11<00:32, 9381.13it/s] 25%|██▍       | 98821/400000 [00:11<00:33, 9118.35it/s] 25%|██▍       | 99743/400000 [00:11<00:32, 9148.00it/s] 25%|██▌       | 100701/400000 [00:11<00:32, 9272.86it/s] 25%|██▌       | 101649/400000 [00:11<00:31, 9332.59it/s] 26%|██▌       | 102585/400000 [00:11<00:31, 9337.59it/s] 26%|██▌       | 103520/400000 [00:11<00:31, 9299.98it/s] 26%|██▌       | 104451/400000 [00:11<00:32, 9224.21it/s] 26%|██▋       | 105396/400000 [00:11<00:31, 9289.14it/s] 27%|██▋       | 106342/400000 [00:11<00:31, 9337.41it/s] 27%|██▋       | 107286/400000 [00:12<00:31, 9365.03it/s] 27%|██▋       | 108223/400000 [00:12<00:31, 9186.91it/s] 27%|██▋       | 109190/400000 [00:12<00:31, 9325.98it/s] 28%|██▊       | 110155/400000 [00:12<00:30, 9418.38it/s] 28%|██▊       | 111099/400000 [00:12<00:30, 9422.31it/s] 28%|██▊       | 112042/400000 [00:12<00:30, 9401.09it/s] 28%|██▊       | 112983/400000 [00:12<00:31, 9210.58it/s] 28%|██▊       | 113953/400000 [00:12<00:30, 9351.41it/s] 29%|██▊       | 114932/400000 [00:12<00:30, 9478.27it/s] 29%|██▉       | 115893/400000 [00:13<00:29, 9516.18it/s] 29%|██▉       | 116846/400000 [00:13<00:30, 9314.13it/s] 29%|██▉       | 117780/400000 [00:13<00:31, 9049.69it/s] 30%|██▉       | 118730/400000 [00:13<00:30, 9180.16it/s] 30%|██▉       | 119651/400000 [00:13<00:31, 8988.70it/s] 30%|███       | 120553/400000 [00:13<00:31, 8818.71it/s] 30%|███       | 121438/400000 [00:13<00:32, 8669.46it/s] 31%|███       | 122329/400000 [00:13<00:31, 8740.00it/s] 31%|███       | 123270/400000 [00:13<00:30, 8929.07it/s] 31%|███       | 124200/400000 [00:13<00:30, 9034.96it/s] 31%|███▏      | 125106/400000 [00:14<00:30, 8942.41it/s] 32%|███▏      | 126016/400000 [00:14<00:30, 8987.69it/s] 32%|███▏      | 126916/400000 [00:14<00:30, 8941.97it/s] 32%|███▏      | 127847/400000 [00:14<00:30, 9048.73it/s] 32%|███▏      | 128753/400000 [00:14<00:30, 9003.68it/s] 32%|███▏      | 129655/400000 [00:14<00:30, 8812.26it/s] 33%|███▎      | 130538/400000 [00:14<00:30, 8693.16it/s] 33%|███▎      | 131409/400000 [00:14<00:31, 8505.01it/s] 33%|███▎      | 132286/400000 [00:14<00:31, 8581.19it/s] 33%|███▎      | 133146/400000 [00:14<00:31, 8551.14it/s] 34%|███▎      | 134003/400000 [00:15<00:31, 8527.35it/s] 34%|███▎      | 134857/400000 [00:15<00:31, 8381.76it/s] 34%|███▍      | 135731/400000 [00:15<00:31, 8485.34it/s] 34%|███▍      | 136633/400000 [00:15<00:30, 8637.01it/s] 34%|███▍      | 137499/400000 [00:15<00:30, 8576.26it/s] 35%|███▍      | 138382/400000 [00:15<00:30, 8650.44it/s] 35%|███▍      | 139248/400000 [00:15<00:30, 8577.63it/s] 35%|███▌      | 140146/400000 [00:15<00:29, 8693.53it/s] 35%|███▌      | 141041/400000 [00:15<00:29, 8767.34it/s] 35%|███▌      | 141938/400000 [00:15<00:29, 8825.44it/s] 36%|███▌      | 142822/400000 [00:16<00:29, 8744.24it/s] 36%|███▌      | 143698/400000 [00:16<00:29, 8636.24it/s] 36%|███▌      | 144585/400000 [00:16<00:29, 8698.58it/s] 36%|███▋      | 145456/400000 [00:16<00:29, 8581.09it/s] 37%|███▋      | 146315/400000 [00:16<00:30, 8435.35it/s] 37%|███▋      | 147160/400000 [00:16<00:30, 8342.70it/s] 37%|███▋      | 148018/400000 [00:16<00:29, 8412.29it/s] 37%|███▋      | 148952/400000 [00:16<00:28, 8667.32it/s] 37%|███▋      | 149822/400000 [00:16<00:28, 8671.98it/s] 38%|███▊      | 150740/400000 [00:17<00:28, 8815.01it/s] 38%|███▊      | 151624/400000 [00:17<00:28, 8588.67it/s] 38%|███▊      | 152486/400000 [00:17<00:29, 8531.83it/s] 38%|███▊      | 153349/400000 [00:17<00:28, 8560.19it/s] 39%|███▊      | 154207/400000 [00:17<00:28, 8552.65it/s] 39%|███▉      | 155090/400000 [00:17<00:28, 8632.10it/s] 39%|███▉      | 155955/400000 [00:17<00:28, 8560.54it/s] 39%|███▉      | 156825/400000 [00:17<00:28, 8601.89it/s] 39%|███▉      | 157727/400000 [00:17<00:27, 8722.06it/s] 40%|███▉      | 158600/400000 [00:17<00:27, 8714.26it/s] 40%|███▉      | 159472/400000 [00:18<00:27, 8680.74it/s] 40%|████      | 160349/400000 [00:18<00:27, 8706.86it/s] 40%|████      | 161220/400000 [00:18<00:27, 8603.14it/s] 41%|████      | 162108/400000 [00:18<00:27, 8684.29it/s] 41%|████      | 162979/400000 [00:18<00:27, 8691.25it/s] 41%|████      | 163858/400000 [00:18<00:27, 8720.01it/s] 41%|████      | 164775/400000 [00:18<00:26, 8849.80it/s] 41%|████▏     | 165661/400000 [00:18<00:27, 8618.84it/s] 42%|████▏     | 166576/400000 [00:18<00:26, 8769.23it/s] 42%|████▏     | 167455/400000 [00:18<00:27, 8601.74it/s] 42%|████▏     | 168322/400000 [00:19<00:26, 8619.06it/s] 42%|████▏     | 169233/400000 [00:19<00:26, 8759.48it/s] 43%|████▎     | 170111/400000 [00:19<00:26, 8754.72it/s] 43%|████▎     | 171026/400000 [00:19<00:25, 8867.54it/s] 43%|████▎     | 171914/400000 [00:19<00:26, 8747.11it/s] 43%|████▎     | 172790/400000 [00:19<00:25, 8750.97it/s] 43%|████▎     | 173666/400000 [00:19<00:25, 8749.79it/s] 44%|████▎     | 174549/400000 [00:19<00:25, 8772.72it/s] 44%|████▍     | 175427/400000 [00:19<00:25, 8746.91it/s] 44%|████▍     | 176302/400000 [00:19<00:25, 8670.15it/s] 44%|████▍     | 177170/400000 [00:20<00:26, 8448.86it/s] 45%|████▍     | 178017/400000 [00:20<00:26, 8265.55it/s] 45%|████▍     | 178850/400000 [00:20<00:26, 8284.50it/s] 45%|████▍     | 179680/400000 [00:20<00:26, 8278.23it/s] 45%|████▌     | 180509/400000 [00:20<00:26, 8150.18it/s] 45%|████▌     | 181331/400000 [00:20<00:26, 8170.80it/s] 46%|████▌     | 182175/400000 [00:20<00:26, 8247.39it/s] 46%|████▌     | 183001/400000 [00:20<00:26, 8114.95it/s] 46%|████▌     | 183838/400000 [00:20<00:26, 8188.69it/s] 46%|████▌     | 184695/400000 [00:20<00:25, 8295.06it/s] 46%|████▋     | 185526/400000 [00:21<00:26, 8212.57it/s] 47%|████▋     | 186383/400000 [00:21<00:25, 8314.86it/s] 47%|████▋     | 187216/400000 [00:21<00:26, 8111.50it/s] 47%|████▋     | 188081/400000 [00:21<00:25, 8265.50it/s] 47%|████▋     | 188942/400000 [00:21<00:25, 8364.52it/s] 47%|████▋     | 189781/400000 [00:21<00:25, 8332.46it/s] 48%|████▊     | 190648/400000 [00:21<00:24, 8429.22it/s] 48%|████▊     | 191493/400000 [00:21<00:25, 8325.51it/s] 48%|████▊     | 192343/400000 [00:21<00:24, 8376.12it/s] 48%|████▊     | 193251/400000 [00:22<00:24, 8573.78it/s] 49%|████▊     | 194162/400000 [00:22<00:23, 8721.37it/s] 49%|████▉     | 195102/400000 [00:22<00:22, 8914.42it/s] 49%|████▉     | 196030/400000 [00:22<00:22, 9019.36it/s] 49%|████▉     | 196938/400000 [00:22<00:22, 9035.58it/s] 49%|████▉     | 197865/400000 [00:22<00:22, 9104.01it/s] 50%|████▉     | 198777/400000 [00:22<00:22, 9046.66it/s] 50%|████▉     | 199721/400000 [00:22<00:21, 9158.98it/s] 50%|█████     | 200649/400000 [00:22<00:21, 9194.44it/s] 50%|█████     | 201570/400000 [00:22<00:21, 9070.76it/s] 51%|█████     | 202478/400000 [00:23<00:22, 8968.60it/s] 51%|█████     | 203376/400000 [00:23<00:21, 8953.51it/s] 51%|█████     | 204272/400000 [00:23<00:22, 8758.75it/s] 51%|█████▏    | 205166/400000 [00:23<00:22, 8807.01it/s] 52%|█████▏    | 206048/400000 [00:23<00:22, 8787.21it/s] 52%|█████▏    | 206942/400000 [00:23<00:21, 8830.27it/s] 52%|█████▏    | 207862/400000 [00:23<00:21, 8935.30it/s] 52%|█████▏    | 208784/400000 [00:23<00:21, 9017.35it/s] 52%|█████▏    | 209695/400000 [00:23<00:21, 9044.00it/s] 53%|█████▎    | 210600/400000 [00:23<00:21, 8908.31it/s] 53%|█████▎    | 211492/400000 [00:24<00:21, 8721.12it/s] 53%|█████▎    | 212391/400000 [00:24<00:21, 8798.25it/s] 53%|█████▎    | 213311/400000 [00:24<00:20, 8915.02it/s] 54%|█████▎    | 214226/400000 [00:24<00:20, 8980.95it/s] 54%|█████▍    | 215126/400000 [00:24<00:21, 8734.07it/s] 54%|█████▍    | 216023/400000 [00:24<00:20, 8801.73it/s] 54%|█████▍    | 216915/400000 [00:24<00:20, 8836.64it/s] 54%|█████▍    | 217831/400000 [00:24<00:20, 8930.41it/s] 55%|█████▍    | 218726/400000 [00:24<00:20, 8714.47it/s] 55%|█████▍    | 219600/400000 [00:24<00:20, 8705.19it/s] 55%|█████▌    | 220522/400000 [00:25<00:20, 8852.41it/s] 55%|█████▌    | 221450/400000 [00:25<00:19, 8974.72it/s] 56%|█████▌    | 222374/400000 [00:25<00:19, 9052.63it/s] 56%|█████▌    | 223285/400000 [00:25<00:19, 9069.39it/s] 56%|█████▌    | 224193/400000 [00:25<00:19, 8920.58it/s] 56%|█████▋    | 225087/400000 [00:25<00:20, 8690.08it/s] 56%|█████▋    | 225959/400000 [00:25<00:20, 8492.82it/s] 57%|█████▋    | 226811/400000 [00:25<00:20, 8476.31it/s] 57%|█████▋    | 227661/400000 [00:25<00:20, 8409.22it/s] 57%|█████▋    | 228511/400000 [00:25<00:20, 8435.34it/s] 57%|█████▋    | 229356/400000 [00:26<00:20, 8215.39it/s] 58%|█████▊    | 230253/400000 [00:26<00:20, 8426.01it/s] 58%|█████▊    | 231129/400000 [00:26<00:19, 8522.32it/s] 58%|█████▊    | 231984/400000 [00:26<00:20, 8296.59it/s] 58%|█████▊    | 232817/400000 [00:26<00:20, 8295.87it/s] 58%|█████▊    | 233744/400000 [00:26<00:19, 8564.93it/s] 59%|█████▊    | 234694/400000 [00:26<00:18, 8825.00it/s] 59%|█████▉    | 235616/400000 [00:26<00:18, 8938.34it/s] 59%|█████▉    | 236521/400000 [00:26<00:18, 8970.95it/s] 59%|█████▉    | 237421/400000 [00:27<00:18, 8879.25it/s] 60%|█████▉    | 238311/400000 [00:27<00:18, 8782.33it/s] 60%|█████▉    | 239191/400000 [00:27<00:18, 8711.74it/s] 60%|██████    | 240064/400000 [00:27<00:18, 8662.12it/s] 60%|██████    | 240932/400000 [00:27<00:18, 8519.80it/s] 60%|██████    | 241786/400000 [00:27<00:18, 8436.09it/s] 61%|██████    | 242631/400000 [00:27<00:18, 8368.54it/s] 61%|██████    | 243476/400000 [00:27<00:18, 8392.29it/s] 61%|██████    | 244316/400000 [00:27<00:18, 8285.34it/s] 61%|██████▏   | 245146/400000 [00:27<00:18, 8262.93it/s] 62%|██████▏   | 246006/400000 [00:28<00:18, 8361.20it/s] 62%|██████▏   | 246843/400000 [00:28<00:18, 8337.20it/s] 62%|██████▏   | 247705/400000 [00:28<00:18, 8419.83it/s] 62%|██████▏   | 248548/400000 [00:28<00:18, 8384.24it/s] 62%|██████▏   | 249387/400000 [00:28<00:18, 8300.63it/s] 63%|██████▎   | 250218/400000 [00:28<00:18, 8216.91it/s] 63%|██████▎   | 251041/400000 [00:28<00:18, 8194.95it/s] 63%|██████▎   | 251861/400000 [00:28<00:18, 8189.68it/s] 63%|██████▎   | 252681/400000 [00:28<00:18, 8149.97it/s] 63%|██████▎   | 253497/400000 [00:28<00:18, 8116.91it/s] 64%|██████▎   | 254309/400000 [00:29<00:18, 8090.03it/s] 64%|██████▍   | 255127/400000 [00:29<00:17, 8116.00it/s] 64%|██████▍   | 255939/400000 [00:29<00:17, 8109.59it/s] 64%|██████▍   | 256751/400000 [00:29<00:17, 8081.24it/s] 64%|██████▍   | 257560/400000 [00:29<00:17, 7917.26it/s] 65%|██████▍   | 258353/400000 [00:29<00:18, 7668.32it/s] 65%|██████▍   | 259139/400000 [00:29<00:18, 7724.45it/s] 65%|██████▌   | 260019/400000 [00:29<00:17, 8017.15it/s] 65%|██████▌   | 260919/400000 [00:29<00:16, 8288.40it/s] 65%|██████▌   | 261753/400000 [00:29<00:16, 8200.15it/s] 66%|██████▌   | 262577/400000 [00:30<00:17, 8016.30it/s] 66%|██████▌   | 263393/400000 [00:30<00:16, 8055.93it/s] 66%|██████▌   | 264202/400000 [00:30<00:17, 7980.77it/s] 66%|██████▋   | 265041/400000 [00:30<00:16, 8096.84it/s] 66%|██████▋   | 265937/400000 [00:30<00:16, 8335.61it/s] 67%|██████▋   | 266774/400000 [00:30<00:16, 8233.63it/s] 67%|██████▋   | 267600/400000 [00:30<00:16, 8214.65it/s] 67%|██████▋   | 268424/400000 [00:30<00:16, 8009.88it/s] 67%|██████▋   | 269228/400000 [00:30<00:16, 7889.22it/s] 68%|██████▊   | 270019/400000 [00:31<00:16, 7874.32it/s] 68%|██████▊   | 270817/400000 [00:31<00:16, 7903.25it/s] 68%|██████▊   | 271637/400000 [00:31<00:16, 7987.55it/s] 68%|██████▊   | 272501/400000 [00:31<00:15, 8172.20it/s] 68%|██████▊   | 273320/400000 [00:31<00:15, 8145.34it/s] 69%|██████▊   | 274167/400000 [00:31<00:15, 8239.31it/s] 69%|██████▊   | 274993/400000 [00:31<00:15, 8134.37it/s] 69%|██████▉   | 275864/400000 [00:31<00:14, 8298.69it/s] 69%|██████▉   | 276707/400000 [00:31<00:14, 8335.43it/s] 69%|██████▉   | 277576/400000 [00:31<00:14, 8437.81it/s] 70%|██████▉   | 278433/400000 [00:32<00:14, 8475.40it/s] 70%|██████▉   | 279282/400000 [00:32<00:14, 8239.95it/s] 70%|███████   | 280132/400000 [00:32<00:14, 8313.44it/s] 70%|███████   | 280965/400000 [00:32<00:14, 8286.52it/s] 70%|███████   | 281795/400000 [00:32<00:14, 8182.07it/s] 71%|███████   | 282615/400000 [00:32<00:14, 8098.33it/s] 71%|███████   | 283426/400000 [00:32<00:14, 8086.16it/s] 71%|███████   | 284347/400000 [00:32<00:13, 8392.01it/s] 71%|███████▏  | 285277/400000 [00:32<00:13, 8644.44it/s] 72%|███████▏  | 286180/400000 [00:32<00:13, 8754.79it/s] 72%|███████▏  | 287059/400000 [00:33<00:13, 8670.85it/s] 72%|███████▏  | 287957/400000 [00:33<00:12, 8760.16it/s] 72%|███████▏  | 288856/400000 [00:33<00:12, 8826.32it/s] 72%|███████▏  | 289794/400000 [00:33<00:12, 8983.08it/s] 73%|███████▎  | 290731/400000 [00:33<00:12, 9093.64it/s] 73%|███████▎  | 291673/400000 [00:33<00:11, 9187.50it/s] 73%|███████▎  | 292594/400000 [00:33<00:11, 9107.06it/s] 73%|███████▎  | 293506/400000 [00:33<00:11, 8989.88it/s] 74%|███████▎  | 294430/400000 [00:33<00:11, 9062.49it/s] 74%|███████▍  | 295338/400000 [00:33<00:11, 9001.67it/s] 74%|███████▍  | 296239/400000 [00:34<00:11, 8966.31it/s] 74%|███████▍  | 297137/400000 [00:34<00:11, 8842.33it/s] 75%|███████▍  | 298050/400000 [00:34<00:11, 8926.25it/s] 75%|███████▍  | 298979/400000 [00:34<00:11, 9032.00it/s] 75%|███████▍  | 299884/400000 [00:34<00:11, 9013.82it/s] 75%|███████▌  | 300786/400000 [00:34<00:11, 8751.08it/s] 75%|███████▌  | 301664/400000 [00:34<00:11, 8686.14it/s] 76%|███████▌  | 302538/400000 [00:34<00:11, 8699.48it/s] 76%|███████▌  | 303410/400000 [00:34<00:11, 8696.17it/s] 76%|███████▌  | 304282/400000 [00:34<00:10, 8702.86it/s] 76%|███████▋  | 305153/400000 [00:35<00:11, 8504.34it/s] 77%|███████▋  | 306046/400000 [00:35<00:10, 8624.74it/s] 77%|███████▋  | 306931/400000 [00:35<00:10, 8688.80it/s] 77%|███████▋  | 307801/400000 [00:35<00:10, 8601.39it/s] 77%|███████▋  | 308663/400000 [00:35<00:10, 8579.48it/s] 77%|███████▋  | 309527/400000 [00:35<00:10, 8597.02it/s] 78%|███████▊  | 310452/400000 [00:35<00:10, 8781.01it/s] 78%|███████▊  | 311353/400000 [00:35<00:10, 8845.37it/s] 78%|███████▊  | 312239/400000 [00:35<00:10, 8720.37it/s] 78%|███████▊  | 313113/400000 [00:35<00:10, 8578.53it/s] 78%|███████▊  | 313973/400000 [00:36<00:10, 8364.48it/s] 79%|███████▊  | 314885/400000 [00:36<00:09, 8575.66it/s] 79%|███████▉  | 315820/400000 [00:36<00:09, 8792.13it/s] 79%|███████▉  | 316732/400000 [00:36<00:09, 8887.59it/s] 79%|███████▉  | 317644/400000 [00:36<00:09, 8955.79it/s] 80%|███████▉  | 318542/400000 [00:36<00:09, 8941.15it/s] 80%|███████▉  | 319449/400000 [00:36<00:08, 8978.02it/s] 80%|████████  | 320381/400000 [00:36<00:08, 9077.47it/s] 80%|████████  | 321290/400000 [00:36<00:08, 8949.60it/s] 81%|████████  | 322191/400000 [00:37<00:08, 8966.87it/s] 81%|████████  | 323089/400000 [00:37<00:08, 8874.63it/s] 81%|████████  | 323986/400000 [00:37<00:08, 8900.24it/s] 81%|████████  | 324879/400000 [00:37<00:08, 8907.01it/s] 81%|████████▏ | 325796/400000 [00:37<00:08, 8983.81it/s] 82%|████████▏ | 326695/400000 [00:37<00:08, 8775.35it/s] 82%|████████▏ | 327574/400000 [00:37<00:08, 8740.26it/s] 82%|████████▏ | 328450/400000 [00:37<00:08, 8678.60it/s] 82%|████████▏ | 329386/400000 [00:37<00:07, 8871.32it/s] 83%|████████▎ | 330314/400000 [00:37<00:07, 8987.85it/s] 83%|████████▎ | 331215/400000 [00:38<00:07, 8973.98it/s] 83%|████████▎ | 332124/400000 [00:38<00:07, 9005.90it/s] 83%|████████▎ | 333026/400000 [00:38<00:07, 8977.37it/s] 83%|████████▎ | 333944/400000 [00:38<00:07, 9034.86it/s] 84%|████████▎ | 334848/400000 [00:38<00:07, 8872.33it/s] 84%|████████▍ | 335737/400000 [00:38<00:07, 8824.48it/s] 84%|████████▍ | 336650/400000 [00:38<00:07, 8911.30it/s] 84%|████████▍ | 337542/400000 [00:38<00:07, 8860.86it/s] 85%|████████▍ | 338434/400000 [00:38<00:06, 8877.93it/s] 85%|████████▍ | 339323/400000 [00:38<00:06, 8803.28it/s] 85%|████████▌ | 340229/400000 [00:39<00:06, 8875.70it/s] 85%|████████▌ | 341124/400000 [00:39<00:06, 8893.21it/s] 86%|████████▌ | 342014/400000 [00:39<00:06, 8682.41it/s] 86%|████████▌ | 342894/400000 [00:39<00:06, 8716.31it/s] 86%|████████▌ | 343786/400000 [00:39<00:06, 8774.11it/s] 86%|████████▌ | 344668/400000 [00:39<00:06, 8785.57it/s] 86%|████████▋ | 345568/400000 [00:39<00:06, 8847.60it/s] 87%|████████▋ | 346454/400000 [00:39<00:06, 8257.04it/s] 87%|████████▋ | 347288/400000 [00:39<00:06, 8233.14it/s] 87%|████████▋ | 348118/400000 [00:39<00:06, 8198.26it/s] 87%|████████▋ | 349027/400000 [00:40<00:06, 8444.86it/s] 87%|████████▋ | 349937/400000 [00:40<00:05, 8629.91it/s] 88%|████████▊ | 350823/400000 [00:40<00:05, 8696.32it/s] 88%|████████▊ | 351726/400000 [00:40<00:05, 8791.47it/s] 88%|████████▊ | 352608/400000 [00:40<00:05, 8657.79it/s] 88%|████████▊ | 353487/400000 [00:40<00:05, 8695.01it/s] 89%|████████▊ | 354367/400000 [00:40<00:05, 8723.53it/s] 89%|████████▉ | 355245/400000 [00:40<00:05, 8737.73it/s] 89%|████████▉ | 356154/400000 [00:40<00:04, 8840.10it/s] 89%|████████▉ | 357057/400000 [00:40<00:04, 8894.95it/s] 89%|████████▉ | 357948/400000 [00:41<00:04, 8827.75it/s] 90%|████████▉ | 358872/400000 [00:41<00:04, 8944.90it/s] 90%|████████▉ | 359768/400000 [00:41<00:04, 8896.78it/s] 90%|█████████ | 360659/400000 [00:41<00:04, 8694.44it/s] 90%|█████████ | 361541/400000 [00:41<00:04, 8730.80it/s] 91%|█████████ | 362434/400000 [00:41<00:04, 8788.68it/s] 91%|█████████ | 363327/400000 [00:41<00:04, 8830.34it/s] 91%|█████████ | 364211/400000 [00:41<00:04, 8793.31it/s] 91%|█████████▏| 365094/400000 [00:41<00:03, 8804.00it/s] 92%|█████████▏| 366039/400000 [00:41<00:03, 8988.15it/s] 92%|█████████▏| 366939/400000 [00:42<00:03, 8981.00it/s] 92%|█████████▏| 367838/400000 [00:42<00:03, 8685.16it/s] 92%|█████████▏| 368730/400000 [00:42<00:03, 8753.47it/s] 92%|█████████▏| 369643/400000 [00:42<00:03, 8860.30it/s] 93%|█████████▎| 370542/400000 [00:42<00:03, 8896.84it/s] 93%|█████████▎| 371473/400000 [00:42<00:03, 9014.68it/s] 93%|█████████▎| 372376/400000 [00:42<00:03, 8909.64it/s] 93%|█████████▎| 373269/400000 [00:42<00:03, 8820.14it/s] 94%|█████████▎| 374211/400000 [00:42<00:02, 8989.45it/s] 94%|█████████▍| 375112/400000 [00:43<00:02, 8953.83it/s] 94%|█████████▍| 376009/400000 [00:43<00:02, 8792.13it/s] 94%|█████████▍| 376890/400000 [00:43<00:02, 8630.58it/s] 94%|█████████▍| 377755/400000 [00:43<00:02, 8635.71it/s] 95%|█████████▍| 378662/400000 [00:43<00:02, 8759.12it/s] 95%|█████████▍| 379543/400000 [00:43<00:02, 8773.01it/s] 95%|█████████▌| 380432/400000 [00:43<00:02, 8806.55it/s] 95%|█████████▌| 381314/400000 [00:43<00:02, 8491.77it/s] 96%|█████████▌| 382167/400000 [00:43<00:02, 8068.63it/s] 96%|█████████▌| 383021/400000 [00:43<00:02, 8204.19it/s] 96%|█████████▌| 383847/400000 [00:44<00:01, 8198.08it/s] 96%|█████████▌| 384671/400000 [00:44<00:01, 8184.99it/s] 96%|█████████▋| 385492/400000 [00:44<00:01, 8114.69it/s] 97%|█████████▋| 386310/400000 [00:44<00:01, 8133.31it/s] 97%|█████████▋| 387166/400000 [00:44<00:01, 8255.80it/s] 97%|█████████▋| 387993/400000 [00:44<00:01, 8169.33it/s] 97%|█████████▋| 388821/400000 [00:44<00:01, 8200.44it/s] 97%|█████████▋| 389642/400000 [00:44<00:01, 8124.10it/s] 98%|█████████▊| 390456/400000 [00:44<00:01, 8115.24it/s] 98%|█████████▊| 391313/400000 [00:44<00:01, 8244.59it/s] 98%|█████████▊| 392139/400000 [00:45<00:00, 8178.83it/s] 98%|█████████▊| 392985/400000 [00:45<00:00, 8259.02it/s] 98%|█████████▊| 393812/400000 [00:45<00:00, 7939.19it/s] 99%|█████████▊| 394610/400000 [00:45<00:00, 7915.08it/s] 99%|█████████▉| 395437/400000 [00:45<00:00, 8016.44it/s] 99%|█████████▉| 396251/400000 [00:45<00:00, 8052.85it/s] 99%|█████████▉| 397107/400000 [00:45<00:00, 8197.45it/s] 99%|█████████▉| 397982/400000 [00:45<00:00, 8354.37it/s]100%|█████████▉| 398820/400000 [00:45<00:00, 8189.85it/s]100%|█████████▉| 399641/400000 [00:46<00:00, 8037.18it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8686.82it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fbbe4dfeb70>, <torchtext.data.dataset.TabularDataset object at 0x7fbbe4dfecc0>, <torchtext.vocab.Vocab object at 0x7fbbe4dfebe0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 28.38 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 41.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.33 file/s]2020-07-10 00:19:43.077352: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-10 00:19:43.081473: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-10 00:19:43.081639: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558d41b85ba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-10 00:19:43.081655: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2777088/9912422 [00:00<00:00, 25514713.09it/s]9920512it [00:00, 34995200.35it/s]                             
0it [00:00, ?it/s]32768it [00:00, 597732.19it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 467558.95it/s]1654784it [00:00, 11412023.48it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166662.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
