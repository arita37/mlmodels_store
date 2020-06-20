
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb420d93ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb420d93ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb48c34a048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb48c34a048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb4a6078e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb4a6078e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb439bc88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb439bc88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb439bc88c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3817472/9912422 [00:00<00:00, 38045769.25it/s]9920512it [00:00, 35720724.52it/s]                             
0it [00:00, ?it/s]32768it [00:00, 614263.29it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 480440.43it/s]1654784it [00:00, 9098838.96it/s]                          
0it [00:00, ?it/s]8192it [00:00, 204577.05it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb420d6e710>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb438b89860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb439bc8510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb439bc8510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb439bc8510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:59,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:58,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:08, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 13.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 65.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.91 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.91 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.04 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ url]
0 examples [00:00, ? examples/s]2020-06-20 18:08:56.681883: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 18:08:56.694293: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-06-20 18:08:56.694746: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d89b7a5e00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 18:08:56.694773: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
45 examples [00:00, 446.83 examples/s]163 examples [00:00, 549.19 examples/s]282 examples [00:00, 654.98 examples/s]401 examples [00:00, 756.95 examples/s]514 examples [00:00, 839.96 examples/s]627 examples [00:00, 907.56 examples/s]739 examples [00:00, 960.55 examples/s]854 examples [00:00, 1008.68 examples/s]964 examples [00:00, 1033.21 examples/s]1074 examples [00:01, 1051.04 examples/s]1185 examples [00:01, 1065.60 examples/s]1296 examples [00:01, 1076.49 examples/s]1405 examples [00:01, 1066.28 examples/s]1513 examples [00:01, 1056.19 examples/s]1622 examples [00:01, 1065.09 examples/s]1734 examples [00:01, 1079.60 examples/s]1849 examples [00:01, 1097.99 examples/s]1960 examples [00:01, 1099.43 examples/s]2071 examples [00:01, 1102.02 examples/s]2182 examples [00:02, 1103.03 examples/s]2293 examples [00:02, 1099.70 examples/s]2404 examples [00:02, 1088.01 examples/s]2515 examples [00:02, 1093.92 examples/s]2629 examples [00:02, 1105.71 examples/s]2743 examples [00:02, 1114.52 examples/s]2858 examples [00:02, 1123.79 examples/s]2971 examples [00:02, 1091.24 examples/s]3084 examples [00:02, 1101.21 examples/s]3205 examples [00:02, 1130.01 examples/s]3319 examples [00:03, 1130.96 examples/s]3436 examples [00:03, 1141.66 examples/s]3557 examples [00:03, 1160.82 examples/s]3674 examples [00:03, 1122.41 examples/s]3787 examples [00:03, 1085.15 examples/s]3903 examples [00:03, 1104.07 examples/s]4022 examples [00:03, 1126.30 examples/s]4138 examples [00:03, 1136.19 examples/s]4252 examples [00:03, 1135.62 examples/s]4366 examples [00:03, 1133.13 examples/s]4480 examples [00:04, 1122.31 examples/s]4595 examples [00:04, 1129.06 examples/s]4714 examples [00:04, 1144.83 examples/s]4835 examples [00:04, 1161.79 examples/s]4952 examples [00:04, 1157.14 examples/s]5068 examples [00:04, 1148.07 examples/s]5183 examples [00:04, 1125.87 examples/s]5303 examples [00:04, 1144.40 examples/s]5419 examples [00:04, 1146.39 examples/s]5534 examples [00:04, 1143.76 examples/s]5649 examples [00:05, 1128.93 examples/s]5765 examples [00:05, 1136.52 examples/s]5879 examples [00:05, 1128.72 examples/s]5999 examples [00:05, 1146.69 examples/s]6119 examples [00:05, 1160.12 examples/s]6236 examples [00:05, 1152.32 examples/s]6353 examples [00:05, 1155.16 examples/s]6469 examples [00:05, 1136.14 examples/s]6583 examples [00:05, 1127.87 examples/s]6698 examples [00:06, 1132.23 examples/s]6812 examples [00:06, 1126.13 examples/s]6925 examples [00:06, 1067.86 examples/s]7033 examples [00:06, 1042.28 examples/s]7138 examples [00:06, 958.76 examples/s] 7238 examples [00:06, 968.56 examples/s]7337 examples [00:06, 940.38 examples/s]7433 examples [00:06, 943.33 examples/s]7534 examples [00:06, 962.36 examples/s]7637 examples [00:06, 979.51 examples/s]7736 examples [00:07, 975.44 examples/s]7850 examples [00:07, 1017.51 examples/s]7967 examples [00:07, 1058.08 examples/s]8085 examples [00:07, 1091.26 examples/s]8200 examples [00:07, 1106.65 examples/s]8312 examples [00:07, 1100.45 examples/s]8427 examples [00:07, 1113.56 examples/s]8544 examples [00:07, 1127.31 examples/s]8663 examples [00:07, 1142.68 examples/s]8786 examples [00:07, 1166.04 examples/s]8903 examples [00:08, 1166.88 examples/s]9020 examples [00:08, 1159.04 examples/s]9137 examples [00:08, 1160.04 examples/s]9256 examples [00:08, 1168.30 examples/s]9373 examples [00:08, 1145.09 examples/s]9488 examples [00:08, 1121.46 examples/s]9604 examples [00:08, 1132.51 examples/s]9718 examples [00:08, 1116.91 examples/s]9839 examples [00:08, 1142.44 examples/s]9954 examples [00:09, 1126.26 examples/s]10067 examples [00:09, 1085.42 examples/s]10181 examples [00:09, 1099.40 examples/s]10294 examples [00:09, 1108.00 examples/s]10411 examples [00:09, 1113.45 examples/s]10526 examples [00:09, 1121.78 examples/s]10640 examples [00:09, 1126.90 examples/s]10759 examples [00:09, 1143.74 examples/s]10877 examples [00:09, 1152.65 examples/s]10998 examples [00:09, 1166.70 examples/s]11115 examples [00:10, 1166.65 examples/s]11232 examples [00:10, 1158.45 examples/s]11348 examples [00:10, 1156.85 examples/s]11464 examples [00:10, 1121.26 examples/s]11581 examples [00:10, 1133.06 examples/s]11697 examples [00:10, 1138.55 examples/s]11812 examples [00:10, 1135.38 examples/s]11932 examples [00:10, 1151.44 examples/s]12050 examples [00:10, 1157.83 examples/s]12166 examples [00:10, 1153.17 examples/s]12284 examples [00:11, 1160.03 examples/s]12401 examples [00:11, 1153.63 examples/s]12521 examples [00:11, 1165.11 examples/s]12638 examples [00:11, 1162.61 examples/s]12755 examples [00:11, 1135.78 examples/s]12871 examples [00:11, 1141.51 examples/s]12988 examples [00:11, 1148.19 examples/s]13106 examples [00:11, 1157.16 examples/s]13226 examples [00:11, 1167.96 examples/s]13347 examples [00:11, 1179.83 examples/s]13466 examples [00:12, 1165.72 examples/s]13583 examples [00:12, 1142.67 examples/s]13702 examples [00:12, 1153.60 examples/s]13818 examples [00:12, 1153.19 examples/s]13934 examples [00:12, 1148.39 examples/s]14050 examples [00:12, 1150.72 examples/s]14166 examples [00:12, 1146.74 examples/s]14288 examples [00:12, 1165.67 examples/s]14408 examples [00:12, 1173.27 examples/s]14531 examples [00:12, 1187.31 examples/s]14650 examples [00:13, 1181.67 examples/s]14769 examples [00:13, 1163.41 examples/s]14886 examples [00:13, 1157.89 examples/s]15003 examples [00:13, 1160.59 examples/s]15120 examples [00:13, 1156.25 examples/s]15236 examples [00:13, 1149.27 examples/s]15351 examples [00:13, 1145.50 examples/s]15466 examples [00:13, 1146.76 examples/s]15581 examples [00:13, 1139.69 examples/s]15701 examples [00:14, 1156.88 examples/s]15821 examples [00:14, 1168.71 examples/s]15938 examples [00:14, 1157.61 examples/s]16058 examples [00:14, 1168.15 examples/s]16176 examples [00:14, 1168.75 examples/s]16295 examples [00:14, 1174.82 examples/s]16413 examples [00:14, 1168.28 examples/s]16530 examples [00:14, 1166.60 examples/s]16647 examples [00:14, 1164.09 examples/s]16769 examples [00:14, 1178.10 examples/s]16890 examples [00:15, 1184.99 examples/s]17009 examples [00:15, 1177.33 examples/s]17127 examples [00:15, 1124.00 examples/s]17247 examples [00:15, 1145.02 examples/s]17365 examples [00:15, 1154.92 examples/s]17481 examples [00:15, 1145.19 examples/s]17597 examples [00:15, 1147.74 examples/s]17712 examples [00:15, 1144.96 examples/s]17828 examples [00:15, 1147.75 examples/s]17945 examples [00:15, 1153.08 examples/s]18061 examples [00:16, 1153.92 examples/s]18183 examples [00:16, 1169.66 examples/s]18301 examples [00:16, 1166.23 examples/s]18421 examples [00:16, 1175.41 examples/s]18539 examples [00:16, 1157.68 examples/s]18655 examples [00:16, 1154.10 examples/s]18773 examples [00:16, 1160.01 examples/s]18890 examples [00:16, 1152.74 examples/s]19011 examples [00:16, 1166.64 examples/s]19131 examples [00:16, 1176.43 examples/s]19251 examples [00:17, 1180.37 examples/s]19372 examples [00:17, 1187.55 examples/s]19491 examples [00:17, 1185.07 examples/s]19610 examples [00:17, 1176.90 examples/s]19729 examples [00:17, 1179.45 examples/s]19847 examples [00:17, 1163.18 examples/s]19968 examples [00:17, 1175.13 examples/s]20086 examples [00:17, 1115.84 examples/s]20199 examples [00:17, 1117.09 examples/s]20320 examples [00:17, 1141.01 examples/s]20439 examples [00:18, 1153.19 examples/s]20560 examples [00:18, 1167.47 examples/s]20678 examples [00:18, 1122.60 examples/s]20793 examples [00:18, 1129.85 examples/s]20913 examples [00:18, 1148.57 examples/s]21032 examples [00:18, 1160.32 examples/s]21152 examples [00:18, 1170.69 examples/s]21270 examples [00:18, 1171.29 examples/s]21388 examples [00:18, 1168.31 examples/s]21506 examples [00:19, 1171.24 examples/s]21624 examples [00:19, 1171.77 examples/s]21745 examples [00:19, 1182.40 examples/s]21864 examples [00:19, 1180.92 examples/s]21983 examples [00:19, 1130.81 examples/s]22101 examples [00:19, 1142.63 examples/s]22217 examples [00:19, 1145.81 examples/s]22341 examples [00:19, 1172.47 examples/s]22460 examples [00:19, 1174.97 examples/s]22581 examples [00:19, 1182.88 examples/s]22704 examples [00:20, 1195.21 examples/s]22825 examples [00:20, 1199.41 examples/s]22949 examples [00:20, 1209.92 examples/s]23071 examples [00:20, 1197.52 examples/s]23191 examples [00:20, 1183.27 examples/s]23313 examples [00:20, 1193.74 examples/s]23433 examples [00:20, 1195.37 examples/s]23553 examples [00:20, 1179.62 examples/s]23672 examples [00:20, 1174.49 examples/s]23790 examples [00:20, 1158.71 examples/s]23907 examples [00:21, 1161.02 examples/s]24029 examples [00:21, 1175.65 examples/s]24153 examples [00:21, 1191.54 examples/s]24273 examples [00:21, 1145.40 examples/s]24390 examples [00:21, 1152.38 examples/s]24506 examples [00:21, 1147.88 examples/s]24628 examples [00:21, 1166.63 examples/s]24745 examples [00:21, 1160.92 examples/s]24862 examples [00:21, 1159.32 examples/s]24982 examples [00:21, 1170.84 examples/s]25100 examples [00:22, 1173.39 examples/s]25222 examples [00:22, 1186.94 examples/s]25342 examples [00:22, 1188.36 examples/s]25461 examples [00:22, 1174.59 examples/s]25579 examples [00:22, 1173.60 examples/s]25697 examples [00:22, 1166.70 examples/s]25816 examples [00:22, 1172.99 examples/s]25937 examples [00:22, 1182.00 examples/s]26056 examples [00:22, 1164.22 examples/s]26175 examples [00:22, 1171.50 examples/s]26293 examples [00:23, 1172.92 examples/s]26411 examples [00:23, 1170.02 examples/s]26530 examples [00:23, 1173.67 examples/s]26648 examples [00:23, 1157.89 examples/s]26764 examples [00:23, 1152.77 examples/s]26882 examples [00:23, 1159.81 examples/s]27002 examples [00:23, 1171.00 examples/s]27120 examples [00:23, 1167.59 examples/s]27237 examples [00:23, 1151.68 examples/s]27353 examples [00:24, 1146.05 examples/s]27471 examples [00:24, 1155.93 examples/s]27591 examples [00:24, 1166.31 examples/s]27708 examples [00:24, 1163.57 examples/s]27825 examples [00:24, 1103.86 examples/s]27937 examples [00:24, 1098.19 examples/s]28055 examples [00:24, 1120.61 examples/s]28176 examples [00:24, 1145.07 examples/s]28294 examples [00:24, 1154.44 examples/s]28410 examples [00:24, 1147.56 examples/s]28528 examples [00:25, 1155.08 examples/s]28646 examples [00:25, 1160.19 examples/s]28769 examples [00:25, 1177.91 examples/s]28888 examples [00:25, 1179.21 examples/s]29007 examples [00:25, 1172.55 examples/s]29126 examples [00:25, 1174.50 examples/s]29244 examples [00:25, 1174.02 examples/s]29363 examples [00:25, 1177.93 examples/s]29482 examples [00:25, 1180.79 examples/s]29601 examples [00:25, 1152.09 examples/s]29717 examples [00:26, 1142.41 examples/s]29835 examples [00:26, 1151.75 examples/s]29957 examples [00:26, 1169.69 examples/s]30075 examples [00:26, 1115.37 examples/s]30189 examples [00:26, 1122.56 examples/s]30303 examples [00:26, 1125.27 examples/s]30423 examples [00:26, 1145.15 examples/s]30543 examples [00:26, 1160.45 examples/s]30663 examples [00:26, 1169.96 examples/s]30781 examples [00:26, 1166.45 examples/s]30899 examples [00:27, 1169.26 examples/s]31017 examples [00:27, 1159.71 examples/s]31135 examples [00:27, 1164.35 examples/s]31252 examples [00:27, 1164.98 examples/s]31369 examples [00:27, 1126.33 examples/s]31482 examples [00:27, 1124.01 examples/s]31602 examples [00:27, 1144.33 examples/s]31724 examples [00:27, 1164.37 examples/s]31844 examples [00:27, 1173.31 examples/s]31962 examples [00:28, 1172.15 examples/s]32080 examples [00:28, 1170.52 examples/s]32202 examples [00:28, 1183.81 examples/s]32321 examples [00:28, 1185.55 examples/s]32442 examples [00:28, 1191.29 examples/s]32562 examples [00:28, 1178.99 examples/s]32680 examples [00:28, 1174.57 examples/s]32802 examples [00:28, 1187.59 examples/s]32922 examples [00:28, 1189.90 examples/s]33042 examples [00:28, 1188.45 examples/s]33161 examples [00:29, 1167.76 examples/s]33281 examples [00:29, 1176.43 examples/s]33399 examples [00:29, 1174.11 examples/s]33517 examples [00:29, 1173.48 examples/s]33635 examples [00:29, 1171.43 examples/s]33753 examples [00:29, 1161.50 examples/s]33870 examples [00:29, 1161.01 examples/s]33991 examples [00:29, 1173.09 examples/s]34113 examples [00:29, 1185.98 examples/s]34236 examples [00:29, 1197.16 examples/s]34356 examples [00:30, 1191.61 examples/s]34480 examples [00:30, 1203.76 examples/s]34601 examples [00:30, 1203.29 examples/s]34726 examples [00:30, 1214.47 examples/s]34848 examples [00:30, 1168.97 examples/s]34966 examples [00:30, 1118.93 examples/s]35082 examples [00:30, 1128.47 examples/s]35204 examples [00:30, 1153.67 examples/s]35326 examples [00:30, 1171.20 examples/s]35444 examples [00:30, 1165.33 examples/s]35561 examples [00:31, 1152.95 examples/s]35677 examples [00:31, 1153.34 examples/s]35793 examples [00:31, 1152.91 examples/s]35912 examples [00:31, 1162.94 examples/s]36029 examples [00:31, 1117.80 examples/s]36142 examples [00:31, 1096.75 examples/s]36253 examples [00:31, 1100.19 examples/s]36370 examples [00:31, 1119.94 examples/s]36492 examples [00:31, 1147.54 examples/s]36610 examples [00:31, 1156.59 examples/s]36728 examples [00:32, 1162.38 examples/s]36845 examples [00:32, 1138.45 examples/s]36961 examples [00:32, 1142.36 examples/s]37082 examples [00:32, 1159.08 examples/s]37199 examples [00:32, 1152.30 examples/s]37316 examples [00:32, 1155.44 examples/s]37437 examples [00:32, 1171.10 examples/s]37557 examples [00:32, 1177.93 examples/s]37677 examples [00:32, 1181.67 examples/s]37801 examples [00:33, 1198.01 examples/s]37921 examples [00:33, 1183.62 examples/s]38042 examples [00:33, 1190.84 examples/s]38162 examples [00:33, 1177.51 examples/s]38280 examples [00:33, 1116.54 examples/s]38393 examples [00:33, 1089.61 examples/s]38503 examples [00:33, 1083.87 examples/s]38622 examples [00:33, 1112.16 examples/s]38734 examples [00:33, 1114.29 examples/s]38851 examples [00:33, 1129.73 examples/s]38965 examples [00:34, 1119.58 examples/s]39083 examples [00:34, 1135.35 examples/s]39199 examples [00:34, 1139.23 examples/s]39314 examples [00:34, 1135.94 examples/s]39430 examples [00:34, 1141.01 examples/s]39545 examples [00:34, 1135.58 examples/s]39662 examples [00:34, 1144.41 examples/s]39781 examples [00:34, 1155.17 examples/s]39902 examples [00:34, 1169.28 examples/s]40020 examples [00:34, 1109.21 examples/s]40132 examples [00:35, 1110.58 examples/s]40249 examples [00:35, 1126.37 examples/s]40373 examples [00:35, 1155.98 examples/s]40490 examples [00:35, 1143.97 examples/s]40605 examples [00:35, 1132.46 examples/s]40719 examples [00:35, 1115.72 examples/s]40831 examples [00:35, 1111.14 examples/s]40943 examples [00:35, 1113.54 examples/s]41055 examples [00:35, 1108.53 examples/s]41172 examples [00:35, 1123.63 examples/s]41287 examples [00:36, 1128.62 examples/s]41407 examples [00:36, 1147.10 examples/s]41523 examples [00:36, 1149.85 examples/s]41639 examples [00:36, 1144.48 examples/s]41754 examples [00:36, 1115.85 examples/s]41866 examples [00:36, 1068.98 examples/s]41987 examples [00:36, 1106.44 examples/s]42108 examples [00:36, 1133.22 examples/s]42230 examples [00:36, 1155.78 examples/s]42348 examples [00:37, 1162.80 examples/s]42465 examples [00:37, 1155.60 examples/s]42584 examples [00:37, 1164.37 examples/s]42701 examples [00:37, 1155.77 examples/s]42817 examples [00:37, 1129.71 examples/s]42935 examples [00:37, 1143.05 examples/s]43050 examples [00:37, 1128.66 examples/s]43171 examples [00:37, 1149.30 examples/s]43291 examples [00:37, 1161.37 examples/s]43415 examples [00:37, 1182.48 examples/s]43537 examples [00:38, 1190.35 examples/s]43657 examples [00:38, 1172.46 examples/s]43777 examples [00:38, 1178.07 examples/s]43897 examples [00:38, 1182.07 examples/s]44018 examples [00:38, 1189.97 examples/s]44138 examples [00:38, 1186.19 examples/s]44257 examples [00:38, 1179.59 examples/s]44376 examples [00:38, 1174.38 examples/s]44494 examples [00:38, 1167.92 examples/s]44614 examples [00:38, 1176.74 examples/s]44732 examples [00:39, 1174.98 examples/s]44850 examples [00:39, 1163.05 examples/s]44972 examples [00:39, 1177.09 examples/s]45092 examples [00:39, 1181.01 examples/s]45211 examples [00:39, 1182.02 examples/s]45330 examples [00:39, 1179.39 examples/s]45448 examples [00:39, 1147.91 examples/s]45568 examples [00:39, 1162.32 examples/s]45690 examples [00:39, 1178.95 examples/s]45811 examples [00:39, 1186.09 examples/s]45932 examples [00:40, 1191.92 examples/s]46052 examples [00:40, 1190.03 examples/s]46172 examples [00:40, 1187.45 examples/s]46291 examples [00:40, 1184.34 examples/s]46414 examples [00:40, 1195.08 examples/s]46534 examples [00:40, 1190.24 examples/s]46654 examples [00:40, 1189.59 examples/s]46773 examples [00:40, 1178.10 examples/s]46891 examples [00:40, 1178.67 examples/s]47009 examples [00:40, 1166.46 examples/s]47126 examples [00:41, 1157.25 examples/s]47242 examples [00:41, 1153.88 examples/s]47360 examples [00:41, 1160.03 examples/s]47478 examples [00:41, 1162.38 examples/s]47595 examples [00:41, 1153.41 examples/s]47711 examples [00:41, 1143.13 examples/s]47826 examples [00:41, 1141.28 examples/s]47941 examples [00:41, 1143.54 examples/s]48056 examples [00:41, 1124.46 examples/s]48172 examples [00:42, 1134.83 examples/s]48287 examples [00:42, 1139.18 examples/s]48403 examples [00:42, 1143.96 examples/s]48518 examples [00:42, 1145.28 examples/s]48634 examples [00:42, 1148.85 examples/s]48752 examples [00:42, 1157.79 examples/s]48868 examples [00:42, 1137.20 examples/s]48982 examples [00:42, 1100.34 examples/s]49097 examples [00:42, 1113.98 examples/s]49209 examples [00:42, 1114.41 examples/s]49326 examples [00:43, 1128.67 examples/s]49446 examples [00:43, 1146.89 examples/s]49564 examples [00:43, 1154.41 examples/s]49686 examples [00:43, 1170.93 examples/s]49804 examples [00:43, 1164.55 examples/s]49921 examples [00:43, 1152.89 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7680/50000 [00:00<00:00, 76798.43 examples/s] 46%|████▌     | 22894/50000 [00:00<00:00, 90198.34 examples/s] 75%|███████▍  | 37492/50000 [00:00<00:00, 101876.94 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 959.85 examples/s]214 examples [00:00, 1015.81 examples/s]339 examples [00:00, 1074.72 examples/s]463 examples [00:00, 1118.66 examples/s]577 examples [00:00, 1124.15 examples/s]693 examples [00:00, 1132.45 examples/s]803 examples [00:00, 1119.55 examples/s]919 examples [00:00, 1130.60 examples/s]1040 examples [00:00, 1153.28 examples/s]1163 examples [00:01, 1172.26 examples/s]1283 examples [00:01, 1180.25 examples/s]1400 examples [00:01, 1176.14 examples/s]1517 examples [00:01, 1165.36 examples/s]1641 examples [00:01, 1184.44 examples/s]1765 examples [00:01, 1197.97 examples/s]1885 examples [00:01, 1170.00 examples/s]2004 examples [00:01, 1173.89 examples/s]2122 examples [00:01, 1139.66 examples/s]2237 examples [00:01, 1121.26 examples/s]2356 examples [00:02, 1140.10 examples/s]2474 examples [00:02, 1150.34 examples/s]2590 examples [00:02, 1138.63 examples/s]2710 examples [00:02, 1155.15 examples/s]2826 examples [00:02, 1155.90 examples/s]2945 examples [00:02, 1163.05 examples/s]3062 examples [00:02, 1151.31 examples/s]3178 examples [00:02, 1137.55 examples/s]3295 examples [00:02, 1145.10 examples/s]3410 examples [00:02, 1144.72 examples/s]3529 examples [00:03, 1157.84 examples/s]3645 examples [00:03, 1145.36 examples/s]3761 examples [00:03, 1146.91 examples/s]3880 examples [00:03, 1159.17 examples/s]3997 examples [00:03, 1160.92 examples/s]4117 examples [00:03, 1170.32 examples/s]4239 examples [00:03, 1182.71 examples/s]4358 examples [00:03, 1184.34 examples/s]4477 examples [00:03, 1167.87 examples/s]4595 examples [00:03, 1171.05 examples/s]4718 examples [00:04, 1186.97 examples/s]4840 examples [00:04, 1195.02 examples/s]4960 examples [00:04, 1185.00 examples/s]5079 examples [00:04, 1186.07 examples/s]5198 examples [00:04, 1175.75 examples/s]5317 examples [00:04, 1177.27 examples/s]5439 examples [00:04, 1188.25 examples/s]5558 examples [00:04, 1183.14 examples/s]5677 examples [00:04, 1155.87 examples/s]5794 examples [00:04, 1159.84 examples/s]5914 examples [00:05, 1168.72 examples/s]6035 examples [00:05, 1177.90 examples/s]6153 examples [00:05, 1176.39 examples/s]6275 examples [00:05, 1186.82 examples/s]6394 examples [00:05, 1185.40 examples/s]6513 examples [00:05, 1181.62 examples/s]6634 examples [00:05, 1187.76 examples/s]6753 examples [00:05, 1173.43 examples/s]6875 examples [00:05, 1186.72 examples/s]6994 examples [00:05, 1182.21 examples/s]7113 examples [00:06, 1181.64 examples/s]7234 examples [00:06, 1188.08 examples/s]7355 examples [00:06, 1193.07 examples/s]7475 examples [00:06, 1194.11 examples/s]7595 examples [00:06, 1192.22 examples/s]7715 examples [00:06, 1192.97 examples/s]7838 examples [00:06, 1203.66 examples/s]7959 examples [00:06, 1202.39 examples/s]8080 examples [00:06, 1196.79 examples/s]8200 examples [00:07, 1176.08 examples/s]8319 examples [00:07, 1180.10 examples/s]8441 examples [00:07, 1191.43 examples/s]8563 examples [00:07, 1198.53 examples/s]8684 examples [00:07, 1198.29 examples/s]8805 examples [00:07, 1201.77 examples/s]8926 examples [00:07, 1195.64 examples/s]9046 examples [00:07, 1178.67 examples/s]9164 examples [00:07, 1149.80 examples/s]9280 examples [00:07, 1098.72 examples/s]9396 examples [00:08, 1115.70 examples/s]9515 examples [00:08, 1136.28 examples/s]9638 examples [00:08, 1160.44 examples/s]9755 examples [00:08, 1150.91 examples/s]9873 examples [00:08, 1157.54 examples/s]9989 examples [00:08, 1126.58 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEV3EL2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEV3EL2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb439bc88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb439bc88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb439bc88c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb4a5e1e780>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb3d4178b00>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb439bc88c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb439bc88c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb439bc88c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb438b892b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb438b891d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb3b70ea840> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb3b70ea840> 

  function with postional parmater data_info <function split_train_valid at 0x7fb3b70ea840> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb3b70ea950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb3b70ea950> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb3b70ea950> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=62d25cf173cbfe147e81f6162718564ea90a2820ffeb3b951bc1df0312d30a5c
  Stored in directory: /tmp/pip-ephem-wheel-cache-f0_3n6q4/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:05:34, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:18:20, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:04:10, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<7:03:16, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.79M/862M [00:01<4:56:27, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 5.45M/862M [00:01<3:26:45, 69.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.15M/862M [00:01<2:24:13, 98.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.4M/862M [00:01<1:40:33, 141kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:10:08, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<48:55, 286kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:01<34:10, 408kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.8M/862M [00:01<23:53, 580kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:01<16:45, 823kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:02<11:46, 1.17MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:02<08:18, 1.64MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:02<05:52, 2.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<04:12, 3.22MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<04:19, 3.12MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.9M/862M [00:03<03:11, 4.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<06:15, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:59, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<06:33, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<04:56, 2.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:05<03:39, 3.65MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<12:29, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<10:13, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<07:56, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<05:43, 2.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<09:22, 1.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<08:00, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:09<04:17, 3.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<52:00, 254kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<40:08, 329kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<29:17, 451kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<21:00, 628kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<14:48, 888kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<26:19, 499kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<20:07, 653kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<14:42, 893kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<10:30, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<11:32, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<09:45, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<07:32, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<05:28, 2.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<08:15, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<08:33, 1.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<06:35, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<04:56, 2.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<06:27, 2.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<05:44, 2.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<04:42, 2.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<03:30, 3.67MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<02:44, 4.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<58:18, 221kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<42:55, 300kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<30:31, 421kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:21<21:36, 594kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<18:58, 675kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<15:18, 836kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:23<11:13, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<10:04, 1.26MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<09:09, 1.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<06:51, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:59, 2.54MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:25, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:40, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:35, 1.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:48, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:46, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<05:10, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<03:49, 3.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:51, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:32, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:52, 2.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:03, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:17, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:54, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:34, 3.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<19:21, 640kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<15:30, 799kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<11:20, 1.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:09, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:56, 561kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<15:26, 794kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<20:35, 595kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<16:27, 744kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<12:00, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:30, 1.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:20, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:01, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<05:03, 2.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<15:46, 768kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<13:00, 931kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:32, 1.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<06:55, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:24, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:51, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:53, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:22, 2.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:48, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:26, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:53, 2.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:37, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:17, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:06, 1.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:36, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:07, 2.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:28, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:24, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:58, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<03:36, 3.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<18:24, 639kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<15:29, 760kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:02, 977kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<09:04, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:56, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:24, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:09, 2.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:32, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:26, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:30, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:54, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:37, 3.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:42, 2.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:51, 1.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:33, 2.52MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:00<03:18, 3.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<47:02, 243kB/s] .vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<34:45, 329kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<24:46, 460kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<17:25, 652kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<25:25, 447kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<19:42, 576kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<14:10, 800kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<10:06, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<11:11, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<09:22, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:56, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:59, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:23, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:47, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:38, 3.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:48, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:56, 1.88MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:36, 2.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:21, 3.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<11:37, 953kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:55, 1.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:20, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:05, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:49, 1.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:13, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:52, 2.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:01, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:02, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:39, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:29, 3.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:48, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:48, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:25, 2.45MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:17, 3.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:23, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:01, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<04:38, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:24, 3.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:45, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:09, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:49, 2.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:53, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:49, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:30, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:00, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:35, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:10, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<04:11, 2.53MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<03:12, 3.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<02:27, 4.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:28, 918kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:49, 1.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:12, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:16, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:53, 1.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:13, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:50, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:34, 2.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:39, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:16, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:39, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<03:28, 2.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:49, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:04, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:58, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<02:53, 3.54MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<11:17, 909kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:30, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:04, 1.45MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<05:02, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<49:27, 206kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<36:16, 281kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<25:45, 394kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<18:05, 560kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<26:37, 380kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<18:41, 539kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<15:45, 637kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<12:40, 792kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<09:15, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:13, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:51, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:00, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:18, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:51, 2.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:19, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:29, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:33, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:06, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:51, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:44, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:24, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:14, 3.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:51, 760kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<13:14, 738kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:29, 931kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:47, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:38, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<04:07, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<38:55, 249kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<31:06, 312kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<22:52, 424kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<16:25, 589kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<11:39, 828kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:57, 879kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<11:19, 851kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:45, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:20, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:49, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:40, 2.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<09:12, 1.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<10:27, 914kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<08:03, 1.19MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<06:09, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<04:27, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:29, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<08:07, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:32, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:45, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<03:28, 2.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<54:05, 174kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<41:15, 228kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<29:43, 317kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<20:59, 447kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<14:50, 632kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<15:41, 596kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<14:16, 655kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<10:52, 860kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<07:50, 1.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<05:39, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:49, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<09:27, 981kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:39, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:52, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:14, 2.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:11, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<09:04, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:10, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<05:17, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:50, 2.38MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:43, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<08:38, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:58, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:24, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:56, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:29, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:03, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:44, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:13, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<10:26, 862kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<08:49, 1.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:55, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:14, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:48, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:15, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:35, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:59, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:47, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:32, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:41, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:25, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:17, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:56, 2.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<02:54, 3.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:51, 1.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<08:35, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:43, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:02, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:33, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:59, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:38, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:07, 2.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:03, 2.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:02, 1.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:41, 997kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:50, 1.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:58, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<03:38, 2.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:26, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:23, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:47, 1.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:16, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:59, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:00, 2.84MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:04, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:59, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:48, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:51, 2.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:26, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:04, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:59, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:38, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:44, 3.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:50, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:32, 982kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:38, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:58, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:37, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:48, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:04, 1.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:34, 1.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:10, 1.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:04, 2.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:18, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:26, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:08, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:44, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:47, 2.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:36, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<08:05, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:25, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:51, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:39, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:49, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:23, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:02, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:34, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:21, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:35, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:48, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<09:37, 836kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:43, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:38, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:07, 1.94MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:52, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<08:03, 990kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:34, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:48, 1.65MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:33, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:26, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:21, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:01, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:15, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:20, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<08:02, 973kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<06:24, 1.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:56, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:39, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:47, 2.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:16, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:28, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:07, 2.47MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:49, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<06:32, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<05:24, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:57, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:58, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:48, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:39, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:26, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:12, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:10, 2.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:33, 2.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:03, 3.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:58, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<05:58, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<04:38, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:24, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:31, 2.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<10:34, 709kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<10:28, 715kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<08:05, 925kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:54, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:18, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:10, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<13:49, 537kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<11:12, 662kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:10, 905kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:52, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<04:14, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<14:40, 501kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<13:17, 553kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<09:55, 740kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:26, 985kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:22, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<03:53, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<10:15, 710kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<10:11, 714kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:42, 943kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:49, 1.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:13, 1.71MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:05, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<17:56, 402kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<15:33, 464kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<11:37, 620kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<08:16, 867kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<05:54, 1.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<09:42, 736kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<09:44, 733kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<07:32, 945kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:27, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:59, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:23, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:40, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:21, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:07, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:11, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<02:19, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<17:59, 389kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<16:55, 414kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<12:57, 540kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<09:16, 752kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:41, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:50, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:19, 833kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:52, 881kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:56, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:24, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:29, 2.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<09:41, 708kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<10:37, 646kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<08:26, 814kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:07, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:27, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:41, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:52, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:33, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:24, 1.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:34, 2.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:06, 3.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<01:45, 3.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<24:15, 277kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<18:49, 357kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<13:37, 493kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<09:39, 693kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<06:53, 966kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<17:54, 372kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<15:48, 421kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<11:55, 558kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<08:33, 776kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<06:12, 1.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:38, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:30, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:45, 976kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:32, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:00, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<03:39, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:43, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:08, 3.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<14:25, 452kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<13:21, 488kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<10:14, 637kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<07:43, 843kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:36, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:08, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<03:14, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:47, 950kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:46, 952kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:13, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:59, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:02, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:37, 2.44MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:05, 3.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:55, 3.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:19, 872kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<08:37, 739kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<06:56, 919kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:09, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:57, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:04, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:27, 2.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:02, 3.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:45, 934kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:06, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:35, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:18, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<03:31, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:33, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:19, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:04, 3.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<01:56, 3.22MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<01:47, 3.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<01:45, 3.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<1:30:25, 69.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<1:04:26, 96.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<45:32, 137kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<32:15, 193kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<22:59, 270kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<16:34, 375kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<12:02, 515kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<08:50, 701kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:36, 935kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<05:02, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<18:22, 336kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<15:02, 411kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<11:01, 559kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<08:11, 753kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<06:18, 977kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:50, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:47, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:08, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:33, 2.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:17, 2.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<01:56, 3.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<10:12, 598kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<09:30, 643kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<07:15, 840kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:29, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:11, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:25, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:43, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:24, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:01, 3.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:54, 3.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<07:20, 822kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<07:16, 831kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:40, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:22, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:27, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:48, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:19, 2.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:04, 2.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:48, 3.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:53, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:22, 937kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<05:04, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:00, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:22, 1.77MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:48, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:19, 2.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:05, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<01:48, 3.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<01:43, 3.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<01:33, 3.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<11:04, 533kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<08:45, 673kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<06:34, 896kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:03, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:57, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:11, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:35, 2.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:11, 2.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:53, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<01:50, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:21, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:47, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:36, 1.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:56, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:47, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:12, 2.62MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:02, 2.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<01:55, 3.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:48, 3.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<01:45, 3.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<15:36, 369kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<12:09, 474kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<08:54, 646kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:49, 843kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:10, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:04, 1.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:19, 1.72MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:42, 2.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:19, 2.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:00, 2.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:49, 3.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<07:00, 812kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<07:04, 805kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:32, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:15, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:46, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:16, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:00, 2.81MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:52, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:51, 3.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:47, 3.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<09:58, 564kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<07:56, 709kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:57, 941kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:30, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:32, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:52, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:19, 2.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:03, 2.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<01:44, 3.20MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:07, 908kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:06, 910kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:45, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:43, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:58, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:27, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:06, 2.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:46, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:37, 3.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:29, 3.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:47, 625kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:06, 677kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:11, 886kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:40, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:39, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:01, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:31, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<02:11, 2.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:57, 2.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:44, 3.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:34, 3.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<15:17, 355kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<12:28, 435kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<09:15, 584kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:49, 791kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:04, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:54, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:01, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:03, 2.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<28:33, 187kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<21:52, 245kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<15:47, 338kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<11:22, 469kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<08:13, 647kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<06:04, 875kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<04:34, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:31, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:05, 745kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<06:41, 789kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<05:11, 1.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:52, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:05, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:29, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:01, 2.59MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:43, 3.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:31, 3.44MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:36, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:50, 1.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:50, 1.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:04, 1.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:28, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:04, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:49, 2.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:36, 3.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:25, 3.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:16, 4.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:35, 921kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:25, 948kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:11, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:13, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:31, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<01:59, 2.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:39, 3.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<05:40, 895kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:44, 753kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:24, 936kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:04, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:05, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:24, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:55, 2.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:33, 3.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:38, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:24, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:24, 1.47MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:35, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:00, 2.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:37, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:19, 3.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<13:33, 364kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<11:41, 422kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<08:42, 567kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<06:17, 781kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:33, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<03:22, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<02:32, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<08:02, 606kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<08:18, 586kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<06:25, 757kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:46, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:35, 1.35MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:45, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:09, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:43, 2.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:25, 3.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<17:10, 279kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<14:10, 339kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<10:28, 458kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:32, 634kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:27, 873kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:01, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<03:00, 1.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:44, 999kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:30, 858kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:21, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:16, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:29, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:54, 2.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:31, 3.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:16, 3.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<15:31, 300kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<12:58, 359kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<09:35, 485kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:54, 672kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:57, 932kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:41, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<10:03, 457kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<08:56, 514kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:43, 683kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:52, 937kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:34, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:39, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:23, 840kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:39, 800kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:23, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:15, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:52, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:28, 3.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<08:24, 530kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<07:35, 587kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<05:41, 781kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:13, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:05, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:46, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:29, 976kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<04:48, 912kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:47, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:48, 1.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:06, 2.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:38, 2.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:19, 3.26MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:44, 640kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<06:23, 675kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:53, 880kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<03:33, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:40, 1.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:34, 2.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<08:28, 501kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<07:41, 553kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<05:43, 741kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<04:15, 994kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:06, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:20, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:46, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:12, 992kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:30, 927kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:56, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:55, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:15, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:41, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<01:23, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:43, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:54, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:15, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:44, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:21, 3.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:05, 3.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:06, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:42, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:59, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:14, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:42, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:19, 2.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:25, 732kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:18, 749kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:04, 971kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:00, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:14, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:40, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:46, 819kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:58, 785kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:50, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:49, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:16, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:41, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:22, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:49, 1.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:45, 806kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:52, 990kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:53, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<02:12, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:41, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:22, 2.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:07, 3.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:26, 849kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<04:49, 780kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:48, 988kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:48, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:08, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:39, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:18, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:05, 3.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<15:11, 243kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<12:11, 303kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<08:52, 416kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:29, 567kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<04:46, 771kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<03:27, 1.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<02:34, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:57, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:19, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:51, 941kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:04, 1.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:20, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:45, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:23, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:06, 3.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:37, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:11, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:33, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:54, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:29, 2.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:08, 3.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<00:57, 3.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:59, 700kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:49, 723kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:42, 940kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:43, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:00, 1.71MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:31, 2.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<06:27, 529kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:50, 585kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:29, 760kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:18, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:27, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:51<01:49, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:24, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:54, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:19, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:41, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:03, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:12, 2.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<00:57, 3.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:37, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:00, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:26, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:52, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:26, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:07, 2.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<00:54, 3.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:16, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:49, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:16, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:45, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:18, 2.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:03, 3.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:50, 3.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:01, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:20, 942kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:36, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:56, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:29, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:08, 2.70MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:54, 3.38MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:04, 1.00MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:20, 920kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:37, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:56, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:26, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:05, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:54, 769kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:50, 784kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:03, 982kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:23, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:49, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:06, 2.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:54, 3.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:58, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:29, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:00, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:30, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:08, 2.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:53, 3.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:22, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:40, 1.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:06, 1.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:34, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:11, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:48, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:04, 910kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:43, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:17, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:47, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:25, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:06, 2.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:51, 3.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:41, 3.99MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<06:41, 408kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<05:35, 488kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:08, 659kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:00, 904kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:10, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:35, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:17, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:25, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:54, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:26, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:05, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:49, 3.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:53, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:44, 946kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:18, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:41, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:14, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:56, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:06, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:53, 874kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:22, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:46, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:17, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:57, 2.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:00, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:54, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:41, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:18, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:59, 2.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:43, 3.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:55, 1.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:34, 927kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:04, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:35, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:09, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:51, 2.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:56, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:24, 962kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:57, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:26, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:24, 1.59MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:54, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:34, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:09, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:51, 2.59MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:48, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:28, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:04, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:47, 2.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:26, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:48, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:27, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:03, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:48, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:31, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:34, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:19, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:01, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:45, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:34, 3.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:30, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:29, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:07, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:50, 2.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:59, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:14, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:00, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:43, 2.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:04, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:17, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:01, 1.79MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:44, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:53, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:37, 2.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:53, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:38, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:27, 3.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:08, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:05, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:57, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:41, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:30, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:19, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:19, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:00, 1.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:42, 2.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:18, 1.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:12, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:54, 1.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:38, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:58, 356kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<03:17, 431kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<02:23, 587kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:50, 761kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:25, 978kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:12, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:00, 1.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:50, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:47, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:41, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:40, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:35, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:33, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:07, 1.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:59, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:50, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:43, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:39, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:34, 2.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:34, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:30, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:32, 2.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:30, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:29, 2.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:29, 2.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:29, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:28, 2.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:24, 911kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:10, 1.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:56, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:42, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:33, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:30, 2.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:30, 2.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:27, 2.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:28, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:26, 2.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:26, 2.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:01, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:54, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:45, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:33, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:30, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:28, 2.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:24, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:25, 2.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:23, 2.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:23, 2.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:22, 3.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:56, 587kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:40, 678kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:20, 851kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:04, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:52, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:42, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:35, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:31, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:27, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:23, 2.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:22, 2.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:21, 3.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:20, 3.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:00, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:57, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:45, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:39, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:31, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:29, 2.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:24, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:24, 2.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:21, 2.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:22, 2.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:20, 3.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:20, 3.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:18, 3.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:14, 447kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:47, 560kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:20, 746kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:00, 983kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:46, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:29, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:24, 2.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:21, 2.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:15, 746kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:13, 760kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:57, 972kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:42, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:33, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:26, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:21, 2.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:18, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:16, 3.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:56, 926kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:54, 952kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:42, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:32, 1.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:24, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:19, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:15, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:57, 839kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:03, 751kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:50, 947kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:36, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:27, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:15, 2.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<15:42, 46.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<11:16, 64.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<07:54, 91.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<05:27, 130kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<03:44, 184kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<02:34, 261kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<02:17, 288kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:49, 360kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:19, 490kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:55, 685kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:39, 942kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:28, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:35, 1.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:35, 998kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:19, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:10, 3.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:35, 871kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:42, 739kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:33, 923kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:23, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:16, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:22, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:23, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:19, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:14, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 3.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:21, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:25, 908kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:20, 1.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:14, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:09, 2.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:11, 1.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:15, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:12, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 1.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:05, 2.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:53, 276kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:42, 346kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:29, 476kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:19, 670kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:12, 940kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:14, 727kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:13, 770kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 934kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 958kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:02, 1.73MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.07MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 156/400000 [00:00<04:16, 1559.73it/s]  0%|          | 1123/400000 [00:00<03:11, 2084.06it/s]  1%|          | 2153/400000 [00:00<02:25, 2739.57it/s]  1%|          | 3187/400000 [00:00<01:52, 3514.40it/s]  1%|          | 4187/400000 [00:00<01:30, 4363.02it/s]  1%|▏         | 5209/400000 [00:00<01:14, 5268.77it/s]  2%|▏         | 6208/400000 [00:00<01:04, 6138.96it/s]  2%|▏         | 7213/400000 [00:00<00:56, 6949.27it/s]  2%|▏         | 8291/400000 [00:00<00:50, 7778.13it/s]  2%|▏         | 9358/400000 [00:01<00:46, 8464.84it/s]  3%|▎         | 10390/400000 [00:01<00:43, 8947.18it/s]  3%|▎         | 11401/400000 [00:01<00:42, 9240.71it/s]  3%|▎         | 12448/400000 [00:01<00:40, 9578.04it/s]  3%|▎         | 13481/400000 [00:01<00:39, 9791.61it/s]  4%|▎         | 14505/400000 [00:01<00:39, 9815.59it/s]  4%|▍         | 15521/400000 [00:01<00:38, 9915.02it/s]  4%|▍         | 16535/400000 [00:01<00:39, 9807.35it/s]  4%|▍         | 17532/400000 [00:01<00:39, 9789.09it/s]  5%|▍         | 18550/400000 [00:01<00:38, 9902.22it/s]  5%|▍         | 19549/400000 [00:02<00:39, 9735.00it/s]  5%|▌         | 20608/400000 [00:02<00:38, 9976.05it/s]  5%|▌         | 21612/400000 [00:02<00:38, 9886.43it/s]  6%|▌         | 22636/400000 [00:02<00:37, 9989.21it/s]  6%|▌         | 23649/400000 [00:02<00:37, 10028.67it/s]  6%|▌         | 24655/400000 [00:02<00:37, 10005.02it/s]  6%|▋         | 25674/400000 [00:02<00:37, 10059.26it/s]  7%|▋         | 26682/400000 [00:02<00:37, 9925.50it/s]   7%|▋         | 27676/400000 [00:02<00:37, 9871.40it/s]  7%|▋         | 28720/400000 [00:02<00:37, 10034.45it/s]  7%|▋         | 29725/400000 [00:03<00:37, 9918.41it/s]   8%|▊         | 30719/400000 [00:03<00:37, 9922.62it/s]  8%|▊         | 31769/400000 [00:03<00:36, 10087.40it/s]  8%|▊         | 32780/400000 [00:03<00:37, 9920.40it/s]   8%|▊         | 33774/400000 [00:03<00:37, 9691.47it/s]  9%|▊         | 34746/400000 [00:03<00:37, 9645.78it/s]  9%|▉         | 35766/400000 [00:03<00:37, 9804.42it/s]  9%|▉         | 36783/400000 [00:03<00:36, 9909.44it/s]  9%|▉         | 37818/400000 [00:03<00:36, 10035.77it/s] 10%|▉         | 38895/400000 [00:03<00:35, 10243.56it/s] 10%|▉         | 39922/400000 [00:04<00:35, 10173.70it/s] 10%|█         | 40952/400000 [00:04<00:35, 10208.83it/s] 10%|█         | 41974/400000 [00:04<00:35, 10145.99it/s] 11%|█         | 43001/400000 [00:04<00:35, 10181.97it/s] 11%|█         | 44020/400000 [00:04<00:35, 10080.80it/s] 11%|█▏        | 45045/400000 [00:04<00:35, 10129.84it/s] 12%|█▏        | 46059/400000 [00:04<00:35, 10001.04it/s] 12%|█▏        | 47091/400000 [00:04<00:34, 10092.22it/s] 12%|█▏        | 48139/400000 [00:04<00:34, 10203.83it/s] 12%|█▏        | 49190/400000 [00:04<00:34, 10293.04it/s] 13%|█▎        | 50221/400000 [00:05<00:33, 10291.34it/s] 13%|█▎        | 51253/400000 [00:05<00:33, 10296.99it/s] 13%|█▎        | 52284/400000 [00:05<00:34, 10179.55it/s] 13%|█▎        | 53303/400000 [00:05<00:34, 9957.94it/s]  14%|█▎        | 54301/400000 [00:05<00:35, 9771.90it/s] 14%|█▍        | 55280/400000 [00:05<00:35, 9641.93it/s] 14%|█▍        | 56284/400000 [00:05<00:35, 9757.95it/s] 14%|█▍        | 57305/400000 [00:05<00:34, 9888.97it/s] 15%|█▍        | 58296/400000 [00:05<00:35, 9750.94it/s] 15%|█▍        | 59291/400000 [00:05<00:34, 9807.85it/s] 15%|█▌        | 60273/400000 [00:06<00:34, 9738.40it/s] 15%|█▌        | 61278/400000 [00:06<00:34, 9827.80it/s] 16%|█▌        | 62269/400000 [00:06<00:34, 9851.87it/s] 16%|█▌        | 63255/400000 [00:06<00:34, 9735.21it/s] 16%|█▌        | 64266/400000 [00:06<00:34, 9842.92it/s] 16%|█▋        | 65264/400000 [00:06<00:33, 9882.55it/s] 17%|█▋        | 66336/400000 [00:06<00:32, 10119.35it/s] 17%|█▋        | 67350/400000 [00:06<00:32, 10087.69it/s] 17%|█▋        | 68361/400000 [00:06<00:33, 9835.97it/s]  17%|█▋        | 69347/400000 [00:07<00:33, 9819.40it/s] 18%|█▊        | 70365/400000 [00:07<00:33, 9923.72it/s] 18%|█▊        | 71419/400000 [00:07<00:32, 10098.55it/s] 18%|█▊        | 72441/400000 [00:07<00:32, 10133.02it/s] 18%|█▊        | 73456/400000 [00:07<00:32, 9897.15it/s]  19%|█▊        | 74448/400000 [00:07<00:33, 9821.27it/s] 19%|█▉        | 75451/400000 [00:07<00:32, 9881.53it/s] 19%|█▉        | 76441/400000 [00:07<00:33, 9801.02it/s] 19%|█▉        | 77426/400000 [00:07<00:32, 9815.38it/s] 20%|█▉        | 78446/400000 [00:07<00:32, 9927.42it/s] 20%|█▉        | 79447/400000 [00:08<00:32, 9949.78it/s] 20%|██        | 80443/400000 [00:08<00:32, 9912.29it/s] 20%|██        | 81482/400000 [00:08<00:31, 10049.27it/s] 21%|██        | 82488/400000 [00:08<00:32, 9892.69it/s]  21%|██        | 83479/400000 [00:08<00:32, 9654.98it/s] 21%|██        | 84447/400000 [00:08<00:32, 9565.81it/s] 21%|██▏       | 85406/400000 [00:08<00:33, 9406.63it/s] 22%|██▏       | 86425/400000 [00:08<00:32, 9626.00it/s] 22%|██▏       | 87391/400000 [00:08<00:32, 9563.20it/s] 22%|██▏       | 88461/400000 [00:08<00:31, 9875.91it/s] 22%|██▏       | 89453/400000 [00:09<00:32, 9559.24it/s] 23%|██▎       | 90468/400000 [00:09<00:31, 9727.21it/s] 23%|██▎       | 91527/400000 [00:09<00:30, 9969.45it/s] 23%|██▎       | 92529/400000 [00:09<00:31, 9885.74it/s] 23%|██▎       | 93531/400000 [00:09<00:30, 9925.30it/s] 24%|██▎       | 94585/400000 [00:09<00:30, 10101.69it/s] 24%|██▍       | 95603/400000 [00:09<00:30, 10125.02it/s] 24%|██▍       | 96653/400000 [00:09<00:29, 10234.50it/s] 24%|██▍       | 97678/400000 [00:09<00:30, 9995.90it/s]  25%|██▍       | 98753/400000 [00:09<00:29, 10209.40it/s] 25%|██▍       | 99791/400000 [00:10<00:29, 10257.84it/s] 25%|██▌       | 100836/400000 [00:10<00:29, 10313.97it/s] 25%|██▌       | 101869/400000 [00:10<00:28, 10288.99it/s] 26%|██▌       | 102899/400000 [00:10<00:29, 9957.57it/s]  26%|██▌       | 103898/400000 [00:10<00:30, 9711.30it/s] 26%|██▌       | 104907/400000 [00:10<00:30, 9819.51it/s] 26%|██▋       | 105934/400000 [00:10<00:29, 9949.87it/s] 27%|██▋       | 106935/400000 [00:10<00:29, 9966.30it/s] 27%|██▋       | 107934/400000 [00:10<00:30, 9704.57it/s] 27%|██▋       | 108908/400000 [00:11<00:30, 9588.67it/s] 27%|██▋       | 109870/400000 [00:11<00:30, 9438.57it/s] 28%|██▊       | 110816/400000 [00:11<00:30, 9358.09it/s] 28%|██▊       | 111754/400000 [00:11<00:31, 9273.03it/s] 28%|██▊       | 112683/400000 [00:11<00:31, 9167.13it/s] 28%|██▊       | 113653/400000 [00:11<00:30, 9318.92it/s] 29%|██▊       | 114651/400000 [00:11<00:30, 9507.24it/s] 29%|██▉       | 115604/400000 [00:11<00:30, 9454.39it/s] 29%|██▉       | 116625/400000 [00:11<00:29, 9666.57it/s] 29%|██▉       | 117614/400000 [00:11<00:29, 9730.61it/s] 30%|██▉       | 118658/400000 [00:12<00:28, 9931.41it/s] 30%|██▉       | 119654/400000 [00:12<00:28, 9733.21it/s] 30%|███       | 120645/400000 [00:12<00:28, 9784.04it/s] 30%|███       | 121692/400000 [00:12<00:27, 9980.00it/s] 31%|███       | 122693/400000 [00:12<00:28, 9848.22it/s] 31%|███       | 123690/400000 [00:12<00:27, 9884.06it/s] 31%|███       | 124698/400000 [00:12<00:27, 9939.90it/s] 31%|███▏      | 125694/400000 [00:12<00:27, 9926.22it/s] 32%|███▏      | 126696/400000 [00:12<00:27, 9951.99it/s] 32%|███▏      | 127692/400000 [00:12<00:27, 9874.49it/s] 32%|███▏      | 128698/400000 [00:13<00:27, 9927.67it/s] 32%|███▏      | 129692/400000 [00:13<00:28, 9547.25it/s] 33%|███▎      | 130660/400000 [00:13<00:28, 9585.10it/s] 33%|███▎      | 131656/400000 [00:13<00:27, 9692.18it/s] 33%|███▎      | 132628/400000 [00:13<00:28, 9503.97it/s] 33%|███▎      | 133628/400000 [00:13<00:27, 9645.09it/s] 34%|███▎      | 134595/400000 [00:13<00:27, 9602.25it/s] 34%|███▍      | 135560/400000 [00:13<00:27, 9615.71it/s] 34%|███▍      | 136523/400000 [00:13<00:27, 9488.11it/s] 34%|███▍      | 137500/400000 [00:13<00:27, 9567.78it/s] 35%|███▍      | 138458/400000 [00:14<00:27, 9561.15it/s] 35%|███▍      | 139441/400000 [00:14<00:27, 9637.56it/s] 35%|███▌      | 140485/400000 [00:14<00:26, 9863.89it/s] 35%|███▌      | 141510/400000 [00:14<00:25, 9975.10it/s] 36%|███▌      | 142510/400000 [00:14<00:26, 9892.92it/s] 36%|███▌      | 143552/400000 [00:14<00:25, 10043.69it/s] 36%|███▌      | 144558/400000 [00:14<00:25, 9998.13it/s]  36%|███▋      | 145587/400000 [00:14<00:25, 10082.04it/s] 37%|███▋      | 146597/400000 [00:14<00:25, 9941.90it/s]  37%|███▋      | 147593/400000 [00:14<00:25, 9934.35it/s] 37%|███▋      | 148588/400000 [00:15<00:25, 9909.74it/s] 37%|███▋      | 149582/400000 [00:15<00:25, 9918.66it/s] 38%|███▊      | 150575/400000 [00:15<00:25, 9870.06it/s] 38%|███▊      | 151563/400000 [00:15<00:25, 9784.75it/s] 38%|███▊      | 152595/400000 [00:15<00:24, 9936.72it/s] 38%|███▊      | 153590/400000 [00:15<00:24, 9894.16it/s] 39%|███▊      | 154636/400000 [00:15<00:24, 10057.18it/s] 39%|███▉      | 155646/400000 [00:15<00:24, 10066.79it/s] 39%|███▉      | 156654/400000 [00:15<00:24, 10014.92it/s] 39%|███▉      | 157657/400000 [00:15<00:24, 9990.09it/s]  40%|███▉      | 158692/400000 [00:16<00:23, 10094.43it/s] 40%|███▉      | 159703/400000 [00:16<00:24, 9971.71it/s]  40%|████      | 160752/400000 [00:16<00:23, 10120.56it/s] 40%|████      | 161774/400000 [00:16<00:23, 10147.85it/s] 41%|████      | 162795/400000 [00:16<00:23, 10164.93it/s] 41%|████      | 163813/400000 [00:16<00:23, 10127.34it/s] 41%|████      | 164864/400000 [00:16<00:22, 10238.26it/s] 41%|████▏     | 165902/400000 [00:16<00:22, 10279.11it/s] 42%|████▏     | 166931/400000 [00:16<00:22, 10265.08it/s] 42%|████▏     | 167958/400000 [00:17<00:23, 9870.96it/s]  42%|████▏     | 168949/400000 [00:17<00:23, 9851.51it/s] 42%|████▏     | 169943/400000 [00:17<00:23, 9875.87it/s] 43%|████▎     | 170933/400000 [00:17<00:24, 9407.76it/s] 43%|████▎     | 171888/400000 [00:17<00:24, 9447.60it/s] 43%|████▎     | 172892/400000 [00:17<00:23, 9617.25it/s] 43%|████▎     | 173858/400000 [00:17<00:23, 9506.96it/s] 44%|████▎     | 174817/400000 [00:17<00:23, 9530.72it/s] 44%|████▍     | 175854/400000 [00:17<00:22, 9767.12it/s] 44%|████▍     | 176867/400000 [00:17<00:22, 9872.15it/s] 44%|████▍     | 177857/400000 [00:18<00:22, 9687.59it/s] 45%|████▍     | 178829/400000 [00:18<00:23, 9492.73it/s] 45%|████▍     | 179885/400000 [00:18<00:22, 9787.76it/s] 45%|████▌     | 180902/400000 [00:18<00:22, 9897.58it/s] 45%|████▌     | 181913/400000 [00:18<00:21, 9957.01it/s] 46%|████▌     | 182911/400000 [00:18<00:22, 9822.98it/s] 46%|████▌     | 183896/400000 [00:18<00:22, 9816.38it/s] 46%|████▌     | 184949/400000 [00:18<00:21, 10018.14it/s] 47%|████▋     | 186004/400000 [00:18<00:21, 10169.86it/s] 47%|████▋     | 187023/400000 [00:18<00:21, 10110.69it/s] 47%|████▋     | 188055/400000 [00:19<00:20, 10170.22it/s] 47%|████▋     | 189074/400000 [00:19<00:20, 10141.48it/s] 48%|████▊     | 190089/400000 [00:19<00:20, 10117.74it/s] 48%|████▊     | 191124/400000 [00:19<00:20, 10185.09it/s] 48%|████▊     | 192144/400000 [00:19<00:20, 10166.73it/s] 48%|████▊     | 193176/400000 [00:19<00:20, 10211.19it/s] 49%|████▊     | 194198/400000 [00:19<00:20, 10168.97it/s] 49%|████▉     | 195217/400000 [00:19<00:20, 10173.95it/s] 49%|████▉     | 196267/400000 [00:19<00:19, 10267.37it/s] 49%|████▉     | 197295/400000 [00:19<00:20, 9685.84it/s]  50%|████▉     | 198325/400000 [00:20<00:20, 9861.89it/s] 50%|████▉     | 199317/400000 [00:20<00:20, 9854.52it/s] 50%|█████     | 200316/400000 [00:20<00:20, 9892.22it/s] 50%|█████     | 201309/400000 [00:20<00:20, 9874.42it/s] 51%|█████     | 202299/400000 [00:20<00:20, 9800.72it/s] 51%|█████     | 203281/400000 [00:20<00:20, 9647.55it/s] 51%|█████     | 204248/400000 [00:20<00:20, 9492.08it/s] 51%|█████▏    | 205239/400000 [00:20<00:20, 9611.71it/s] 52%|█████▏    | 206218/400000 [00:20<00:20, 9664.43it/s] 52%|█████▏    | 207225/400000 [00:20<00:19, 9782.43it/s] 52%|█████▏    | 208214/400000 [00:21<00:19, 9811.76it/s] 52%|█████▏    | 209229/400000 [00:21<00:19, 9910.00it/s] 53%|█████▎    | 210284/400000 [00:21<00:18, 10091.58it/s] 53%|█████▎    | 211295/400000 [00:21<00:18, 10002.83it/s] 53%|█████▎    | 212297/400000 [00:21<00:18, 9966.00it/s]  53%|█████▎    | 213295/400000 [00:21<00:18, 9931.18it/s] 54%|█████▎    | 214314/400000 [00:21<00:18, 10007.26it/s] 54%|█████▍    | 215323/400000 [00:21<00:18, 10030.49it/s] 54%|█████▍    | 216349/400000 [00:21<00:18, 10097.12it/s] 54%|█████▍    | 217360/400000 [00:22<00:18, 10040.61it/s] 55%|█████▍    | 218384/400000 [00:22<00:17, 10097.65it/s] 55%|█████▍    | 219395/400000 [00:22<00:17, 10086.24it/s] 55%|█████▌    | 220404/400000 [00:22<00:17, 10063.10it/s] 55%|█████▌    | 221436/400000 [00:22<00:17, 10136.24it/s] 56%|█████▌    | 222450/400000 [00:22<00:17, 9909.85it/s]  56%|█████▌    | 223443/400000 [00:22<00:18, 9801.81it/s] 56%|█████▌    | 224472/400000 [00:22<00:17, 9942.24it/s] 56%|█████▋    | 225482/400000 [00:22<00:17, 9986.53it/s] 57%|█████▋    | 226482/400000 [00:22<00:17, 9962.21it/s] 57%|█████▋    | 227479/400000 [00:23<00:17, 9908.48it/s] 57%|█████▋    | 228471/400000 [00:23<00:17, 9808.39it/s] 57%|█████▋    | 229459/400000 [00:23<00:17, 9827.95it/s] 58%|█████▊    | 230488/400000 [00:23<00:17, 9961.51it/s] 58%|█████▊    | 231485/400000 [00:23<00:17, 9887.76it/s] 58%|█████▊    | 232475/400000 [00:23<00:17, 9515.19it/s] 58%|█████▊    | 233430/400000 [00:23<00:17, 9504.75it/s] 59%|█████▊    | 234413/400000 [00:23<00:17, 9598.91it/s] 59%|█████▉    | 235402/400000 [00:23<00:17, 9681.70it/s] 59%|█████▉    | 236419/400000 [00:23<00:16, 9820.79it/s] 59%|█████▉    | 237403/400000 [00:24<00:16, 9708.55it/s] 60%|█████▉    | 238376/400000 [00:24<00:16, 9656.59it/s] 60%|█████▉    | 239343/400000 [00:24<00:16, 9559.19it/s] 60%|██████    | 240353/400000 [00:24<00:16, 9713.35it/s] 60%|██████    | 241380/400000 [00:24<00:16, 9873.21it/s] 61%|██████    | 242369/400000 [00:24<00:16, 9709.46it/s] 61%|██████    | 243355/400000 [00:24<00:16, 9751.94it/s] 61%|██████    | 244386/400000 [00:24<00:15, 9911.62it/s] 61%|██████▏   | 245447/400000 [00:24<00:15, 10110.96it/s] 62%|██████▏   | 246488/400000 [00:24<00:15, 10196.75it/s] 62%|██████▏   | 247510/400000 [00:25<00:15, 10155.63it/s] 62%|██████▏   | 248527/400000 [00:25<00:14, 10159.45it/s] 62%|██████▏   | 249544/400000 [00:25<00:14, 10140.58it/s] 63%|██████▎   | 250597/400000 [00:25<00:14, 10253.14it/s] 63%|██████▎   | 251629/400000 [00:25<00:14, 10271.92it/s] 63%|██████▎   | 252657/400000 [00:25<00:14, 10059.25it/s] 63%|██████▎   | 253665/400000 [00:25<00:14, 10005.24it/s] 64%|██████▎   | 254671/400000 [00:25<00:14, 10019.31it/s] 64%|██████▍   | 255700/400000 [00:25<00:14, 10096.35it/s] 64%|██████▍   | 256766/400000 [00:25<00:13, 10257.87it/s] 64%|██████▍   | 257793/400000 [00:26<00:14, 10082.82it/s] 65%|██████▍   | 258803/400000 [00:26<00:14, 10001.98it/s] 65%|██████▍   | 259805/400000 [00:26<00:14, 9977.31it/s]  65%|██████▌   | 260804/400000 [00:26<00:14, 9879.63it/s] 65%|██████▌   | 261811/400000 [00:26<00:13, 9934.34it/s] 66%|██████▌   | 262806/400000 [00:26<00:13, 9862.82it/s] 66%|██████▌   | 263823/400000 [00:26<00:13, 9951.16it/s] 66%|██████▌   | 264819/400000 [00:26<00:13, 9932.90it/s] 66%|██████▋   | 265893/400000 [00:26<00:13, 10160.76it/s] 67%|██████▋   | 266931/400000 [00:26<00:13, 10224.81it/s] 67%|██████▋   | 267955/400000 [00:27<00:13, 10071.14it/s] 67%|██████▋   | 269000/400000 [00:27<00:12, 10178.71it/s] 68%|██████▊   | 270020/400000 [00:27<00:12, 10113.62it/s] 68%|██████▊   | 271036/400000 [00:27<00:12, 10127.48it/s] 68%|██████▊   | 272050/400000 [00:27<00:12, 9854.98it/s]  68%|██████▊   | 273038/400000 [00:27<00:13, 9602.68it/s] 69%|██████▊   | 274002/400000 [00:27<00:13, 9517.21it/s] 69%|██████▊   | 274968/400000 [00:27<00:13, 9557.18it/s] 69%|██████▉   | 276017/400000 [00:27<00:12, 9817.73it/s] 69%|██████▉   | 277046/400000 [00:28<00:12, 9954.01it/s] 70%|██████▉   | 278044/400000 [00:28<00:12, 9765.43it/s] 70%|██████▉   | 279062/400000 [00:28<00:12, 9885.54it/s] 70%|███████   | 280109/400000 [00:28<00:11, 10053.07it/s] 70%|███████   | 281152/400000 [00:28<00:11, 10160.94it/s] 71%|███████   | 282170/400000 [00:28<00:11, 10057.11it/s] 71%|███████   | 283178/400000 [00:28<00:11, 9843.98it/s]  71%|███████   | 284228/400000 [00:28<00:11, 10029.40it/s] 71%|███████▏  | 285250/400000 [00:28<00:11, 10085.32it/s] 72%|███████▏  | 286261/400000 [00:28<00:11, 10074.69it/s] 72%|███████▏  | 287307/400000 [00:29<00:11, 10184.94it/s] 72%|███████▏  | 288327/400000 [00:29<00:11, 10116.68it/s] 72%|███████▏  | 289373/400000 [00:29<00:10, 10217.06it/s] 73%|███████▎  | 290416/400000 [00:29<00:10, 10277.96it/s] 73%|███████▎  | 291456/400000 [00:29<00:10, 10313.18it/s] 73%|███████▎  | 292488/400000 [00:29<00:10, 10290.60it/s] 73%|███████▎  | 293518/400000 [00:29<00:10, 9917.06it/s]  74%|███████▎  | 294531/400000 [00:29<00:10, 9977.44it/s] 74%|███████▍  | 295584/400000 [00:29<00:10, 10135.27it/s] 74%|███████▍  | 296609/400000 [00:29<00:10, 10169.22it/s] 74%|███████▍  | 297628/400000 [00:30<00:10, 9881.73it/s]  75%|███████▍  | 298620/400000 [00:30<00:10, 9731.67it/s] 75%|███████▍  | 299660/400000 [00:30<00:10, 9920.61it/s] 75%|███████▌  | 300688/400000 [00:30<00:09, 10022.68it/s] 75%|███████▌  | 301693/400000 [00:30<00:09, 9976.23it/s]  76%|███████▌  | 302693/400000 [00:30<00:09, 9809.32it/s] 76%|███████▌  | 303676/400000 [00:30<00:09, 9694.42it/s] 76%|███████▌  | 304691/400000 [00:30<00:09, 9825.44it/s] 76%|███████▋  | 305711/400000 [00:30<00:09, 9934.54it/s] 77%|███████▋  | 306772/400000 [00:30<00:09, 10126.28it/s] 77%|███████▋  | 307832/400000 [00:31<00:08, 10262.84it/s] 77%|███████▋  | 308861/400000 [00:31<00:08, 10170.89it/s] 77%|███████▋  | 309892/400000 [00:31<00:08, 10207.22it/s] 78%|███████▊  | 310935/400000 [00:31<00:08, 10270.20it/s] 78%|███████▊  | 311963/400000 [00:31<00:08, 10216.31it/s] 78%|███████▊  | 312986/400000 [00:31<00:08, 10180.79it/s] 79%|███████▊  | 314005/400000 [00:31<00:08, 9969.53it/s]  79%|███████▉  | 315033/400000 [00:31<00:08, 10059.53it/s] 79%|███████▉  | 316063/400000 [00:31<00:08, 10127.11it/s] 79%|███████▉  | 317081/400000 [00:31<00:08, 10140.97it/s] 80%|███████▉  | 318096/400000 [00:32<00:08, 10051.97it/s] 80%|███████▉  | 319102/400000 [00:32<00:08, 9698.82it/s]  80%|████████  | 320103/400000 [00:32<00:08, 9789.76it/s] 80%|████████  | 321141/400000 [00:32<00:07, 9959.62it/s] 81%|████████  | 322140/400000 [00:32<00:07, 9754.77it/s] 81%|████████  | 323119/400000 [00:32<00:07, 9755.43it/s] 81%|████████  | 324097/400000 [00:32<00:07, 9698.20it/s] 81%|████████▏ | 325073/400000 [00:32<00:07, 9714.04it/s] 82%|████████▏ | 326059/400000 [00:32<00:07, 9756.20it/s] 82%|████████▏ | 327058/400000 [00:33<00:07, 9823.54it/s] 82%|████████▏ | 328044/400000 [00:33<00:07, 9832.72it/s] 82%|████████▏ | 329051/400000 [00:33<00:07, 9901.12it/s] 83%|████████▎ | 330099/400000 [00:33<00:06, 10065.34it/s] 83%|████████▎ | 331171/400000 [00:33<00:06, 10252.41it/s] 83%|████████▎ | 332198/400000 [00:33<00:06, 10050.98it/s] 83%|████████▎ | 333206/400000 [00:33<00:06, 9760.10it/s]  84%|████████▎ | 334188/400000 [00:33<00:06, 9775.70it/s] 84%|████████▍ | 335168/400000 [00:33<00:06, 9676.93it/s] 84%|████████▍ | 336138/400000 [00:33<00:06, 9613.26it/s] 84%|████████▍ | 337101/400000 [00:34<00:06, 9571.52it/s] 85%|████████▍ | 338060/400000 [00:34<00:06, 9399.44it/s] 85%|████████▍ | 339002/400000 [00:34<00:06, 9305.17it/s] 85%|████████▍ | 339934/400000 [00:34<00:06, 9268.39it/s] 85%|████████▌ | 340863/400000 [00:34<00:06, 9274.10it/s] 85%|████████▌ | 341825/400000 [00:34<00:06, 9373.77it/s] 86%|████████▌ | 342773/400000 [00:34<00:06, 9404.24it/s] 86%|████████▌ | 343787/400000 [00:34<00:05, 9611.07it/s] 86%|████████▌ | 344814/400000 [00:34<00:05, 9797.73it/s] 86%|████████▋ | 345843/400000 [00:34<00:05, 9937.32it/s] 87%|████████▋ | 346877/400000 [00:35<00:05, 10052.69it/s] 87%|████████▋ | 347884/400000 [00:35<00:05, 9948.58it/s]  87%|████████▋ | 348921/400000 [00:35<00:05, 10069.12it/s] 87%|████████▋ | 349930/400000 [00:35<00:05, 9977.52it/s]  88%|████████▊ | 350929/400000 [00:35<00:04, 9838.98it/s] 88%|████████▊ | 351915/400000 [00:35<00:04, 9643.62it/s] 88%|████████▊ | 352882/400000 [00:35<00:05, 9298.38it/s] 88%|████████▊ | 353838/400000 [00:35<00:04, 9373.57it/s] 89%|████████▊ | 354856/400000 [00:35<00:04, 9599.84it/s] 89%|████████▉ | 355883/400000 [00:35<00:04, 9789.67it/s] 89%|████████▉ | 356866/400000 [00:36<00:04, 9732.83it/s] 89%|████████▉ | 357842/400000 [00:36<00:04, 9673.72it/s] 90%|████████▉ | 358887/400000 [00:36<00:04, 9892.23it/s] 90%|████████▉ | 359901/400000 [00:36<00:04, 9965.22it/s] 90%|█████████ | 360922/400000 [00:36<00:03, 10035.44it/s] 90%|█████████ | 361951/400000 [00:36<00:03, 10109.38it/s] 91%|█████████ | 362964/400000 [00:36<00:03, 9829.55it/s]  91%|█████████ | 363950/400000 [00:36<00:03, 9742.54it/s] 91%|█████████ | 364984/400000 [00:36<00:03, 9914.24it/s] 92%|█████████▏| 366023/400000 [00:37<00:03, 10051.85it/s] 92%|█████████▏| 367031/400000 [00:37<00:03, 9927.75it/s]  92%|█████████▏| 368026/400000 [00:37<00:03, 9730.71it/s] 92%|█████████▏| 369017/400000 [00:37<00:03, 9782.00it/s] 92%|█████████▏| 369997/400000 [00:37<00:03, 9595.96it/s] 93%|█████████▎| 370959/400000 [00:37<00:03, 9508.73it/s] 93%|█████████▎| 371927/400000 [00:37<00:02, 9558.72it/s] 93%|█████████▎| 372884/400000 [00:37<00:02, 9459.44it/s] 93%|█████████▎| 373883/400000 [00:37<00:02, 9610.12it/s] 94%|█████████▎| 374934/400000 [00:37<00:02, 9861.06it/s] 94%|█████████▍| 375936/400000 [00:38<00:02, 9907.87it/s] 94%|█████████▍| 376929/400000 [00:38<00:02, 9910.32it/s] 94%|█████████▍| 377922/400000 [00:38<00:02, 9733.37it/s] 95%|█████████▍| 378943/400000 [00:38<00:02, 9869.57it/s] 95%|█████████▍| 379988/400000 [00:38<00:01, 10036.48it/s] 95%|█████████▌| 380994/400000 [00:38<00:01, 9983.69it/s]  95%|█████████▌| 381994/400000 [00:38<00:01, 9801.45it/s] 96%|█████████▌| 382976/400000 [00:38<00:01, 9770.92it/s] 96%|█████████▌| 383955/400000 [00:38<00:01, 9725.22it/s] 96%|█████████▋| 385001/400000 [00:38<00:01, 9932.85it/s] 97%|█████████▋| 386037/400000 [00:39<00:01, 10056.31it/s] 97%|█████████▋| 387079/400000 [00:39<00:01, 10161.74it/s] 97%|█████████▋| 388097/400000 [00:39<00:01, 9931.83it/s]  97%|█████████▋| 389115/400000 [00:39<00:01, 10004.84it/s] 98%|█████████▊| 390157/400000 [00:39<00:00, 10125.20it/s] 98%|█████████▊| 391181/400000 [00:39<00:00, 10159.03it/s] 98%|█████████▊| 392198/400000 [00:39<00:00, 10114.62it/s] 98%|█████████▊| 393211/400000 [00:39<00:00, 9969.94it/s]  99%|█████████▊| 394210/400000 [00:39<00:00, 9922.26it/s] 99%|█████████▉| 395267/400000 [00:39<00:00, 10107.15it/s] 99%|█████████▉| 396320/400000 [00:40<00:00, 10227.37it/s] 99%|█████████▉| 397362/400000 [00:40<00:00, 10283.25it/s]100%|█████████▉| 398392/400000 [00:40<00:00, 10221.96it/s]100%|█████████▉| 399415/400000 [00:40<00:00, 10157.01it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9893.57it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb4a5dfa5c0>, <torchtext.data.dataset.TabularDataset object at 0x7fb4a5dfa470>, <torchtext.vocab.Vocab object at 0x7fb4a5dfa550>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.13 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.78 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 17.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 17.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.09 file/s]2020-06-20 18:17:46.569902: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 18:17:46.573376: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-06-20 18:17:46.573547: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561626038d20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 18:17:46.573560: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:21, 120955.82it/s] 83%|████████▎ | 8273920/9912422 [00:00<00:09, 172685.61it/s]9920512it [00:00, 39661055.83it/s]                           
0it [00:00, ?it/s]32768it [00:00, 597750.39it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470928.64it/s]1654784it [00:00, 11793265.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 167507.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
