
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb04ce5a048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb04ce5a048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb0b8a011e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb0b8a011e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb0d6d49ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb0d6d49ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb065d2f620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb065d2f620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb065d2f620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:19, 123724.36it/s] 77%|███████▋  | 7585792/9912422 [00:00<00:13, 176624.44it/s]9920512it [00:00, 39706775.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 855576.50it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464943.86it/s]1654784it [00:00, 11482159.94it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186461.06it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb063288908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb04cd62ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb065d2f268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb065d2f268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb065d2f268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:09,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:48,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 12.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 28.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 67.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.66 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.20 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.20 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.30 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.23s/ url]
0 examples [00:00, ? examples/s]2020-07-12 00:08:58.198652: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-12 00:08:58.263487: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-12 00:08:58.263716: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c5378b4f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 00:08:58.263735: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:01,  1.55s/ examples]115 examples [00:01,  1.09s/ examples]226 examples [00:01,  1.31 examples/s]320 examples [00:01,  1.87 examples/s]427 examples [00:01,  2.67 examples/s]536 examples [00:02,  3.82 examples/s]643 examples [00:02,  5.44 examples/s]747 examples [00:02,  7.76 examples/s]852 examples [00:02, 11.05 examples/s]959 examples [00:02, 15.71 examples/s]1076 examples [00:02, 22.31 examples/s]1192 examples [00:02, 31.62 examples/s]1303 examples [00:02, 44.62 examples/s]1418 examples [00:02, 62.69 examples/s]1534 examples [00:02, 87.51 examples/s]1646 examples [00:03, 120.61 examples/s]1756 examples [00:03, 164.33 examples/s]1865 examples [00:03, 217.80 examples/s]1971 examples [00:03, 285.85 examples/s]2086 examples [00:03, 368.90 examples/s]2200 examples [00:03, 462.39 examples/s]2310 examples [00:03, 559.10 examples/s]2425 examples [00:03, 660.46 examples/s]2541 examples [00:03, 757.09 examples/s]2658 examples [00:03, 846.35 examples/s]2772 examples [00:04, 914.98 examples/s]2886 examples [00:04, 968.84 examples/s]2999 examples [00:04, 990.52 examples/s]3110 examples [00:04, 1023.12 examples/s]3221 examples [00:04, 1023.71 examples/s]3334 examples [00:04, 1052.35 examples/s]3445 examples [00:04, 1066.87 examples/s]3555 examples [00:04, 1062.74 examples/s]3664 examples [00:04, 1052.52 examples/s]3771 examples [00:05, 1055.32 examples/s]3882 examples [00:05, 1069.21 examples/s]3995 examples [00:05, 1086.04 examples/s]4105 examples [00:05, 1088.48 examples/s]4215 examples [00:05, 1091.27 examples/s]4326 examples [00:05, 1094.72 examples/s]4436 examples [00:05, 1068.30 examples/s]4544 examples [00:05, 1063.27 examples/s]4652 examples [00:05, 1065.76 examples/s]4761 examples [00:05, 1071.96 examples/s]4876 examples [00:06, 1091.86 examples/s]4986 examples [00:06, 1093.96 examples/s]5101 examples [00:06, 1108.12 examples/s]5212 examples [00:06, 1096.19 examples/s]5325 examples [00:06, 1103.47 examples/s]5437 examples [00:06, 1107.56 examples/s]5551 examples [00:06, 1115.10 examples/s]5663 examples [00:06, 1105.24 examples/s]5774 examples [00:06, 1046.47 examples/s]5881 examples [00:06, 1051.32 examples/s]5993 examples [00:07, 1070.54 examples/s]6104 examples [00:07, 1081.48 examples/s]6215 examples [00:07, 1088.07 examples/s]6328 examples [00:07, 1098.67 examples/s]6439 examples [00:07, 1094.13 examples/s]6549 examples [00:07, 1063.15 examples/s]6661 examples [00:07, 1077.45 examples/s]6772 examples [00:07, 1086.99 examples/s]6881 examples [00:07, 1073.47 examples/s]6994 examples [00:07, 1088.91 examples/s]7108 examples [00:08, 1102.30 examples/s]7223 examples [00:08, 1114.86 examples/s]7335 examples [00:08, 1114.06 examples/s]7447 examples [00:08, 1102.07 examples/s]7559 examples [00:08, 1105.43 examples/s]7674 examples [00:08, 1116.97 examples/s]7788 examples [00:08, 1123.32 examples/s]7901 examples [00:08, 1111.02 examples/s]8016 examples [00:08, 1119.87 examples/s]8129 examples [00:09, 1121.68 examples/s]8245 examples [00:09, 1132.10 examples/s]8360 examples [00:09, 1135.20 examples/s]8474 examples [00:09, 1118.45 examples/s]8588 examples [00:09, 1121.96 examples/s]8702 examples [00:09, 1125.26 examples/s]8815 examples [00:09, 1123.37 examples/s]8930 examples [00:09, 1130.79 examples/s]9044 examples [00:09, 1112.55 examples/s]9156 examples [00:09, 1086.89 examples/s]9265 examples [00:10, 1085.72 examples/s]9375 examples [00:10, 1089.48 examples/s]9485 examples [00:10, 1074.27 examples/s]9593 examples [00:10, 1075.04 examples/s]9706 examples [00:10, 1090.62 examples/s]9816 examples [00:10, 1092.60 examples/s]9926 examples [00:10, 1070.26 examples/s]10034 examples [00:10, 1036.38 examples/s]10138 examples [00:10, 1033.58 examples/s]10251 examples [00:10, 1058.44 examples/s]10364 examples [00:11, 1077.60 examples/s]10475 examples [00:11, 1085.91 examples/s]10588 examples [00:11, 1098.31 examples/s]10699 examples [00:11, 1097.61 examples/s]10813 examples [00:11, 1107.40 examples/s]10929 examples [00:11, 1120.06 examples/s]11045 examples [00:11, 1131.18 examples/s]11161 examples [00:11, 1138.60 examples/s]11275 examples [00:11, 1119.40 examples/s]11389 examples [00:11, 1122.57 examples/s]11502 examples [00:12, 1028.39 examples/s]11607 examples [00:12, 991.41 examples/s] 11708 examples [00:12, 983.49 examples/s]11808 examples [00:12, 977.87 examples/s]11907 examples [00:12, 946.22 examples/s]12003 examples [00:12, 852.27 examples/s]12094 examples [00:12, 868.24 examples/s]12190 examples [00:12, 893.59 examples/s]12281 examples [00:12, 872.28 examples/s]12370 examples [00:13, 858.36 examples/s]12470 examples [00:13, 894.89 examples/s]12583 examples [00:13, 953.33 examples/s]12695 examples [00:13, 996.67 examples/s]12808 examples [00:13, 1033.06 examples/s]12917 examples [00:13, 1049.40 examples/s]13024 examples [00:13, 1036.33 examples/s]13134 examples [00:13, 1054.34 examples/s]13241 examples [00:13, 1056.53 examples/s]13357 examples [00:14, 1084.14 examples/s]13470 examples [00:14, 1096.45 examples/s]13584 examples [00:14, 1107.57 examples/s]13699 examples [00:14, 1118.28 examples/s]13812 examples [00:14, 1119.57 examples/s]13927 examples [00:14, 1127.41 examples/s]14041 examples [00:14, 1128.31 examples/s]14154 examples [00:14, 1121.80 examples/s]14267 examples [00:14, 1114.39 examples/s]14379 examples [00:14, 1097.92 examples/s]14489 examples [00:15, 1086.96 examples/s]14601 examples [00:15, 1096.45 examples/s]14716 examples [00:15, 1111.05 examples/s]14829 examples [00:15, 1116.61 examples/s]14941 examples [00:15, 1117.07 examples/s]15057 examples [00:15, 1128.22 examples/s]15170 examples [00:15, 1121.75 examples/s]15283 examples [00:15, 1102.93 examples/s]15394 examples [00:15, 1052.04 examples/s]15500 examples [00:15, 1034.11 examples/s]15609 examples [00:16, 1050.23 examples/s]15716 examples [00:16, 1054.22 examples/s]15822 examples [00:16, 1037.35 examples/s]15926 examples [00:16, 1011.44 examples/s]16029 examples [00:16, 1015.75 examples/s]16136 examples [00:16, 1029.59 examples/s]16240 examples [00:16, 1007.16 examples/s]16341 examples [00:16, 979.99 examples/s] 16440 examples [00:16, 969.17 examples/s]16543 examples [00:16, 984.27 examples/s]16651 examples [00:17, 1009.06 examples/s]16757 examples [00:17, 1023.27 examples/s]16867 examples [00:17, 1044.81 examples/s]16973 examples [00:17, 1046.64 examples/s]17081 examples [00:17, 1056.05 examples/s]17192 examples [00:17, 1069.83 examples/s]17303 examples [00:17, 1079.74 examples/s]17415 examples [00:17, 1089.12 examples/s]17525 examples [00:17, 1087.36 examples/s]17634 examples [00:18, 1084.61 examples/s]17747 examples [00:18, 1096.59 examples/s]17857 examples [00:18, 1091.11 examples/s]17967 examples [00:18, 1078.06 examples/s]18077 examples [00:18, 1082.54 examples/s]18186 examples [00:18, 1080.06 examples/s]18295 examples [00:18, 1080.32 examples/s]18404 examples [00:18, 1079.49 examples/s]18515 examples [00:18, 1087.03 examples/s]18624 examples [00:18, 1078.72 examples/s]18732 examples [00:19, 1073.74 examples/s]18841 examples [00:19, 1077.96 examples/s]18950 examples [00:19, 1078.92 examples/s]19060 examples [00:19, 1082.60 examples/s]19172 examples [00:19, 1090.97 examples/s]19282 examples [00:19, 1093.03 examples/s]19393 examples [00:19, 1095.81 examples/s]19505 examples [00:19, 1101.55 examples/s]19616 examples [00:19, 1071.96 examples/s]19724 examples [00:19, 1068.54 examples/s]19833 examples [00:20, 1073.99 examples/s]19949 examples [00:20, 1097.81 examples/s]20059 examples [00:20, 1053.92 examples/s]20165 examples [00:20, 1047.58 examples/s]20281 examples [00:20, 1076.91 examples/s]20395 examples [00:20, 1092.77 examples/s]20506 examples [00:20, 1097.08 examples/s]20616 examples [00:20, 1096.91 examples/s]20727 examples [00:20, 1098.72 examples/s]20840 examples [00:20, 1105.25 examples/s]20952 examples [00:21, 1107.82 examples/s]21063 examples [00:21, 1107.09 examples/s]21174 examples [00:21, 1097.93 examples/s]21287 examples [00:21, 1105.74 examples/s]21404 examples [00:21, 1122.83 examples/s]21518 examples [00:21, 1126.49 examples/s]21635 examples [00:21, 1137.72 examples/s]21751 examples [00:21, 1142.08 examples/s]21866 examples [00:21, 1082.63 examples/s]21981 examples [00:21, 1101.53 examples/s]22092 examples [00:22, 1086.24 examples/s]22205 examples [00:22, 1098.04 examples/s]22316 examples [00:22, 1083.54 examples/s]22425 examples [00:22, 1075.41 examples/s]22533 examples [00:22, 1071.27 examples/s]22645 examples [00:22, 1083.78 examples/s]22759 examples [00:22, 1099.72 examples/s]22870 examples [00:22, 1054.17 examples/s]22978 examples [00:22, 1060.25 examples/s]23093 examples [00:23, 1084.22 examples/s]23204 examples [00:23, 1089.37 examples/s]23314 examples [00:23, 1089.86 examples/s]23427 examples [00:23, 1100.31 examples/s]23541 examples [00:23, 1110.59 examples/s]23658 examples [00:23, 1127.17 examples/s]23772 examples [00:23, 1129.71 examples/s]23887 examples [00:23, 1135.22 examples/s]24003 examples [00:23, 1139.96 examples/s]24118 examples [00:23, 1137.25 examples/s]24232 examples [00:24, 1128.16 examples/s]24345 examples [00:24, 1122.17 examples/s]24459 examples [00:24, 1127.19 examples/s]24572 examples [00:24, 1071.24 examples/s]24680 examples [00:24, 1031.94 examples/s]24791 examples [00:24, 1051.47 examples/s]24906 examples [00:24, 1078.68 examples/s]25021 examples [00:24, 1098.92 examples/s]25133 examples [00:24, 1103.59 examples/s]25245 examples [00:24, 1107.85 examples/s]25357 examples [00:25, 1095.87 examples/s]25467 examples [00:25, 1076.61 examples/s]25575 examples [00:25, 1057.00 examples/s]25681 examples [00:25, 1033.52 examples/s]25791 examples [00:25, 1052.55 examples/s]25897 examples [00:25, 1034.92 examples/s]26011 examples [00:25, 1062.59 examples/s]26122 examples [00:25, 1073.71 examples/s]26230 examples [00:25, 1070.00 examples/s]26338 examples [00:25, 1052.25 examples/s]26452 examples [00:26, 1076.86 examples/s]26565 examples [00:26, 1089.60 examples/s]26678 examples [00:26, 1098.68 examples/s]26789 examples [00:26, 1079.59 examples/s]26904 examples [00:26, 1097.72 examples/s]27017 examples [00:26, 1104.80 examples/s]27129 examples [00:26, 1108.88 examples/s]27243 examples [00:26, 1115.74 examples/s]27355 examples [00:26, 1081.56 examples/s]27468 examples [00:27, 1094.41 examples/s]27578 examples [00:27, 1073.13 examples/s]27689 examples [00:27, 1081.41 examples/s]27798 examples [00:27, 1070.96 examples/s]27906 examples [00:27, 1068.75 examples/s]28014 examples [00:27, 1070.56 examples/s]28125 examples [00:27, 1079.70 examples/s]28241 examples [00:27, 1101.12 examples/s]28352 examples [00:27, 1096.41 examples/s]28468 examples [00:27, 1112.06 examples/s]28582 examples [00:28, 1119.38 examples/s]28695 examples [00:28, 1100.73 examples/s]28811 examples [00:28, 1116.25 examples/s]28923 examples [00:28, 1077.23 examples/s]29039 examples [00:28, 1100.26 examples/s]29156 examples [00:28, 1120.07 examples/s]29269 examples [00:28, 1114.83 examples/s]29383 examples [00:28, 1121.35 examples/s]29496 examples [00:28, 1114.64 examples/s]29608 examples [00:28, 1088.83 examples/s]29723 examples [00:29, 1105.84 examples/s]29834 examples [00:29, 1103.22 examples/s]29945 examples [00:29, 1102.03 examples/s]30056 examples [00:29, 1044.25 examples/s]30173 examples [00:29, 1078.23 examples/s]30288 examples [00:29, 1097.86 examples/s]30401 examples [00:29, 1105.66 examples/s]30513 examples [00:29, 1108.49 examples/s]30625 examples [00:29, 1054.85 examples/s]30735 examples [00:30, 1067.76 examples/s]30843 examples [00:30, 1048.28 examples/s]30956 examples [00:30, 1070.86 examples/s]31069 examples [00:30, 1085.92 examples/s]31178 examples [00:30, 1070.81 examples/s]31293 examples [00:30, 1092.80 examples/s]31408 examples [00:30, 1107.33 examples/s]31522 examples [00:30, 1115.41 examples/s]31634 examples [00:30, 1114.84 examples/s]31746 examples [00:30, 1104.92 examples/s]31857 examples [00:31, 1098.34 examples/s]31967 examples [00:31, 1085.00 examples/s]32076 examples [00:31, 1080.50 examples/s]32188 examples [00:31, 1089.36 examples/s]32300 examples [00:31, 1097.35 examples/s]32413 examples [00:31, 1105.23 examples/s]32526 examples [00:31, 1112.26 examples/s]32638 examples [00:31, 1104.49 examples/s]32749 examples [00:31, 1104.87 examples/s]32860 examples [00:31, 1086.99 examples/s]32969 examples [00:32, 1050.89 examples/s]33079 examples [00:32, 1063.07 examples/s]33190 examples [00:32, 1074.00 examples/s]33301 examples [00:32, 1082.47 examples/s]33410 examples [00:32, 1069.25 examples/s]33520 examples [00:32, 1077.85 examples/s]33633 examples [00:32, 1091.95 examples/s]33749 examples [00:32, 1109.36 examples/s]33861 examples [00:32, 1112.16 examples/s]33975 examples [00:32, 1118.57 examples/s]34090 examples [00:33, 1126.99 examples/s]34203 examples [00:33, 1125.17 examples/s]34316 examples [00:33, 1120.47 examples/s]34429 examples [00:33, 1106.03 examples/s]34542 examples [00:33, 1112.74 examples/s]34659 examples [00:33, 1128.22 examples/s]34772 examples [00:33, 1118.48 examples/s]34884 examples [00:33, 1110.67 examples/s]34999 examples [00:33, 1120.45 examples/s]35112 examples [00:33, 1102.66 examples/s]35228 examples [00:34, 1116.78 examples/s]35340 examples [00:34, 1114.96 examples/s]35453 examples [00:34, 1118.80 examples/s]35569 examples [00:34, 1130.84 examples/s]35684 examples [00:34, 1133.80 examples/s]35798 examples [00:34, 1132.76 examples/s]35913 examples [00:34, 1135.60 examples/s]36027 examples [00:34, 1116.81 examples/s]36140 examples [00:34, 1120.17 examples/s]36253 examples [00:35, 1085.02 examples/s]36362 examples [00:35, 1036.77 examples/s]36472 examples [00:35, 1053.45 examples/s]36581 examples [00:35, 1063.52 examples/s]36692 examples [00:35, 1074.62 examples/s]36804 examples [00:35, 1086.88 examples/s]36919 examples [00:35, 1102.41 examples/s]37032 examples [00:35, 1109.49 examples/s]37144 examples [00:35, 1101.64 examples/s]37255 examples [00:35, 1095.63 examples/s]37369 examples [00:36, 1108.48 examples/s]37484 examples [00:36, 1117.81 examples/s]37596 examples [00:36, 1105.94 examples/s]37707 examples [00:36, 1095.24 examples/s]37817 examples [00:36, 1087.42 examples/s]37930 examples [00:36, 1099.80 examples/s]38044 examples [00:36, 1109.82 examples/s]38159 examples [00:36, 1118.81 examples/s]38271 examples [00:36, 1111.51 examples/s]38385 examples [00:36, 1119.54 examples/s]38499 examples [00:37, 1125.51 examples/s]38615 examples [00:37, 1133.94 examples/s]38729 examples [00:37, 1134.50 examples/s]38843 examples [00:37, 1112.11 examples/s]38955 examples [00:37, 1109.94 examples/s]39070 examples [00:37, 1119.38 examples/s]39183 examples [00:37, 1099.11 examples/s]39294 examples [00:37, 1093.58 examples/s]39404 examples [00:37, 1092.05 examples/s]39514 examples [00:37, 1056.83 examples/s]39626 examples [00:38, 1074.23 examples/s]39734 examples [00:38, 1048.12 examples/s]39845 examples [00:38, 1065.08 examples/s]39952 examples [00:38, 1062.54 examples/s]40059 examples [00:38, 1008.85 examples/s]40171 examples [00:38, 1038.60 examples/s]40280 examples [00:38, 1050.21 examples/s]40389 examples [00:38, 1059.88 examples/s]40502 examples [00:38, 1077.59 examples/s]40616 examples [00:39, 1093.11 examples/s]40732 examples [00:39, 1111.00 examples/s]40848 examples [00:39, 1124.15 examples/s]40961 examples [00:39, 1125.24 examples/s]41074 examples [00:39, 1126.05 examples/s]41187 examples [00:39, 1124.54 examples/s]41300 examples [00:39, 1125.96 examples/s]41414 examples [00:39, 1128.55 examples/s]41528 examples [00:39, 1130.94 examples/s]41642 examples [00:39, 1101.08 examples/s]41753 examples [00:40, 1102.76 examples/s]41864 examples [00:40, 1098.70 examples/s]41977 examples [00:40, 1107.59 examples/s]42088 examples [00:40, 1106.73 examples/s]42199 examples [00:40, 1105.10 examples/s]42312 examples [00:40, 1112.20 examples/s]42424 examples [00:40, 1111.91 examples/s]42537 examples [00:40, 1114.86 examples/s]42649 examples [00:40, 1102.39 examples/s]42760 examples [00:40, 1030.57 examples/s]42865 examples [00:41, 1023.80 examples/s]42972 examples [00:41, 1034.78 examples/s]43077 examples [00:41, 1036.72 examples/s]43182 examples [00:41, 1032.17 examples/s]43292 examples [00:41, 1049.11 examples/s]43401 examples [00:41, 1059.93 examples/s]43513 examples [00:41, 1076.47 examples/s]43625 examples [00:41, 1088.15 examples/s]43735 examples [00:41, 1089.68 examples/s]43847 examples [00:41, 1095.62 examples/s]43959 examples [00:42, 1100.95 examples/s]44070 examples [00:42, 1103.10 examples/s]44182 examples [00:42, 1107.60 examples/s]44297 examples [00:42, 1118.99 examples/s]44409 examples [00:42, 1088.07 examples/s]44523 examples [00:42, 1102.98 examples/s]44634 examples [00:42, 1092.79 examples/s]44744 examples [00:42, 1075.85 examples/s]44854 examples [00:42, 1080.75 examples/s]44963 examples [00:43, 1075.81 examples/s]45071 examples [00:43, 1067.08 examples/s]45179 examples [00:43, 1069.68 examples/s]45287 examples [00:43, 1058.40 examples/s]45395 examples [00:43, 1064.50 examples/s]45505 examples [00:43, 1073.11 examples/s]45617 examples [00:43, 1084.76 examples/s]45729 examples [00:43, 1093.74 examples/s]45839 examples [00:43, 1080.51 examples/s]45948 examples [00:43, 1082.43 examples/s]46060 examples [00:44, 1091.71 examples/s]46172 examples [00:44, 1098.89 examples/s]46282 examples [00:44, 1063.53 examples/s]46389 examples [00:44, 1049.94 examples/s]46500 examples [00:44, 1066.91 examples/s]46609 examples [00:44, 1070.98 examples/s]46723 examples [00:44, 1090.32 examples/s]46837 examples [00:44, 1103.35 examples/s]46950 examples [00:44, 1111.00 examples/s]47062 examples [00:44, 1076.15 examples/s]47174 examples [00:45, 1087.71 examples/s]47284 examples [00:45, 1087.66 examples/s]47396 examples [00:45, 1095.98 examples/s]47509 examples [00:45, 1104.64 examples/s]47620 examples [00:45, 1101.13 examples/s]47734 examples [00:45, 1111.66 examples/s]47846 examples [00:45, 1101.15 examples/s]47957 examples [00:45, 1092.96 examples/s]48067 examples [00:45, 1041.81 examples/s]48172 examples [00:46, 955.45 examples/s] 48270 examples [00:46, 919.02 examples/s]48371 examples [00:46, 944.21 examples/s]48483 examples [00:46, 989.13 examples/s]48584 examples [00:46, 949.97 examples/s]48687 examples [00:46, 971.39 examples/s]48798 examples [00:46, 1007.88 examples/s]48910 examples [00:46, 1037.23 examples/s]49022 examples [00:46, 1058.62 examples/s]49134 examples [00:46, 1074.13 examples/s]49243 examples [00:47, 1076.96 examples/s]49352 examples [00:47, 1055.45 examples/s]49458 examples [00:47, 1043.51 examples/s]49563 examples [00:47, 1007.15 examples/s]49670 examples [00:47, 1025.11 examples/s]49785 examples [00:47, 1057.57 examples/s]49898 examples [00:47, 1075.97 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7525/50000 [00:00<00:00, 75244.51 examples/s] 42%|████▏     | 21059/50000 [00:00<00:00, 86806.58 examples/s] 69%|██████▉   | 34420/50000 [00:00<00:00, 96998.67 examples/s] 96%|█████████▌| 48037/50000 [00:00<00:00, 106159.55 examples/s]                                                                0 examples [00:00, ? examples/s]95 examples [00:00, 943.82 examples/s]208 examples [00:00, 991.51 examples/s]312 examples [00:00, 1003.25 examples/s]425 examples [00:00, 1038.04 examples/s]538 examples [00:00, 1061.85 examples/s]652 examples [00:00, 1083.67 examples/s]761 examples [00:00, 1083.92 examples/s]866 examples [00:00, 1073.29 examples/s]969 examples [00:00, 1054.17 examples/s]1082 examples [00:01, 1075.31 examples/s]1194 examples [00:01, 1085.93 examples/s]1307 examples [00:01, 1098.32 examples/s]1423 examples [00:01, 1112.78 examples/s]1534 examples [00:01, 1102.48 examples/s]1648 examples [00:01, 1113.13 examples/s]1760 examples [00:01, 1114.40 examples/s]1872 examples [00:01, 1111.59 examples/s]1984 examples [00:01, 1071.95 examples/s]2092 examples [00:01, 1073.61 examples/s]2201 examples [00:02, 1077.44 examples/s]2311 examples [00:02, 1081.13 examples/s]2420 examples [00:02, 1049.07 examples/s]2526 examples [00:02, 1032.67 examples/s]2635 examples [00:02, 1049.18 examples/s]2742 examples [00:02, 1053.19 examples/s]2853 examples [00:02, 1067.93 examples/s]2962 examples [00:02, 1072.00 examples/s]3070 examples [00:02, 1072.70 examples/s]3178 examples [00:02, 1056.37 examples/s]3286 examples [00:03, 1061.85 examples/s]3394 examples [00:03, 1064.33 examples/s]3501 examples [00:03, 1049.85 examples/s]3614 examples [00:03, 1071.89 examples/s]3729 examples [00:03, 1092.22 examples/s]3844 examples [00:03, 1106.52 examples/s]3955 examples [00:03, 1083.30 examples/s]4067 examples [00:03, 1092.77 examples/s]4177 examples [00:03, 1075.56 examples/s]4291 examples [00:03, 1093.15 examples/s]4402 examples [00:04, 1097.82 examples/s]4513 examples [00:04, 1101.01 examples/s]4627 examples [00:04, 1108.12 examples/s]4738 examples [00:04, 1096.68 examples/s]4853 examples [00:04, 1110.71 examples/s]4965 examples [00:04, 1084.65 examples/s]5081 examples [00:04, 1104.70 examples/s]5192 examples [00:04, 1095.12 examples/s]5302 examples [00:04, 1096.06 examples/s]5418 examples [00:04, 1112.51 examples/s]5531 examples [00:05, 1117.04 examples/s]5643 examples [00:05, 1116.67 examples/s]5755 examples [00:05, 1110.73 examples/s]5871 examples [00:05, 1124.10 examples/s]5987 examples [00:05, 1132.94 examples/s]6101 examples [00:05, 1126.34 examples/s]6217 examples [00:05, 1133.73 examples/s]6335 examples [00:05, 1145.58 examples/s]6450 examples [00:05, 1129.12 examples/s]6564 examples [00:06, 1117.24 examples/s]6676 examples [00:06, 1110.36 examples/s]6792 examples [00:06, 1122.28 examples/s]6908 examples [00:06, 1132.48 examples/s]7022 examples [00:06, 1129.11 examples/s]7135 examples [00:06, 1122.02 examples/s]7248 examples [00:06, 1107.28 examples/s]7362 examples [00:06, 1114.29 examples/s]7474 examples [00:06, 1107.86 examples/s]7585 examples [00:06, 1086.34 examples/s]7695 examples [00:07, 1090.12 examples/s]7805 examples [00:07, 1088.75 examples/s]7916 examples [00:07, 1093.30 examples/s]8026 examples [00:07, 1090.53 examples/s]8136 examples [00:07, 1088.20 examples/s]8248 examples [00:07, 1094.73 examples/s]8359 examples [00:07, 1097.75 examples/s]8470 examples [00:07, 1100.94 examples/s]8581 examples [00:07, 1099.03 examples/s]8691 examples [00:07, 1089.66 examples/s]8800 examples [00:08, 1067.04 examples/s]8907 examples [00:08, 1043.79 examples/s]9012 examples [00:08, 1032.87 examples/s]9116 examples [00:08, 1015.24 examples/s]9227 examples [00:08, 1040.62 examples/s]9338 examples [00:08, 1058.39 examples/s]9448 examples [00:08, 1069.08 examples/s]9558 examples [00:08, 1078.15 examples/s]9668 examples [00:08, 1083.17 examples/s]9777 examples [00:08, 1075.95 examples/s]9885 examples [00:09, 1077.11 examples/s]9993 examples [00:09, 1063.33 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6ES9X2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete6ES9X2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb065d2f620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb065d2f620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb065d2f620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb0001b4438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb0001f3b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb065d2f620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb065d2f620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb065d2f620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb0001f47b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb04cd62358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fafec093400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fafec093400> 

  function with postional parmater data_info <function split_train_valid at 0x7fafec093400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fafec093510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fafec093510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fafec093510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=71b25e36dffb716cdc0e1cbf7e42cc5e9475b1945a0a78b5c05e51d9eff7c0a4
  Stored in directory: /tmp/pip-ephem-wheel-cache-xvqpvhis/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:44:58, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:09:29, 14.8kB/s].vector_cache/glove.6B.zip:   0%|          | 172k/862M [00:01<11:22:38, 21.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 713k/862M [00:01<7:58:36, 30.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.70M/862M [00:01<5:34:28, 42.8kB/s].vector_cache/glove.6B.zip:   1%|          | 5.70M/862M [00:01<3:53:27, 61.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.1M/862M [00:01<2:42:28, 87.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<1:53:18, 125kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.8M/862M [00:01<1:19:04, 178kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:01<55:04, 254kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:01<38:32, 361kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.8M/862M [00:01<26:56, 514kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:02<18:51, 730kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.3M/862M [00:02<13:13, 1.04MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<09:18, 1.46MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:02<06:34, 2.06MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.8M/862M [00:03<05:26, 2.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<05:42, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<08:19, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<06:47, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<05:01, 2.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<07:21, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:52, 1.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<05:10, 2.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:07<03:45, 3.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<2:16:54, 97.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<1:39:13, 134kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<1:10:16, 189kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<49:19, 268kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<38:06, 347kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<28:00, 472kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:11<19:54, 662kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<16:58, 774kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<14:33, 903kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<10:44, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<07:47, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<09:03, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:15<07:41, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:15<05:42, 2.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<07:01, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:17<07:40, 1.69MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:17<05:56, 2.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<04:22, 2.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<07:18, 1.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<06:27, 2.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<04:50, 2.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<06:22, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<07:03, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<05:33, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:21<04:02, 3.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<2:30:08, 85.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<1:46:09, 121kB/s] .vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:23<1:14:29, 171kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<55:00, 231kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<41:05, 310kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:25<29:22, 433kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<20:36, 615kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:16:04, 166kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<54:33, 232kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<38:24, 329kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<29:38, 425kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<23:31, 535kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<17:01, 739kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<12:02, 1.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<14:40, 854kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<11:37, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<08:27, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:40, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:24, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<05:27, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<03:58, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:01:50, 200kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<46:01, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<32:45, 378kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<23:01, 536kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:42, 567kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<16:31, 745kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<11:52, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:00, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:18, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:51, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<05:38, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<23:08, 526kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<17:29, 696kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<12:33, 968kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<11:26, 1.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<10:35, 1.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<08:03, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<05:46, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<22:29, 535kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<17:02, 706kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<12:13, 982kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<11:12, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:09, 910kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<10:27, 1.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<07:37, 1.57MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:55, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:05, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:15, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:30, 2.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<1:05:17, 181kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<48:10, 245kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<34:12, 345kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<24:00, 490kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<25:29, 461kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<20:18, 579kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<14:45, 796kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<10:25, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<17:44, 659kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<14:37, 799kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<10:47, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<09:03, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:56, 1.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<59:06, 196kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<43:49, 264kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<31:14, 369kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<21:54, 524kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:37:31, 118kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<1:10:41, 162kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<50:01, 229kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<35:06, 326kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<31:30, 362kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<25:30, 448kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<18:41, 610kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<13:15, 858kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:12, 859kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:28, 909kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<09:26, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<09:24, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:19, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:19, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:47, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:31, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:40, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:37, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:20, 1.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:35, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:42, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:24, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:38, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:47, 1.41MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:24, 1.31MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:42, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:10, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:49, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:05, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:25, 2.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:46, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:30, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:50, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:13, 2.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:16, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<08:02, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:12, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:29, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:22, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:53, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:12, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<04:29, 2.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<07:44, 1.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:03, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:19, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:36, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:49, 1.35MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:19, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:27, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<07:30, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<07:55, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:08, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<04:27, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:48, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:59, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<07:31, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<07:56, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:28, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:41, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:53, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:03, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<07:27, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:51, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:15, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<07:42, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:08, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:21, 1.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:35, 2.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:24, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:39, 1.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<05:59, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<04:19, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:36, 1.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:52, 1.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<06:08, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:27, 2.24MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:36, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:44, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:59, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:20, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<07:26, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<07:43, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<04:22, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:27, 1.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<07:35, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:49, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:15, 2.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:36, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:24, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:59, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:38, 2.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:32, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:11, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:46, 2.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:08, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:29, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:50, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:14, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<07:13, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:29, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:44, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:12, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:23, 1.76MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:11, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:55, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:35, 2.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:53, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<07:06, 1.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:01, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:19, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:24, 1.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:44, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:09, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<07:15, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<07:19, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:08, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:04, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:47, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:36, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:22, 2.71MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<06:36, 1.38MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<06:51, 1.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:15, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:47, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<06:53, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:03, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:24, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:55, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<07:06, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<07:02, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:28, 1.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:56, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<07:28, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<07:17, 1.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:32, 1.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:59, 2.22MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:14, 1.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:37, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<05:06, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:52, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:23, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:12, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:11, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:04, 2.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<06:02, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:15, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:30, 2.47MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<07:05, 1.22MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<06:52, 1.26MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:16, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:48, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<07:14, 1.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<07:24, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:39, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:18, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<05:36, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:23, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:11, 2.65MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:18, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<07:45, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:50, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:11, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:37, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:19, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:51, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:29, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<13:30, 614kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<10:52, 762kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<07:56, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<05:48, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:58, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:47, 1.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:26, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:11, 2.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<25:52, 315kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<19:55, 409kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<14:22, 566kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<10:06, 801kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<12:37, 641kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<10:20, 782kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:32, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<05:21, 1.50MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<08:30, 942kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:24, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:32, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:57, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<31:30, 252kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<23:58, 332kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<17:09, 463kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<12:03, 655kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:59, 657kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<10:18, 764kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:36, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:25, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:27, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:54, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:23, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:10, 2.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:16, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:47, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<05:10, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:42, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<10:09, 755kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:19, 922kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<06:03, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:19, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:59, 846kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:50, 970kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<05:48, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:08, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:35, 1.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:08, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:36, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:30, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:21, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:04, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:57, 2.51MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<05:11, 1.43MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:00, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:51, 2.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<37:24, 196kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<27:43, 264kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<19:41, 371kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<13:48, 527kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<13:49, 525kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<11:12, 648kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<08:08, 889kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<05:45, 1.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<08:30, 845kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<07:27, 963kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:32, 1.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<03:56, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:05, 881kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:09, 995kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:22, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<36:42, 192kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<27:10, 260kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<19:21, 364kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<13:32, 516kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<59:17, 118kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<42:26, 165kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<29:49, 233kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<20:51, 332kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<22:22, 309kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<16:56, 408kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<12:09, 568kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:35, 714kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<07:59, 857kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:53, 1.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:13, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:52, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:42, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:42, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:48, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:57, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:10, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:24, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:41, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:58, 2.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:32, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:41, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:43, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:14, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:23, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:39, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:56, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:15, 1.98MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:31, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:50, 3.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:53, 1.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<04:36, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:27, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:28, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<10:39, 591kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<09:48, 643kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<07:21, 856kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:14, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<03:45, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<10:58, 568kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<08:48, 707kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<06:23, 972kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<04:30, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<12:31, 492kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<11:03, 557kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<08:18, 740kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<05:54, 1.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:36, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:03, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:48, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:39, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:48, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:50, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:24, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:27, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:41, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:51, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:03, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:23, 2.45MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:38, 2.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:53, 2.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:17, 2.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:38, 3.49MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<12:45, 450kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<09:58, 576kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<07:10, 799kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<05:06, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:30, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<04:55, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:41, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:30, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:29, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:41, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:48, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:59, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:18, 2.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:40, 3.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:48, 1.44MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:54, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:55, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:01, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:19, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:41, 3.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:58, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:44, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:51, 1.86MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<02:02, 2.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<29:27, 179kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<21:32, 244kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<15:17, 344kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<11:28, 453kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:57, 580kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:29, 798kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<04:33, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<10:20, 496kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<08:09, 629kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:55, 863kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:57, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:22, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:17, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:20, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<06:40, 748kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:33, 896kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<04:06, 1.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<02:53, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<1:38:16, 50.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<1:09:38, 70.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<48:49, 100kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<33:55, 143kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<29:45, 163kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<21:41, 224kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<15:20, 315kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<10:40, 448kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<35:40, 134kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<25:50, 185kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<18:15, 261kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<13:26, 351kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<10:16, 459kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<07:23, 636kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<05:09, 901kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<17:57, 259kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<13:25, 346kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<09:34, 483kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<06:40, 686kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<29:03, 158kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<21:10, 216kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<14:58, 305kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<10:26, 433kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<11:37, 388kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<08:56, 504kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<06:24, 701kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<04:31, 987kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<05:19, 835kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<04:31, 983kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:20, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:22<02:22, 1.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<07:08, 613kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:46, 757kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:12, 1.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:58, 1.45MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:06, 842kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:20, 990kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:03, 851kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:58, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:20, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:27, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:31, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:30, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:23, 2.96MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<05:46, 710kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<04:44, 863kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:27, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:28, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:21, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:04, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:19, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:16, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:58, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:22, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:46, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:57, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:04, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:35, 2.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:10, 3.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:14, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:54, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:22, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:43, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:10, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:11, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:40, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:11, 3.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<05:07, 719kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:31, 813kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:21, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:23, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:53, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:37, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:59, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:58, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:36, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:08, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:32, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:01, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:01, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:31, 2.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:07, 3.07MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:45, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:10, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:34, 2.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:51, 1.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:52, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:26, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:04, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:40, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:44, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:21, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:08, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:46, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:17, 2.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:45, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:46, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:21, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<00:58, 3.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<06:45, 454kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<05:47, 529kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<04:16, 716kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<03:01, 1.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:55, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:32, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:52, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:19, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:24, 664kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<03:35, 815kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:36, 1.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:50, 1.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:19, 860kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:49, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:04, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:49, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:45, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:19, 2.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<00:56, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<04:15, 640kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:54, 695kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:56, 923kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:05, 1.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:06, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:55, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:26, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:02, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:40, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:36, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:13, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<00:52, 2.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:26, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<02:35, 973kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:59, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:25, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:35, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:27, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:07, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:47, 2.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:57, 805kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:53, 822kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:13, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:34, 1.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:43, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:36, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:12, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:12, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:13, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:56, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:27, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:10, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:53, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:59, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:03, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:49, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:58, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:45, 2.66MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:32, 3.66MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<14:56, 131kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<10:46, 182kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<07:33, 257kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<05:29, 346kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<04:29, 421kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<03:16, 576kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<02:17, 810kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<02:00, 908kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:42, 1.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:15, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:52, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:28, 709kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:02, 862kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:29, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:18, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:27, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:08, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:49, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:00, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:58, 1.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:44, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:32, 2.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:56, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:55, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:41, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:29, 3.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:36, 411kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:59, 494kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:12, 665kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:32, 932kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:26, 975kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:15, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:56, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:51, 1.56MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:50, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:38, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:38, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:53, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:43, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:31, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:41, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:41, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:31, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:33, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:46, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:37, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:26, 2.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:34, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:40, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:33, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:23, 2.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:31, 1.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:23, 2.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:24, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:35, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:28, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:20, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:26, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:26, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:20, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:20, 2.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:20, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:14, 2.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:29, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:27, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:20, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:25, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:20, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:18, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:08, 3.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:18, 342kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:59, 450kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:40, 626kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:29, 775kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:28, 800kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:21, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:14, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:14, 1.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:12, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:08, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:05, 2.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 1.80MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.32MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<11:19:14,  9.81it/s]  0%|          | 900/400000 [00:00<7:54:37, 14.01it/s]  0%|          | 1791/400000 [00:00<5:31:42, 20.01it/s]  1%|          | 2655/400000 [00:00<3:51:55, 28.55it/s]  1%|          | 3564/400000 [00:00<2:42:11, 40.74it/s]  1%|          | 4464/400000 [00:00<1:53:29, 58.08it/s]  1%|▏         | 5347/400000 [00:00<1:19:29, 82.74it/s]  2%|▏         | 6290/400000 [00:00<55:43, 117.76it/s]   2%|▏         | 7156/400000 [00:00<39:08, 167.25it/s]  2%|▏         | 8012/400000 [00:01<27:34, 236.94it/s]  2%|▏         | 8902/400000 [00:01<19:28, 334.67it/s]  2%|▏         | 9762/400000 [00:01<13:50, 470.00it/s]  3%|▎         | 10659/400000 [00:01<09:52, 656.68it/s]  3%|▎         | 11578/400000 [00:01<07:06, 910.22it/s]  3%|▎         | 12459/400000 [00:01<05:11, 1244.43it/s]  3%|▎         | 13351/400000 [00:01<03:50, 1677.44it/s]  4%|▎         | 14233/400000 [00:01<02:54, 2214.75it/s]  4%|▍         | 15151/400000 [00:01<02:14, 2867.41it/s]  4%|▍         | 16051/400000 [00:01<01:46, 3603.99it/s]  4%|▍         | 16966/400000 [00:02<01:26, 4404.13it/s]  4%|▍         | 17883/400000 [00:02<01:13, 5217.46it/s]  5%|▍         | 18788/400000 [00:02<01:04, 5940.37it/s]  5%|▍         | 19685/400000 [00:02<00:58, 6473.46it/s]  5%|▌         | 20559/400000 [00:02<00:54, 6945.01it/s]  5%|▌         | 21440/400000 [00:02<00:51, 7414.09it/s]  6%|▌         | 22340/400000 [00:02<00:48, 7827.72it/s]  6%|▌         | 23240/400000 [00:02<00:46, 8145.43it/s]  6%|▌         | 24164/400000 [00:02<00:44, 8445.41it/s]  6%|▋         | 25061/400000 [00:02<00:44, 8509.30it/s]  6%|▋         | 25949/400000 [00:03<00:44, 8404.97it/s]  7%|▋         | 26950/400000 [00:03<00:42, 8829.63it/s]  7%|▋         | 27914/400000 [00:03<00:41, 9057.16it/s]  7%|▋         | 28838/400000 [00:03<00:41, 9028.47it/s]  7%|▋         | 29811/400000 [00:03<00:40, 9227.33it/s]  8%|▊         | 30755/400000 [00:03<00:39, 9288.13it/s]  8%|▊         | 31691/400000 [00:03<00:39, 9286.32it/s]  8%|▊         | 32625/400000 [00:03<00:39, 9248.84it/s]  8%|▊         | 33566/400000 [00:03<00:39, 9296.04it/s]  9%|▊         | 34499/400000 [00:03<00:39, 9281.13it/s]  9%|▉         | 35465/400000 [00:04<00:38, 9391.45it/s]  9%|▉         | 36406/400000 [00:04<00:39, 9277.18it/s]  9%|▉         | 37336/400000 [00:04<00:39, 9073.85it/s] 10%|▉         | 38246/400000 [00:04<00:39, 9058.81it/s] 10%|▉         | 39154/400000 [00:04<00:41, 8727.07it/s] 10%|█         | 40078/400000 [00:04<00:40, 8874.00it/s] 10%|█         | 40987/400000 [00:04<00:40, 8937.01it/s] 10%|█         | 41883/400000 [00:04<00:40, 8871.29it/s] 11%|█         | 42774/400000 [00:04<00:40, 8880.48it/s] 11%|█         | 43664/400000 [00:04<00:40, 8797.36it/s] 11%|█         | 44545/400000 [00:05<00:40, 8796.80it/s] 11%|█▏        | 45448/400000 [00:05<00:39, 8863.87it/s] 12%|█▏        | 46336/400000 [00:05<00:40, 8798.40it/s] 12%|█▏        | 47226/400000 [00:05<00:39, 8826.96it/s] 12%|█▏        | 48110/400000 [00:05<00:40, 8710.11it/s] 12%|█▏        | 48982/400000 [00:05<00:40, 8599.80it/s] 12%|█▏        | 49872/400000 [00:05<00:40, 8686.17it/s] 13%|█▎        | 50752/400000 [00:05<00:40, 8717.93it/s] 13%|█▎        | 51665/400000 [00:05<00:39, 8836.85it/s] 13%|█▎        | 52550/400000 [00:05<00:40, 8668.15it/s] 13%|█▎        | 53466/400000 [00:06<00:39, 8809.29it/s] 14%|█▎        | 54384/400000 [00:06<00:38, 8914.49it/s] 14%|█▍        | 55277/400000 [00:06<00:39, 8755.26it/s] 14%|█▍        | 56155/400000 [00:06<00:39, 8681.94it/s] 14%|█▍        | 57025/400000 [00:06<00:39, 8662.47it/s] 14%|█▍        | 57974/400000 [00:06<00:38, 8893.28it/s] 15%|█▍        | 58866/400000 [00:06<00:38, 8873.29it/s] 15%|█▍        | 59755/400000 [00:06<00:38, 8854.73it/s] 15%|█▌        | 60642/400000 [00:06<00:38, 8848.66it/s] 15%|█▌        | 61556/400000 [00:07<00:37, 8931.83it/s] 16%|█▌        | 62450/400000 [00:07<00:37, 8931.59it/s] 16%|█▌        | 63370/400000 [00:07<00:37, 9009.95it/s] 16%|█▌        | 64272/400000 [00:07<00:37, 8901.07it/s] 16%|█▋        | 65163/400000 [00:07<00:38, 8801.33it/s] 17%|█▋        | 66044/400000 [00:07<00:37, 8799.58it/s] 17%|█▋        | 66944/400000 [00:07<00:37, 8858.29it/s] 17%|█▋        | 67852/400000 [00:07<00:37, 8922.63it/s] 17%|█▋        | 68745/400000 [00:07<00:37, 8854.01it/s] 17%|█▋        | 69631/400000 [00:07<00:37, 8850.26it/s] 18%|█▊        | 70517/400000 [00:08<00:38, 8634.27it/s] 18%|█▊        | 71426/400000 [00:08<00:37, 8764.67it/s] 18%|█▊        | 72336/400000 [00:08<00:36, 8860.85it/s] 18%|█▊        | 73224/400000 [00:08<00:37, 8686.56it/s] 19%|█▊        | 74153/400000 [00:08<00:36, 8857.27it/s] 19%|█▉        | 75060/400000 [00:08<00:36, 8918.92it/s] 19%|█▉        | 75957/400000 [00:08<00:36, 8933.49it/s] 19%|█▉        | 76852/400000 [00:08<00:36, 8885.14it/s] 19%|█▉        | 77742/400000 [00:08<00:36, 8821.48it/s] 20%|█▉        | 78634/400000 [00:08<00:36, 8848.12it/s] 20%|█▉        | 79520/400000 [00:09<00:36, 8824.03it/s] 20%|██        | 80403/400000 [00:09<00:36, 8814.05it/s] 20%|██        | 81316/400000 [00:09<00:35, 8904.34it/s] 21%|██        | 82207/400000 [00:09<00:35, 8831.43it/s] 21%|██        | 83098/400000 [00:09<00:35, 8854.43it/s] 21%|██        | 83984/400000 [00:09<00:36, 8751.09it/s] 21%|██        | 84874/400000 [00:09<00:35, 8771.68it/s] 21%|██▏       | 85793/400000 [00:09<00:35, 8890.64it/s] 22%|██▏       | 86683/400000 [00:09<00:35, 8855.97it/s] 22%|██▏       | 87580/400000 [00:09<00:35, 8888.86it/s] 22%|██▏       | 88477/400000 [00:10<00:34, 8912.99it/s] 22%|██▏       | 89393/400000 [00:10<00:34, 8985.51it/s] 23%|██▎       | 90304/400000 [00:10<00:34, 9021.49it/s] 23%|██▎       | 91207/400000 [00:10<00:35, 8661.15it/s] 23%|██▎       | 92077/400000 [00:10<00:35, 8633.44it/s] 23%|██▎       | 92980/400000 [00:10<00:35, 8746.41it/s] 23%|██▎       | 93857/400000 [00:10<00:35, 8570.37it/s] 24%|██▎       | 94763/400000 [00:10<00:35, 8710.57it/s] 24%|██▍       | 95637/400000 [00:10<00:35, 8586.95it/s] 24%|██▍       | 96553/400000 [00:10<00:34, 8749.41it/s] 24%|██▍       | 97454/400000 [00:11<00:34, 8821.45it/s] 25%|██▍       | 98338/400000 [00:11<00:36, 8213.34it/s] 25%|██▍       | 99252/400000 [00:11<00:35, 8468.63it/s] 25%|██▌       | 100148/400000 [00:11<00:34, 8608.74it/s] 25%|██▌       | 101093/400000 [00:11<00:33, 8844.04it/s] 26%|██▌       | 102040/400000 [00:11<00:33, 9021.93it/s] 26%|██▌       | 102958/400000 [00:11<00:32, 9066.30it/s] 26%|██▌       | 103869/400000 [00:11<00:33, 8938.81it/s] 26%|██▌       | 104770/400000 [00:11<00:32, 8957.60it/s] 26%|██▋       | 105668/400000 [00:12<00:32, 8922.53it/s] 27%|██▋       | 106562/400000 [00:12<00:33, 8820.89it/s] 27%|██▋       | 107464/400000 [00:12<00:32, 8877.06it/s] 27%|██▋       | 108353/400000 [00:12<00:33, 8826.08it/s] 27%|██▋       | 109237/400000 [00:12<00:34, 8510.55it/s] 28%|██▊       | 110141/400000 [00:12<00:33, 8662.06it/s] 28%|██▊       | 111067/400000 [00:12<00:32, 8833.14it/s] 28%|██▊       | 111964/400000 [00:12<00:32, 8871.05it/s] 28%|██▊       | 112888/400000 [00:12<00:31, 8976.11it/s] 28%|██▊       | 113788/400000 [00:12<00:32, 8889.07it/s] 29%|██▊       | 114679/400000 [00:13<00:33, 8641.47it/s] 29%|██▉       | 115627/400000 [00:13<00:32, 8876.27it/s] 29%|██▉       | 116583/400000 [00:13<00:31, 9068.29it/s] 29%|██▉       | 117494/400000 [00:13<00:31, 9011.19it/s] 30%|██▉       | 118424/400000 [00:13<00:30, 9095.75it/s] 30%|██▉       | 119336/400000 [00:13<00:31, 8866.54it/s] 30%|███       | 120258/400000 [00:13<00:31, 8969.29it/s] 30%|███       | 121163/400000 [00:13<00:31, 8992.71it/s] 31%|███       | 122089/400000 [00:13<00:30, 9070.55it/s] 31%|███       | 122998/400000 [00:13<00:31, 8933.60it/s] 31%|███       | 123907/400000 [00:14<00:30, 8977.64it/s] 31%|███       | 124806/400000 [00:14<00:33, 8251.45it/s] 31%|███▏      | 125684/400000 [00:14<00:32, 8402.03it/s] 32%|███▏      | 126553/400000 [00:14<00:32, 8484.71it/s] 32%|███▏      | 127488/400000 [00:14<00:31, 8726.88it/s] 32%|███▏      | 128423/400000 [00:14<00:30, 8903.23it/s] 32%|███▏      | 129319/400000 [00:14<00:30, 8845.48it/s] 33%|███▎      | 130208/400000 [00:14<00:31, 8651.05it/s] 33%|███▎      | 131077/400000 [00:14<00:31, 8508.78it/s] 33%|███▎      | 131990/400000 [00:15<00:30, 8683.62it/s] 33%|███▎      | 132918/400000 [00:15<00:30, 8852.40it/s] 33%|███▎      | 133880/400000 [00:15<00:29, 9067.45it/s] 34%|███▎      | 134791/400000 [00:15<00:29, 9077.20it/s] 34%|███▍      | 135702/400000 [00:15<00:29, 8957.71it/s] 34%|███▍      | 136600/400000 [00:15<00:29, 8859.67it/s] 34%|███▍      | 137488/400000 [00:15<00:30, 8628.78it/s] 35%|███▍      | 138367/400000 [00:15<00:30, 8675.27it/s] 35%|███▍      | 139327/400000 [00:15<00:29, 8932.90it/s] 35%|███▌      | 140224/400000 [00:15<00:29, 8810.17it/s] 35%|███▌      | 141108/400000 [00:16<00:30, 8445.87it/s] 35%|███▌      | 141978/400000 [00:16<00:30, 8519.18it/s] 36%|███▌      | 142926/400000 [00:16<00:29, 8784.13it/s] 36%|███▌      | 143810/400000 [00:16<00:29, 8626.45it/s] 36%|███▌      | 144739/400000 [00:16<00:28, 8812.35it/s] 36%|███▋      | 145624/400000 [00:16<00:30, 8374.99it/s] 37%|███▋      | 146548/400000 [00:16<00:29, 8616.05it/s] 37%|███▋      | 147490/400000 [00:16<00:28, 8838.85it/s] 37%|███▋      | 148402/400000 [00:16<00:28, 8918.86it/s] 37%|███▋      | 149299/400000 [00:16<00:28, 8919.51it/s] 38%|███▊      | 150195/400000 [00:17<00:28, 8899.35it/s] 38%|███▊      | 151088/400000 [00:17<00:28, 8790.47it/s] 38%|███▊      | 152023/400000 [00:17<00:27, 8950.20it/s] 38%|███▊      | 152925/400000 [00:17<00:27, 8969.12it/s] 38%|███▊      | 153824/400000 [00:17<00:27, 8895.89it/s] 39%|███▊      | 154715/400000 [00:17<00:27, 8889.78it/s] 39%|███▉      | 155605/400000 [00:17<00:27, 8772.56it/s] 39%|███▉      | 156484/400000 [00:17<00:28, 8563.27it/s] 39%|███▉      | 157343/400000 [00:17<00:28, 8503.55it/s] 40%|███▉      | 158195/400000 [00:18<00:28, 8363.17it/s] 40%|███▉      | 159033/400000 [00:18<00:29, 8257.50it/s] 40%|███▉      | 159900/400000 [00:18<00:28, 8375.56it/s] 40%|████      | 160780/400000 [00:18<00:28, 8497.99it/s] 40%|████      | 161632/400000 [00:18<00:28, 8454.91it/s] 41%|████      | 162541/400000 [00:18<00:27, 8634.17it/s] 41%|████      | 163478/400000 [00:18<00:26, 8841.47it/s] 41%|████      | 164383/400000 [00:18<00:26, 8902.24it/s] 41%|████▏     | 165275/400000 [00:18<00:26, 8869.23it/s] 42%|████▏     | 166164/400000 [00:18<00:26, 8783.44it/s] 42%|████▏     | 167044/400000 [00:19<00:26, 8693.21it/s] 42%|████▏     | 167921/400000 [00:19<00:26, 8713.74it/s] 42%|████▏     | 168794/400000 [00:19<00:26, 8717.07it/s] 42%|████▏     | 169750/400000 [00:19<00:25, 8953.49it/s] 43%|████▎     | 170661/400000 [00:19<00:25, 8998.84it/s] 43%|████▎     | 171615/400000 [00:19<00:24, 9152.41it/s] 43%|████▎     | 172554/400000 [00:19<00:24, 9220.43it/s] 43%|████▎     | 173480/400000 [00:19<00:24, 9230.84it/s] 44%|████▎     | 174469/400000 [00:19<00:23, 9417.35it/s] 44%|████▍     | 175413/400000 [00:19<00:23, 9382.92it/s] 44%|████▍     | 176353/400000 [00:20<00:23, 9319.71it/s] 44%|████▍     | 177286/400000 [00:20<00:24, 9221.17it/s] 45%|████▍     | 178211/400000 [00:20<00:24, 9229.18it/s] 45%|████▍     | 179171/400000 [00:20<00:23, 9335.49it/s] 45%|████▌     | 180106/400000 [00:20<00:23, 9323.73it/s] 45%|████▌     | 181039/400000 [00:20<00:23, 9157.53it/s] 45%|████▌     | 181991/400000 [00:20<00:23, 9263.31it/s] 46%|████▌     | 182940/400000 [00:20<00:23, 9328.08it/s] 46%|████▌     | 183931/400000 [00:20<00:22, 9495.03it/s] 46%|████▌     | 184897/400000 [00:20<00:22, 9540.79it/s] 46%|████▋     | 185853/400000 [00:21<00:23, 9078.20it/s] 47%|████▋     | 186767/400000 [00:21<00:24, 8747.95it/s] 47%|████▋     | 187732/400000 [00:21<00:23, 8998.86it/s] 47%|████▋     | 188794/400000 [00:21<00:22, 9429.43it/s] 47%|████▋     | 189787/400000 [00:21<00:21, 9571.20it/s] 48%|████▊     | 190772/400000 [00:21<00:21, 9652.70it/s] 48%|████▊     | 191743/400000 [00:21<00:21, 9526.25it/s] 48%|████▊     | 192717/400000 [00:21<00:21, 9588.01it/s] 48%|████▊     | 193679/400000 [00:21<00:21, 9506.10it/s] 49%|████▊     | 194632/400000 [00:21<00:21, 9399.91it/s] 49%|████▉     | 195574/400000 [00:22<00:22, 9244.38it/s] 49%|████▉     | 196560/400000 [00:22<00:21, 9419.26it/s] 49%|████▉     | 197505/400000 [00:22<00:21, 9329.78it/s] 50%|████▉     | 198453/400000 [00:22<00:21, 9371.52it/s] 50%|████▉     | 199400/400000 [00:22<00:21, 9399.37it/s] 50%|█████     | 200410/400000 [00:22<00:20, 9597.92it/s] 50%|█████     | 201372/400000 [00:22<00:20, 9473.50it/s] 51%|█████     | 202350/400000 [00:22<00:20, 9560.87it/s] 51%|█████     | 203328/400000 [00:22<00:20, 9623.29it/s] 51%|█████     | 204318/400000 [00:22<00:20, 9704.33it/s] 51%|█████▏    | 205290/400000 [00:23<00:20, 9529.95it/s] 52%|█████▏    | 206245/400000 [00:23<00:20, 9447.10it/s] 52%|█████▏    | 207240/400000 [00:23<00:20, 9590.67it/s] 52%|█████▏    | 208201/400000 [00:23<00:20, 9577.29it/s] 52%|█████▏    | 209160/400000 [00:23<00:20, 9457.40it/s] 53%|█████▎    | 210119/400000 [00:23<00:19, 9494.65it/s] 53%|█████▎    | 211070/400000 [00:23<00:19, 9486.15it/s] 53%|█████▎    | 212049/400000 [00:23<00:19, 9572.14it/s] 53%|█████▎    | 213063/400000 [00:23<00:19, 9735.04it/s] 54%|█████▎    | 214097/400000 [00:24<00:18, 9907.10it/s] 54%|█████▍    | 215090/400000 [00:24<00:18, 9757.29it/s] 54%|█████▍    | 216074/400000 [00:24<00:18, 9781.35it/s] 54%|█████▍    | 217100/400000 [00:24<00:18, 9920.10it/s] 55%|█████▍    | 218094/400000 [00:24<00:18, 9845.13it/s] 55%|█████▍    | 219084/400000 [00:24<00:18, 9859.99it/s] 55%|█████▌    | 220071/400000 [00:24<00:18, 9718.26it/s] 55%|█████▌    | 221044/400000 [00:24<00:18, 9687.11it/s] 56%|█████▌    | 222014/400000 [00:24<00:18, 9390.63it/s] 56%|█████▌    | 222956/400000 [00:24<00:19, 9236.90it/s] 56%|█████▌    | 223882/400000 [00:25<00:19, 9156.27it/s] 56%|█████▌    | 224878/400000 [00:25<00:18, 9383.29it/s] 56%|█████▋    | 225844/400000 [00:25<00:18, 9462.63it/s] 57%|█████▋    | 226814/400000 [00:25<00:18, 9532.44it/s] 57%|█████▋    | 227775/400000 [00:25<00:18, 9554.26it/s] 57%|█████▋    | 228732/400000 [00:25<00:18, 9240.91it/s] 57%|█████▋    | 229660/400000 [00:25<00:18, 9181.09it/s] 58%|█████▊    | 230581/400000 [00:25<00:18, 9171.94it/s] 58%|█████▊    | 231607/400000 [00:25<00:17, 9472.04it/s] 58%|█████▊    | 232558/400000 [00:25<00:17, 9364.84it/s] 58%|█████▊    | 233498/400000 [00:26<00:17, 9363.16it/s] 59%|█████▊    | 234468/400000 [00:26<00:17, 9460.78it/s] 59%|█████▉    | 235452/400000 [00:26<00:17, 9569.52it/s] 59%|█████▉    | 236437/400000 [00:26<00:16, 9651.10it/s] 59%|█████▉    | 237404/400000 [00:26<00:17, 9415.95it/s] 60%|█████▉    | 238398/400000 [00:26<00:16, 9566.41it/s] 60%|█████▉    | 239391/400000 [00:26<00:16, 9670.30it/s] 60%|██████    | 240360/400000 [00:26<00:16, 9669.42it/s] 60%|██████    | 241329/400000 [00:26<00:16, 9654.97it/s] 61%|██████    | 242315/400000 [00:26<00:16, 9714.25it/s] 61%|██████    | 243288/400000 [00:27<00:16, 9467.00it/s] 61%|██████    | 244306/400000 [00:27<00:16, 9668.55it/s] 61%|██████▏   | 245354/400000 [00:27<00:15, 9895.91it/s] 62%|██████▏   | 246386/400000 [00:27<00:15, 10017.13it/s] 62%|██████▏   | 247391/400000 [00:27<00:15, 9984.54it/s]  62%|██████▏   | 248398/400000 [00:27<00:15, 10008.37it/s] 62%|██████▏   | 249409/400000 [00:27<00:15, 10037.59it/s] 63%|██████▎   | 250414/400000 [00:27<00:14, 9987.48it/s]  63%|██████▎   | 251414/400000 [00:27<00:14, 9954.28it/s] 63%|██████▎   | 252410/400000 [00:27<00:14, 9840.18it/s] 63%|██████▎   | 253395/400000 [00:28<00:15, 9478.84it/s] 64%|██████▎   | 254347/400000 [00:28<00:15, 9189.91it/s] 64%|██████▍   | 255271/400000 [00:28<00:15, 9124.98it/s] 64%|██████▍   | 256187/400000 [00:28<00:16, 8952.34it/s] 64%|██████▍   | 257086/400000 [00:28<00:16, 8725.20it/s] 65%|██████▍   | 258012/400000 [00:28<00:15, 8878.42it/s] 65%|██████▍   | 258903/400000 [00:28<00:16, 8770.60it/s] 65%|██████▍   | 259817/400000 [00:28<00:15, 8877.13it/s] 65%|██████▌   | 260735/400000 [00:28<00:15, 8965.53it/s] 65%|██████▌   | 261740/400000 [00:29<00:14, 9264.56it/s] 66%|██████▌   | 262671/400000 [00:29<00:14, 9270.79it/s] 66%|██████▌   | 263629/400000 [00:29<00:14, 9358.55it/s] 66%|██████▌   | 264608/400000 [00:29<00:14, 9481.71it/s] 66%|██████▋   | 265617/400000 [00:29<00:13, 9653.79it/s] 67%|██████▋   | 266616/400000 [00:29<00:13, 9751.87it/s] 67%|██████▋   | 267593/400000 [00:29<00:13, 9703.44it/s] 67%|██████▋   | 268565/400000 [00:29<00:13, 9652.73it/s] 67%|██████▋   | 269532/400000 [00:29<00:13, 9559.09it/s] 68%|██████▊   | 270515/400000 [00:29<00:13, 9638.23it/s] 68%|██████▊   | 271480/400000 [00:30<00:13, 9553.24it/s] 68%|██████▊   | 272443/400000 [00:30<00:13, 9573.21it/s] 68%|██████▊   | 273401/400000 [00:30<00:13, 9443.33it/s] 69%|██████▊   | 274374/400000 [00:30<00:13, 9527.13it/s] 69%|██████▉   | 275328/400000 [00:30<00:13, 9100.17it/s] 69%|██████▉   | 276243/400000 [00:30<00:13, 9101.53it/s] 69%|██████▉   | 277157/400000 [00:30<00:13, 8995.72it/s] 70%|██████▉   | 278110/400000 [00:30<00:13, 9148.78it/s] 70%|██████▉   | 279028/400000 [00:30<00:13, 9151.20it/s] 70%|██████▉   | 279945/400000 [00:30<00:13, 8988.26it/s] 70%|███████   | 280880/400000 [00:31<00:13, 9093.54it/s] 70%|███████   | 281859/400000 [00:31<00:12, 9290.25it/s] 71%|███████   | 282860/400000 [00:31<00:12, 9493.69it/s] 71%|███████   | 283817/400000 [00:31<00:12, 9516.06it/s] 71%|███████   | 284774/400000 [00:31<00:12, 9530.83it/s] 71%|███████▏  | 285814/400000 [00:31<00:11, 9774.47it/s] 72%|███████▏  | 286794/400000 [00:31<00:11, 9709.69it/s] 72%|███████▏  | 287767/400000 [00:31<00:11, 9611.59it/s] 72%|███████▏  | 288730/400000 [00:31<00:11, 9414.86it/s] 72%|███████▏  | 289693/400000 [00:31<00:11, 9476.20it/s] 73%|███████▎  | 290649/400000 [00:32<00:11, 9499.75it/s] 73%|███████▎  | 291601/400000 [00:32<00:11, 9402.51it/s] 73%|███████▎  | 292543/400000 [00:32<00:11, 9318.54it/s] 73%|███████▎  | 293496/400000 [00:32<00:11, 9379.46it/s] 74%|███████▎  | 294435/400000 [00:32<00:11, 9324.23it/s] 74%|███████▍  | 295368/400000 [00:32<00:11, 9266.43it/s] 74%|███████▍  | 296301/400000 [00:32<00:11, 9283.44it/s] 74%|███████▍  | 297230/400000 [00:32<00:11, 9141.12it/s] 75%|███████▍  | 298145/400000 [00:32<00:11, 9054.63it/s] 75%|███████▍  | 299052/400000 [00:33<00:11, 9033.39it/s] 75%|███████▍  | 299956/400000 [00:33<00:11, 8794.07it/s] 75%|███████▌  | 300838/400000 [00:33<00:11, 8724.22it/s] 75%|███████▌  | 301719/400000 [00:33<00:11, 8748.10it/s] 76%|███████▌  | 302595/400000 [00:33<00:11, 8743.63it/s] 76%|███████▌  | 303500/400000 [00:33<00:10, 8833.08it/s] 76%|███████▌  | 304419/400000 [00:33<00:10, 8935.11it/s] 76%|███████▋  | 305314/400000 [00:33<00:10, 8839.85it/s] 77%|███████▋  | 306237/400000 [00:33<00:10, 8951.49it/s] 77%|███████▋  | 307181/400000 [00:33<00:10, 9091.57it/s] 77%|███████▋  | 308092/400000 [00:34<00:10, 8839.66it/s] 77%|███████▋  | 308979/400000 [00:34<00:10, 8773.09it/s] 77%|███████▋  | 309862/400000 [00:34<00:10, 8787.63it/s] 78%|███████▊  | 310810/400000 [00:34<00:09, 8982.44it/s] 78%|███████▊  | 311776/400000 [00:34<00:09, 9173.86it/s] 78%|███████▊  | 312696/400000 [00:34<00:09, 9169.68it/s] 78%|███████▊  | 313649/400000 [00:34<00:09, 9274.77it/s] 79%|███████▊  | 314578/400000 [00:34<00:09, 9041.63it/s] 79%|███████▉  | 315503/400000 [00:34<00:09, 9100.85it/s] 79%|███████▉  | 316415/400000 [00:34<00:09, 9063.97it/s] 79%|███████▉  | 317323/400000 [00:35<00:09, 8866.67it/s] 80%|███████▉  | 318213/400000 [00:35<00:09, 8874.78it/s] 80%|███████▉  | 319169/400000 [00:35<00:08, 9068.68it/s] 80%|████████  | 320114/400000 [00:35<00:08, 9178.88it/s] 80%|████████  | 321100/400000 [00:35<00:08, 9372.30it/s] 81%|████████  | 322040/400000 [00:35<00:08, 9351.94it/s] 81%|████████  | 323039/400000 [00:35<00:08, 9532.25it/s] 81%|████████  | 324027/400000 [00:35<00:07, 9632.50it/s] 81%|████████▏ | 325040/400000 [00:35<00:07, 9776.52it/s] 82%|████████▏ | 326020/400000 [00:35<00:07, 9779.70it/s] 82%|████████▏ | 327000/400000 [00:36<00:07, 9708.29it/s] 82%|████████▏ | 327988/400000 [00:36<00:07, 9758.55it/s] 82%|████████▏ | 328965/400000 [00:36<00:07, 9696.93it/s] 83%|████████▎ | 330005/400000 [00:36<00:07, 9897.22it/s] 83%|████████▎ | 330997/400000 [00:36<00:07, 9375.14it/s] 83%|████████▎ | 331942/400000 [00:36<00:07, 9385.55it/s] 83%|████████▎ | 332889/400000 [00:36<00:07, 9410.54it/s] 83%|████████▎ | 333834/400000 [00:36<00:07, 9298.34it/s] 84%|████████▎ | 334801/400000 [00:36<00:06, 9406.29it/s] 84%|████████▍ | 335750/400000 [00:36<00:06, 9429.28it/s] 84%|████████▍ | 336695/400000 [00:37<00:06, 9295.37it/s] 84%|████████▍ | 337704/400000 [00:37<00:06, 9519.05it/s] 85%|████████▍ | 338665/400000 [00:37<00:06, 9544.44it/s] 85%|████████▍ | 339642/400000 [00:37<00:06, 9610.50it/s] 85%|████████▌ | 340627/400000 [00:37<00:06, 9680.46it/s] 85%|████████▌ | 341604/400000 [00:37<00:06, 9705.39it/s] 86%|████████▌ | 342668/400000 [00:37<00:05, 9965.18it/s] 86%|████████▌ | 343667/400000 [00:37<00:05, 9744.31it/s] 86%|████████▌ | 344645/400000 [00:37<00:05, 9583.23it/s] 86%|████████▋ | 345606/400000 [00:38<00:05, 9266.11it/s] 87%|████████▋ | 346537/400000 [00:38<00:06, 8908.85it/s] 87%|████████▋ | 347469/400000 [00:38<00:05, 9026.28it/s] 87%|████████▋ | 348379/400000 [00:38<00:05, 9045.96it/s] 87%|████████▋ | 349287/400000 [00:38<00:05, 9043.54it/s] 88%|████████▊ | 350194/400000 [00:38<00:05, 8929.79it/s] 88%|████████▊ | 351204/400000 [00:38<00:05, 9250.29it/s] 88%|████████▊ | 352134/400000 [00:38<00:05, 9261.38it/s] 88%|████████▊ | 353064/400000 [00:38<00:05, 9261.91it/s] 88%|████████▊ | 353998/400000 [00:38<00:04, 9284.68it/s] 89%|████████▉ | 355011/400000 [00:39<00:04, 9517.33it/s] 89%|████████▉ | 355966/400000 [00:39<00:04, 9171.84it/s] 89%|████████▉ | 356888/400000 [00:39<00:04, 9032.86it/s] 89%|████████▉ | 357826/400000 [00:39<00:04, 9133.16it/s] 90%|████████▉ | 358743/400000 [00:39<00:04, 8983.53it/s] 90%|████████▉ | 359644/400000 [00:39<00:04, 8413.91it/s] 90%|█████████ | 360495/400000 [00:39<00:05, 7567.52it/s] 90%|█████████ | 361442/400000 [00:39<00:04, 8052.05it/s] 91%|█████████ | 362389/400000 [00:39<00:04, 8429.58it/s] 91%|█████████ | 363254/400000 [00:40<00:04, 8338.81it/s] 91%|█████████ | 364160/400000 [00:40<00:04, 8540.58it/s] 91%|█████████▏| 365166/400000 [00:40<00:03, 8944.80it/s] 92%|█████████▏| 366133/400000 [00:40<00:03, 9148.65it/s] 92%|█████████▏| 367100/400000 [00:40<00:03, 9298.93it/s] 92%|█████████▏| 368038/400000 [00:40<00:03, 9137.60it/s] 92%|█████████▏| 368958/400000 [00:40<00:03, 8872.05it/s] 92%|█████████▏| 369852/400000 [00:40<00:03, 8271.14it/s] 93%|█████████▎| 370794/400000 [00:40<00:03, 8583.30it/s] 93%|█████████▎| 371683/400000 [00:40<00:03, 8672.51it/s] 93%|█████████▎| 372559/400000 [00:41<00:03, 8616.14it/s] 93%|█████████▎| 373477/400000 [00:41<00:03, 8775.79it/s] 94%|█████████▎| 374360/400000 [00:41<00:02, 8670.97it/s] 94%|█████████▍| 375314/400000 [00:41<00:02, 8912.73it/s] 94%|█████████▍| 376224/400000 [00:41<00:02, 8966.13it/s] 94%|█████████▍| 377136/400000 [00:41<00:02, 9008.97it/s] 95%|█████████▍| 378068/400000 [00:41<00:02, 9099.67it/s] 95%|█████████▍| 379003/400000 [00:41<00:02, 9172.02it/s] 95%|█████████▍| 379928/400000 [00:41<00:02, 9192.90it/s] 95%|█████████▌| 380914/400000 [00:41<00:02, 9381.01it/s] 95%|█████████▌| 381854/400000 [00:42<00:01, 9380.94it/s] 96%|█████████▌| 382825/400000 [00:42<00:01, 9476.60it/s] 96%|█████████▌| 383774/400000 [00:42<00:01, 9382.03it/s] 96%|█████████▌| 384714/400000 [00:42<00:01, 9080.90it/s] 96%|█████████▋| 385625/400000 [00:42<00:01, 9045.48it/s] 97%|█████████▋| 386603/400000 [00:42<00:01, 9253.51it/s] 97%|█████████▋| 387572/400000 [00:42<00:01, 9378.24it/s] 97%|█████████▋| 388528/400000 [00:42<00:01, 9430.25it/s] 97%|█████████▋| 389502/400000 [00:42<00:01, 9520.03it/s] 98%|█████████▊| 390456/400000 [00:43<00:01, 9489.54it/s] 98%|█████████▊| 391406/400000 [00:43<00:00, 9371.04it/s] 98%|█████████▊| 392345/400000 [00:43<00:00, 9289.14it/s] 98%|█████████▊| 393275/400000 [00:43<00:00, 9252.66it/s] 99%|█████████▊| 394261/400000 [00:43<00:00, 9426.62it/s] 99%|█████████▉| 395205/400000 [00:43<00:00, 9322.99it/s] 99%|█████████▉| 396139/400000 [00:43<00:00, 9245.01it/s] 99%|█████████▉| 397083/400000 [00:43<00:00, 9300.85it/s]100%|█████████▉| 398030/400000 [00:43<00:00, 9348.59it/s]100%|█████████▉| 398966/400000 [00:43<00:00, 9313.19it/s]100%|█████████▉| 399898/400000 [00:44<00:00, 9289.80it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 9084.28it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fafec0b1b38>, <torchtext.data.dataset.TabularDataset object at 0x7fafec0b1c88>, <torchtext.vocab.Vocab object at 0x7fafec0b1ba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.47 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.81 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.20 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.63 file/s]2020-07-12 00:18:00.825258: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-12 00:18:00.828753: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-12 00:18:00.829455: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f1d32708e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 00:18:00.829469: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:22, 120446.38it/s] 74%|███████▍  | 7331840/9912422 [00:00<00:15, 171942.29it/s]9920512it [00:00, 38667581.55it/s]                           
0it [00:00, ?it/s]32768it [00:00, 558268.29it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457706.72it/s]1654784it [00:00, 11242258.98it/s]                         
0it [00:00, ?it/s]8192it [00:00, 175371.89it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
