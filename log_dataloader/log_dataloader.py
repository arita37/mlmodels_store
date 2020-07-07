
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f52f83560d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f52f83560d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f53636a01e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f53636a01e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f53819e8e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f53819e8e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f53109ce620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f53109ce620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f53109ce620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 131613.23it/s] 81%|████████  | 8028160/9912422 [00:00<00:10, 187886.56it/s]9920512it [00:00, 41376486.66it/s]                           
0it [00:00, ?it/s]32768it [00:00, 677059.19it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468121.64it/s]1654784it [00:00, 12043914.68it/s]                         
0it [00:00, ?it/s]8192it [00:00, 237979.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f530e009908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52f7a00ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f53109ce268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f53109ce268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f53109ce268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:39,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:38,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:37,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:37,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:36,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:36,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:35,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:34,  1.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:06,  2.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:03,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:03,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:43,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:43,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:43,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:42,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:42,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:42,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:41,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:28,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:27,  4.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:19,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:18,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:18,  6.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:13,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:12,  8.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 27.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 27.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 56.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.23 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.06s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 79.23 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.06s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.06s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.99 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.07s/ url]
0 examples [00:00, ? examples/s]2020-07-07 06:10:30.985368: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-07 06:10:30.999011: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-07 06:10:30.999197: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f1a514f6a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 06:10:30.999217: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.70 examples/s]87 examples [00:00, 13.79 examples/s]179 examples [00:00, 19.58 examples/s]268 examples [00:00, 27.70 examples/s]358 examples [00:00, 39.05 examples/s]444 examples [00:00, 54.72 examples/s]533 examples [00:00, 76.16 examples/s]618 examples [00:00, 104.75 examples/s]706 examples [00:00, 142.32 examples/s]792 examples [00:01, 189.71 examples/s]878 examples [00:01, 247.52 examples/s]965 examples [00:01, 314.83 examples/s]1051 examples [00:01, 388.20 examples/s]1136 examples [00:01, 453.18 examples/s]1218 examples [00:01, 523.13 examples/s]1308 examples [00:01, 597.31 examples/s]1400 examples [00:01, 666.16 examples/s]1487 examples [00:01, 697.32 examples/s]1571 examples [00:01, 717.63 examples/s]1653 examples [00:02, 745.04 examples/s]1739 examples [00:02, 774.99 examples/s]1823 examples [00:02, 792.83 examples/s]1912 examples [00:02, 817.89 examples/s]1998 examples [00:02, 828.20 examples/s]2087 examples [00:02, 843.49 examples/s]2175 examples [00:02, 854.08 examples/s]2262 examples [00:02, 839.13 examples/s]2351 examples [00:02, 852.94 examples/s]2439 examples [00:02, 860.24 examples/s]2528 examples [00:03, 868.18 examples/s]2616 examples [00:03, 844.02 examples/s]2706 examples [00:03, 857.92 examples/s]2796 examples [00:03, 867.88 examples/s]2884 examples [00:03, 867.63 examples/s]2976 examples [00:03, 881.27 examples/s]3068 examples [00:03, 890.33 examples/s]3158 examples [00:03, 879.14 examples/s]3247 examples [00:03, 878.74 examples/s]3335 examples [00:03, 877.03 examples/s]3423 examples [00:04, 875.45 examples/s]3511 examples [00:04, 871.53 examples/s]3599 examples [00:04, 854.92 examples/s]3685 examples [00:04, 834.29 examples/s]3769 examples [00:04, 810.70 examples/s]3854 examples [00:04, 820.08 examples/s]3942 examples [00:04, 836.33 examples/s]4032 examples [00:04, 854.15 examples/s]4126 examples [00:04, 877.71 examples/s]4223 examples [00:05, 902.34 examples/s]4314 examples [00:05, 902.08 examples/s]4409 examples [00:05, 915.83 examples/s]4501 examples [00:05, 908.45 examples/s]4593 examples [00:05, 891.27 examples/s]4683 examples [00:05, 865.39 examples/s]4770 examples [00:05, 850.72 examples/s]4861 examples [00:05, 866.66 examples/s]4950 examples [00:05, 871.05 examples/s]5038 examples [00:05, 872.96 examples/s]5130 examples [00:06, 885.37 examples/s]5219 examples [00:06, 873.90 examples/s]5307 examples [00:06, 875.55 examples/s]5395 examples [00:06, 842.58 examples/s]5480 examples [00:06, 806.48 examples/s]5568 examples [00:06, 824.82 examples/s]5657 examples [00:06, 841.20 examples/s]5745 examples [00:06, 849.61 examples/s]5831 examples [00:06, 818.80 examples/s]5915 examples [00:07, 824.42 examples/s]5998 examples [00:07, 824.51 examples/s]6084 examples [00:07, 834.34 examples/s]6170 examples [00:07, 841.80 examples/s]6255 examples [00:07, 843.76 examples/s]6340 examples [00:07, 822.72 examples/s]6427 examples [00:07, 834.35 examples/s]6514 examples [00:07, 843.71 examples/s]6599 examples [00:07, 841.84 examples/s]6686 examples [00:07, 849.51 examples/s]6772 examples [00:08, 842.61 examples/s]6860 examples [00:08, 852.70 examples/s]6947 examples [00:08, 855.64 examples/s]7037 examples [00:08, 866.50 examples/s]7124 examples [00:08, 850.02 examples/s]7210 examples [00:08, 817.35 examples/s]7293 examples [00:08, 817.88 examples/s]7376 examples [00:08, 819.32 examples/s]7465 examples [00:08, 836.93 examples/s]7552 examples [00:08, 845.90 examples/s]7638 examples [00:09, 848.24 examples/s]7723 examples [00:09, 847.45 examples/s]7812 examples [00:09, 858.89 examples/s]7900 examples [00:09, 863.81 examples/s]7988 examples [00:09, 867.08 examples/s]8075 examples [00:09, 864.29 examples/s]8166 examples [00:09, 877.47 examples/s]8255 examples [00:09, 878.36 examples/s]8343 examples [00:09, 834.61 examples/s]8427 examples [00:09, 818.20 examples/s]8510 examples [00:10, 800.75 examples/s]8594 examples [00:10, 811.34 examples/s]8678 examples [00:10, 818.81 examples/s]8766 examples [00:10, 835.09 examples/s]8850 examples [00:10, 833.88 examples/s]8934 examples [00:10, 796.25 examples/s]9022 examples [00:10, 818.45 examples/s]9108 examples [00:10, 829.65 examples/s]9195 examples [00:10, 838.95 examples/s]9281 examples [00:11, 844.05 examples/s]9366 examples [00:11, 840.07 examples/s]9458 examples [00:11, 861.82 examples/s]9550 examples [00:11, 878.18 examples/s]9639 examples [00:11, 873.93 examples/s]9727 examples [00:11, 857.30 examples/s]9814 examples [00:11, 860.34 examples/s]9903 examples [00:11, 867.41 examples/s]9990 examples [00:11, 862.66 examples/s]10077 examples [00:11, 785.02 examples/s]10162 examples [00:12, 803.28 examples/s]10248 examples [00:12, 818.56 examples/s]10336 examples [00:12, 833.67 examples/s]10426 examples [00:12, 850.84 examples/s]10516 examples [00:12, 863.26 examples/s]10607 examples [00:12, 874.92 examples/s]10695 examples [00:12, 869.65 examples/s]10783 examples [00:12, 851.48 examples/s]10870 examples [00:12, 855.88 examples/s]10956 examples [00:12, 853.66 examples/s]11042 examples [00:13, 842.17 examples/s]11128 examples [00:13, 845.50 examples/s]11214 examples [00:13, 847.87 examples/s]11301 examples [00:13, 852.94 examples/s]11388 examples [00:13, 855.02 examples/s]11480 examples [00:13, 871.31 examples/s]11568 examples [00:13, 830.27 examples/s]11661 examples [00:13, 856.60 examples/s]11749 examples [00:13, 863.23 examples/s]11841 examples [00:13, 877.92 examples/s]11932 examples [00:14, 886.85 examples/s]12021 examples [00:14, 877.45 examples/s]12115 examples [00:14, 893.71 examples/s]12205 examples [00:14, 895.17 examples/s]12298 examples [00:14, 904.58 examples/s]12389 examples [00:14, 898.92 examples/s]12479 examples [00:14, 898.29 examples/s]12569 examples [00:14, 890.12 examples/s]12659 examples [00:14, 866.51 examples/s]12746 examples [00:15, 862.85 examples/s]12835 examples [00:15, 868.85 examples/s]12922 examples [00:15, 866.07 examples/s]13013 examples [00:15, 877.92 examples/s]13103 examples [00:15, 883.63 examples/s]13193 examples [00:15, 887.39 examples/s]13285 examples [00:15, 895.36 examples/s]13375 examples [00:15, 893.63 examples/s]13467 examples [00:15, 899.53 examples/s]13559 examples [00:15, 903.79 examples/s]13650 examples [00:16, 879.00 examples/s]13739 examples [00:16, 874.15 examples/s]13829 examples [00:16, 880.75 examples/s]13918 examples [00:16, 863.30 examples/s]14006 examples [00:16, 865.97 examples/s]14095 examples [00:16, 871.49 examples/s]14183 examples [00:16, 873.43 examples/s]14271 examples [00:16, 854.95 examples/s]14365 examples [00:16, 877.39 examples/s]14453 examples [00:16, 872.09 examples/s]14543 examples [00:17, 879.03 examples/s]14632 examples [00:17, 876.95 examples/s]14720 examples [00:17, 869.02 examples/s]14809 examples [00:17, 874.54 examples/s]14897 examples [00:17, 865.30 examples/s]14984 examples [00:17, 833.29 examples/s]15068 examples [00:17, 824.36 examples/s]15156 examples [00:17, 838.10 examples/s]15247 examples [00:17, 857.45 examples/s]15337 examples [00:17, 868.19 examples/s]15425 examples [00:18, 868.28 examples/s]15513 examples [00:18, 870.58 examples/s]15601 examples [00:18, 868.20 examples/s]15688 examples [00:18, 866.28 examples/s]15775 examples [00:18, 862.47 examples/s]15864 examples [00:18, 868.36 examples/s]15951 examples [00:18, 855.15 examples/s]16038 examples [00:18, 857.62 examples/s]16124 examples [00:18, 855.15 examples/s]16214 examples [00:19, 867.40 examples/s]16301 examples [00:19, 865.25 examples/s]16388 examples [00:19, 847.63 examples/s]16478 examples [00:19, 860.29 examples/s]16565 examples [00:19, 839.72 examples/s]16656 examples [00:19, 859.42 examples/s]16746 examples [00:19, 871.09 examples/s]16834 examples [00:19, 872.35 examples/s]16922 examples [00:19, 861.39 examples/s]17012 examples [00:19, 870.07 examples/s]17104 examples [00:20, 881.73 examples/s]17193 examples [00:20, 864.56 examples/s]17280 examples [00:20, 838.49 examples/s]17369 examples [00:20, 853.18 examples/s]17461 examples [00:20, 870.79 examples/s]17549 examples [00:20, 858.64 examples/s]17637 examples [00:20, 863.25 examples/s]17724 examples [00:20, 847.56 examples/s]17812 examples [00:20, 855.70 examples/s]17900 examples [00:20, 862.24 examples/s]17987 examples [00:21, 845.97 examples/s]18075 examples [00:21, 853.56 examples/s]18161 examples [00:21, 839.29 examples/s]18249 examples [00:21, 848.06 examples/s]18340 examples [00:21, 863.98 examples/s]18434 examples [00:21, 883.39 examples/s]18527 examples [00:21, 895.01 examples/s]18617 examples [00:21, 890.32 examples/s]18707 examples [00:21, 861.09 examples/s]18794 examples [00:22, 836.95 examples/s]18886 examples [00:22, 860.03 examples/s]18976 examples [00:22, 870.10 examples/s]19064 examples [00:22, 872.45 examples/s]19157 examples [00:22, 888.32 examples/s]19247 examples [00:22, 887.32 examples/s]19337 examples [00:22, 890.74 examples/s]19429 examples [00:22, 898.25 examples/s]19519 examples [00:22, 879.83 examples/s]19611 examples [00:22, 890.76 examples/s]19708 examples [00:23, 912.70 examples/s]19804 examples [00:23, 924.46 examples/s]19903 examples [00:23, 943.09 examples/s]19998 examples [00:23, 929.24 examples/s]20092 examples [00:23, 865.26 examples/s]20180 examples [00:23, 861.88 examples/s]20271 examples [00:23, 873.25 examples/s]20362 examples [00:23, 883.53 examples/s]20451 examples [00:23, 881.15 examples/s]20540 examples [00:23, 865.79 examples/s]20627 examples [00:24, 858.05 examples/s]20714 examples [00:24, 839.55 examples/s]20799 examples [00:24, 838.88 examples/s]20884 examples [00:24, 835.08 examples/s]20968 examples [00:24, 833.62 examples/s]21054 examples [00:24, 841.16 examples/s]21139 examples [00:24, 837.10 examples/s]21223 examples [00:24, 837.82 examples/s]21309 examples [00:24, 842.95 examples/s]21396 examples [00:24, 850.53 examples/s]21483 examples [00:25, 856.15 examples/s]21571 examples [00:25, 861.53 examples/s]21658 examples [00:25, 862.09 examples/s]21746 examples [00:25, 865.71 examples/s]21833 examples [00:25, 865.69 examples/s]21921 examples [00:25, 868.47 examples/s]22010 examples [00:25, 871.93 examples/s]22098 examples [00:25, 869.77 examples/s]22185 examples [00:25, 860.16 examples/s]22272 examples [00:26, 845.65 examples/s]22357 examples [00:26, 839.83 examples/s]22444 examples [00:26, 848.24 examples/s]22534 examples [00:26, 861.56 examples/s]22622 examples [00:26, 866.84 examples/s]22710 examples [00:26, 869.08 examples/s]22797 examples [00:26, 824.42 examples/s]22885 examples [00:26, 838.71 examples/s]22975 examples [00:26, 854.45 examples/s]23064 examples [00:26, 861.62 examples/s]23151 examples [00:27, 863.43 examples/s]23240 examples [00:27, 868.61 examples/s]23328 examples [00:27, 871.44 examples/s]23416 examples [00:27, 849.90 examples/s]23502 examples [00:27, 846.64 examples/s]23587 examples [00:27, 846.88 examples/s]23678 examples [00:27, 864.44 examples/s]23765 examples [00:27, 862.41 examples/s]23852 examples [00:27, 860.87 examples/s]23940 examples [00:27, 864.77 examples/s]24027 examples [00:28, 852.12 examples/s]24117 examples [00:28, 863.93 examples/s]24208 examples [00:28, 874.94 examples/s]24296 examples [00:28, 874.21 examples/s]24384 examples [00:28, 857.50 examples/s]24470 examples [00:28, 840.79 examples/s]24562 examples [00:28, 861.61 examples/s]24649 examples [00:28, 860.78 examples/s]24736 examples [00:28, 857.93 examples/s]24822 examples [00:28, 853.70 examples/s]24910 examples [00:29, 861.21 examples/s]24997 examples [00:29, 860.52 examples/s]25084 examples [00:29, 850.00 examples/s]25171 examples [00:29, 855.86 examples/s]25257 examples [00:29, 832.13 examples/s]25341 examples [00:29, 818.02 examples/s]25423 examples [00:29, 812.24 examples/s]25505 examples [00:29, 796.11 examples/s]25585 examples [00:29, 797.10 examples/s]25673 examples [00:30, 817.73 examples/s]25760 examples [00:30, 831.31 examples/s]25850 examples [00:30, 848.12 examples/s]25936 examples [00:30, 845.03 examples/s]26023 examples [00:30, 850.92 examples/s]26109 examples [00:30, 839.37 examples/s]26198 examples [00:30, 852.55 examples/s]26286 examples [00:30, 860.54 examples/s]26373 examples [00:30, 862.97 examples/s]26461 examples [00:30, 867.92 examples/s]26551 examples [00:31, 876.14 examples/s]26639 examples [00:31, 871.33 examples/s]26730 examples [00:31, 880.62 examples/s]26824 examples [00:31, 896.27 examples/s]26914 examples [00:31, 879.43 examples/s]27010 examples [00:31, 901.26 examples/s]27104 examples [00:31, 911.76 examples/s]27196 examples [00:31, 905.43 examples/s]27287 examples [00:31, 902.94 examples/s]27378 examples [00:31, 884.91 examples/s]27467 examples [00:32, 845.58 examples/s]27556 examples [00:32, 857.79 examples/s]27646 examples [00:32, 869.36 examples/s]27736 examples [00:32, 876.79 examples/s]27824 examples [00:32, 863.63 examples/s]27911 examples [00:32, 850.15 examples/s]27997 examples [00:32, 843.06 examples/s]28082 examples [00:32, 830.83 examples/s]28166 examples [00:32, 817.48 examples/s]28251 examples [00:32, 823.82 examples/s]28337 examples [00:33, 833.72 examples/s]28424 examples [00:33, 841.80 examples/s]28509 examples [00:33, 841.95 examples/s]28595 examples [00:33, 845.98 examples/s]28680 examples [00:33, 838.61 examples/s]28766 examples [00:33, 842.87 examples/s]28851 examples [00:33, 844.62 examples/s]28938 examples [00:33, 850.71 examples/s]29025 examples [00:33, 855.69 examples/s]29113 examples [00:34, 859.92 examples/s]29201 examples [00:34, 864.19 examples/s]29291 examples [00:34, 873.96 examples/s]29381 examples [00:34, 881.10 examples/s]29470 examples [00:34, 879.32 examples/s]29558 examples [00:34, 861.55 examples/s]29645 examples [00:34, 860.98 examples/s]29732 examples [00:34, 862.84 examples/s]29821 examples [00:34, 869.07 examples/s]29912 examples [00:34, 880.93 examples/s]30001 examples [00:35, 820.24 examples/s]30084 examples [00:35, 792.84 examples/s]30174 examples [00:35, 821.87 examples/s]30263 examples [00:35, 840.55 examples/s]30354 examples [00:35, 858.15 examples/s]30441 examples [00:35, 844.32 examples/s]30526 examples [00:35, 792.37 examples/s]30614 examples [00:35, 814.79 examples/s]30702 examples [00:35, 832.46 examples/s]30790 examples [00:35, 844.31 examples/s]30875 examples [00:36, 845.11 examples/s]30960 examples [00:36, 829.92 examples/s]31048 examples [00:36, 842.69 examples/s]31139 examples [00:36, 861.74 examples/s]31230 examples [00:36, 873.74 examples/s]31319 examples [00:36, 876.99 examples/s]31411 examples [00:36, 887.15 examples/s]31500 examples [00:36, 874.86 examples/s]31588 examples [00:36, 874.48 examples/s]31676 examples [00:37, 866.32 examples/s]31763 examples [00:37, 861.67 examples/s]31851 examples [00:37, 866.43 examples/s]31939 examples [00:37, 869.64 examples/s]32030 examples [00:37, 878.77 examples/s]32118 examples [00:37, 857.31 examples/s]32205 examples [00:37, 859.12 examples/s]32292 examples [00:37, 837.32 examples/s]32376 examples [00:37, 805.86 examples/s]32460 examples [00:37, 815.02 examples/s]32542 examples [00:38, 790.62 examples/s]32630 examples [00:38, 812.89 examples/s]32716 examples [00:38, 824.24 examples/s]32803 examples [00:38, 836.42 examples/s]32893 examples [00:38, 852.43 examples/s]32979 examples [00:38, 848.21 examples/s]33065 examples [00:38, 850.68 examples/s]33156 examples [00:38, 866.90 examples/s]33243 examples [00:38, 867.31 examples/s]33330 examples [00:38, 856.54 examples/s]33416 examples [00:39, 852.55 examples/s]33503 examples [00:39, 854.79 examples/s]33589 examples [00:39, 848.58 examples/s]33674 examples [00:39, 847.31 examples/s]33760 examples [00:39, 849.59 examples/s]33845 examples [00:39, 836.12 examples/s]33931 examples [00:39, 842.33 examples/s]34023 examples [00:39, 863.49 examples/s]34112 examples [00:39, 870.60 examples/s]34200 examples [00:39, 872.54 examples/s]34292 examples [00:40, 883.79 examples/s]34383 examples [00:40, 891.15 examples/s]34473 examples [00:40, 887.56 examples/s]34567 examples [00:40, 902.12 examples/s]34660 examples [00:40, 908.42 examples/s]34751 examples [00:40, 894.75 examples/s]34841 examples [00:40, 893.78 examples/s]34931 examples [00:40, 879.66 examples/s]35020 examples [00:40, 873.26 examples/s]35108 examples [00:41, 858.05 examples/s]35194 examples [00:41, 857.65 examples/s]35280 examples [00:41, 836.80 examples/s]35365 examples [00:41, 838.81 examples/s]35454 examples [00:41, 851.95 examples/s]35545 examples [00:41, 868.23 examples/s]35632 examples [00:41, 853.77 examples/s]35718 examples [00:41, 833.49 examples/s]35805 examples [00:41, 843.77 examples/s]35895 examples [00:41, 858.12 examples/s]35982 examples [00:42, 856.48 examples/s]36068 examples [00:42, 856.45 examples/s]36160 examples [00:42, 874.01 examples/s]36249 examples [00:42, 877.24 examples/s]36338 examples [00:42, 879.90 examples/s]36427 examples [00:42, 864.35 examples/s]36514 examples [00:42, 856.78 examples/s]36600 examples [00:42, 857.23 examples/s]36688 examples [00:42, 861.35 examples/s]36778 examples [00:42, 871.78 examples/s]36868 examples [00:43, 877.86 examples/s]36956 examples [00:43, 872.68 examples/s]37044 examples [00:43, 870.42 examples/s]37134 examples [00:43, 879.06 examples/s]37224 examples [00:43, 884.90 examples/s]37313 examples [00:43, 880.08 examples/s]37402 examples [00:43, 863.04 examples/s]37489 examples [00:43, 857.16 examples/s]37578 examples [00:43, 863.75 examples/s]37667 examples [00:43, 870.73 examples/s]37758 examples [00:44, 881.83 examples/s]37847 examples [00:44, 875.06 examples/s]37935 examples [00:44, 872.40 examples/s]38023 examples [00:44, 827.50 examples/s]38112 examples [00:44, 842.92 examples/s]38204 examples [00:44, 863.71 examples/s]38291 examples [00:44, 818.85 examples/s]38377 examples [00:44, 828.11 examples/s]38465 examples [00:44, 839.98 examples/s]38550 examples [00:45, 839.83 examples/s]38640 examples [00:45, 854.85 examples/s]38726 examples [00:45, 841.74 examples/s]38811 examples [00:45, 843.19 examples/s]38900 examples [00:45, 856.34 examples/s]38986 examples [00:45, 852.94 examples/s]39076 examples [00:45, 866.19 examples/s]39165 examples [00:45, 869.30 examples/s]39255 examples [00:45, 876.20 examples/s]39346 examples [00:45, 885.21 examples/s]39435 examples [00:46, 882.31 examples/s]39524 examples [00:46, 870.67 examples/s]39612 examples [00:46, 865.63 examples/s]39700 examples [00:46, 867.69 examples/s]39788 examples [00:46, 870.58 examples/s]39876 examples [00:46, 853.92 examples/s]39962 examples [00:46, 837.48 examples/s]40046 examples [00:46, 775.80 examples/s]40127 examples [00:46, 784.95 examples/s]40207 examples [00:46, 780.23 examples/s]40290 examples [00:47, 794.12 examples/s]40374 examples [00:47, 805.46 examples/s]40461 examples [00:47, 823.17 examples/s]40545 examples [00:47, 826.06 examples/s]40630 examples [00:47, 830.39 examples/s]40717 examples [00:47, 841.56 examples/s]40802 examples [00:47, 839.47 examples/s]40891 examples [00:47, 851.48 examples/s]40977 examples [00:47, 852.98 examples/s]41065 examples [00:47, 858.91 examples/s]41151 examples [00:48, 850.61 examples/s]41237 examples [00:48, 852.29 examples/s]41325 examples [00:48, 859.77 examples/s]41418 examples [00:48, 877.21 examples/s]41507 examples [00:48, 878.48 examples/s]41600 examples [00:48, 891.52 examples/s]41694 examples [00:48, 902.74 examples/s]41785 examples [00:48, 898.24 examples/s]41879 examples [00:48, 908.40 examples/s]41973 examples [00:49, 915.27 examples/s]42065 examples [00:49, 902.56 examples/s]42160 examples [00:49, 913.37 examples/s]42252 examples [00:49, 904.21 examples/s]42344 examples [00:49, 907.13 examples/s]42438 examples [00:49, 915.71 examples/s]42530 examples [00:49, 890.24 examples/s]42620 examples [00:49, 891.94 examples/s]42711 examples [00:49, 896.92 examples/s]42801 examples [00:49, 891.50 examples/s]42891 examples [00:50, 878.20 examples/s]42982 examples [00:50, 884.81 examples/s]43071 examples [00:50, 870.72 examples/s]43163 examples [00:50, 883.64 examples/s]43252 examples [00:50, 880.65 examples/s]43343 examples [00:50, 885.43 examples/s]43432 examples [00:50, 877.36 examples/s]43521 examples [00:50, 878.55 examples/s]43609 examples [00:50, 860.69 examples/s]43696 examples [00:50, 863.42 examples/s]43783 examples [00:51, 831.73 examples/s]43867 examples [00:51, 824.54 examples/s]43951 examples [00:51, 827.38 examples/s]44038 examples [00:51, 839.32 examples/s]44127 examples [00:51, 851.17 examples/s]44213 examples [00:51, 839.61 examples/s]44298 examples [00:51, 836.55 examples/s]44387 examples [00:51, 850.28 examples/s]44473 examples [00:51, 838.88 examples/s]44558 examples [00:51, 841.06 examples/s]44644 examples [00:52, 845.68 examples/s]44729 examples [00:52, 822.91 examples/s]44812 examples [00:52, 819.32 examples/s]44900 examples [00:52, 836.40 examples/s]44984 examples [00:52, 835.58 examples/s]45070 examples [00:52, 841.94 examples/s]45156 examples [00:52, 846.52 examples/s]45241 examples [00:52, 825.64 examples/s]45324 examples [00:52, 774.64 examples/s]45406 examples [00:53, 786.97 examples/s]45494 examples [00:53, 811.24 examples/s]45585 examples [00:53, 838.16 examples/s]45674 examples [00:53, 853.03 examples/s]45763 examples [00:53, 861.33 examples/s]45850 examples [00:53, 833.93 examples/s]45934 examples [00:53, 798.00 examples/s]46015 examples [00:53, 792.54 examples/s]46098 examples [00:53, 802.88 examples/s]46186 examples [00:53, 823.47 examples/s]46274 examples [00:54, 839.35 examples/s]46361 examples [00:54, 847.56 examples/s]46449 examples [00:54, 855.08 examples/s]46535 examples [00:54, 854.15 examples/s]46622 examples [00:54, 857.04 examples/s]46709 examples [00:54, 860.77 examples/s]46796 examples [00:54, 844.58 examples/s]46882 examples [00:54, 847.32 examples/s]46970 examples [00:54, 854.47 examples/s]47056 examples [00:54, 855.83 examples/s]47142 examples [00:55, 856.00 examples/s]47230 examples [00:55, 860.63 examples/s]47319 examples [00:55, 867.90 examples/s]47406 examples [00:55, 867.83 examples/s]47495 examples [00:55, 872.21 examples/s]47585 examples [00:55, 878.43 examples/s]47673 examples [00:55, 877.63 examples/s]47761 examples [00:55, 872.93 examples/s]47850 examples [00:55, 877.55 examples/s]47940 examples [00:55, 880.64 examples/s]48031 examples [00:56, 887.47 examples/s]48120 examples [00:56, 860.23 examples/s]48207 examples [00:56, 828.49 examples/s]48291 examples [00:56, 831.41 examples/s]48379 examples [00:56, 844.24 examples/s]48464 examples [00:56, 808.03 examples/s]48550 examples [00:56, 821.36 examples/s]48635 examples [00:56, 825.00 examples/s]48721 examples [00:56, 834.13 examples/s]48805 examples [00:57, 818.85 examples/s]48898 examples [00:57, 847.74 examples/s]48990 examples [00:57, 867.72 examples/s]49080 examples [00:57, 875.26 examples/s]49170 examples [00:57, 880.52 examples/s]49261 examples [00:57, 887.24 examples/s]49352 examples [00:57, 893.33 examples/s]49443 examples [00:57, 895.63 examples/s]49533 examples [00:57, 891.07 examples/s]49623 examples [00:57, 886.53 examples/s]49712 examples [00:58, 886.27 examples/s]49801 examples [00:58, 874.68 examples/s]49889 examples [00:58, 849.39 examples/s]49975 examples [00:58, 838.85 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5878/50000 [00:00<00:00, 58778.65 examples/s] 33%|███▎      | 16439/50000 [00:00<00:00, 67797.05 examples/s] 55%|█████▍    | 27498/50000 [00:00<00:00, 76699.47 examples/s] 77%|███████▋  | 38499/50000 [00:00<00:00, 84360.86 examples/s] 99%|█████████▊| 49306/50000 [00:00<00:00, 90302.96 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 695.26 examples/s]159 examples [00:00, 743.40 examples/s]247 examples [00:00, 778.28 examples/s]336 examples [00:00, 808.18 examples/s]423 examples [00:00, 824.32 examples/s]511 examples [00:00, 839.27 examples/s]597 examples [00:00, 843.68 examples/s]677 examples [00:00, 802.86 examples/s]764 examples [00:00, 821.09 examples/s]851 examples [00:01, 835.10 examples/s]939 examples [00:01, 846.11 examples/s]1024 examples [00:01, 845.89 examples/s]1111 examples [00:01, 850.02 examples/s]1201 examples [00:01, 862.97 examples/s]1291 examples [00:01, 871.40 examples/s]1378 examples [00:01, 856.41 examples/s]1464 examples [00:01, 835.54 examples/s]1551 examples [00:01, 843.94 examples/s]1643 examples [00:01, 862.66 examples/s]1730 examples [00:02, 861.82 examples/s]1817 examples [00:02, 860.12 examples/s]1904 examples [00:02, 858.88 examples/s]1992 examples [00:02, 862.07 examples/s]2084 examples [00:02, 877.47 examples/s]2172 examples [00:02, 869.79 examples/s]2264 examples [00:02, 880.58 examples/s]2356 examples [00:02, 890.32 examples/s]2446 examples [00:02, 892.04 examples/s]2537 examples [00:02, 895.06 examples/s]2627 examples [00:03, 894.45 examples/s]2717 examples [00:03, 893.51 examples/s]2807 examples [00:03, 889.87 examples/s]2897 examples [00:03, 889.82 examples/s]2988 examples [00:03, 894.42 examples/s]3078 examples [00:03, 890.75 examples/s]3170 examples [00:03, 896.66 examples/s]3260 examples [00:03, 857.33 examples/s]3347 examples [00:03, 853.82 examples/s]3433 examples [00:03, 846.00 examples/s]3518 examples [00:04, 847.15 examples/s]3605 examples [00:04, 852.19 examples/s]3693 examples [00:04, 859.70 examples/s]3781 examples [00:04, 865.57 examples/s]3871 examples [00:04, 875.54 examples/s]3961 examples [00:04, 880.93 examples/s]4051 examples [00:04, 886.04 examples/s]4140 examples [00:04, 879.79 examples/s]4229 examples [00:04, 850.47 examples/s]4319 examples [00:04, 862.82 examples/s]4407 examples [00:05, 865.97 examples/s]4498 examples [00:05, 876.78 examples/s]4586 examples [00:05, 872.99 examples/s]4675 examples [00:05, 877.34 examples/s]4763 examples [00:05, 878.11 examples/s]4851 examples [00:05, 844.67 examples/s]4936 examples [00:05, 832.64 examples/s]5020 examples [00:05, 828.83 examples/s]5109 examples [00:05, 843.08 examples/s]5201 examples [00:06, 862.24 examples/s]5288 examples [00:06, 864.26 examples/s]5378 examples [00:06, 874.30 examples/s]5467 examples [00:06, 876.89 examples/s]5555 examples [00:06, 876.33 examples/s]5645 examples [00:06, 881.79 examples/s]5734 examples [00:06, 870.32 examples/s]5822 examples [00:06, 860.33 examples/s]5911 examples [00:06, 866.89 examples/s]5998 examples [00:06, 865.58 examples/s]6091 examples [00:07, 883.46 examples/s]6182 examples [00:07, 890.88 examples/s]6272 examples [00:07, 892.40 examples/s]6362 examples [00:07, 883.76 examples/s]6453 examples [00:07, 889.87 examples/s]6543 examples [00:07, 858.72 examples/s]6630 examples [00:07, 860.22 examples/s]6722 examples [00:07, 874.90 examples/s]6812 examples [00:07, 881.66 examples/s]6901 examples [00:07, 882.61 examples/s]6992 examples [00:08, 889.52 examples/s]7082 examples [00:08, 870.55 examples/s]7172 examples [00:08, 876.58 examples/s]7260 examples [00:08, 876.51 examples/s]7351 examples [00:08, 885.98 examples/s]7441 examples [00:08, 889.99 examples/s]7532 examples [00:08, 893.94 examples/s]7622 examples [00:08, 892.41 examples/s]7712 examples [00:08, 876.78 examples/s]7800 examples [00:08, 876.28 examples/s]7888 examples [00:09, 873.23 examples/s]7976 examples [00:09, 869.34 examples/s]8066 examples [00:09, 877.19 examples/s]8155 examples [00:09, 876.22 examples/s]8243 examples [00:09, 868.98 examples/s]8330 examples [00:09, 837.62 examples/s]8419 examples [00:09, 850.43 examples/s]8506 examples [00:09, 856.20 examples/s]8592 examples [00:09, 853.96 examples/s]8678 examples [00:10, 849.51 examples/s]8765 examples [00:10, 854.00 examples/s]8851 examples [00:10, 840.28 examples/s]8939 examples [00:10, 849.22 examples/s]9026 examples [00:10, 852.00 examples/s]9115 examples [00:10, 860.05 examples/s]9202 examples [00:10, 860.78 examples/s]9291 examples [00:10, 867.43 examples/s]9380 examples [00:10, 873.15 examples/s]9468 examples [00:10, 864.53 examples/s]9557 examples [00:11, 871.36 examples/s]9647 examples [00:11, 878.75 examples/s]9736 examples [00:11, 880.60 examples/s]9825 examples [00:11, 878.71 examples/s]9916 examples [00:11, 886.23 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 99%|█████████▉| 9930/10000 [00:00<00:00, 99295.36 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKH8YR4/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKH8YR4/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f53109ce620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f53109ce620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f53109ce620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5299e295c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5299e65cc0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f53109ce620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f53109ce620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f53109ce620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5299e879e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f52f7a00358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5298145158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5298145158> 

  function with postional parmater data_info <function split_train_valid at 0x7f5298145158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5298145268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5298145268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5298145268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=245bd17eb31a70b23f3afc55db916b70f74e807393660a98363c8d0e98ed3979
  Stored in directory: /tmp/pip-ephem-wheel-cache-k4j91_pb/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:23:18, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:30:09, 16.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:12:19, 23.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:09:08, 33.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:59:38, 47.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.38M/862M [00:01<3:28:25, 68.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:25:24, 97.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:41:17, 139kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:01<1:10:39, 198kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<49:14, 283kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:01<34:27, 402kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:01<24:02, 573kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.2M/862M [00:02<16:53, 812kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:02<11:50, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:02<08:22, 1.62MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<06:13, 2.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<06:15, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<08:02, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<06:26, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<04:44, 2.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<07:01, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:26, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<04:53, 2.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<06:14, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<07:13, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<05:40, 2.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:08<04:08, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<10:22, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<08:38, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<06:20, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<07:29, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<06:35, 1.99MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:12<04:53, 2.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<06:31, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:54, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:14<04:28, 2.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<06:11, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<05:39, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<04:17, 3.02MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:47, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<05:19, 2.43MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:18<03:52, 3.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<11:35, 1.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<10:53, 1.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<08:12, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<05:53, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<11:35, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<09:27, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<06:56, 1.84MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:49, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<08:04, 1.58MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<06:13, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:29, 2.83MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<13:59, 906kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<10:53, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:58, 1.59MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<05:42, 2.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<54:46, 230kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<40:55, 308kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<29:09, 432kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<20:34, 610kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<18:41, 671kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<14:23, 871kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:22, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:09, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:45, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:28, 1.67MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:20, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<48:35, 255kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<35:17, 351kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<24:56, 496kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<20:16, 608kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:27, 798kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<11:04, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<10:39, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:42, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:23, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:20, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:22, 1.91MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:43, 2.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:11, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:48, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:17, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<03:51, 3.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<08:11, 1.47MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:58, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:43, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:17, 2.78MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:49, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:17, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:00, 2.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:36, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:07, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:50, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:11, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:20, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<04:57, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:45, 3.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:21, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:46, 2.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:37, 3.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:17, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:53, 2.37MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:42, 3.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:23, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:08, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:48, 2.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:33, 3.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:27, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:42, 2.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<04:16, 2.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:39, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:19, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:00, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:19, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:43, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:48, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:38, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:34, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:39, 3.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<02:41, 4.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<1:07:58, 163kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<49:53, 222kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<35:27, 313kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<24:51, 444kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:35:57, 115kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:08:17, 161kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<47:58, 229kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<36:02, 304kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<26:21, 416kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<18:40, 585kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<15:36, 698kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<13:11, 825kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:45, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<06:54, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<34:57, 310kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<25:22, 426kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<18:11, 594kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<12:47, 841kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<52:17, 206kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<38:56, 276kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<28:18, 380kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<20:02, 536kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<16:08, 662kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<12:23, 863kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:55, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:42, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:17, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:20, 1.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<04:31, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<31:05, 339kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<22:39, 465kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<16:20, 645kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<11:30, 912kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<20:28, 512kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<16:29, 635kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:03, 868kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<10:05, 1.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:07, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:56, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:34, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:42, 1.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:13, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:47, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:34, 2.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:52, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:29, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:17, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:12, 3.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:15, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:43, 2.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:34, 2.83MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:51, 2.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:27, 1.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:21, 2.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:09, 3.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<1:13:33, 136kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<52:29, 190kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<36:55, 270kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<28:03, 354kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<21:42, 458kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<15:40, 633kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<12:30, 788kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:46, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:04, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:14, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:02, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:26, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:47, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:29, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:14, 2.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<10:45, 898kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:31, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:11, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:34, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:17, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:01, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:44, 2.56MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:45, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:19, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:15, 2.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:29, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:00, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:55, 2.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<02:53, 3.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:38, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:56, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:41, 2.54MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:45, 1.96MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:03, 2.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<02:56, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:04, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<07:07, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:49, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<39:37, 232kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<28:39, 320kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<20:14, 452kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<16:14, 561kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<13:13, 689kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<09:39, 942kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<06:51, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<08:48, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:06, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:09, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:42, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:55, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:39, 2.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:39, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:06, 2.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:16, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:47, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:48, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:45, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:50, 3.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:35, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:36, 2.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:36, 3.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<07:45, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<06:19, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:38, 1.85MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:13, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:31, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:22, 2.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<04:21, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:47, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:43, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<02:42, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:51, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:58, 1.69MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:39, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:00, 2.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:00, 2.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:04, 2.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:36, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<02:36, 3.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<7:54:51, 17.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<5:32:59, 24.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<3:52:33, 35.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<2:43:58, 49.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<1:56:24, 70.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<1:21:46, 99.5kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<58:11, 139kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<41:32, 194kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<29:11, 276kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<22:12, 361kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<16:21, 490kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<11:36, 688kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:58, 796kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<08:36, 924kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:25, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<04:33, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<7:37:54, 17.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<5:21:05, 24.5kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<3:44:12, 35.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<2:38:03, 49.4kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<1:52:11, 69.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<1:18:44, 99.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<54:56, 141kB/s]   .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<43:00, 180kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<30:52, 251kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<21:44, 355kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<16:56, 453kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<13:25, 571kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<09:46, 783kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<08:01, 947kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:23, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<04:37, 1.64MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:59, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:01, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:50, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<02:46, 2.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:59, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:57, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:39, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:19, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:32, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:32, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:32, 2.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<54:43, 134kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<39:01, 188kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<27:24, 266kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<20:47, 349kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<16:01, 453kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<11:31, 629kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<08:06, 889kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<12:37, 570kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<09:34, 751kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<06:51, 1.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:27, 1.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:58, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:31, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<03:14, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<6:46:42, 17.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<4:45:10, 24.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<3:19:03, 35.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<2:20:14, 49.8kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<1:39:32, 70.2kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<1:09:54, 99.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<49:41, 139kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<35:27, 195kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<24:54, 277kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<18:57, 361kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<13:49, 495kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<09:49, 694kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<08:26, 803kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<07:17, 929kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<05:23, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:52, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:47, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:02, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<02:59, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:37, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:53, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:01, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:11, 3.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:59, 941kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:34, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:03, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:20, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:21, 1.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:22, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:23, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:16, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:04, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:47, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:06, 3.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:42, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:03, 3.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:54, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<03:18, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:35, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:51, 3.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<12:19, 500kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<09:15, 665kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:35, 929kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<06:00, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:49, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:30, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:52, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:56, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:03, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:05, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:03, 2.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:49, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:10, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:27, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:50, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:57, 1.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:40, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:59, 2.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:44, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:30, 2.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:53, 3.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:39, 2.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:01, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:23, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:22, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:48, 3.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:32, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:54, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:17, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:38, 3.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:01, 683kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:05, 898kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:21, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<03:06, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<12:22, 437kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<09:45, 554kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<07:05, 760kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<04:58, 1.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<5:11:36, 17.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<3:38:23, 24.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<2:32:16, 34.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:47:06, 49.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:15:25, 69.8kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<52:39, 99.5kB/s]  .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<37:50, 138kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<27:22, 190kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<19:19, 269kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<13:37, 379kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<10:40, 481kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<07:59, 641kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<05:42, 895kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:08, 986kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:37, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:27, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:27, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<17:13, 290kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<12:28, 400kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<08:48, 564kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<07:17, 677kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<06:06, 807kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:28, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<03:10, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:33, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:40, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:41, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:59, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:35, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:55, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:26, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:40, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:04, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:29, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:34, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:58, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:11, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:35, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:45, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:07, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:31, 2.97MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:55, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:12, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:20, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:40, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:46, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:07, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:31, 2.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:47, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:06, 1.41MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:15, 1.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:35, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:41, 1.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:05, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:08, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:55, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:26, 2.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:58, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:12, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:43, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:14, 3.31MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<05:48, 708kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<04:29, 915kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<03:12, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:10, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:02, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:19, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:39, 2.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<3:50:51, 17.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<2:41:44, 24.5kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<1:52:34, 35.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:18:59, 49.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<55:36, 70.1kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<38:45, 99.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<27:47, 138kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<19:48, 193kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<13:52, 274kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<10:30, 358kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<08:07, 463kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<05:51, 640kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:38, 796kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:37, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<02:36, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:34, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:58, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:45<01:24, 2.53MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<3:26:34, 17.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<2:24:42, 24.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<1:40:38, 35.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<1:10:32, 49.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<49:38, 70.2kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<34:34, 100kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<24:46, 138kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<18:01, 190kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<12:44, 267kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<09:19, 359kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:51, 488kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<04:50, 687kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:07, 798kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:32, 924kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:37, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:20, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:26, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:44, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:51, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:27, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:30, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:18, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<00:58, 3.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<00:43, 4.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<12:37, 238kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<09:08, 329kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<06:24, 465kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<05:09, 570kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:12, 698kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<03:03, 956kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:10, 1.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:22, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:57, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:25, 1.99MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:38, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:43, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:19, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:56, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:18, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:53, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:22, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:34, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:38, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:15, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:54, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:46, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:06, 2.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:21, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:11, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:53, 2.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:20, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:44, 3.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<03:39, 654kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:45, 864kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:58, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:54, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:48, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:22, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:18, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:08, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:51, 2.61MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:10, 1.87MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:59, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:01, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:56, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:42, 2.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:57, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:53, 2.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:39, 3.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:03, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:49, 2.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:35, 3.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:33, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:17, 1.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:55, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:04, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:08, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:52, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:53, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:48, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:35, 2.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:30, 3.21MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:41, 610kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:02, 800kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:26, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:21, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:06, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:48, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:54, 1.66MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:44, 2.00MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:36, 2.46MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<00:25, 3.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<12:44, 112kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<09:11, 156kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<06:26, 220kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<04:34, 299kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<03:19, 409kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:19, 575kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:52, 689kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:34, 825kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:09, 1.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:58, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:48, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:34, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:40, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:35, 1.97MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:25, 2.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:36, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:28, 2.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:19, 3.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<08:36, 119kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<06:05, 167kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<04:10, 237kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<03:02, 314kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<02:19, 409kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:38, 569kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<01:06, 805kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:12, 401kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:37, 540kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:07, 758kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:46, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<03:34, 229kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<02:39, 306kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:52, 428kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:20, 556kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:00, 733kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:42, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:28, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<03:01, 225kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<02:14, 302kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:34, 423kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<01:01, 600kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:57, 313kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:24, 427kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:57, 602kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:45, 716kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:38, 845kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:27, 1.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:18, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:21, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:17, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:11, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:08, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:08, 1.87MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 3.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:30, 256kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:22, 340kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:15, 471kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:08, 662kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 775kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 995kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.36MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<110:48:26,  1.00it/s]  0%|          | 754/400000 [00:01<77:25:24,  1.43it/s]  0%|          | 1464/400000 [00:01<54:06:17,  2.05it/s]  1%|          | 2211/400000 [00:01<37:48:24,  2.92it/s]  1%|          | 2951/400000 [00:01<26:25:12,  4.17it/s]  1%|          | 3715/400000 [00:01<18:27:45,  5.96it/s]  1%|          | 4469/400000 [00:01<12:54:13,  8.51it/s]  1%|▏         | 5238/400000 [00:01<9:01:09, 12.16it/s]   2%|▏         | 6009/400000 [00:01<6:18:19, 17.36it/s]  2%|▏         | 6784/400000 [00:01<4:24:33, 24.77it/s]  2%|▏         | 7583/400000 [00:01<3:05:03, 35.34it/s]  2%|▏         | 8347/400000 [00:02<2:09:32, 50.39it/s]  2%|▏         | 9112/400000 [00:02<1:30:45, 71.78it/s]  2%|▏         | 9871/400000 [00:02<1:03:42, 102.07it/s]  3%|▎         | 10624/400000 [00:02<44:45, 144.98it/s]   3%|▎         | 11381/400000 [00:02<31:31, 205.42it/s]  3%|▎         | 12126/400000 [00:02<22:20, 289.26it/s]  3%|▎         | 12837/400000 [00:02<15:53, 406.14it/s]  3%|▎         | 13581/400000 [00:02<11:21, 566.93it/s]  4%|▎         | 14298/400000 [00:02<08:12, 782.70it/s]  4%|▍         | 15033/400000 [00:03<06:00, 1069.26it/s]  4%|▍         | 15761/400000 [00:03<04:27, 1437.00it/s]  4%|▍         | 16545/400000 [00:03<03:21, 1903.20it/s]  4%|▍         | 17311/400000 [00:03<02:35, 2457.04it/s]  5%|▍         | 18059/400000 [00:03<02:04, 3064.14it/s]  5%|▍         | 18800/400000 [00:03<01:42, 3708.85it/s]  5%|▍         | 19537/400000 [00:03<01:27, 4337.19it/s]  5%|▌         | 20314/400000 [00:03<01:15, 4997.73it/s]  5%|▌         | 21075/400000 [00:03<01:08, 5570.76it/s]  5%|▌         | 21825/400000 [00:03<01:03, 5964.10it/s]  6%|▌         | 22564/400000 [00:04<01:00, 6202.42it/s]  6%|▌         | 23287/400000 [00:04<00:58, 6443.43it/s]  6%|▌         | 24006/400000 [00:04<00:57, 6576.11it/s]  6%|▌         | 24716/400000 [00:04<00:55, 6711.86it/s]  6%|▋         | 25446/400000 [00:04<00:54, 6877.44it/s]  7%|▋         | 26178/400000 [00:04<00:53, 7003.80it/s]  7%|▋         | 26899/400000 [00:04<00:52, 7062.92it/s]  7%|▋         | 27620/400000 [00:04<00:54, 6854.45it/s]  7%|▋         | 28350/400000 [00:04<00:53, 6981.09it/s]  7%|▋         | 29060/400000 [00:04<00:52, 7014.97it/s]  7%|▋         | 29768/400000 [00:05<00:53, 6936.23it/s]  8%|▊         | 30484/400000 [00:05<00:52, 6998.47it/s]  8%|▊         | 31245/400000 [00:05<00:51, 7171.12it/s]  8%|▊         | 31966/400000 [00:05<00:51, 7172.25it/s]  8%|▊         | 32724/400000 [00:05<00:50, 7288.49it/s]  8%|▊         | 33463/400000 [00:05<00:50, 7318.23it/s]  9%|▊         | 34197/400000 [00:05<00:50, 7308.45it/s]  9%|▊         | 34966/400000 [00:05<00:49, 7417.95it/s]  9%|▉         | 35727/400000 [00:05<00:48, 7472.96it/s]  9%|▉         | 36476/400000 [00:05<00:48, 7471.04it/s]  9%|▉         | 37224/400000 [00:06<00:48, 7447.44it/s]  9%|▉         | 37970/400000 [00:06<00:48, 7429.88it/s] 10%|▉         | 38758/400000 [00:06<00:47, 7558.06it/s] 10%|▉         | 39522/400000 [00:06<00:47, 7579.97it/s] 10%|█         | 40281/400000 [00:06<00:47, 7571.80it/s] 10%|█         | 41064/400000 [00:06<00:46, 7645.26it/s] 10%|█         | 41832/400000 [00:06<00:46, 7655.48it/s] 11%|█         | 42598/400000 [00:06<00:47, 7451.87it/s] 11%|█         | 43357/400000 [00:06<00:47, 7490.97it/s] 11%|█         | 44137/400000 [00:06<00:46, 7579.33it/s] 11%|█         | 44896/400000 [00:07<00:47, 7533.37it/s] 11%|█▏        | 45651/400000 [00:07<00:47, 7476.48it/s] 12%|█▏        | 46422/400000 [00:07<00:46, 7544.84it/s] 12%|█▏        | 47193/400000 [00:07<00:46, 7593.22it/s] 12%|█▏        | 47953/400000 [00:07<00:46, 7544.24it/s] 12%|█▏        | 48708/400000 [00:07<00:46, 7516.58it/s] 12%|█▏        | 49460/400000 [00:07<00:47, 7372.44it/s] 13%|█▎        | 50199/400000 [00:07<00:47, 7375.04it/s] 13%|█▎        | 50938/400000 [00:07<00:47, 7338.40it/s] 13%|█▎        | 51698/400000 [00:08<00:46, 7414.90it/s] 13%|█▎        | 52460/400000 [00:08<00:46, 7474.40it/s] 13%|█▎        | 53208/400000 [00:08<00:46, 7407.43it/s] 13%|█▎        | 53950/400000 [00:08<00:46, 7396.11it/s] 14%|█▎        | 54690/400000 [00:08<00:47, 7267.58it/s] 14%|█▍        | 55453/400000 [00:08<00:46, 7370.79it/s] 14%|█▍        | 56191/400000 [00:08<00:46, 7358.53it/s] 14%|█▍        | 56928/400000 [00:08<00:46, 7302.10it/s] 14%|█▍        | 57659/400000 [00:08<00:47, 7227.15it/s] 15%|█▍        | 58394/400000 [00:08<00:47, 7261.31it/s] 15%|█▍        | 59128/400000 [00:09<00:46, 7282.22it/s] 15%|█▍        | 59857/400000 [00:09<00:46, 7261.89it/s] 15%|█▌        | 60584/400000 [00:09<00:47, 7198.15it/s] 15%|█▌        | 61324/400000 [00:09<00:46, 7256.27it/s] 16%|█▌        | 62050/400000 [00:09<00:46, 7256.59it/s] 16%|█▌        | 62776/400000 [00:09<00:46, 7213.57it/s] 16%|█▌        | 63498/400000 [00:09<00:46, 7214.79it/s] 16%|█▌        | 64231/400000 [00:09<00:46, 7247.89it/s] 16%|█▌        | 64978/400000 [00:09<00:45, 7311.57it/s] 16%|█▋        | 65712/400000 [00:09<00:45, 7317.55it/s] 17%|█▋        | 66444/400000 [00:10<00:46, 7134.50it/s] 17%|█▋        | 67206/400000 [00:10<00:45, 7273.52it/s] 17%|█▋        | 67938/400000 [00:10<00:45, 7287.05it/s] 17%|█▋        | 68668/400000 [00:10<00:45, 7274.00it/s] 17%|█▋        | 69410/400000 [00:10<00:45, 7314.91it/s] 18%|█▊        | 70143/400000 [00:10<00:45, 7311.12it/s] 18%|█▊        | 70875/400000 [00:10<00:45, 7255.35it/s] 18%|█▊        | 71601/400000 [00:10<00:45, 7243.35it/s] 18%|█▊        | 72330/400000 [00:10<00:45, 7254.15it/s] 18%|█▊        | 73056/400000 [00:10<00:45, 7184.00it/s] 18%|█▊        | 73792/400000 [00:11<00:45, 7232.96it/s] 19%|█▊        | 74518/400000 [00:11<00:44, 7240.65it/s] 19%|█▉        | 75243/400000 [00:11<00:45, 7154.64it/s] 19%|█▉        | 75984/400000 [00:11<00:44, 7227.43it/s] 19%|█▉        | 76744/400000 [00:11<00:44, 7333.46it/s] 19%|█▉        | 77489/400000 [00:11<00:43, 7367.79it/s] 20%|█▉        | 78253/400000 [00:11<00:43, 7446.95it/s] 20%|█▉        | 79005/400000 [00:11<00:42, 7468.42it/s] 20%|█▉        | 79753/400000 [00:11<00:42, 7458.74it/s] 20%|██        | 80503/400000 [00:11<00:42, 7468.45it/s] 20%|██        | 81251/400000 [00:12<00:42, 7466.66it/s] 20%|██        | 81998/400000 [00:12<00:43, 7357.90it/s] 21%|██        | 82741/400000 [00:12<00:43, 7377.88it/s] 21%|██        | 83480/400000 [00:12<00:42, 7364.99it/s] 21%|██        | 84217/400000 [00:12<00:43, 7311.99it/s] 21%|██        | 84949/400000 [00:12<00:43, 7283.40it/s] 21%|██▏       | 85678/400000 [00:12<00:43, 7284.87it/s] 22%|██▏       | 86407/400000 [00:12<00:43, 7232.36it/s] 22%|██▏       | 87133/400000 [00:12<00:43, 7240.46it/s] 22%|██▏       | 87863/400000 [00:12<00:43, 7255.57it/s] 22%|██▏       | 88592/400000 [00:13<00:42, 7265.18it/s] 22%|██▏       | 89340/400000 [00:13<00:42, 7327.71it/s] 23%|██▎       | 90073/400000 [00:13<00:42, 7280.44it/s] 23%|██▎       | 90802/400000 [00:13<00:42, 7279.31it/s] 23%|██▎       | 91554/400000 [00:13<00:41, 7348.48it/s] 23%|██▎       | 92290/400000 [00:13<00:42, 7257.31it/s] 23%|██▎       | 93046/400000 [00:13<00:41, 7343.40it/s] 23%|██▎       | 93790/400000 [00:13<00:41, 7370.62it/s] 24%|██▎       | 94530/400000 [00:13<00:41, 7378.62it/s] 24%|██▍       | 95269/400000 [00:13<00:41, 7315.48it/s] 24%|██▍       | 96001/400000 [00:14<00:41, 7306.04it/s] 24%|██▍       | 96735/400000 [00:14<00:41, 7310.18it/s] 24%|██▍       | 97467/400000 [00:14<00:41, 7219.33it/s] 25%|██▍       | 98190/400000 [00:14<00:41, 7209.51it/s] 25%|██▍       | 98916/400000 [00:14<00:41, 7223.34it/s] 25%|██▍       | 99639/400000 [00:14<00:42, 7145.16it/s] 25%|██▌       | 100354/400000 [00:14<00:42, 7117.56it/s] 25%|██▌       | 101067/400000 [00:14<00:42, 7020.15it/s] 25%|██▌       | 101810/400000 [00:14<00:41, 7136.09it/s] 26%|██▌       | 102546/400000 [00:14<00:41, 7200.56it/s] 26%|██▌       | 103302/400000 [00:15<00:40, 7303.43it/s] 26%|██▌       | 104034/400000 [00:15<00:40, 7249.34it/s] 26%|██▌       | 104760/400000 [00:15<00:41, 7095.91it/s] 26%|██▋       | 105485/400000 [00:15<00:41, 7140.40it/s] 27%|██▋       | 106262/400000 [00:15<00:40, 7317.81it/s] 27%|██▋       | 106996/400000 [00:15<00:40, 7295.14it/s] 27%|██▋       | 107738/400000 [00:15<00:39, 7330.48it/s] 27%|██▋       | 108472/400000 [00:15<00:40, 7254.85it/s] 27%|██▋       | 109219/400000 [00:15<00:39, 7315.20it/s] 27%|██▋       | 109952/400000 [00:16<00:40, 7248.51it/s] 28%|██▊       | 110692/400000 [00:16<00:39, 7292.64it/s] 28%|██▊       | 111444/400000 [00:16<00:39, 7356.46it/s] 28%|██▊       | 112181/400000 [00:16<00:39, 7323.74it/s] 28%|██▊       | 112914/400000 [00:16<00:39, 7306.25it/s] 28%|██▊       | 113645/400000 [00:16<00:39, 7304.64it/s] 29%|██▊       | 114376/400000 [00:16<00:40, 7071.29it/s] 29%|██▉       | 115146/400000 [00:16<00:39, 7245.11it/s] 29%|██▉       | 115899/400000 [00:16<00:38, 7326.53it/s] 29%|██▉       | 116634/400000 [00:16<00:39, 7256.82it/s] 29%|██▉       | 117362/400000 [00:17<00:39, 7219.60it/s] 30%|██▉       | 118086/400000 [00:17<00:39, 7141.76it/s] 30%|██▉       | 118802/400000 [00:17<00:39, 7114.36it/s] 30%|██▉       | 119561/400000 [00:17<00:38, 7250.60it/s] 30%|███       | 120288/400000 [00:17<00:38, 7217.78it/s] 30%|███       | 121030/400000 [00:17<00:38, 7276.28it/s] 30%|███       | 121759/400000 [00:17<00:39, 7126.35it/s] 31%|███       | 122475/400000 [00:17<00:38, 7135.31it/s] 31%|███       | 123201/400000 [00:17<00:38, 7171.77it/s] 31%|███       | 123919/400000 [00:17<00:38, 7159.51it/s] 31%|███       | 124660/400000 [00:18<00:38, 7230.70it/s] 31%|███▏      | 125437/400000 [00:18<00:37, 7383.17it/s] 32%|███▏      | 126189/400000 [00:18<00:36, 7420.50it/s] 32%|███▏      | 126932/400000 [00:18<00:37, 7377.12it/s] 32%|███▏      | 127671/400000 [00:18<00:37, 7330.22it/s] 32%|███▏      | 128429/400000 [00:18<00:36, 7403.04it/s] 32%|███▏      | 129192/400000 [00:18<00:36, 7468.01it/s] 32%|███▏      | 129975/400000 [00:18<00:35, 7572.13it/s] 33%|███▎      | 130733/400000 [00:18<00:35, 7555.01it/s] 33%|███▎      | 131490/400000 [00:18<00:35, 7471.81it/s] 33%|███▎      | 132268/400000 [00:19<00:35, 7560.96it/s] 33%|███▎      | 133025/400000 [00:19<00:35, 7445.23it/s] 33%|███▎      | 133771/400000 [00:19<00:36, 7383.26it/s] 34%|███▎      | 134511/400000 [00:19<00:35, 7382.08it/s] 34%|███▍      | 135281/400000 [00:19<00:35, 7471.92it/s] 34%|███▍      | 136029/400000 [00:19<00:35, 7385.81it/s] 34%|███▍      | 136797/400000 [00:19<00:35, 7469.51it/s] 34%|███▍      | 137575/400000 [00:19<00:34, 7558.35it/s] 35%|███▍      | 138332/400000 [00:19<00:34, 7518.97it/s] 35%|███▍      | 139085/400000 [00:19<00:34, 7500.43it/s] 35%|███▍      | 139836/400000 [00:20<00:35, 7423.01it/s] 35%|███▌      | 140579/400000 [00:20<00:35, 7403.77it/s] 35%|███▌      | 141350/400000 [00:20<00:34, 7492.40it/s] 36%|███▌      | 142100/400000 [00:20<00:34, 7469.63it/s] 36%|███▌      | 142848/400000 [00:20<00:34, 7449.24it/s] 36%|███▌      | 143613/400000 [00:20<00:34, 7505.77it/s] 36%|███▌      | 144364/400000 [00:20<00:34, 7391.21it/s] 36%|███▋      | 145109/400000 [00:20<00:34, 7408.72it/s] 36%|███▋      | 145851/400000 [00:20<00:34, 7388.27it/s] 37%|███▋      | 146591/400000 [00:20<00:34, 7372.04it/s] 37%|███▋      | 147329/400000 [00:21<00:34, 7285.43it/s] 37%|███▋      | 148068/400000 [00:21<00:34, 7314.13it/s] 37%|███▋      | 148800/400000 [00:21<00:34, 7302.71it/s] 37%|███▋      | 149531/400000 [00:21<00:34, 7268.35it/s] 38%|███▊      | 150259/400000 [00:21<00:34, 7246.90it/s] 38%|███▊      | 150988/400000 [00:21<00:34, 7259.26it/s] 38%|███▊      | 151715/400000 [00:21<00:34, 7189.13it/s] 38%|███▊      | 152443/400000 [00:21<00:34, 7214.55it/s] 38%|███▊      | 153182/400000 [00:21<00:33, 7263.98it/s] 38%|███▊      | 153918/400000 [00:22<00:33, 7290.53it/s] 39%|███▊      | 154669/400000 [00:22<00:33, 7352.42it/s] 39%|███▉      | 155427/400000 [00:22<00:32, 7417.18it/s] 39%|███▉      | 156185/400000 [00:22<00:32, 7465.20it/s] 39%|███▉      | 156932/400000 [00:22<00:33, 7347.04it/s] 39%|███▉      | 157690/400000 [00:22<00:32, 7415.04it/s] 40%|███▉      | 158433/400000 [00:22<00:32, 7340.70it/s] 40%|███▉      | 159174/400000 [00:22<00:32, 7326.67it/s] 40%|███▉      | 159912/400000 [00:22<00:32, 7340.58it/s] 40%|████      | 160653/400000 [00:22<00:32, 7359.39it/s] 40%|████      | 161396/400000 [00:23<00:32, 7379.48it/s] 41%|████      | 162136/400000 [00:23<00:32, 7382.71it/s] 41%|████      | 162875/400000 [00:23<00:32, 7348.55it/s] 41%|████      | 163624/400000 [00:23<00:31, 7390.00it/s] 41%|████      | 164364/400000 [00:23<00:32, 7332.06it/s] 41%|████▏     | 165112/400000 [00:23<00:31, 7373.72it/s] 41%|████▏     | 165851/400000 [00:23<00:31, 7376.15it/s] 42%|████▏     | 166597/400000 [00:23<00:31, 7398.90it/s] 42%|████▏     | 167375/400000 [00:23<00:30, 7508.24it/s] 42%|████▏     | 168127/400000 [00:23<00:31, 7412.90it/s] 42%|████▏     | 168883/400000 [00:24<00:31, 7453.69it/s] 42%|████▏     | 169651/400000 [00:24<00:30, 7515.82it/s] 43%|████▎     | 170404/400000 [00:24<00:30, 7513.68it/s] 43%|████▎     | 171156/400000 [00:24<00:30, 7410.33it/s] 43%|████▎     | 171898/400000 [00:24<00:30, 7373.55it/s] 43%|████▎     | 172660/400000 [00:24<00:30, 7445.01it/s] 43%|████▎     | 173405/400000 [00:24<00:30, 7406.15it/s] 44%|████▎     | 174146/400000 [00:24<00:30, 7406.74it/s] 44%|████▎     | 174913/400000 [00:24<00:30, 7482.27it/s] 44%|████▍     | 175662/400000 [00:24<00:30, 7465.35it/s] 44%|████▍     | 176420/400000 [00:25<00:29, 7498.01it/s] 44%|████▍     | 177187/400000 [00:25<00:29, 7548.05it/s] 44%|████▍     | 177944/400000 [00:25<00:29, 7554.46it/s] 45%|████▍     | 178700/400000 [00:25<00:29, 7492.94it/s] 45%|████▍     | 179456/400000 [00:25<00:29, 7511.62it/s] 45%|████▌     | 180241/400000 [00:25<00:28, 7609.43it/s] 45%|████▌     | 181003/400000 [00:25<00:28, 7589.01it/s] 45%|████▌     | 181785/400000 [00:25<00:28, 7655.26it/s] 46%|████▌     | 182569/400000 [00:25<00:28, 7708.55it/s] 46%|████▌     | 183341/400000 [00:25<00:28, 7645.46it/s] 46%|████▌     | 184106/400000 [00:26<00:28, 7447.00it/s] 46%|████▌     | 184853/400000 [00:26<00:28, 7448.16it/s] 46%|████▋     | 185617/400000 [00:26<00:28, 7503.85it/s] 47%|████▋     | 186369/400000 [00:26<00:28, 7473.74it/s] 47%|████▋     | 187117/400000 [00:26<00:28, 7353.11it/s] 47%|████▋     | 187854/400000 [00:26<00:28, 7339.81it/s] 47%|████▋     | 188589/400000 [00:26<00:28, 7309.70it/s] 47%|████▋     | 189343/400000 [00:26<00:28, 7376.15it/s] 48%|████▊     | 190082/400000 [00:26<00:28, 7314.90it/s] 48%|████▊     | 190814/400000 [00:26<00:28, 7220.36it/s] 48%|████▊     | 191549/400000 [00:27<00:28, 7256.62it/s] 48%|████▊     | 192312/400000 [00:27<00:28, 7361.77it/s] 48%|████▊     | 193079/400000 [00:27<00:27, 7449.26it/s] 48%|████▊     | 193825/400000 [00:27<00:28, 7284.39it/s] 49%|████▊     | 194555/400000 [00:27<00:28, 7226.26it/s] 49%|████▉     | 195286/400000 [00:27<00:28, 7249.34it/s] 49%|████▉     | 196012/400000 [00:27<00:28, 7200.91it/s] 49%|████▉     | 196733/400000 [00:27<00:28, 7145.37it/s] 49%|████▉     | 197462/400000 [00:27<00:28, 7187.35it/s] 50%|████▉     | 198195/400000 [00:27<00:27, 7228.80it/s] 50%|████▉     | 198932/400000 [00:28<00:27, 7268.11it/s] 50%|████▉     | 199698/400000 [00:28<00:27, 7380.41it/s] 50%|█████     | 200475/400000 [00:28<00:26, 7492.93it/s] 50%|█████     | 201232/400000 [00:28<00:26, 7513.87it/s] 50%|█████     | 201993/400000 [00:28<00:26, 7540.31it/s] 51%|█████     | 202748/400000 [00:28<00:26, 7439.77it/s] 51%|█████     | 203503/400000 [00:28<00:26, 7472.11it/s] 51%|█████     | 204251/400000 [00:28<00:26, 7412.87it/s] 51%|█████▏    | 205013/400000 [00:28<00:26, 7470.98it/s] 51%|█████▏    | 205766/400000 [00:28<00:25, 7488.16it/s] 52%|█████▏    | 206527/400000 [00:29<00:25, 7522.17it/s] 52%|█████▏    | 207286/400000 [00:29<00:25, 7539.84it/s] 52%|█████▏    | 208041/400000 [00:29<00:26, 7341.14it/s] 52%|█████▏    | 208777/400000 [00:29<00:26, 7294.71it/s] 52%|█████▏    | 209525/400000 [00:29<00:25, 7347.51it/s] 53%|█████▎    | 210291/400000 [00:29<00:25, 7436.66it/s] 53%|█████▎    | 211036/400000 [00:29<00:25, 7405.64it/s] 53%|█████▎    | 211836/400000 [00:29<00:24, 7574.04it/s] 53%|█████▎    | 212613/400000 [00:29<00:24, 7628.66it/s] 53%|█████▎    | 213394/400000 [00:30<00:24, 7682.07it/s] 54%|█████▎    | 214177/400000 [00:30<00:24, 7724.95it/s] 54%|█████▎    | 214951/400000 [00:30<00:24, 7619.87it/s] 54%|█████▍    | 215714/400000 [00:30<00:24, 7615.93it/s] 54%|█████▍    | 216497/400000 [00:30<00:23, 7675.35it/s] 54%|█████▍    | 217266/400000 [00:30<00:24, 7470.81it/s] 55%|█████▍    | 218015/400000 [00:30<00:24, 7448.12it/s] 55%|█████▍    | 218774/400000 [00:30<00:24, 7487.86it/s] 55%|█████▍    | 219524/400000 [00:30<00:24, 7478.69it/s] 55%|█████▌    | 220281/400000 [00:30<00:23, 7504.85it/s] 55%|█████▌    | 221032/400000 [00:31<00:24, 7389.18it/s] 55%|█████▌    | 221772/400000 [00:31<00:24, 7325.90it/s] 56%|█████▌    | 222526/400000 [00:31<00:24, 7387.18it/s] 56%|█████▌    | 223266/400000 [00:31<00:23, 7389.64it/s] 56%|█████▌    | 224009/400000 [00:31<00:23, 7399.85it/s] 56%|█████▌    | 224750/400000 [00:31<00:23, 7355.79it/s] 56%|█████▋    | 225489/400000 [00:31<00:23, 7364.14it/s] 57%|█████▋    | 226226/400000 [00:31<00:23, 7326.37it/s] 57%|█████▋    | 226959/400000 [00:31<00:23, 7258.06it/s] 57%|█████▋    | 227686/400000 [00:31<00:23, 7233.67it/s] 57%|█████▋    | 228420/400000 [00:32<00:23, 7263.43it/s] 57%|█████▋    | 229161/400000 [00:32<00:23, 7306.49it/s] 57%|█████▋    | 229892/400000 [00:32<00:23, 7304.92it/s] 58%|█████▊    | 230623/400000 [00:32<00:23, 7272.84it/s] 58%|█████▊    | 231371/400000 [00:32<00:22, 7332.61it/s] 58%|█████▊    | 232105/400000 [00:32<00:23, 7161.85it/s] 58%|█████▊    | 232823/400000 [00:32<00:25, 6604.69it/s] 58%|█████▊    | 233557/400000 [00:32<00:24, 6808.85it/s] 59%|█████▊    | 234246/400000 [00:32<00:26, 6258.33it/s] 59%|█████▊    | 234965/400000 [00:33<00:25, 6508.88it/s] 59%|█████▉    | 235647/400000 [00:33<00:24, 6597.81it/s] 59%|█████▉    | 236344/400000 [00:33<00:24, 6701.76it/s] 59%|█████▉    | 237030/400000 [00:33<00:24, 6747.28it/s] 59%|█████▉    | 237750/400000 [00:33<00:23, 6876.72it/s] 60%|█████▉    | 238477/400000 [00:33<00:23, 6989.39it/s] 60%|█████▉    | 239243/400000 [00:33<00:22, 7177.65it/s] 60%|█████▉    | 239994/400000 [00:33<00:21, 7273.85it/s] 60%|██████    | 240781/400000 [00:33<00:21, 7442.82it/s] 60%|██████    | 241557/400000 [00:33<00:21, 7532.86it/s] 61%|██████    | 242313/400000 [00:34<00:21, 7448.07it/s] 61%|██████    | 243060/400000 [00:34<00:21, 7331.79it/s] 61%|██████    | 243849/400000 [00:34<00:20, 7488.72it/s] 61%|██████    | 244600/400000 [00:34<00:21, 7317.32it/s] 61%|██████▏   | 245340/400000 [00:34<00:21, 7341.00it/s] 62%|██████▏   | 246082/400000 [00:34<00:20, 7362.02it/s] 62%|██████▏   | 246833/400000 [00:34<00:20, 7404.40it/s] 62%|██████▏   | 247577/400000 [00:34<00:20, 7412.85it/s] 62%|██████▏   | 248345/400000 [00:34<00:20, 7489.11it/s] 62%|██████▏   | 249095/400000 [00:34<00:20, 7489.59it/s] 62%|██████▏   | 249856/400000 [00:35<00:19, 7524.05it/s] 63%|██████▎   | 250609/400000 [00:35<00:19, 7519.08it/s] 63%|██████▎   | 251362/400000 [00:35<00:19, 7472.70it/s] 63%|██████▎   | 252113/400000 [00:35<00:19, 7480.30it/s] 63%|██████▎   | 252875/400000 [00:35<00:19, 7519.89it/s] 63%|██████▎   | 253628/400000 [00:35<00:19, 7484.73it/s] 64%|██████▎   | 254377/400000 [00:35<00:19, 7355.65it/s] 64%|██████▍   | 255114/400000 [00:35<00:20, 7122.81it/s] 64%|██████▍   | 255829/400000 [00:35<00:20, 6942.41it/s] 64%|██████▍   | 256526/400000 [00:35<00:20, 6850.20it/s] 64%|██████▍   | 257214/400000 [00:36<00:20, 6856.57it/s] 64%|██████▍   | 257933/400000 [00:36<00:20, 6951.45it/s] 65%|██████▍   | 258651/400000 [00:36<00:20, 7018.38it/s] 65%|██████▍   | 259383/400000 [00:36<00:19, 7105.42it/s] 65%|██████▌   | 260095/400000 [00:36<00:19, 7068.66it/s] 65%|██████▌   | 260814/400000 [00:36<00:19, 7102.42it/s] 65%|██████▌   | 261543/400000 [00:36<00:19, 7156.00it/s] 66%|██████▌   | 262295/400000 [00:36<00:18, 7260.17it/s] 66%|██████▌   | 263033/400000 [00:36<00:18, 7292.45it/s] 66%|██████▌   | 263763/400000 [00:36<00:18, 7189.47it/s] 66%|██████▌   | 264483/400000 [00:37<00:19, 7075.63it/s] 66%|██████▋   | 265212/400000 [00:37<00:18, 7137.31it/s] 66%|██████▋   | 265934/400000 [00:37<00:18, 7159.73it/s] 67%|██████▋   | 266666/400000 [00:37<00:18, 7206.48it/s] 67%|██████▋   | 267405/400000 [00:37<00:18, 7258.98it/s] 67%|██████▋   | 268132/400000 [00:37<00:18, 7187.73it/s] 67%|██████▋   | 268852/400000 [00:37<00:18, 7009.65it/s] 67%|██████▋   | 269555/400000 [00:37<00:18, 7008.01it/s] 68%|██████▊   | 270257/400000 [00:37<00:18, 7006.26it/s] 68%|██████▊   | 271025/400000 [00:37<00:17, 7195.23it/s] 68%|██████▊   | 271752/400000 [00:38<00:17, 7215.92it/s] 68%|██████▊   | 272475/400000 [00:38<00:17, 7093.79it/s] 68%|██████▊   | 273261/400000 [00:38<00:17, 7306.10it/s] 68%|██████▊   | 274000/400000 [00:38<00:17, 7330.39it/s] 69%|██████▊   | 274774/400000 [00:38<00:16, 7446.49it/s] 69%|██████▉   | 275522/400000 [00:38<00:16, 7455.65it/s] 69%|██████▉   | 276269/400000 [00:38<00:16, 7341.72it/s] 69%|██████▉   | 277053/400000 [00:38<00:16, 7482.03it/s] 69%|██████▉   | 277803/400000 [00:38<00:16, 7372.47it/s] 70%|██████▉   | 278562/400000 [00:39<00:16, 7436.28it/s] 70%|██████▉   | 279309/400000 [00:39<00:16, 7444.21it/s] 70%|███████   | 280094/400000 [00:39<00:15, 7559.64it/s] 70%|███████   | 280851/400000 [00:39<00:15, 7450.90it/s] 70%|███████   | 281603/400000 [00:39<00:15, 7470.25it/s] 71%|███████   | 282370/400000 [00:39<00:15, 7528.33it/s] 71%|███████   | 283124/400000 [00:39<00:15, 7441.88it/s] 71%|███████   | 283907/400000 [00:39<00:15, 7553.94it/s] 71%|███████   | 284664/400000 [00:39<00:15, 7533.04it/s] 71%|███████▏  | 285418/400000 [00:39<00:15, 7439.92it/s] 72%|███████▏  | 286179/400000 [00:40<00:15, 7488.71it/s] 72%|███████▏  | 286934/400000 [00:40<00:15, 7505.88it/s] 72%|███████▏  | 287699/400000 [00:40<00:14, 7547.80it/s] 72%|███████▏  | 288455/400000 [00:40<00:14, 7547.49it/s] 72%|███████▏  | 289210/400000 [00:40<00:14, 7503.76it/s] 72%|███████▏  | 289966/400000 [00:40<00:14, 7519.18it/s] 73%|███████▎  | 290723/400000 [00:40<00:14, 7531.42it/s] 73%|███████▎  | 291477/400000 [00:40<00:14, 7498.11it/s] 73%|███████▎  | 292247/400000 [00:40<00:14, 7553.07it/s] 73%|███████▎  | 293003/400000 [00:40<00:14, 7554.89it/s] 73%|███████▎  | 293767/400000 [00:41<00:14, 7579.64it/s] 74%|███████▎  | 294526/400000 [00:41<00:14, 7456.34it/s] 74%|███████▍  | 295284/400000 [00:41<00:13, 7491.32it/s] 74%|███████▍  | 296035/400000 [00:41<00:13, 7496.53it/s] 74%|███████▍  | 296785/400000 [00:41<00:13, 7441.27it/s] 74%|███████▍  | 297530/400000 [00:41<00:13, 7373.92it/s] 75%|███████▍  | 298268/400000 [00:41<00:14, 7142.05it/s] 75%|███████▍  | 298985/400000 [00:41<00:14, 7123.62it/s] 75%|███████▍  | 299740/400000 [00:41<00:13, 7246.00it/s] 75%|███████▌  | 300498/400000 [00:41<00:13, 7341.91it/s] 75%|███████▌  | 301235/400000 [00:42<00:13, 7349.75it/s] 75%|███████▌  | 301975/400000 [00:42<00:13, 7364.65it/s] 76%|███████▌  | 302713/400000 [00:42<00:13, 7319.76it/s] 76%|███████▌  | 303452/400000 [00:42<00:13, 7340.11it/s] 76%|███████▌  | 304187/400000 [00:42<00:13, 7340.05it/s] 76%|███████▌  | 304922/400000 [00:42<00:13, 7261.88it/s] 76%|███████▋  | 305649/400000 [00:42<00:13, 7121.76it/s] 77%|███████▋  | 306363/400000 [00:42<00:13, 7101.21it/s] 77%|███████▋  | 307074/400000 [00:42<00:13, 7083.17it/s] 77%|███████▋  | 307830/400000 [00:42<00:12, 7218.44it/s] 77%|███████▋  | 308558/400000 [00:43<00:12, 7236.15it/s] 77%|███████▋  | 309295/400000 [00:43<00:12, 7273.96it/s] 78%|███████▊  | 310054/400000 [00:43<00:12, 7363.52it/s] 78%|███████▊  | 310791/400000 [00:43<00:12, 7317.10it/s] 78%|███████▊  | 311545/400000 [00:43<00:11, 7380.97it/s] 78%|███████▊  | 312300/400000 [00:43<00:11, 7429.19it/s] 78%|███████▊  | 313044/400000 [00:43<00:11, 7350.00it/s] 78%|███████▊  | 313780/400000 [00:43<00:11, 7297.63it/s] 79%|███████▊  | 314521/400000 [00:43<00:11, 7328.38it/s] 79%|███████▉  | 315257/400000 [00:43<00:11, 7335.31it/s] 79%|███████▉  | 316014/400000 [00:44<00:11, 7401.27it/s] 79%|███████▉  | 316755/400000 [00:44<00:11, 7379.58it/s] 79%|███████▉  | 317519/400000 [00:44<00:11, 7453.56it/s] 80%|███████▉  | 318268/400000 [00:44<00:10, 7463.02it/s] 80%|███████▉  | 319018/400000 [00:44<00:10, 7473.04it/s] 80%|███████▉  | 319766/400000 [00:44<00:10, 7463.02it/s] 80%|████████  | 320513/400000 [00:44<00:10, 7372.19it/s] 80%|████████  | 321272/400000 [00:44<00:10, 7434.41it/s] 81%|████████  | 322033/400000 [00:44<00:10, 7485.52it/s] 81%|████████  | 322782/400000 [00:44<00:10, 7472.42it/s] 81%|████████  | 323530/400000 [00:45<00:10, 7464.12it/s] 81%|████████  | 324277/400000 [00:45<00:10, 7435.59it/s] 81%|████████▏ | 325021/400000 [00:45<00:10, 7400.27it/s] 81%|████████▏ | 325762/400000 [00:45<00:10, 7361.88it/s] 82%|████████▏ | 326499/400000 [00:45<00:10, 7333.61it/s] 82%|████████▏ | 327233/400000 [00:45<00:09, 7292.31it/s] 82%|████████▏ | 327963/400000 [00:45<00:10, 7155.27it/s] 82%|████████▏ | 328680/400000 [00:45<00:09, 7134.34it/s] 82%|████████▏ | 329434/400000 [00:45<00:09, 7249.78it/s] 83%|████████▎ | 330180/400000 [00:45<00:09, 7309.53it/s] 83%|████████▎ | 330912/400000 [00:46<00:09, 7284.13it/s] 83%|████████▎ | 331641/400000 [00:46<00:09, 7272.81it/s] 83%|████████▎ | 332369/400000 [00:46<00:09, 7217.21it/s] 83%|████████▎ | 333096/400000 [00:46<00:09, 7230.44it/s] 83%|████████▎ | 333842/400000 [00:46<00:09, 7295.25it/s] 84%|████████▎ | 334611/400000 [00:46<00:08, 7407.62it/s] 84%|████████▍ | 335353/400000 [00:46<00:08, 7306.95it/s] 84%|████████▍ | 336085/400000 [00:46<00:08, 7258.88it/s] 84%|████████▍ | 336812/400000 [00:46<00:08, 7196.96it/s] 84%|████████▍ | 337573/400000 [00:47<00:08, 7314.51it/s] 85%|████████▍ | 338323/400000 [00:47<00:08, 7368.11it/s] 85%|████████▍ | 339070/400000 [00:47<00:08, 7395.52it/s] 85%|████████▍ | 339811/400000 [00:47<00:08, 7397.85it/s] 85%|████████▌ | 340552/400000 [00:47<00:08, 7296.52it/s] 85%|████████▌ | 341284/400000 [00:47<00:08, 7303.50it/s] 86%|████████▌ | 342020/400000 [00:47<00:07, 7318.62it/s] 86%|████████▌ | 342753/400000 [00:47<00:07, 7185.98it/s] 86%|████████▌ | 343473/400000 [00:47<00:07, 7168.68it/s] 86%|████████▌ | 344202/400000 [00:47<00:07, 7204.47it/s] 86%|████████▌ | 344953/400000 [00:48<00:07, 7292.46it/s] 86%|████████▋ | 345710/400000 [00:48<00:07, 7372.73it/s] 87%|████████▋ | 346469/400000 [00:48<00:07, 7435.01it/s] 87%|████████▋ | 347266/400000 [00:48<00:06, 7586.28it/s] 87%|████████▋ | 348041/400000 [00:48<00:06, 7632.33it/s] 87%|████████▋ | 348818/400000 [00:48<00:06, 7671.89it/s] 87%|████████▋ | 349604/400000 [00:48<00:06, 7724.37it/s] 88%|████████▊ | 350377/400000 [00:48<00:06, 7585.05it/s] 88%|████████▊ | 351159/400000 [00:48<00:06, 7652.23it/s] 88%|████████▊ | 351926/400000 [00:48<00:06, 7644.89it/s] 88%|████████▊ | 352692/400000 [00:49<00:06, 7614.32it/s] 88%|████████▊ | 353454/400000 [00:49<00:06, 7602.09it/s] 89%|████████▊ | 354215/400000 [00:49<00:06, 7601.71it/s] 89%|████████▊ | 354976/400000 [00:49<00:05, 7564.88it/s] 89%|████████▉ | 355733/400000 [00:49<00:05, 7545.09it/s] 89%|████████▉ | 356491/400000 [00:49<00:05, 7554.52it/s] 89%|████████▉ | 357269/400000 [00:49<00:05, 7618.75it/s] 90%|████████▉ | 358032/400000 [00:49<00:05, 7569.02it/s] 90%|████████▉ | 358792/400000 [00:49<00:05, 7575.22it/s] 90%|████████▉ | 359552/400000 [00:49<00:05, 7579.80it/s] 90%|█████████ | 360311/400000 [00:50<00:05, 7570.96it/s] 90%|█████████ | 361088/400000 [00:50<00:05, 7628.39it/s] 90%|█████████ | 361851/400000 [00:50<00:05, 7531.81it/s] 91%|█████████ | 362605/400000 [00:50<00:04, 7486.51it/s] 91%|█████████ | 363363/400000 [00:50<00:04, 7512.73it/s] 91%|█████████ | 364115/400000 [00:50<00:04, 7507.42it/s] 91%|█████████ | 364866/400000 [00:50<00:04, 7416.57it/s] 91%|█████████▏| 365613/400000 [00:50<00:04, 7429.96it/s] 92%|█████████▏| 366357/400000 [00:50<00:04, 7412.60it/s] 92%|█████████▏| 367099/400000 [00:50<00:04, 7381.28it/s] 92%|█████████▏| 367838/400000 [00:51<00:04, 7369.78it/s] 92%|█████████▏| 368580/400000 [00:51<00:04, 7382.85it/s] 92%|█████████▏| 369327/400000 [00:51<00:04, 7407.65it/s] 93%|█████████▎| 370068/400000 [00:51<00:04, 7392.09it/s] 93%|█████████▎| 370808/400000 [00:51<00:03, 7331.71it/s] 93%|█████████▎| 371542/400000 [00:51<00:03, 7298.23it/s] 93%|█████████▎| 372286/400000 [00:51<00:03, 7340.18it/s] 93%|█████████▎| 373021/400000 [00:51<00:03, 7309.17it/s] 93%|█████████▎| 373753/400000 [00:51<00:03, 7252.35it/s] 94%|█████████▎| 374486/400000 [00:51<00:03, 7274.13it/s] 94%|█████████▍| 375214/400000 [00:52<00:03, 7180.76it/s] 94%|█████████▍| 375982/400000 [00:52<00:03, 7323.41it/s] 94%|█████████▍| 376718/400000 [00:52<00:03, 7332.70it/s] 94%|█████████▍| 377467/400000 [00:52<00:03, 7378.86it/s] 95%|█████████▍| 378206/400000 [00:52<00:02, 7352.92it/s] 95%|█████████▍| 378942/400000 [00:52<00:02, 7265.65it/s] 95%|█████████▍| 379714/400000 [00:52<00:02, 7395.77it/s] 95%|█████████▌| 380471/400000 [00:52<00:02, 7444.97it/s] 95%|█████████▌| 381217/400000 [00:52<00:02, 7409.48it/s] 96%|█████████▌| 382003/400000 [00:52<00:02, 7538.01it/s] 96%|█████████▌| 382785/400000 [00:53<00:02, 7617.93it/s] 96%|█████████▌| 383548/400000 [00:53<00:02, 7566.99it/s] 96%|█████████▌| 384321/400000 [00:53<00:02, 7595.98it/s] 96%|█████████▋| 385087/400000 [00:53<00:01, 7613.14it/s] 96%|█████████▋| 385849/400000 [00:53<00:01, 7554.28it/s] 97%|█████████▋| 386632/400000 [00:53<00:01, 7632.58it/s] 97%|█████████▋| 387406/400000 [00:53<00:01, 7662.64it/s] 97%|█████████▋| 388175/400000 [00:53<00:01, 7669.91it/s] 97%|█████████▋| 388943/400000 [00:53<00:01, 7589.86it/s] 97%|█████████▋| 389703/400000 [00:53<00:01, 7457.33it/s] 98%|█████████▊| 390473/400000 [00:54<00:01, 7527.08it/s] 98%|█████████▊| 391247/400000 [00:54<00:01, 7589.07it/s] 98%|█████████▊| 392025/400000 [00:54<00:01, 7644.93it/s] 98%|█████████▊| 392796/400000 [00:54<00:00, 7663.68it/s] 98%|█████████▊| 393563/400000 [00:54<00:00, 7583.25it/s] 99%|█████████▊| 394327/400000 [00:54<00:00, 7599.28it/s] 99%|█████████▉| 395088/400000 [00:54<00:00, 7572.91it/s] 99%|█████████▉| 395846/400000 [00:54<00:00, 7539.83it/s] 99%|█████████▉| 396620/400000 [00:54<00:00, 7597.57it/s] 99%|█████████▉| 397408/400000 [00:54<00:00, 7679.88it/s]100%|█████████▉| 398188/400000 [00:55<00:00, 7714.68it/s]100%|█████████▉| 398980/400000 [00:55<00:00, 7773.17it/s]100%|█████████▉| 399758/400000 [00:55<00:00, 7703.16it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7228.57it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5298161a90>, <torchtext.data.dataset.TabularDataset object at 0x7f5298161be0>, <torchtext.vocab.Vocab object at 0x7f5298161b00>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.57 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.57 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.57 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.69 file/s]2020-07-07 06:20:00.447331: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-07 06:20:00.451998: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-07 06:20:00.452201: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1757ea8e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-07 06:20:00.452219: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144599.83it/s] 41%|████      | 4038656/9912422 [00:00<00:28, 206253.35it/s]9920512it [00:00, 34932915.10it/s]                           
0it [00:00, ?it/s]32768it [00:00, 742736.61it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1029901.68it/s]1654784it [00:00, 12434193.16it/s]                           
0it [00:00, ?it/s]8192it [00:00, 179659.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
