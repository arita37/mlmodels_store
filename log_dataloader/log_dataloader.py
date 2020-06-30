
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f30793baea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f30793baea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f30e46791e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f30e46791e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f31029c1e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f31029c1e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f30919a1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f30919a1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f30919a1488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144226.24it/s] 80%|████████  | 7979008/9912422 [00:00<00:09, 205877.46it/s]9920512it [00:00, 43181761.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 590023.76it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 490141.53it/s]1654784it [00:00, 11766315.55it/s]                         
0it [00:00, ?it/s]8192it [00:00, 201678.35it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3078b4e550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f308ee7c6a0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f30919a10d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f30919a10d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f30919a10d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:55,  2.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:55,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:54,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:53,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:53,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  4.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:37,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:37,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:36,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  4.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:25,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:25,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 10.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:00<00:05, 19.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 61.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 66.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 71.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 71.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 74.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 74.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 77.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 80.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 81.43 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 81.43 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.55 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ url]
0 examples [00:00, ? examples/s]2020-06-30 00:11:12.272075: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-30 00:11:12.354502: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-06-30 00:11:12.388108: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fc2ff22950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-30 00:11:12.388185: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.27 examples/s]97 examples [00:00,  4.66 examples/s]200 examples [00:00,  6.64 examples/s]305 examples [00:00,  9.46 examples/s]410 examples [00:00, 13.46 examples/s]515 examples [00:00, 19.13 examples/s]620 examples [00:00, 27.11 examples/s]725 examples [00:01, 38.31 examples/s]829 examples [00:01, 53.87 examples/s]935 examples [00:01, 75.31 examples/s]1036 examples [00:01, 104.16 examples/s]1136 examples [00:01, 142.31 examples/s]1242 examples [00:01, 192.17 examples/s]1343 examples [00:01, 253.29 examples/s]1448 examples [00:01, 327.91 examples/s]1552 examples [00:01, 412.17 examples/s]1654 examples [00:01, 497.08 examples/s]1755 examples [00:02, 580.21 examples/s]1857 examples [00:02, 665.61 examples/s]1957 examples [00:02, 728.80 examples/s]2061 examples [00:02, 799.18 examples/s]2168 examples [00:02, 864.29 examples/s]2274 examples [00:02, 913.84 examples/s]2382 examples [00:02, 957.37 examples/s]2487 examples [00:02, 979.43 examples/s]2592 examples [00:02, 993.74 examples/s]2696 examples [00:02, 999.90 examples/s]2801 examples [00:03, 1014.19 examples/s]2907 examples [00:03, 1025.12 examples/s]3012 examples [00:03, 1032.23 examples/s]3117 examples [00:03, 1006.57 examples/s]3222 examples [00:03, 1017.81 examples/s]3325 examples [00:03, 1006.10 examples/s]3427 examples [00:03, 999.82 examples/s] 3528 examples [00:03, 974.00 examples/s]3632 examples [00:03, 990.42 examples/s]3735 examples [00:03, 1000.71 examples/s]3840 examples [00:04, 1014.29 examples/s]3945 examples [00:04, 1024.69 examples/s]4050 examples [00:04, 1032.14 examples/s]4154 examples [00:04, 1012.45 examples/s]4256 examples [00:04, 998.58 examples/s] 4360 examples [00:04, 1009.95 examples/s]4467 examples [00:04, 1027.00 examples/s]4570 examples [00:04, 1016.39 examples/s]4673 examples [00:04, 1018.86 examples/s]4778 examples [00:04, 1026.65 examples/s]4881 examples [00:05, 1016.58 examples/s]4983 examples [00:05, 998.25 examples/s] 5083 examples [00:05, 994.40 examples/s]5188 examples [00:05, 1010.23 examples/s]5293 examples [00:05, 1019.90 examples/s]5400 examples [00:05, 1033.54 examples/s]5507 examples [00:05, 1041.40 examples/s]5612 examples [00:05, 1042.75 examples/s]5717 examples [00:05, 1041.57 examples/s]5823 examples [00:06, 1046.44 examples/s]5928 examples [00:06, 1043.77 examples/s]6033 examples [00:06, 1044.91 examples/s]6140 examples [00:06, 1049.82 examples/s]6246 examples [00:06, 1042.46 examples/s]6351 examples [00:06, 1041.31 examples/s]6457 examples [00:06, 1046.22 examples/s]6562 examples [00:06, 982.41 examples/s] 6662 examples [00:06, 970.00 examples/s]6760 examples [00:06, 945.87 examples/s]6864 examples [00:07, 971.97 examples/s]6970 examples [00:07, 996.69 examples/s]7076 examples [00:07, 1013.76 examples/s]7183 examples [00:07, 1029.02 examples/s]7291 examples [00:07, 1041.58 examples/s]7399 examples [00:07, 1050.72 examples/s]7506 examples [00:07, 1056.09 examples/s]7613 examples [00:07, 1058.41 examples/s]7719 examples [00:07, 1025.07 examples/s]7822 examples [00:07, 1001.79 examples/s]7925 examples [00:08, 1009.81 examples/s]8030 examples [00:08, 1019.69 examples/s]8136 examples [00:08, 1030.34 examples/s]8240 examples [00:08, 1026.45 examples/s]8343 examples [00:08, 976.57 examples/s] 8447 examples [00:08, 993.56 examples/s]8548 examples [00:08, 996.46 examples/s]8652 examples [00:08, 1007.01 examples/s]8758 examples [00:08, 1020.26 examples/s]8865 examples [00:09, 1032.81 examples/s]8972 examples [00:09, 1042.04 examples/s]9079 examples [00:09, 1049.01 examples/s]9187 examples [00:09, 1056.17 examples/s]9293 examples [00:09, 1041.50 examples/s]9398 examples [00:09, 1035.98 examples/s]9505 examples [00:09, 1043.13 examples/s]9612 examples [00:09, 1050.49 examples/s]9718 examples [00:09, 1039.94 examples/s]9823 examples [00:09, 1041.85 examples/s]9928 examples [00:10, 1031.62 examples/s]10032 examples [00:10, 944.40 examples/s]10129 examples [00:10, 950.59 examples/s]10235 examples [00:10, 979.48 examples/s]10337 examples [00:10, 989.82 examples/s]10444 examples [00:10, 1009.92 examples/s]10550 examples [00:10, 1023.54 examples/s]10657 examples [00:10, 1036.12 examples/s]10763 examples [00:10, 1043.01 examples/s]10868 examples [00:10, 1007.78 examples/s]10973 examples [00:11, 1017.55 examples/s]11080 examples [00:11, 1031.05 examples/s]11188 examples [00:11, 1042.58 examples/s]11294 examples [00:11, 1046.97 examples/s]11399 examples [00:11, 1035.24 examples/s]11504 examples [00:11, 1038.07 examples/s]11608 examples [00:11, 1023.01 examples/s]11715 examples [00:11, 1034.36 examples/s]11819 examples [00:11, 965.34 examples/s] 11923 examples [00:12, 985.82 examples/s]12028 examples [00:12, 1004.00 examples/s]12133 examples [00:12, 1015.03 examples/s]12237 examples [00:12, 1019.77 examples/s]12340 examples [00:12, 1017.85 examples/s]12446 examples [00:12, 1028.35 examples/s]12553 examples [00:12, 1040.47 examples/s]12661 examples [00:12, 1051.74 examples/s]12768 examples [00:12, 1057.08 examples/s]12876 examples [00:12, 1061.81 examples/s]12983 examples [00:13, 1058.35 examples/s]13090 examples [00:13, 1059.01 examples/s]13197 examples [00:13, 1061.46 examples/s]13305 examples [00:13, 1065.58 examples/s]13412 examples [00:13, 1050.28 examples/s]13518 examples [00:13, 1013.54 examples/s]13621 examples [00:13, 1017.92 examples/s]13724 examples [00:13, 992.62 examples/s] 13828 examples [00:13, 1005.04 examples/s]13931 examples [00:13, 1010.36 examples/s]14033 examples [00:14, 991.60 examples/s] 14139 examples [00:14, 1009.23 examples/s]14242 examples [00:14, 1013.28 examples/s]14347 examples [00:14, 1023.75 examples/s]14452 examples [00:14, 1031.21 examples/s]14556 examples [00:14, 1024.12 examples/s]14663 examples [00:14, 1035.34 examples/s]14767 examples [00:14, 1009.77 examples/s]14872 examples [00:14, 1021.15 examples/s]14977 examples [00:14, 1028.00 examples/s]15080 examples [00:15, 1008.21 examples/s]15181 examples [00:15, 1005.00 examples/s]15287 examples [00:15, 1019.12 examples/s]15393 examples [00:15, 1030.89 examples/s]15497 examples [00:15, 1012.06 examples/s]15599 examples [00:15, 973.89 examples/s] 15705 examples [00:15, 996.06 examples/s]15812 examples [00:15, 1016.86 examples/s]15918 examples [00:15, 1027.36 examples/s]16022 examples [00:16, 1018.81 examples/s]16127 examples [00:16, 1025.29 examples/s]16232 examples [00:16, 1030.29 examples/s]16338 examples [00:16, 1036.17 examples/s]16443 examples [00:16, 1040.24 examples/s]16548 examples [00:16, 1033.20 examples/s]16652 examples [00:16, 1011.26 examples/s]16754 examples [00:16, 974.98 examples/s] 16859 examples [00:16, 993.79 examples/s]16959 examples [00:16, 993.60 examples/s]17059 examples [00:17, 920.16 examples/s]17153 examples [00:17, 863.54 examples/s]17253 examples [00:17, 898.44 examples/s]17349 examples [00:17, 914.39 examples/s]17448 examples [00:17, 935.66 examples/s]17545 examples [00:17, 944.04 examples/s]17646 examples [00:17, 960.56 examples/s]17743 examples [00:17, 942.77 examples/s]17838 examples [00:17, 939.28 examples/s]17935 examples [00:17, 948.12 examples/s]18038 examples [00:18, 968.83 examples/s]18138 examples [00:18, 977.22 examples/s]18241 examples [00:18, 992.08 examples/s]18345 examples [00:18, 1005.00 examples/s]18446 examples [00:18, 953.21 examples/s] 18548 examples [00:18, 971.39 examples/s]18652 examples [00:18, 989.33 examples/s]18754 examples [00:18, 998.25 examples/s]18859 examples [00:18, 1012.04 examples/s]18963 examples [00:19, 1019.89 examples/s]19066 examples [00:19, 1022.35 examples/s]19169 examples [00:19, 1024.06 examples/s]19275 examples [00:19, 1032.12 examples/s]19379 examples [00:19, 1031.47 examples/s]19483 examples [00:19, 1027.46 examples/s]19586 examples [00:19, 1023.32 examples/s]19689 examples [00:19, 1017.25 examples/s]19791 examples [00:19, 990.96 examples/s] 19891 examples [00:19, 978.09 examples/s]19994 examples [00:20, 988.61 examples/s]20093 examples [00:20, 938.14 examples/s]20197 examples [00:20, 965.15 examples/s]20300 examples [00:20, 983.13 examples/s]20405 examples [00:20, 1000.89 examples/s]20509 examples [00:20, 1009.56 examples/s]20612 examples [00:20, 1013.86 examples/s]20717 examples [00:20, 1023.50 examples/s]20820 examples [00:20, 1017.68 examples/s]20925 examples [00:20, 1026.29 examples/s]21028 examples [00:21, 1026.31 examples/s]21131 examples [00:21, 1022.59 examples/s]21236 examples [00:21, 1029.08 examples/s]21341 examples [00:21, 1033.84 examples/s]21445 examples [00:21, 1000.74 examples/s]21546 examples [00:21, 983.48 examples/s] 21645 examples [00:21, 980.62 examples/s]21747 examples [00:21, 990.85 examples/s]21848 examples [00:21, 995.71 examples/s]21954 examples [00:21, 1011.70 examples/s]22057 examples [00:22, 1016.73 examples/s]22162 examples [00:22, 1024.69 examples/s]22268 examples [00:22, 1032.35 examples/s]22373 examples [00:22, 1036.80 examples/s]22477 examples [00:22, 1036.06 examples/s]22581 examples [00:22, 1018.47 examples/s]22683 examples [00:22, 1010.27 examples/s]22785 examples [00:22, 936.43 examples/s] 22880 examples [00:22, 881.78 examples/s]22970 examples [00:23, 826.16 examples/s]23055 examples [00:23, 829.78 examples/s]23159 examples [00:23, 882.13 examples/s]23261 examples [00:23, 918.42 examples/s]23366 examples [00:23, 953.91 examples/s]23465 examples [00:23, 956.52 examples/s]23565 examples [00:23, 966.92 examples/s]23663 examples [00:23, 954.18 examples/s]23762 examples [00:23, 961.85 examples/s]23859 examples [00:23, 958.82 examples/s]23956 examples [00:24, 958.81 examples/s]24058 examples [00:24, 974.18 examples/s]24156 examples [00:24, 969.39 examples/s]24254 examples [00:24, 921.19 examples/s]24347 examples [00:24, 922.73 examples/s]24445 examples [00:24, 937.69 examples/s]24540 examples [00:24, 937.13 examples/s]24634 examples [00:24, 914.89 examples/s]24732 examples [00:24, 932.09 examples/s]24833 examples [00:25, 952.20 examples/s]24934 examples [00:25, 966.77 examples/s]25034 examples [00:25, 975.94 examples/s]25132 examples [00:25, 961.46 examples/s]25235 examples [00:25, 979.94 examples/s]25337 examples [00:25, 990.31 examples/s]25438 examples [00:25, 994.30 examples/s]25541 examples [00:25, 1003.06 examples/s]25642 examples [00:25, 1001.19 examples/s]25747 examples [00:25, 1013.83 examples/s]25849 examples [00:26, 1008.48 examples/s]25950 examples [00:26, 993.29 examples/s] 26050 examples [00:26, 938.33 examples/s]26145 examples [00:26, 924.30 examples/s]26239 examples [00:26, 927.41 examples/s]26334 examples [00:26, 933.65 examples/s]26433 examples [00:26, 949.35 examples/s]26537 examples [00:26, 973.54 examples/s]26642 examples [00:26, 993.69 examples/s]26742 examples [00:26, 937.05 examples/s]26837 examples [00:27, 927.80 examples/s]26937 examples [00:27, 948.20 examples/s]27040 examples [00:27, 970.92 examples/s]27138 examples [00:27, 971.14 examples/s]27236 examples [00:27, 916.04 examples/s]27337 examples [00:27, 939.19 examples/s]27432 examples [00:27, 888.03 examples/s]27526 examples [00:27, 902.27 examples/s]27625 examples [00:27, 925.78 examples/s]27719 examples [00:28, 922.56 examples/s]27812 examples [00:28, 883.76 examples/s]27904 examples [00:28, 892.73 examples/s]28004 examples [00:28, 922.28 examples/s]28103 examples [00:28, 940.31 examples/s]28198 examples [00:28, 911.03 examples/s]28292 examples [00:28, 917.75 examples/s]28387 examples [00:28, 927.00 examples/s]28486 examples [00:28, 942.53 examples/s]28589 examples [00:28, 965.58 examples/s]28692 examples [00:29, 983.19 examples/s]28796 examples [00:29, 997.97 examples/s]28897 examples [00:29, 922.99 examples/s]29004 examples [00:29, 960.62 examples/s]29108 examples [00:29, 982.70 examples/s]29208 examples [00:29, 975.03 examples/s]29314 examples [00:29, 998.58 examples/s]29415 examples [00:29, 933.29 examples/s]29522 examples [00:29, 968.76 examples/s]29628 examples [00:30, 992.70 examples/s]29733 examples [00:30, 1007.13 examples/s]29839 examples [00:30, 1020.17 examples/s]29942 examples [00:30, 1021.24 examples/s]30045 examples [00:30, 962.53 examples/s] 30152 examples [00:30, 990.34 examples/s]30256 examples [00:30, 1003.95 examples/s]30362 examples [00:30, 1019.38 examples/s]30468 examples [00:30, 1029.54 examples/s]30574 examples [00:30, 1037.38 examples/s]30679 examples [00:31, 1035.89 examples/s]30783 examples [00:31, 1014.12 examples/s]30885 examples [00:31, 1001.42 examples/s]30986 examples [00:31, 1002.84 examples/s]31088 examples [00:31, 1006.70 examples/s]31189 examples [00:31, 971.35 examples/s] 31287 examples [00:31, 944.52 examples/s]31391 examples [00:31, 969.82 examples/s]31494 examples [00:31, 984.59 examples/s]31596 examples [00:32, 992.83 examples/s]31696 examples [00:32, 975.71 examples/s]31800 examples [00:32, 992.99 examples/s]31900 examples [00:32, 951.15 examples/s]32000 examples [00:32, 963.39 examples/s]32100 examples [00:32, 973.48 examples/s]32204 examples [00:32, 990.91 examples/s]32307 examples [00:32, 1002.24 examples/s]32408 examples [00:32, 963.11 examples/s] 32512 examples [00:32, 982.43 examples/s]32616 examples [00:33, 998.29 examples/s]32719 examples [00:33, 1007.58 examples/s]32823 examples [00:33, 1016.65 examples/s]32925 examples [00:33, 1005.02 examples/s]33026 examples [00:33, 996.86 examples/s] 33130 examples [00:33, 1008.35 examples/s]33231 examples [00:33, 1000.70 examples/s]33332 examples [00:33, 1001.15 examples/s]33436 examples [00:33, 1011.11 examples/s]33538 examples [00:33, 988.61 examples/s] 33644 examples [00:34, 1006.57 examples/s]33745 examples [00:34, 1006.63 examples/s]33851 examples [00:34, 1020.95 examples/s]33955 examples [00:34, 1025.65 examples/s]34059 examples [00:34, 1027.55 examples/s]34162 examples [00:34, 1019.95 examples/s]34266 examples [00:34, 1025.81 examples/s]34370 examples [00:34, 1029.22 examples/s]34473 examples [00:34, 941.24 examples/s] 34577 examples [00:35, 966.75 examples/s]34680 examples [00:35, 983.35 examples/s]34783 examples [00:35, 996.75 examples/s]34887 examples [00:35, 1008.24 examples/s]34989 examples [00:35, 977.81 examples/s] 35093 examples [00:35, 995.29 examples/s]35197 examples [00:35, 1007.80 examples/s]35301 examples [00:35, 1016.98 examples/s]35405 examples [00:35, 1022.26 examples/s]35508 examples [00:35, 1021.74 examples/s]35611 examples [00:36, 988.19 examples/s] 35713 examples [00:36, 995.55 examples/s]35816 examples [00:36, 1003.86 examples/s]35917 examples [00:36, 1004.64 examples/s]36020 examples [00:36, 1009.76 examples/s]36122 examples [00:36, 998.90 examples/s] 36225 examples [00:36, 1007.09 examples/s]36326 examples [00:36, 976.10 examples/s] 36424 examples [00:36, 947.49 examples/s]36522 examples [00:36, 954.30 examples/s]36618 examples [00:37, 900.83 examples/s]36721 examples [00:37, 935.10 examples/s]36825 examples [00:37, 963.56 examples/s]36926 examples [00:37, 976.11 examples/s]37027 examples [00:37, 984.46 examples/s]37127 examples [00:37, 987.75 examples/s]37227 examples [00:37, 990.17 examples/s]37329 examples [00:37, 997.77 examples/s]37429 examples [00:37, 969.04 examples/s]37527 examples [00:38, 952.41 examples/s]37623 examples [00:38, 919.34 examples/s]37718 examples [00:38, 928.16 examples/s]37821 examples [00:38, 955.23 examples/s]37917 examples [00:38, 921.60 examples/s]38010 examples [00:38, 919.97 examples/s]38116 examples [00:38, 955.98 examples/s]38220 examples [00:38, 977.43 examples/s]38325 examples [00:38, 995.71 examples/s]38427 examples [00:38, 1002.51 examples/s]38532 examples [00:39, 1014.52 examples/s]38634 examples [00:39, 1014.59 examples/s]38736 examples [00:39, 1011.78 examples/s]38844 examples [00:39, 1028.72 examples/s]38948 examples [00:39, 988.49 examples/s] 39048 examples [00:39, 975.89 examples/s]39153 examples [00:39, 995.86 examples/s]39258 examples [00:39, 1010.75 examples/s]39360 examples [00:39, 1008.01 examples/s]39464 examples [00:39, 1015.56 examples/s]39568 examples [00:40, 1020.83 examples/s]39673 examples [00:40, 1027.73 examples/s]39778 examples [00:40, 1033.77 examples/s]39882 examples [00:40, 1033.62 examples/s]39986 examples [00:40, 1034.61 examples/s]40090 examples [00:40, 957.36 examples/s] 40194 examples [00:40, 979.00 examples/s]40299 examples [00:40, 998.55 examples/s]40403 examples [00:40, 1009.66 examples/s]40505 examples [00:40, 1002.43 examples/s]40606 examples [00:41, 960.17 examples/s] 40710 examples [00:41, 981.80 examples/s]40810 examples [00:41, 986.77 examples/s]40910 examples [00:41, 987.47 examples/s]41010 examples [00:41, 967.70 examples/s]41113 examples [00:41, 984.85 examples/s]41217 examples [00:41, 1000.60 examples/s]41320 examples [00:41, 1008.94 examples/s]41424 examples [00:41, 1017.79 examples/s]41526 examples [00:42, 1012.95 examples/s]41628 examples [00:42, 1004.55 examples/s]41730 examples [00:42, 1007.21 examples/s]41834 examples [00:42, 1014.10 examples/s]41936 examples [00:42, 1008.38 examples/s]42037 examples [00:42, 994.83 examples/s] 42138 examples [00:42, 998.18 examples/s]42238 examples [00:42, 970.76 examples/s]42341 examples [00:42, 986.09 examples/s]42444 examples [00:42, 996.13 examples/s]42545 examples [00:43, 1000.10 examples/s]42647 examples [00:43, 1005.88 examples/s]42751 examples [00:43, 1015.56 examples/s]42855 examples [00:43, 1020.91 examples/s]42959 examples [00:43, 1023.70 examples/s]43062 examples [00:43, 1010.25 examples/s]43166 examples [00:43, 1016.20 examples/s]43268 examples [00:43, 992.21 examples/s] 43373 examples [00:43, 1007.32 examples/s]43476 examples [00:43, 1012.30 examples/s]43578 examples [00:44, 1008.61 examples/s]43679 examples [00:44, 1007.44 examples/s]43780 examples [00:44, 995.82 examples/s] 43880 examples [00:44, 948.53 examples/s]43976 examples [00:44, 934.95 examples/s]44072 examples [00:44, 939.81 examples/s]44176 examples [00:44, 967.47 examples/s]44276 examples [00:44, 974.68 examples/s]44377 examples [00:44, 984.69 examples/s]44484 examples [00:44, 1005.26 examples/s]44585 examples [00:45, 1001.70 examples/s]44689 examples [00:45, 1011.61 examples/s]44793 examples [00:45, 1019.70 examples/s]44897 examples [00:45, 1024.40 examples/s]45000 examples [00:45, 1016.99 examples/s]45102 examples [00:45, 991.89 examples/s] 45203 examples [00:45, 996.34 examples/s]45308 examples [00:45, 1009.19 examples/s]45410 examples [00:45, 965.68 examples/s] 45513 examples [00:46, 981.67 examples/s]45612 examples [00:46, 982.77 examples/s]45716 examples [00:46, 996.87 examples/s]45819 examples [00:46, 1004.33 examples/s]45922 examples [00:46, 1009.40 examples/s]46024 examples [00:46, 1005.65 examples/s]46125 examples [00:46, 1005.96 examples/s]46227 examples [00:46, 1008.53 examples/s]46334 examples [00:46, 1023.33 examples/s]46440 examples [00:46, 1032.90 examples/s]46545 examples [00:47, 1037.53 examples/s]46649 examples [00:47, 1031.50 examples/s]46754 examples [00:47, 1036.04 examples/s]46860 examples [00:47, 1040.51 examples/s]46965 examples [00:47, 1030.14 examples/s]47069 examples [00:47, 958.75 examples/s] 47168 examples [00:47, 966.52 examples/s]47273 examples [00:47, 989.41 examples/s]47379 examples [00:47, 1007.75 examples/s]47481 examples [00:47, 1003.59 examples/s]47584 examples [00:48, 1009.30 examples/s]47686 examples [00:48, 1001.52 examples/s]47791 examples [00:48, 1013.47 examples/s]47896 examples [00:48, 1023.41 examples/s]47999 examples [00:48, 1021.51 examples/s]48102 examples [00:48, 1011.43 examples/s]48204 examples [00:48, 1010.52 examples/s]48308 examples [00:48, 1018.70 examples/s]48412 examples [00:48, 1024.40 examples/s]48516 examples [00:48, 1027.31 examples/s]48619 examples [00:49, 1027.84 examples/s]48722 examples [00:49, 990.73 examples/s] 48828 examples [00:49, 1008.13 examples/s]48934 examples [00:49, 1022.25 examples/s]49037 examples [00:49, 1016.66 examples/s]49141 examples [00:49, 1023.17 examples/s]49244 examples [00:49, 1021.74 examples/s]49348 examples [00:49, 1026.93 examples/s]49451 examples [00:49, 1013.82 examples/s]49555 examples [00:50, 1020.60 examples/s]49659 examples [00:50, 1024.68 examples/s]49762 examples [00:50, 1022.19 examples/s]49865 examples [00:50, 1024.32 examples/s]49969 examples [00:50, 1027.16 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6562/50000 [00:00<00:00, 65616.78 examples/s] 39%|███▉      | 19376/50000 [00:00<00:00, 76867.51 examples/s] 63%|██████▎   | 31424/50000 [00:00<00:00, 86229.72 examples/s] 88%|████████▊ | 44091/50000 [00:00<00:00, 95362.69 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 857.54 examples/s]193 examples [00:00, 909.83 examples/s]299 examples [00:00, 949.50 examples/s]407 examples [00:00, 984.64 examples/s]516 examples [00:00, 1012.56 examples/s]618 examples [00:00, 1014.03 examples/s]715 examples [00:00, 999.16 examples/s] 817 examples [00:00, 1005.01 examples/s]923 examples [00:00, 1018.29 examples/s]1028 examples [00:01, 1026.99 examples/s]1136 examples [00:01, 1040.32 examples/s]1244 examples [00:01, 1049.78 examples/s]1351 examples [00:01, 1052.78 examples/s]1459 examples [00:01, 1056.86 examples/s]1565 examples [00:01, 1027.40 examples/s]1668 examples [00:01, 1024.42 examples/s]1776 examples [00:01, 1039.44 examples/s]1882 examples [00:01, 1045.23 examples/s]1989 examples [00:01, 1051.93 examples/s]2098 examples [00:02, 1060.64 examples/s]2206 examples [00:02, 1064.49 examples/s]2314 examples [00:02, 1068.01 examples/s]2421 examples [00:02, 1047.94 examples/s]2527 examples [00:02, 1049.07 examples/s]2633 examples [00:02, 1050.12 examples/s]2740 examples [00:02, 1053.72 examples/s]2846 examples [00:02, 1012.36 examples/s]2950 examples [00:02, 1017.86 examples/s]3056 examples [00:02, 1028.05 examples/s]3162 examples [00:03, 1036.96 examples/s]3266 examples [00:03, 1028.52 examples/s]3369 examples [00:03, 1006.20 examples/s]3474 examples [00:03, 1015.51 examples/s]3581 examples [00:03, 1029.65 examples/s]3688 examples [00:03, 1040.41 examples/s]3793 examples [00:03, 1011.03 examples/s]3897 examples [00:03, 1017.77 examples/s]4004 examples [00:03, 1031.50 examples/s]4108 examples [00:03, 1031.04 examples/s]4213 examples [00:04, 1034.33 examples/s]4317 examples [00:04, 1028.45 examples/s]4423 examples [00:04, 1036.92 examples/s]4529 examples [00:04, 1041.45 examples/s]4636 examples [00:04, 1049.08 examples/s]4743 examples [00:04, 1053.77 examples/s]4849 examples [00:04, 1047.11 examples/s]4954 examples [00:04, 1033.10 examples/s]5059 examples [00:04, 1037.11 examples/s]5165 examples [00:04, 1043.15 examples/s]5271 examples [00:05, 1048.03 examples/s]5376 examples [00:05, 1047.51 examples/s]5482 examples [00:05, 1049.56 examples/s]5587 examples [00:05, 1049.07 examples/s]5692 examples [00:05, 1039.95 examples/s]5797 examples [00:05, 1025.66 examples/s]5903 examples [00:05, 1033.03 examples/s]6007 examples [00:05, 995.83 examples/s] 6111 examples [00:05, 1008.27 examples/s]6217 examples [00:06, 1021.31 examples/s]6323 examples [00:06, 1032.05 examples/s]6428 examples [00:06, 1034.64 examples/s]6532 examples [00:06, 1025.66 examples/s]6635 examples [00:06, 1010.56 examples/s]6741 examples [00:06, 1023.02 examples/s]6844 examples [00:06, 990.48 examples/s] 6949 examples [00:06, 1004.50 examples/s]7055 examples [00:06, 1020.11 examples/s]7158 examples [00:06, 1002.84 examples/s]7266 examples [00:07, 1022.65 examples/s]7369 examples [00:07, 979.71 examples/s] 7468 examples [00:07, 982.45 examples/s]7567 examples [00:07, 980.99 examples/s]7666 examples [00:07, 981.73 examples/s]7765 examples [00:07, 977.91 examples/s]7863 examples [00:07, 956.89 examples/s]7960 examples [00:07, 958.88 examples/s]8064 examples [00:07, 979.94 examples/s]8163 examples [00:07, 968.49 examples/s]8268 examples [00:08, 989.10 examples/s]8375 examples [00:08, 1009.76 examples/s]8477 examples [00:08, 1012.41 examples/s]8579 examples [00:08, 1005.85 examples/s]8684 examples [00:08, 1017.65 examples/s]8789 examples [00:08, 1025.02 examples/s]8892 examples [00:08, 1016.02 examples/s]8999 examples [00:08, 1029.21 examples/s]9103 examples [00:08, 1009.69 examples/s]9205 examples [00:08, 1004.39 examples/s]9313 examples [00:09, 1023.83 examples/s]9421 examples [00:09, 1037.92 examples/s]9529 examples [00:09, 1048.91 examples/s]9636 examples [00:09, 1053.02 examples/s]9742 examples [00:09, 1038.96 examples/s]9847 examples [00:09, 1016.29 examples/s]9953 examples [00:09, 1027.38 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUY8M7O/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUY8M7O/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f30919a1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f30919a1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f30919a1488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3078b4e550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f301ae09048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f30919a1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f30919a1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f30919a1488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f301ae09048>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f308ee7c0f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f30180de158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f30180de158> 

  function with postional parmater data_info <function split_train_valid at 0x7f30180de158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f30180de268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f30180de268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f30180de268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=09443b8c6f12cb5e7870907205c3d3b7a8573613ba2740866b6de977e9b7cdba
  Stored in directory: /tmp/pip-ephem-wheel-cache-cq2k_f_m/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:49:10, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:12:15, 14.8kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:23:18, 21.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 885k/862M [00:01<7:58:47, 30.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.63M/862M [00:01<5:34:43, 42.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.45M/862M [00:01<3:53:22, 61.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.82M/862M [00:01<2:42:50, 87.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:01<1:53:17, 125kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:19:11, 178kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<55:09, 253kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.5M/862M [00:01<38:30, 361kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.0M/862M [00:01<26:54, 514kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.1M/862M [00:02<18:50, 730kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:02<13:12, 1.04MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.8M/862M [00:02<09:17, 1.46MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.2M/862M [00:02<06:33, 2.06MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<04:47, 2.82MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<03:50, 3.51MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<12:20:43, 18.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<8:39:07, 25.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:04<6:02:38, 37.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<4:18:16, 51.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<3:03:27, 72.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<2:08:53, 104kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:06<1:30:04, 148kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:15:54, 175kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<54:29, 244kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:08<38:21, 346kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<29:53, 443kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<23:36, 561kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<17:04, 774kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:10<12:04, 1.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<17:12, 765kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<13:22, 984kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<09:39, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<09:49, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<09:31, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:18, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<07:13, 1.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<07:47, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:07, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:16<04:24, 2.94MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<1:26:29, 150kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<1:01:51, 210kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<43:32, 297kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<33:25, 386kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<26:07, 494kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<18:56, 680kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:20<13:20, 961kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<1:11:09, 180kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<51:06, 251kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<36:02, 355kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<28:07, 454kB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<20:57, 608kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<14:55, 853kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<13:27, 944kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:41, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:47, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<08:25, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:27, 1.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:32, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:36, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:55, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:23, 2.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:01, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<06:46, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:22, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:45, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:17, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<03:58, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:41, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:03, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:42, 3.31MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<08:06, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:56, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:09, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<06:28, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:48, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<04:21, 2.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<05:53, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:34, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:13, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:47, 3.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<1:29:11, 135kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<1:03:36, 190kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<44:42, 269kB/s]  .vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<34:01, 353kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<25:01, 480kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<17:44, 675kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<13:13, 903kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<8:16:49, 24.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<5:48:02, 34.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<4:03:00, 48.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<2:55:05, 67.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<2:05:01, 94.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<1:28:03, 135kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<1:01:31, 192kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<1:12:40, 162kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<52:02, 227kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<36:38, 321kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<28:20, 414kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<21:00, 558kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<14:55, 785kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<13:10, 886kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<11:35, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:39, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<06:09, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<33:59, 341kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<24:57, 464kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<17:43, 652kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<15:05, 765kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<11:43, 983kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:26, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<08:35, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<08:22, 1.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<06:20, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<04:38, 2.46MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:01, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<06:05, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:29, 2.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<05:48, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:21, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:55, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:35, 3.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<08:08, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:40, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:56, 2.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:05, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:25, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:04, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<06:02, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:42, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:24, 3.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<10:21, 1.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<08:24, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:06, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:51, 1.60MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<07:00, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:27, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:34, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:01, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:47, 2.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<05:10, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:43, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:34, 3.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:03, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:36, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:28, 3.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<04:57, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:38, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:24, 2.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:16, 3.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:55, 1.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:13, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<03:54, 2.71MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:13, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:48, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:35, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:19, 3.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<1:29:19, 118kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<1:03:33, 165kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<44:36, 235kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<31:52, 328kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<7:28:11, 23.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<5:14:01, 33.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<3:39:00, 47.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<2:40:03, 64.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<1:54:12, 90.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:20:17, 129kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<56:17, 184kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<42:11, 244kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<30:35, 337kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<21:37, 475kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<17:29, 585kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<14:19, 715kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:31, 971kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:59, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<07:19, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<05:20, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:06, 1.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:19, 1.59MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:30, 2.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<12:50, 781kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:01, 1.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:15, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:24, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:16, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:36, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<04:00, 2.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<38:39, 256kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<28:03, 352kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<19:50, 497kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<16:09, 608kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:17, 739kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:42, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<06:55, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<08:37, 1.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:01, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:06, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:51, 1.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:46, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:54, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:24, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:19, 2.88MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:35, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:08, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:04, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:01, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:01, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:19, 2.18MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<03:58, 2.37MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:00, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:55, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:54, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:50, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<03:51, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<02:48, 3.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<1:07:45, 136kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<48:20, 190kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<33:58, 270kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<25:50, 354kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<18:58, 481kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<13:28, 675kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:32, 785kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:54, 915kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<07:23, 1.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<05:14, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<17:32, 513kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<13:11, 682kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<09:26, 950kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:40, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<07:52, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:58, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<6:12:03, 23.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<4:20:38, 34.0kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<3:01:38, 48.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<2:12:51, 66.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<1:34:46, 92.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<1:06:40, 132kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<47:49, 183kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<34:22, 254kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<24:12, 359kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<18:51, 459kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<14:57, 579kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<10:53, 794kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:58, 958kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<07:08, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:12, 1.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:37, 1.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:42, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:10, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:31, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:36, 1.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:44, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<02:44, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:14, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:49, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:32, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<02:35, 3.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<32:10, 259kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<24:26, 340kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<17:47, 467kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<12:41, 653kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<10:41, 771kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<08:19, 990kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:03, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<04:20, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<33:27, 244kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<25:27, 321kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<18:45, 436kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<13:15, 615kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<09:21, 866kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<1:03:37, 127kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<45:00, 180kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<31:37, 255kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<23:58, 335kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<17:44, 453kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<12:35, 636kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<10:39, 748kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<08:16, 963kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:58, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<06:02, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:01, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:40, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:24, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<03:53, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:55, 2.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:52, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:30, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:38, 2.92MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:39, 2.11MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:17, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:23, 3.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<56:04, 136kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<39:59, 191kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<28:04, 271kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<21:20, 354kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<16:27, 459kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<11:51, 637kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<08:22, 897kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<08:58, 835kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:02, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:05, 1.46MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:17, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:17, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:54, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:50, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:51, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:32, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:11, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:25, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:27, 2.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:44, 2.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<4:52:46, 24.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<3:25:01, 35.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<2:22:41, 50.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<1:45:25, 67.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<1:15:25, 94.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<53:05, 135kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<36:59, 192kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<32:29, 218kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<23:26, 302kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<16:31, 427kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<13:10, 533kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:55, 707kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<07:05, 985kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:35, 1.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:01, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:33, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<03:14, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<19:59, 344kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<14:15, 482kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<09:59, 682kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<25:15, 270kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<18:22, 371kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<12:58, 522kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<10:37, 635kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<08:06, 831kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<05:47, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:38, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:14, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:57, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<22:30, 294kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<16:25, 402kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<11:36, 566kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<09:37, 679kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<07:23, 884kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:19, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<05:14, 1.24MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:58, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:47, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:41, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 479M/862M [03:26<03:13, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:24, 2.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:09, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:29, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:42, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<01:58, 3.17MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:08, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:32, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:37, 2.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:16, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:32, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:47, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:55, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:20, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:39, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<01:54, 3.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<40:09, 151kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<28:41, 211kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<20:09, 299kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<15:26, 388kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<12:00, 499kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<08:38, 691kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<06:06, 972kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<06:34, 900kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:12, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:45, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:59, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:58, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:02, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:10, 2.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:12, 704kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<06:19, 913kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:56, 1.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:37, 1.58MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:40, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:10, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<4:10:28, 22.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<2:55:17, 32.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<2:01:47, 45.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:29:06, 62.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:03:28, 87.9kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<44:35, 125kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<31:49, 173kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<22:44, 242kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<15:57, 344kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<11:11, 487kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<21:23, 255kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<16:06, 338kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<11:31, 471kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<08:03, 667kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<44:07, 122kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<31:18, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<21:58, 244kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<15:20, 346kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<42:23, 125kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<30:43, 173kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<21:43, 244kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<15:07, 347kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<53:55, 97.2kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<38:13, 137kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<26:45, 195kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<19:48, 261kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<14:22, 359kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<10:08, 507kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<08:15, 618kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<06:17, 810kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:30, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:19, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:03, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:04, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:56, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:54, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:28, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:08, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:16, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:05, 2.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:34, 3.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:31, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<01:57, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:26, 3.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:05, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:54, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:37, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:19, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:44, 2.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:13, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:29, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:58, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:25, 3.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<37:39, 119kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<26:46, 167kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<18:44, 238kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<14:02, 314kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<10:40, 413kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<07:40, 573kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<06:01, 722kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:39, 933kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:21, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:19, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:11, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:24, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:43, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:14, 992kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:23, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<02:27, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:40, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:17, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:40, 2.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:25, 2.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<2:59:48, 22.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<2:05:43, 32.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<1:27:10, 46.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<1:02:51, 63.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<44:48, 89.4kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<31:28, 127kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<21:48, 181kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<43:27, 90.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<30:46, 128kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<21:30, 182kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<15:49, 245kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<11:51, 327kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<08:27, 456kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<05:53, 646kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<3:43:45, 17.0kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<2:36:44, 24.2kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<1:49:04, 34.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<1:16:29, 48.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<53:50, 69.3kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<37:31, 98.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<26:52, 137kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<19:31, 188kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<13:47, 265kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<09:35, 377kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<08:57, 402kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<06:37, 542kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<04:41, 759kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<04:04, 865kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<03:32, 997kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:38, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:23, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<02:01, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:28, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:50, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:35, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:11, 2.82MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:36, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:05, 3.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:32, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:22, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:27, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:41, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:20, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<00:58, 3.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<22:59, 136kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<16:23, 190kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<11:27, 269kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<08:38, 353kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<06:42, 454kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<04:50, 628kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:48, 783kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:58, 1.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:08, 1.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:09, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:18, 2.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:34, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:23, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:01, 2.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:21, 2.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:14, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<00:55, 2.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:26, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:08, 2.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:48, 3.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<06:04, 434kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<04:30, 583kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<03:11, 816kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:48, 915kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:13, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:35, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:14, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<1:53:01, 22.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<1:18:52, 31.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<54:10, 45.1kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<40:24, 60.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<28:44, 84.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<20:07, 120kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<13:52, 171kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<11:19, 209kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<08:08, 290kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<05:42, 410kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<04:31, 509kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<03:14, 706kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:34, 868kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:01, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:27, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:16, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:55, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:08, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:13, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:56, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:40, 3.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:20, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:08, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:50, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:07, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:52, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<00:36, 3.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<22:31, 83.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<15:54, 118kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<11:01, 168kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<07:59, 228kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<05:57, 305kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<04:13, 427kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<02:53, 605kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<1:42:19, 17.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<1:11:32, 24.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<49:22, 34.8kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<34:14, 49.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<24:15, 69.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<16:55, 98.5kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<11:32, 140kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<10:01, 161kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<07:07, 226kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<04:57, 320kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:45, 412kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:55, 526kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:06, 723kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:27, 1.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<02:22, 621kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:48, 813kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:16, 1.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:12, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:07, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:50, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:35, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<06:02, 222kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<04:20, 307kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<03:01, 434kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<02:21, 541kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:46, 715kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:14, 996kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:07, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:01, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:46, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:42, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:36, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:26, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:32, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:29, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:21, 2.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:28, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:24, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:19, 2.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<39:52, 23.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<27:37, 33.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<18:23, 47.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<13:18, 64.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<09:27, 91.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<06:33, 129kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<04:19, 184kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<08:42, 91.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<06:07, 129kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<04:10, 183kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<02:57, 246kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<02:07, 340kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<01:26, 479kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:06, 590kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:54, 720kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:39, 982kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:26, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:50, 706kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:38, 914kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:26, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:24, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:14, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 2.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<22:15, 17.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<15:21, 24.6kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<10:01, 35.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<06:22, 49.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<04:23, 70.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<02:49, 100kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<01:47, 138kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<01:16, 190kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:51, 268kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:29, 360kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:21, 489kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:12, 686kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:08, 797kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 925kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.03MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:50:27,  6.23it/s]  0%|          | 849/400000 [00:00<12:27:58,  8.89it/s]  0%|          | 1644/400000 [00:00<8:42:47, 12.70it/s]  1%|          | 2501/400000 [00:00<6:05:23, 18.13it/s]  1%|          | 3325/400000 [00:00<4:15:29, 25.88it/s]  1%|          | 4155/400000 [00:00<2:58:42, 36.92it/s]  1%|▏         | 5011/400000 [00:00<2:05:03, 52.64it/s]  1%|▏         | 5864/400000 [00:00<1:27:34, 75.00it/s]  2%|▏         | 6728/400000 [00:00<1:01:23, 106.75it/s]  2%|▏         | 7565/400000 [00:01<43:07, 151.67it/s]    2%|▏         | 8415/400000 [00:01<30:21, 215.03it/s]  2%|▏         | 9278/400000 [00:01<21:25, 303.94it/s]  3%|▎         | 10130/400000 [00:01<15:11, 427.66it/s]  3%|▎         | 10970/400000 [00:01<10:50, 597.83it/s]  3%|▎         | 11823/400000 [00:01<07:48, 829.12it/s]  3%|▎         | 12666/400000 [00:01<05:41, 1135.79it/s]  3%|▎         | 13508/400000 [00:01<04:11, 1533.85it/s]  4%|▎         | 14348/400000 [00:01<03:11, 2012.90it/s]  4%|▍         | 15184/400000 [00:01<02:27, 2606.41it/s]  4%|▍         | 16027/400000 [00:02<01:56, 3287.45it/s]  4%|▍         | 16859/400000 [00:02<01:35, 4016.12it/s]  4%|▍         | 17720/400000 [00:02<01:19, 4781.23it/s]  5%|▍         | 18558/400000 [00:02<01:09, 5462.93it/s]  5%|▍         | 19413/400000 [00:02<01:02, 6125.53it/s]  5%|▌         | 20252/400000 [00:02<00:59, 6411.85it/s]  5%|▌         | 21105/400000 [00:02<00:54, 6926.93it/s]  5%|▌         | 21923/400000 [00:02<00:52, 7226.91it/s]  6%|▌         | 22753/400000 [00:02<00:50, 7516.29it/s]  6%|▌         | 23602/400000 [00:02<00:48, 7783.34it/s]  6%|▌         | 24460/400000 [00:03<00:46, 8004.83it/s]  6%|▋         | 25319/400000 [00:03<00:45, 8171.46it/s]  7%|▋         | 26180/400000 [00:03<00:45, 8296.36it/s]  7%|▋         | 27039/400000 [00:03<00:44, 8382.29it/s]  7%|▋         | 27891/400000 [00:03<00:44, 8312.12it/s]  7%|▋         | 28732/400000 [00:03<00:46, 8054.40it/s]  7%|▋         | 29580/400000 [00:03<00:45, 8175.17it/s]  8%|▊         | 30441/400000 [00:03<00:44, 8300.85it/s]  8%|▊         | 31276/400000 [00:03<00:44, 8240.41it/s]  8%|▊         | 32104/400000 [00:04<00:45, 8058.15it/s]  8%|▊         | 32914/400000 [00:04<00:46, 7928.90it/s]  8%|▊         | 33710/400000 [00:04<00:46, 7899.38it/s]  9%|▊         | 34507/400000 [00:04<00:46, 7919.84it/s]  9%|▉         | 35304/400000 [00:04<00:46, 7907.10it/s]  9%|▉         | 36096/400000 [00:04<00:48, 7565.55it/s]  9%|▉         | 36923/400000 [00:04<00:46, 7763.32it/s]  9%|▉         | 37768/400000 [00:04<00:45, 7956.87it/s] 10%|▉         | 38585/400000 [00:04<00:45, 8018.96it/s] 10%|▉         | 39394/400000 [00:04<00:44, 8039.88it/s] 10%|█         | 40201/400000 [00:05<00:46, 7810.24it/s] 10%|█         | 40985/400000 [00:05<00:48, 7439.65it/s] 10%|█         | 41807/400000 [00:05<00:46, 7656.41it/s] 11%|█         | 42579/400000 [00:05<00:47, 7596.63it/s] 11%|█         | 43436/400000 [00:05<00:45, 7863.16it/s] 11%|█         | 44297/400000 [00:05<00:44, 8072.06it/s] 11%|█▏        | 45146/400000 [00:05<00:43, 8192.85it/s] 12%|█▏        | 46001/400000 [00:05<00:42, 8296.46it/s] 12%|█▏        | 46861/400000 [00:05<00:42, 8383.10it/s] 12%|█▏        | 47715/400000 [00:05<00:41, 8428.03it/s] 12%|█▏        | 48578/400000 [00:06<00:41, 8487.61it/s] 12%|█▏        | 49429/400000 [00:06<00:42, 8264.26it/s] 13%|█▎        | 50265/400000 [00:06<00:42, 8290.21it/s] 13%|█▎        | 51130/400000 [00:06<00:41, 8393.10it/s] 13%|█▎        | 51971/400000 [00:06<00:42, 8171.98it/s] 13%|█▎        | 52831/400000 [00:06<00:41, 8294.25it/s] 13%|█▎        | 53685/400000 [00:06<00:41, 8364.79it/s] 14%|█▎        | 54524/400000 [00:06<00:41, 8271.07it/s] 14%|█▍        | 55387/400000 [00:06<00:41, 8374.55it/s] 14%|█▍        | 56253/400000 [00:07<00:40, 8455.71it/s] 14%|█▍        | 57115/400000 [00:07<00:40, 8502.98it/s] 14%|█▍        | 57967/400000 [00:07<00:40, 8506.66it/s] 15%|█▍        | 58819/400000 [00:07<00:40, 8419.82it/s] 15%|█▍        | 59682/400000 [00:07<00:40, 8479.47it/s] 15%|█▌        | 60544/400000 [00:07<00:39, 8518.88it/s] 15%|█▌        | 61407/400000 [00:07<00:39, 8550.83it/s] 16%|█▌        | 62264/400000 [00:07<00:39, 8556.27it/s] 16%|█▌        | 63120/400000 [00:07<00:39, 8429.66it/s] 16%|█▌        | 63964/400000 [00:07<00:39, 8427.96it/s] 16%|█▌        | 64823/400000 [00:08<00:39, 8474.80it/s] 16%|█▋        | 65684/400000 [00:08<00:39, 8513.08it/s] 17%|█▋        | 66549/400000 [00:08<00:38, 8551.76it/s] 17%|█▋        | 67405/400000 [00:08<00:39, 8464.37it/s] 17%|█▋        | 68258/400000 [00:08<00:39, 8482.49it/s] 17%|█▋        | 69118/400000 [00:08<00:38, 8514.77it/s] 17%|█▋        | 69982/400000 [00:08<00:38, 8550.17it/s] 18%|█▊        | 70846/400000 [00:08<00:38, 8574.58it/s] 18%|█▊        | 71704/400000 [00:08<00:39, 8377.96it/s] 18%|█▊        | 72563/400000 [00:08<00:38, 8437.00it/s] 18%|█▊        | 73414/400000 [00:09<00:38, 8457.29it/s] 19%|█▊        | 74279/400000 [00:09<00:38, 8511.72it/s] 19%|█▉        | 75145/400000 [00:09<00:37, 8553.33it/s] 19%|█▉        | 76001/400000 [00:09<00:38, 8343.50it/s] 19%|█▉        | 76842/400000 [00:09<00:38, 8361.94it/s] 19%|█▉        | 77684/400000 [00:09<00:38, 8379.19it/s] 20%|█▉        | 78546/400000 [00:09<00:38, 8447.60it/s] 20%|█▉        | 79392/400000 [00:09<00:39, 8203.07it/s] 20%|██        | 80215/400000 [00:09<00:41, 7786.18it/s] 20%|██        | 81073/400000 [00:09<00:39, 8006.99it/s] 20%|██        | 81929/400000 [00:10<00:38, 8163.37it/s] 21%|██        | 82787/400000 [00:10<00:38, 8283.10it/s] 21%|██        | 83638/400000 [00:10<00:37, 8349.40it/s] 21%|██        | 84476/400000 [00:10<00:37, 8310.21it/s] 21%|██▏       | 85329/400000 [00:10<00:37, 8372.43it/s] 22%|██▏       | 86168/400000 [00:10<00:37, 8324.95it/s] 22%|██▏       | 87032/400000 [00:10<00:37, 8414.49it/s] 22%|██▏       | 87887/400000 [00:10<00:36, 8454.06it/s] 22%|██▏       | 88734/400000 [00:10<00:37, 8298.85it/s] 22%|██▏       | 89573/400000 [00:10<00:37, 8323.83it/s] 23%|██▎       | 90430/400000 [00:11<00:36, 8395.84it/s] 23%|██▎       | 91287/400000 [00:11<00:36, 8444.44it/s] 23%|██▎       | 92133/400000 [00:11<00:36, 8446.15it/s] 23%|██▎       | 92979/400000 [00:11<00:36, 8439.36it/s] 23%|██▎       | 93837/400000 [00:11<00:36, 8480.23it/s] 24%|██▎       | 94698/400000 [00:11<00:35, 8516.34it/s] 24%|██▍       | 95559/400000 [00:11<00:35, 8541.69it/s] 24%|██▍       | 96414/400000 [00:11<00:35, 8533.42it/s] 24%|██▍       | 97268/400000 [00:11<00:35, 8466.80it/s] 25%|██▍       | 98130/400000 [00:11<00:35, 8511.84it/s] 25%|██▍       | 98990/400000 [00:12<00:35, 8537.76it/s] 25%|██▍       | 99844/400000 [00:12<00:35, 8531.32it/s] 25%|██▌       | 100705/400000 [00:12<00:34, 8553.90it/s] 25%|██▌       | 101561/400000 [00:12<00:35, 8450.86it/s] 26%|██▌       | 102423/400000 [00:12<00:35, 8498.00it/s] 26%|██▌       | 103278/400000 [00:12<00:34, 8511.79it/s] 26%|██▌       | 104140/400000 [00:12<00:34, 8542.24it/s] 26%|██▌       | 104995/400000 [00:12<00:34, 8523.41it/s] 26%|██▋       | 105848/400000 [00:12<00:35, 8241.35it/s] 27%|██▋       | 106709/400000 [00:12<00:35, 8348.31it/s] 27%|██▋       | 107570/400000 [00:13<00:34, 8424.00it/s] 27%|██▋       | 108432/400000 [00:13<00:34, 8479.09it/s] 27%|██▋       | 109296/400000 [00:13<00:34, 8525.97it/s] 28%|██▊       | 110150/400000 [00:13<00:34, 8490.31it/s] 28%|██▊       | 111011/400000 [00:13<00:33, 8525.38it/s] 28%|██▊       | 111864/400000 [00:13<00:33, 8510.18it/s] 28%|██▊       | 112716/400000 [00:13<00:34, 8446.68it/s] 28%|██▊       | 113582/400000 [00:13<00:33, 8507.11it/s] 29%|██▊       | 114434/400000 [00:13<00:34, 8230.06it/s] 29%|██▉       | 115297/400000 [00:14<00:34, 8344.91it/s] 29%|██▉       | 116156/400000 [00:14<00:33, 8415.78it/s] 29%|██▉       | 117000/400000 [00:14<00:33, 8383.58it/s] 29%|██▉       | 117863/400000 [00:14<00:33, 8455.15it/s] 30%|██▉       | 118710/400000 [00:14<00:33, 8391.12it/s] 30%|██▉       | 119571/400000 [00:14<00:33, 8452.94it/s] 30%|███       | 120433/400000 [00:14<00:32, 8500.63it/s] 30%|███       | 121294/400000 [00:14<00:32, 8533.10it/s] 31%|███       | 122150/400000 [00:14<00:32, 8539.63it/s] 31%|███       | 123005/400000 [00:14<00:32, 8511.18it/s] 31%|███       | 123861/400000 [00:15<00:32, 8525.59it/s] 31%|███       | 124716/400000 [00:15<00:32, 8531.54it/s] 31%|███▏      | 125577/400000 [00:15<00:32, 8552.91it/s] 32%|███▏      | 126436/400000 [00:15<00:31, 8562.10it/s] 32%|███▏      | 127293/400000 [00:15<00:32, 8516.00it/s] 32%|███▏      | 128151/400000 [00:15<00:31, 8533.18it/s] 32%|███▏      | 129005/400000 [00:15<00:32, 8406.90it/s] 32%|███▏      | 129856/400000 [00:15<00:32, 8437.24it/s] 33%|███▎      | 130718/400000 [00:15<00:31, 8488.63it/s] 33%|███▎      | 131568/400000 [00:15<00:31, 8454.06it/s] 33%|███▎      | 132424/400000 [00:16<00:31, 8482.89it/s] 33%|███▎      | 133278/400000 [00:16<00:31, 8497.00it/s] 34%|███▎      | 134138/400000 [00:16<00:31, 8525.63it/s] 34%|███▎      | 134991/400000 [00:16<00:31, 8445.09it/s] 34%|███▍      | 135836/400000 [00:16<00:31, 8334.91it/s] 34%|███▍      | 136698/400000 [00:16<00:31, 8417.30it/s] 34%|███▍      | 137547/400000 [00:16<00:31, 8438.96it/s] 35%|███▍      | 138409/400000 [00:16<00:30, 8491.59it/s] 35%|███▍      | 139270/400000 [00:16<00:30, 8524.44it/s] 35%|███▌      | 140123/400000 [00:16<00:30, 8451.95it/s] 35%|███▌      | 140969/400000 [00:17<00:30, 8426.98it/s] 35%|███▌      | 141832/400000 [00:17<00:30, 8484.59it/s] 36%|███▌      | 142694/400000 [00:17<00:30, 8524.75it/s] 36%|███▌      | 143557/400000 [00:17<00:29, 8554.62it/s] 36%|███▌      | 144413/400000 [00:17<00:30, 8508.53it/s] 36%|███▋      | 145270/400000 [00:17<00:29, 8526.87it/s] 37%|███▋      | 146123/400000 [00:17<00:30, 8461.44it/s] 37%|███▋      | 146970/400000 [00:17<00:29, 8437.40it/s] 37%|███▋      | 147814/400000 [00:17<00:30, 8226.66it/s] 37%|███▋      | 148638/400000 [00:17<00:30, 8221.35it/s] 37%|███▋      | 149462/400000 [00:18<00:31, 7919.07it/s] 38%|███▊      | 150264/400000 [00:18<00:31, 7947.92it/s] 38%|███▊      | 151108/400000 [00:18<00:30, 8087.00it/s] 38%|███▊      | 151919/400000 [00:18<00:31, 7778.23it/s] 38%|███▊      | 152779/400000 [00:18<00:30, 8007.06it/s] 38%|███▊      | 153629/400000 [00:18<00:30, 8147.45it/s] 39%|███▊      | 154488/400000 [00:18<00:29, 8275.14it/s] 39%|███▉      | 155349/400000 [00:18<00:29, 8372.55it/s] 39%|███▉      | 156189/400000 [00:18<00:29, 8375.33it/s] 39%|███▉      | 157035/400000 [00:18<00:28, 8399.18it/s] 39%|███▉      | 157878/400000 [00:19<00:28, 8405.80it/s] 40%|███▉      | 158734/400000 [00:19<00:28, 8451.44it/s] 40%|███▉      | 159597/400000 [00:19<00:28, 8502.47it/s] 40%|████      | 160456/400000 [00:19<00:28, 8527.23it/s] 40%|████      | 161310/400000 [00:19<00:28, 8510.76it/s] 41%|████      | 162162/400000 [00:19<00:28, 8415.86it/s] 41%|████      | 163016/400000 [00:19<00:28, 8452.53it/s] 41%|████      | 163879/400000 [00:19<00:27, 8504.02it/s] 41%|████      | 164730/400000 [00:19<00:27, 8493.25it/s] 41%|████▏     | 165580/400000 [00:19<00:27, 8459.65it/s] 42%|████▏     | 166427/400000 [00:20<00:27, 8446.14it/s] 42%|████▏     | 167272/400000 [00:20<00:27, 8373.41it/s] 42%|████▏     | 168132/400000 [00:20<00:27, 8439.39it/s] 42%|████▏     | 168994/400000 [00:20<00:27, 8490.10it/s] 42%|████▏     | 169844/400000 [00:20<00:27, 8482.70it/s] 43%|████▎     | 170693/400000 [00:20<00:27, 8460.80it/s] 43%|████▎     | 171540/400000 [00:20<00:27, 8386.05it/s] 43%|████▎     | 172406/400000 [00:20<00:26, 8465.06it/s] 43%|████▎     | 173269/400000 [00:20<00:26, 8513.41it/s] 44%|████▎     | 174131/400000 [00:20<00:26, 8542.44it/s] 44%|████▎     | 174987/400000 [00:21<00:26, 8547.05it/s] 44%|████▍     | 175842/400000 [00:21<00:26, 8317.39it/s] 44%|████▍     | 176676/400000 [00:21<00:26, 8287.78it/s] 44%|████▍     | 177537/400000 [00:21<00:26, 8381.51it/s] 45%|████▍     | 178396/400000 [00:21<00:26, 8440.27it/s] 45%|████▍     | 179241/400000 [00:21<00:26, 8231.56it/s] 45%|████▌     | 180066/400000 [00:21<00:27, 7894.06it/s] 45%|████▌     | 180860/400000 [00:21<00:28, 7823.36it/s] 45%|████▌     | 181718/400000 [00:21<00:27, 8034.41it/s] 46%|████▌     | 182563/400000 [00:22<00:26, 8154.39it/s] 46%|████▌     | 183415/400000 [00:22<00:26, 8258.58it/s] 46%|████▌     | 184274/400000 [00:22<00:25, 8352.49it/s] 46%|████▋     | 185137/400000 [00:22<00:25, 8432.46it/s] 46%|████▋     | 185982/400000 [00:22<00:25, 8420.17it/s] 47%|████▋     | 186843/400000 [00:22<00:25, 8473.78it/s] 47%|████▋     | 187695/400000 [00:22<00:25, 8486.70it/s] 47%|████▋     | 188559/400000 [00:22<00:24, 8529.99it/s] 47%|████▋     | 189413/400000 [00:22<00:24, 8532.94it/s] 48%|████▊     | 190267/400000 [00:22<00:25, 8338.11it/s] 48%|████▊     | 191103/400000 [00:23<00:25, 8241.05it/s] 48%|████▊     | 191929/400000 [00:23<00:25, 8191.43it/s] 48%|████▊     | 192788/400000 [00:23<00:24, 8305.56it/s] 48%|████▊     | 193646/400000 [00:23<00:24, 8384.86it/s] 49%|████▊     | 194486/400000 [00:23<00:24, 8341.41it/s] 49%|████▉     | 195330/400000 [00:23<00:24, 8370.04it/s] 49%|████▉     | 196177/400000 [00:23<00:24, 8399.29it/s] 49%|████▉     | 197040/400000 [00:23<00:23, 8467.09it/s] 49%|████▉     | 197901/400000 [00:23<00:23, 8508.93it/s] 50%|████▉     | 198764/400000 [00:23<00:23, 8543.47it/s] 50%|████▉     | 199619/400000 [00:24<00:23, 8542.23it/s] 50%|█████     | 200474/400000 [00:24<00:23, 8525.26it/s] 50%|█████     | 201338/400000 [00:24<00:23, 8556.94it/s] 51%|█████     | 202200/400000 [00:24<00:23, 8573.93it/s] 51%|█████     | 203062/400000 [00:24<00:22, 8585.82it/s] 51%|█████     | 203922/400000 [00:24<00:22, 8587.58it/s] 51%|█████     | 204781/400000 [00:24<00:22, 8499.62it/s] 51%|█████▏    | 205643/400000 [00:24<00:22, 8532.88it/s] 52%|█████▏    | 206497/400000 [00:24<00:22, 8528.82it/s] 52%|█████▏    | 207351/400000 [00:24<00:23, 8338.18it/s] 52%|█████▏    | 208197/400000 [00:25<00:22, 8374.31it/s] 52%|█████▏    | 209036/400000 [00:25<00:22, 8345.24it/s] 52%|█████▏    | 209889/400000 [00:25<00:22, 8399.76it/s] 53%|█████▎    | 210749/400000 [00:25<00:22, 8457.73it/s] 53%|█████▎    | 211596/400000 [00:25<00:22, 8296.56it/s] 53%|█████▎    | 212449/400000 [00:25<00:22, 8363.44it/s] 53%|█████▎    | 213312/400000 [00:25<00:22, 8440.16it/s] 54%|█████▎    | 214176/400000 [00:25<00:21, 8498.52it/s] 54%|█████▍    | 215031/400000 [00:25<00:21, 8513.62it/s] 54%|█████▍    | 215889/400000 [00:25<00:21, 8533.29it/s] 54%|█████▍    | 216743/400000 [00:26<00:21, 8506.13it/s] 54%|█████▍    | 217594/400000 [00:26<00:21, 8503.04it/s] 55%|█████▍    | 218445/400000 [00:26<00:21, 8467.02it/s] 55%|█████▍    | 219305/400000 [00:26<00:21, 8505.00it/s] 55%|█████▌    | 220166/400000 [00:26<00:21, 8533.78it/s] 55%|█████▌    | 221020/400000 [00:26<00:21, 8517.39it/s] 55%|█████▌    | 221872/400000 [00:26<00:20, 8494.48it/s] 56%|█████▌    | 222733/400000 [00:26<00:20, 8528.36it/s] 56%|█████▌    | 223595/400000 [00:26<00:20, 8554.40it/s] 56%|█████▌    | 224451/400000 [00:26<00:20, 8491.11it/s] 56%|█████▋    | 225307/400000 [00:27<00:20, 8509.46it/s] 57%|█████▋    | 226159/400000 [00:27<00:20, 8457.03it/s] 57%|█████▋    | 227021/400000 [00:27<00:20, 8504.13it/s] 57%|█████▋    | 227872/400000 [00:27<00:20, 8431.53it/s] 57%|█████▋    | 228732/400000 [00:27<00:20, 8479.79it/s] 57%|█████▋    | 229586/400000 [00:27<00:20, 8496.78it/s] 58%|█████▊    | 230436/400000 [00:27<00:19, 8488.16it/s] 58%|█████▊    | 231285/400000 [00:27<00:20, 8413.25it/s] 58%|█████▊    | 232127/400000 [00:27<00:20, 8360.51it/s] 58%|█████▊    | 232974/400000 [00:27<00:19, 8391.16it/s] 58%|█████▊    | 233821/400000 [00:28<00:19, 8413.90it/s] 59%|█████▊    | 234663/400000 [00:28<00:19, 8358.11it/s] 59%|█████▉    | 235500/400000 [00:28<00:19, 8352.04it/s] 59%|█████▉    | 236363/400000 [00:28<00:19, 8432.63it/s] 59%|█████▉    | 237207/400000 [00:28<00:19, 8404.98it/s] 60%|█████▉    | 238065/400000 [00:28<00:19, 8455.94it/s] 60%|█████▉    | 238911/400000 [00:28<00:19, 8363.35it/s] 60%|█████▉    | 239773/400000 [00:28<00:18, 8436.31it/s] 60%|██████    | 240618/400000 [00:28<00:18, 8388.80it/s] 60%|██████    | 241477/400000 [00:28<00:18, 8445.99it/s] 61%|██████    | 242335/400000 [00:29<00:18, 8485.58it/s] 61%|██████    | 243184/400000 [00:29<00:18, 8450.55it/s] 61%|██████    | 244045/400000 [00:29<00:18, 8496.32it/s] 61%|██████    | 244908/400000 [00:29<00:18, 8534.46it/s] 61%|██████▏   | 245762/400000 [00:29<00:18, 8476.48it/s] 62%|██████▏   | 246610/400000 [00:29<00:18, 8309.45it/s] 62%|██████▏   | 247442/400000 [00:29<00:18, 8265.74it/s] 62%|██████▏   | 248289/400000 [00:29<00:18, 8324.57it/s] 62%|██████▏   | 249123/400000 [00:29<00:18, 8313.35it/s] 62%|██████▏   | 249977/400000 [00:30<00:17, 8379.98it/s] 63%|██████▎   | 250835/400000 [00:30<00:17, 8437.89it/s] 63%|██████▎   | 251680/400000 [00:30<00:17, 8314.07it/s] 63%|██████▎   | 252520/400000 [00:30<00:17, 8339.30it/s] 63%|██████▎   | 253377/400000 [00:30<00:17, 8406.21it/s] 64%|██████▎   | 254219/400000 [00:30<00:17, 8393.75it/s] 64%|██████▍   | 255064/400000 [00:30<00:17, 8409.24it/s] 64%|██████▍   | 255906/400000 [00:30<00:17, 8401.32it/s] 64%|██████▍   | 256766/400000 [00:30<00:16, 8459.90it/s] 64%|██████▍   | 257627/400000 [00:30<00:16, 8503.77it/s] 65%|██████▍   | 258489/400000 [00:31<00:16, 8535.39it/s] 65%|██████▍   | 259343/400000 [00:31<00:16, 8527.95it/s] 65%|██████▌   | 260196/400000 [00:31<00:16, 8506.53it/s] 65%|██████▌   | 261054/400000 [00:31<00:16, 8527.73it/s] 65%|██████▌   | 261913/400000 [00:31<00:16, 8544.84it/s] 66%|██████▌   | 262773/400000 [00:31<00:16, 8558.98it/s] 66%|██████▌   | 263629/400000 [00:31<00:15, 8551.82it/s] 66%|██████▌   | 264485/400000 [00:31<00:15, 8551.87it/s] 66%|██████▋   | 265349/400000 [00:31<00:15, 8575.99it/s] 67%|██████▋   | 266207/400000 [00:31<00:15, 8576.55it/s] 67%|██████▋   | 267065/400000 [00:32<00:15, 8577.09it/s] 67%|██████▋   | 267923/400000 [00:32<00:15, 8546.08it/s] 67%|██████▋   | 268778/400000 [00:32<00:15, 8424.90it/s] 67%|██████▋   | 269637/400000 [00:32<00:15, 8471.31it/s] 68%|██████▊   | 270494/400000 [00:32<00:15, 8497.93it/s] 68%|██████▊   | 271353/400000 [00:32<00:15, 8523.46it/s] 68%|██████▊   | 272206/400000 [00:32<00:15, 8513.32it/s] 68%|██████▊   | 273058/400000 [00:32<00:15, 8421.28it/s] 68%|██████▊   | 273914/400000 [00:32<00:14, 8460.03it/s] 69%|██████▊   | 274772/400000 [00:32<00:14, 8495.49it/s] 69%|██████▉   | 275631/400000 [00:33<00:14, 8522.44it/s] 69%|██████▉   | 276486/400000 [00:33<00:14, 8529.50it/s] 69%|██████▉   | 277340/400000 [00:33<00:14, 8521.61it/s] 70%|██████▉   | 278200/400000 [00:33<00:14, 8544.01it/s] 70%|██████▉   | 279055/400000 [00:33<00:14, 8540.28it/s] 70%|██████▉   | 279910/400000 [00:33<00:14, 8529.12it/s] 70%|███████   | 280763/400000 [00:33<00:13, 8526.38it/s] 70%|███████   | 281616/400000 [00:33<00:13, 8497.49it/s] 71%|███████   | 282476/400000 [00:33<00:13, 8526.42it/s] 71%|███████   | 283335/400000 [00:33<00:13, 8545.23it/s] 71%|███████   | 284195/400000 [00:34<00:13, 8561.38it/s] 71%|███████▏  | 285052/400000 [00:34<00:13, 8519.07it/s] 71%|███████▏  | 285904/400000 [00:34<00:13, 8469.32it/s] 72%|███████▏  | 286765/400000 [00:34<00:13, 8510.76it/s] 72%|███████▏  | 287617/400000 [00:34<00:13, 8469.79it/s] 72%|███████▏  | 288476/400000 [00:34<00:13, 8504.61it/s] 72%|███████▏  | 289327/400000 [00:34<00:13, 8445.28it/s] 73%|███████▎  | 290175/400000 [00:34<00:12, 8453.52it/s] 73%|███████▎  | 291032/400000 [00:34<00:12, 8485.88it/s] 73%|███████▎  | 291887/400000 [00:34<00:12, 8502.67it/s] 73%|███████▎  | 292748/400000 [00:35<00:12, 8533.55it/s] 73%|███████▎  | 293609/400000 [00:35<00:12, 8554.62it/s] 74%|███████▎  | 294465/400000 [00:35<00:12, 8528.84it/s] 74%|███████▍  | 295327/400000 [00:35<00:12, 8554.86it/s] 74%|███████▍  | 296184/400000 [00:35<00:12, 8557.83it/s] 74%|███████▍  | 297040/400000 [00:35<00:12, 8541.80it/s] 74%|███████▍  | 297895/400000 [00:35<00:11, 8521.73it/s] 75%|███████▍  | 298748/400000 [00:35<00:11, 8466.28it/s] 75%|███████▍  | 299595/400000 [00:35<00:11, 8455.33it/s] 75%|███████▌  | 300449/400000 [00:35<00:11, 8478.45it/s] 75%|███████▌  | 301305/400000 [00:36<00:11, 8502.05it/s] 76%|███████▌  | 302156/400000 [00:36<00:11, 8422.59it/s] 76%|███████▌  | 303000/400000 [00:36<00:11, 8426.78it/s] 76%|███████▌  | 303851/400000 [00:36<00:11, 8451.37it/s] 76%|███████▌  | 304710/400000 [00:36<00:11, 8490.98it/s] 76%|███████▋  | 305574/400000 [00:36<00:11, 8533.39it/s] 77%|███████▋  | 306439/400000 [00:36<00:10, 8567.30it/s] 77%|███████▋  | 307296/400000 [00:36<00:11, 8365.79it/s] 77%|███████▋  | 308134/400000 [00:36<00:11, 8335.12it/s] 77%|███████▋  | 308992/400000 [00:36<00:10, 8407.03it/s] 77%|███████▋  | 309853/400000 [00:37<00:10, 8465.93it/s] 78%|███████▊  | 310713/400000 [00:37<00:10, 8504.86it/s] 78%|███████▊  | 311564/400000 [00:37<00:10, 8486.07it/s] 78%|███████▊  | 312424/400000 [00:37<00:10, 8518.12it/s] 78%|███████▊  | 313277/400000 [00:37<00:10, 8509.57it/s] 79%|███████▊  | 314135/400000 [00:37<00:10, 8530.56it/s] 79%|███████▊  | 314995/400000 [00:37<00:09, 8549.92it/s] 79%|███████▉  | 315851/400000 [00:37<00:09, 8517.72it/s] 79%|███████▉  | 316703/400000 [00:37<00:09, 8366.81it/s] 79%|███████▉  | 317561/400000 [00:37<00:09, 8429.03it/s] 80%|███████▉  | 318420/400000 [00:38<00:09, 8474.85it/s] 80%|███████▉  | 319275/400000 [00:38<00:09, 8495.05it/s] 80%|████████  | 320125/400000 [00:38<00:09, 8373.96it/s] 80%|████████  | 320986/400000 [00:38<00:09, 8441.97it/s] 80%|████████  | 321841/400000 [00:38<00:09, 8472.65it/s] 81%|████████  | 322698/400000 [00:38<00:09, 8500.52it/s] 81%|████████  | 323557/400000 [00:38<00:08, 8525.34it/s] 81%|████████  | 324410/400000 [00:38<00:09, 8316.10it/s] 81%|████████▏ | 325270/400000 [00:38<00:08, 8398.60it/s] 82%|████████▏ | 326128/400000 [00:38<00:08, 8449.45it/s] 82%|████████▏ | 326989/400000 [00:39<00:08, 8496.35it/s] 82%|████████▏ | 327851/400000 [00:39<00:08, 8531.82it/s] 82%|████████▏ | 328705/400000 [00:39<00:08, 8496.71it/s] 82%|████████▏ | 329561/400000 [00:39<00:08, 8513.23it/s] 83%|████████▎ | 330420/400000 [00:39<00:08, 8535.80it/s] 83%|████████▎ | 331276/400000 [00:39<00:08, 8542.03it/s] 83%|████████▎ | 332131/400000 [00:39<00:07, 8497.25it/s] 83%|████████▎ | 332981/400000 [00:39<00:07, 8468.35it/s] 83%|████████▎ | 333840/400000 [00:39<00:07, 8502.71it/s] 84%|████████▎ | 334691/400000 [00:39<00:07, 8471.54it/s] 84%|████████▍ | 335539/400000 [00:40<00:07, 8414.83it/s] 84%|████████▍ | 336399/400000 [00:40<00:07, 8469.43it/s] 84%|████████▍ | 337247/400000 [00:40<00:07, 8470.12it/s] 85%|████████▍ | 338110/400000 [00:40<00:07, 8516.15it/s] 85%|████████▍ | 338962/400000 [00:40<00:07, 8446.03it/s] 85%|████████▍ | 339807/400000 [00:40<00:07, 8443.28it/s] 85%|████████▌ | 340668/400000 [00:40<00:06, 8491.67it/s] 85%|████████▌ | 341518/400000 [00:40<00:06, 8472.23it/s] 86%|████████▌ | 342378/400000 [00:40<00:06, 8508.56it/s] 86%|████████▌ | 343238/400000 [00:40<00:06, 8533.88it/s] 86%|████████▌ | 344093/400000 [00:41<00:06, 8536.85it/s] 86%|████████▌ | 344956/400000 [00:41<00:06, 8562.92it/s] 86%|████████▋ | 345813/400000 [00:41<00:06, 8465.05it/s] 87%|████████▋ | 346667/400000 [00:41<00:06, 8486.85it/s] 87%|████████▋ | 347530/400000 [00:41<00:06, 8528.47it/s] 87%|████████▋ | 348384/400000 [00:41<00:06, 8458.77it/s] 87%|████████▋ | 349231/400000 [00:41<00:06, 8275.96it/s] 88%|████████▊ | 350060/400000 [00:41<00:06, 8275.10it/s] 88%|████████▊ | 350918/400000 [00:41<00:05, 8362.26it/s] 88%|████████▊ | 351780/400000 [00:42<00:05, 8437.66it/s] 88%|████████▊ | 352641/400000 [00:42<00:05, 8486.66it/s] 88%|████████▊ | 353504/400000 [00:42<00:05, 8529.06it/s] 89%|████████▊ | 354358/400000 [00:42<00:05, 8501.35it/s] 89%|████████▉ | 355209/400000 [00:42<00:05, 8421.68it/s] 89%|████████▉ | 356065/400000 [00:42<00:05, 8460.36it/s] 89%|████████▉ | 356927/400000 [00:42<00:05, 8506.67it/s] 89%|████████▉ | 357786/400000 [00:42<00:04, 8531.44it/s] 90%|████████▉ | 358640/400000 [00:42<00:04, 8486.36it/s] 90%|████████▉ | 359494/400000 [00:42<00:04, 8500.83it/s] 90%|█████████ | 360356/400000 [00:43<00:04, 8535.22it/s] 90%|█████████ | 361216/400000 [00:43<00:04, 8552.61it/s] 91%|█████████ | 362077/400000 [00:43<00:04, 8568.99it/s] 91%|█████████ | 362934/400000 [00:43<00:04, 8506.18it/s] 91%|█████████ | 363785/400000 [00:43<00:04, 8430.31it/s] 91%|█████████ | 364642/400000 [00:43<00:04, 8469.15it/s] 91%|█████████▏| 365502/400000 [00:43<00:04, 8506.98it/s] 92%|█████████▏| 366365/400000 [00:43<00:03, 8541.82it/s] 92%|█████████▏| 367220/400000 [00:43<00:03, 8536.47it/s] 92%|█████████▏| 368075/400000 [00:43<00:03, 8540.28it/s] 92%|█████████▏| 368937/400000 [00:44<00:03, 8563.89it/s] 92%|█████████▏| 369801/400000 [00:44<00:03, 8586.12it/s] 93%|█████████▎| 370660/400000 [00:44<00:03, 8575.28it/s] 93%|█████████▎| 371518/400000 [00:44<00:03, 8372.69it/s] 93%|█████████▎| 372373/400000 [00:44<00:03, 8424.99it/s] 93%|█████████▎| 373230/400000 [00:44<00:03, 8466.29it/s] 94%|█████████▎| 374093/400000 [00:44<00:03, 8514.32it/s] 94%|█████████▎| 374956/400000 [00:44<00:02, 8547.97it/s] 94%|█████████▍| 375812/400000 [00:44<00:02, 8374.23it/s] 94%|█████████▍| 376651/400000 [00:44<00:02, 8340.50it/s] 94%|█████████▍| 377509/400000 [00:45<00:02, 8410.81it/s] 95%|█████████▍| 378371/400000 [00:45<00:02, 8471.23it/s] 95%|█████████▍| 379227/400000 [00:45<00:02, 8497.33it/s] 95%|█████████▌| 380078/400000 [00:45<00:02, 8490.53it/s] 95%|█████████▌| 380928/400000 [00:45<00:02, 8466.87it/s] 95%|█████████▌| 381791/400000 [00:45<00:02, 8512.53it/s] 96%|█████████▌| 382654/400000 [00:45<00:02, 8544.49it/s] 96%|█████████▌| 383509/400000 [00:45<00:01, 8376.43it/s] 96%|█████████▌| 384371/400000 [00:45<00:01, 8446.09it/s] 96%|█████████▋| 385217/400000 [00:45<00:01, 8434.83it/s] 97%|█████████▋| 386075/400000 [00:46<00:01, 8475.30it/s] 97%|█████████▋| 386937/400000 [00:46<00:01, 8515.84it/s] 97%|█████████▋| 387789/400000 [00:46<00:01, 8473.87it/s] 97%|█████████▋| 388648/400000 [00:46<00:01, 8506.90it/s] 97%|█████████▋| 389499/400000 [00:46<00:01, 8480.53it/s] 98%|█████████▊| 390353/400000 [00:46<00:01, 8496.71it/s] 98%|█████████▊| 391203/400000 [00:46<00:01, 8493.43it/s] 98%|█████████▊| 392053/400000 [00:46<00:00, 8472.77it/s] 98%|█████████▊| 392904/400000 [00:46<00:00, 8481.84it/s] 98%|█████████▊| 393753/400000 [00:46<00:00, 8358.82it/s] 99%|█████████▊| 394590/400000 [00:47<00:00, 8281.22it/s] 99%|█████████▉| 395419/400000 [00:47<00:00, 8091.79it/s] 99%|█████████▉| 396256/400000 [00:47<00:00, 8106.50it/s] 99%|█████████▉| 397109/400000 [00:47<00:00, 8225.48it/s] 99%|█████████▉| 397960/400000 [00:47<00:00, 8307.24it/s]100%|█████████▉| 398820/400000 [00:47<00:00, 8392.08it/s]100%|█████████▉| 399682/400000 [00:47<00:00, 8457.46it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8383.55it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f3018101400>, <torchtext.data.dataset.TabularDataset object at 0x7f3018101550>, <torchtext.vocab.Vocab object at 0x7f3018101470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.10 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.10 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.10 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.10 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.31 file/s]2020-06-30 00:20:26.010600: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-30 00:20:26.015071: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095094999 Hz
2020-06-30 00:20:26.015261: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562f33fa8860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-30 00:20:26.015276: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 160937.43it/s] 82%|████████▏ | 8126464/9912422 [00:00<00:07, 229715.23it/s]9920512it [00:00, 45222760.87it/s]                           
0it [00:00, ?it/s]32768it [00:00, 460436.63it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466827.36it/s]1654784it [00:00, 11967554.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 146290.09it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
