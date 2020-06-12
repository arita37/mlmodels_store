
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fef932626a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fef932626a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7feffe82a0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7feffe82a0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff018556ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff018556ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fefac0a5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fefac0a5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fefac0a5950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157625.77it/s] 59%|█████▉    | 5824512/9912422 [00:00<00:18, 224917.22it/s]9920512it [00:00, 40905415.53it/s]                           
0it [00:00, ?it/s]32768it [00:00, 565271.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 445485.96it/s]1654784it [00:00, 11137267.51it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193114.69it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fefa9709b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fefa96fdda0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fefac0a5598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fefac0a5598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fefac0a5598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 15.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 20.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 20.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 55.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 68.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 74.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 79.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 79.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 82.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 85.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.04 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.04 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.30s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.69 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.30s/ url]
0 examples [00:00, ? examples/s]2020-06-12 00:08:48.283553: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-12 00:08:48.348409: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-12 00:08:48.349220: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55716a38c4c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 00:08:48.349239: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.78 examples/s]92 examples [00:00,  2.55 examples/s]188 examples [00:00,  3.63 examples/s]289 examples [00:00,  5.18 examples/s]386 examples [00:00,  7.39 examples/s]492 examples [00:01, 10.52 examples/s]595 examples [00:01, 14.96 examples/s]693 examples [00:01, 21.23 examples/s]792 examples [00:01, 30.06 examples/s]887 examples [00:01, 42.36 examples/s]989 examples [00:01, 59.46 examples/s]1086 examples [00:01, 82.74 examples/s]1187 examples [00:01, 114.16 examples/s]1285 examples [00:01, 153.66 examples/s]1378 examples [00:01, 204.82 examples/s]1471 examples [00:02, 263.01 examples/s]1562 examples [00:02, 334.29 examples/s]1659 examples [00:02, 416.05 examples/s]1751 examples [00:02, 495.29 examples/s]1854 examples [00:02, 585.88 examples/s]1949 examples [00:02, 649.48 examples/s]2045 examples [00:02, 717.44 examples/s]2144 examples [00:02, 781.98 examples/s]2240 examples [00:02, 826.12 examples/s]2335 examples [00:03, 858.85 examples/s]2430 examples [00:03, 861.33 examples/s]2525 examples [00:03, 885.17 examples/s]2619 examples [00:03, 888.57 examples/s]2712 examples [00:03, 885.54 examples/s]2804 examples [00:03, 895.21 examples/s]2898 examples [00:03, 907.20 examples/s]2993 examples [00:03, 918.03 examples/s]3089 examples [00:03, 929.94 examples/s]3183 examples [00:03, 924.21 examples/s]3277 examples [00:04, 926.46 examples/s]3376 examples [00:04, 942.47 examples/s]3471 examples [00:04, 936.46 examples/s]3565 examples [00:04, 936.57 examples/s]3659 examples [00:04, 934.68 examples/s]3753 examples [00:04, 926.70 examples/s]3849 examples [00:04, 933.72 examples/s]3943 examples [00:04, 934.75 examples/s]4037 examples [00:04, 917.84 examples/s]4132 examples [00:04, 925.35 examples/s]4228 examples [00:05, 934.57 examples/s]4322 examples [00:05, 905.60 examples/s]4413 examples [00:05, 906.29 examples/s]4507 examples [00:05, 914.97 examples/s]4599 examples [00:05, 901.79 examples/s]4692 examples [00:05, 909.28 examples/s]4787 examples [00:05, 919.84 examples/s]4883 examples [00:05, 930.38 examples/s]4977 examples [00:05, 927.03 examples/s]5071 examples [00:05, 930.32 examples/s]5166 examples [00:06, 935.06 examples/s]5266 examples [00:06, 953.00 examples/s]5362 examples [00:06, 952.97 examples/s]5460 examples [00:06, 957.64 examples/s]5556 examples [00:06, 935.82 examples/s]5650 examples [00:06, 933.71 examples/s]5744 examples [00:06, 909.72 examples/s]5836 examples [00:06, 909.29 examples/s]5928 examples [00:06, 911.89 examples/s]6020 examples [00:07, 909.22 examples/s]6112 examples [00:07, 910.47 examples/s]6211 examples [00:07, 931.33 examples/s]6311 examples [00:07, 948.90 examples/s]6407 examples [00:07, 951.03 examples/s]6503 examples [00:07, 942.87 examples/s]6598 examples [00:07, 938.05 examples/s]6692 examples [00:07, 936.41 examples/s]6788 examples [00:07, 943.23 examples/s]6883 examples [00:07, 933.99 examples/s]6982 examples [00:08, 948.55 examples/s]7077 examples [00:08, 940.60 examples/s]7172 examples [00:08, 940.93 examples/s]7267 examples [00:08, 942.09 examples/s]7362 examples [00:08, 897.59 examples/s]7453 examples [00:08, 899.05 examples/s]7550 examples [00:08, 918.95 examples/s]7643 examples [00:08, 910.14 examples/s]7736 examples [00:08, 912.41 examples/s]7828 examples [00:08, 904.81 examples/s]7925 examples [00:09, 922.45 examples/s]8020 examples [00:09, 930.49 examples/s]8117 examples [00:09, 940.03 examples/s]8212 examples [00:09, 942.22 examples/s]8309 examples [00:09, 948.82 examples/s]8404 examples [00:09, 936.06 examples/s]8502 examples [00:09, 947.44 examples/s]8602 examples [00:09, 960.80 examples/s]8702 examples [00:09, 970.12 examples/s]8803 examples [00:09, 979.04 examples/s]8901 examples [00:10, 962.88 examples/s]8998 examples [00:10, 953.90 examples/s]9094 examples [00:10, 938.24 examples/s]9188 examples [00:10, 936.90 examples/s]9284 examples [00:10, 942.48 examples/s]9379 examples [00:10, 928.11 examples/s]9475 examples [00:10, 936.97 examples/s]9572 examples [00:10, 945.06 examples/s]9667 examples [00:10, 926.85 examples/s]9763 examples [00:11, 936.33 examples/s]9857 examples [00:11, 933.01 examples/s]9951 examples [00:11, 931.91 examples/s]10045 examples [00:11, 858.07 examples/s]10142 examples [00:11, 886.68 examples/s]10237 examples [00:11, 903.77 examples/s]10330 examples [00:11, 909.98 examples/s]10429 examples [00:11, 931.36 examples/s]10523 examples [00:11, 931.60 examples/s]10621 examples [00:11, 944.97 examples/s]10723 examples [00:12, 965.14 examples/s]10821 examples [00:12, 967.16 examples/s]10918 examples [00:12, 952.50 examples/s]11014 examples [00:12, 949.87 examples/s]11110 examples [00:12, 924.18 examples/s]11203 examples [00:12, 918.22 examples/s]11296 examples [00:12, 915.29 examples/s]11394 examples [00:12, 930.36 examples/s]11497 examples [00:12, 956.87 examples/s]11604 examples [00:12, 986.40 examples/s]11707 examples [00:13, 998.13 examples/s]11808 examples [00:13, 989.08 examples/s]11914 examples [00:13, 1007.28 examples/s]12015 examples [00:13, 980.63 examples/s] 12114 examples [00:13, 909.69 examples/s]12211 examples [00:13, 925.50 examples/s]12305 examples [00:13, 883.01 examples/s]12395 examples [00:13, 816.35 examples/s]12479 examples [00:13, 798.38 examples/s]12567 examples [00:14, 820.13 examples/s]12658 examples [00:14, 843.73 examples/s]12744 examples [00:14, 825.40 examples/s]12829 examples [00:14, 831.41 examples/s]12919 examples [00:14, 848.88 examples/s]13013 examples [00:14, 872.97 examples/s]13101 examples [00:14, 859.62 examples/s]13205 examples [00:14, 905.26 examples/s]13305 examples [00:14, 930.37 examples/s]13403 examples [00:14, 944.01 examples/s]13501 examples [00:15, 954.49 examples/s]13597 examples [00:15, 954.47 examples/s]13695 examples [00:15, 959.98 examples/s]13793 examples [00:15, 963.24 examples/s]13890 examples [00:15, 959.19 examples/s]13991 examples [00:15, 973.15 examples/s]14092 examples [00:15, 981.50 examples/s]14191 examples [00:15, 979.50 examples/s]14293 examples [00:15, 989.88 examples/s]14394 examples [00:15, 995.82 examples/s]14494 examples [00:16, 933.42 examples/s]14589 examples [00:16, 930.11 examples/s]14683 examples [00:16, 921.14 examples/s]14778 examples [00:16, 928.70 examples/s]14874 examples [00:16, 936.25 examples/s]14975 examples [00:16, 956.39 examples/s]15071 examples [00:16, 952.80 examples/s]15168 examples [00:16, 956.56 examples/s]15264 examples [00:16, 955.16 examples/s]15365 examples [00:17, 968.64 examples/s]15462 examples [00:17, 966.85 examples/s]15559 examples [00:17, 963.34 examples/s]15656 examples [00:17, 938.11 examples/s]15755 examples [00:17, 952.64 examples/s]15851 examples [00:17, 945.67 examples/s]15946 examples [00:17, 939.49 examples/s]16041 examples [00:17, 932.19 examples/s]16140 examples [00:17, 947.07 examples/s]16238 examples [00:17, 955.44 examples/s]16334 examples [00:18, 956.33 examples/s]16430 examples [00:18, 942.88 examples/s]16525 examples [00:18, 929.71 examples/s]16621 examples [00:18, 937.08 examples/s]16715 examples [00:18, 903.71 examples/s]16808 examples [00:18, 910.97 examples/s]16903 examples [00:18, 920.58 examples/s]16996 examples [00:18, 909.14 examples/s]17088 examples [00:18, 911.05 examples/s]17180 examples [00:18, 907.27 examples/s]17276 examples [00:19, 921.89 examples/s]17371 examples [00:19, 929.09 examples/s]17465 examples [00:19, 899.43 examples/s]17556 examples [00:19, 900.72 examples/s]17647 examples [00:19, 896.26 examples/s]17738 examples [00:19, 897.85 examples/s]17833 examples [00:19, 911.75 examples/s]17929 examples [00:19, 925.68 examples/s]18027 examples [00:19, 941.23 examples/s]18122 examples [00:20, 919.94 examples/s]18215 examples [00:20, 913.81 examples/s]18307 examples [00:20, 911.63 examples/s]18399 examples [00:20, 902.11 examples/s]18490 examples [00:20, 903.09 examples/s]18587 examples [00:20, 920.11 examples/s]18690 examples [00:20, 949.63 examples/s]18788 examples [00:20, 955.61 examples/s]18884 examples [00:20, 950.91 examples/s]18984 examples [00:20, 964.28 examples/s]19085 examples [00:21, 975.68 examples/s]19183 examples [00:21, 966.91 examples/s]19280 examples [00:21, 956.69 examples/s]19376 examples [00:21, 940.24 examples/s]19472 examples [00:21, 945.93 examples/s]19567 examples [00:21, 941.77 examples/s]19665 examples [00:21, 951.62 examples/s]19762 examples [00:21, 956.37 examples/s]19858 examples [00:21, 954.21 examples/s]19954 examples [00:21, 945.50 examples/s]20049 examples [00:22, 898.88 examples/s]20146 examples [00:22, 916.92 examples/s]20239 examples [00:22, 920.40 examples/s]20332 examples [00:22, 908.41 examples/s]20433 examples [00:22, 934.31 examples/s]20527 examples [00:22, 935.77 examples/s]20628 examples [00:22, 955.42 examples/s]20730 examples [00:22, 972.65 examples/s]20828 examples [00:22, 972.95 examples/s]20935 examples [00:22, 998.69 examples/s]21040 examples [00:23, 1013.45 examples/s]21142 examples [00:23, 1011.85 examples/s]21244 examples [00:23, 1005.54 examples/s]21345 examples [00:23, 953.71 examples/s] 21442 examples [00:23, 937.61 examples/s]21537 examples [00:23, 940.35 examples/s]21636 examples [00:23, 953.42 examples/s]21736 examples [00:23, 966.46 examples/s]21833 examples [00:23, 959.06 examples/s]21933 examples [00:24, 967.48 examples/s]22030 examples [00:24, 953.42 examples/s]22133 examples [00:24, 972.64 examples/s]22231 examples [00:24, 966.74 examples/s]22329 examples [00:24, 968.62 examples/s]22426 examples [00:24, 954.26 examples/s]22530 examples [00:24, 977.06 examples/s]22630 examples [00:24, 981.47 examples/s]22730 examples [00:24, 985.80 examples/s]22830 examples [00:24, 988.84 examples/s]22933 examples [00:25, 999.75 examples/s]23034 examples [00:25, 989.27 examples/s]23134 examples [00:25, 972.88 examples/s]23232 examples [00:25, 958.75 examples/s]23332 examples [00:25, 967.98 examples/s]23432 examples [00:25, 975.40 examples/s]23531 examples [00:25, 978.29 examples/s]23629 examples [00:25, 969.30 examples/s]23733 examples [00:25, 987.65 examples/s]23834 examples [00:25, 991.99 examples/s]23934 examples [00:26, 991.43 examples/s]24038 examples [00:26, 1003.10 examples/s]24139 examples [00:26, 990.39 examples/s] 24239 examples [00:26, 976.94 examples/s]24337 examples [00:26, 929.86 examples/s]24436 examples [00:26, 944.02 examples/s]24531 examples [00:26, 944.02 examples/s]24629 examples [00:26, 954.08 examples/s]24725 examples [00:26, 941.96 examples/s]24820 examples [00:26, 935.48 examples/s]24919 examples [00:27, 949.04 examples/s]25018 examples [00:27, 958.52 examples/s]25119 examples [00:27, 973.21 examples/s]25217 examples [00:27, 972.39 examples/s]25315 examples [00:27, 952.88 examples/s]25411 examples [00:27, 933.09 examples/s]25505 examples [00:27, 913.50 examples/s]25606 examples [00:27, 938.96 examples/s]25701 examples [00:27, 937.79 examples/s]25798 examples [00:28, 944.77 examples/s]25894 examples [00:28, 948.08 examples/s]25994 examples [00:28, 963.01 examples/s]26091 examples [00:28, 961.85 examples/s]26188 examples [00:28, 919.00 examples/s]26285 examples [00:28, 931.69 examples/s]26384 examples [00:28, 945.71 examples/s]26479 examples [00:28, 929.37 examples/s]26574 examples [00:28, 935.39 examples/s]26668 examples [00:28, 917.58 examples/s]26764 examples [00:29, 928.06 examples/s]26863 examples [00:29, 943.37 examples/s]26958 examples [00:29, 943.10 examples/s]27053 examples [00:29, 943.38 examples/s]27148 examples [00:29, 932.51 examples/s]27242 examples [00:29, 909.04 examples/s]27341 examples [00:29, 930.55 examples/s]27435 examples [00:29, 915.81 examples/s]27528 examples [00:29, 918.19 examples/s]27621 examples [00:29, 919.44 examples/s]27714 examples [00:30, 913.82 examples/s]27808 examples [00:30, 919.76 examples/s]27901 examples [00:30, 914.72 examples/s]27997 examples [00:30, 927.60 examples/s]28090 examples [00:30, 922.22 examples/s]28183 examples [00:30, 913.43 examples/s]28275 examples [00:30, 909.71 examples/s]28367 examples [00:30, 901.00 examples/s]28458 examples [00:30, 897.13 examples/s]28548 examples [00:31, 888.81 examples/s]28647 examples [00:31, 916.43 examples/s]28740 examples [00:31, 917.31 examples/s]28832 examples [00:31, 904.39 examples/s]28924 examples [00:31, 907.69 examples/s]29017 examples [00:31, 913.70 examples/s]29109 examples [00:31, 915.22 examples/s]29201 examples [00:31, 916.09 examples/s]29293 examples [00:31, 909.63 examples/s]29387 examples [00:31, 916.13 examples/s]29479 examples [00:32, 901.76 examples/s]29572 examples [00:32, 909.33 examples/s]29665 examples [00:32, 914.75 examples/s]29757 examples [00:32, 909.79 examples/s]29849 examples [00:32, 893.73 examples/s]29939 examples [00:32, 874.20 examples/s]30027 examples [00:32, 820.09 examples/s]30127 examples [00:32, 865.63 examples/s]30227 examples [00:32, 901.88 examples/s]30328 examples [00:32, 930.49 examples/s]30423 examples [00:33, 935.73 examples/s]30518 examples [00:33, 923.67 examples/s]30613 examples [00:33, 929.81 examples/s]30710 examples [00:33, 939.69 examples/s]30805 examples [00:33, 906.33 examples/s]30899 examples [00:33, 913.37 examples/s]30995 examples [00:33, 926.50 examples/s]31092 examples [00:33, 936.05 examples/s]31191 examples [00:33, 949.16 examples/s]31287 examples [00:33, 944.45 examples/s]31382 examples [00:34, 934.85 examples/s]31481 examples [00:34, 948.42 examples/s]31576 examples [00:34, 932.01 examples/s]31674 examples [00:34, 945.18 examples/s]31773 examples [00:34, 955.05 examples/s]31869 examples [00:34, 932.93 examples/s]31968 examples [00:34, 947.77 examples/s]32069 examples [00:34, 963.21 examples/s]32166 examples [00:34, 950.16 examples/s]32262 examples [00:35, 931.17 examples/s]32356 examples [00:35, 925.23 examples/s]32449 examples [00:35, 912.86 examples/s]32541 examples [00:35, 898.51 examples/s]32632 examples [00:35, 895.10 examples/s]32725 examples [00:35, 903.81 examples/s]32816 examples [00:35, 893.59 examples/s]32914 examples [00:35, 916.33 examples/s]33016 examples [00:35, 944.17 examples/s]33113 examples [00:35, 949.27 examples/s]33209 examples [00:36, 931.37 examples/s]33309 examples [00:36, 950.10 examples/s]33408 examples [00:36, 959.65 examples/s]33505 examples [00:36, 957.12 examples/s]33604 examples [00:36, 963.79 examples/s]33701 examples [00:36, 946.86 examples/s]33799 examples [00:36, 955.51 examples/s]33898 examples [00:36, 961.75 examples/s]33995 examples [00:36, 953.19 examples/s]34091 examples [00:36, 952.79 examples/s]34187 examples [00:37, 943.72 examples/s]34288 examples [00:37, 960.74 examples/s]34387 examples [00:37, 968.34 examples/s]34488 examples [00:37, 980.36 examples/s]34590 examples [00:37, 989.93 examples/s]34690 examples [00:37, 984.01 examples/s]34789 examples [00:37, 970.43 examples/s]34887 examples [00:37, 964.01 examples/s]34984 examples [00:37, 952.04 examples/s]35085 examples [00:37, 965.67 examples/s]35184 examples [00:38, 971.80 examples/s]35285 examples [00:38, 980.63 examples/s]35384 examples [00:38, 981.39 examples/s]35483 examples [00:38, 978.24 examples/s]35581 examples [00:38, 947.87 examples/s]35677 examples [00:38, 940.10 examples/s]35772 examples [00:38, 900.67 examples/s]35868 examples [00:38, 917.57 examples/s]35964 examples [00:38, 927.61 examples/s]36058 examples [00:39, 929.23 examples/s]36152 examples [00:39, 923.98 examples/s]36245 examples [00:39, 922.67 examples/s]36341 examples [00:39, 931.33 examples/s]36441 examples [00:39, 950.63 examples/s]36537 examples [00:39, 951.16 examples/s]36633 examples [00:39, 950.88 examples/s]36729 examples [00:39, 943.09 examples/s]36824 examples [00:39, 944.25 examples/s]36921 examples [00:39, 950.55 examples/s]37020 examples [00:40, 961.31 examples/s]37117 examples [00:40, 950.28 examples/s]37217 examples [00:40, 961.32 examples/s]37314 examples [00:40, 931.21 examples/s]37408 examples [00:40, 928.49 examples/s]37506 examples [00:40, 940.88 examples/s]37601 examples [00:40, 917.83 examples/s]37694 examples [00:40, 920.75 examples/s]37789 examples [00:40, 928.85 examples/s]37883 examples [00:40, 923.85 examples/s]37980 examples [00:41, 936.07 examples/s]38074 examples [00:41, 932.47 examples/s]38168 examples [00:41, 925.53 examples/s]38266 examples [00:41, 938.83 examples/s]38360 examples [00:41, 937.43 examples/s]38454 examples [00:41, 929.01 examples/s]38547 examples [00:41, 925.99 examples/s]38640 examples [00:41, 900.46 examples/s]38734 examples [00:41, 911.33 examples/s]38826 examples [00:42, 909.18 examples/s]38919 examples [00:42, 912.35 examples/s]39011 examples [00:42, 914.36 examples/s]39103 examples [00:42, 909.16 examples/s]39194 examples [00:42, 902.22 examples/s]39291 examples [00:42, 919.63 examples/s]39384 examples [00:42, 911.41 examples/s]39476 examples [00:42, 907.20 examples/s]39570 examples [00:42, 914.60 examples/s]39667 examples [00:42, 930.00 examples/s]39769 examples [00:43, 954.32 examples/s]39867 examples [00:43, 961.75 examples/s]39964 examples [00:43, 942.39 examples/s]40059 examples [00:43, 862.88 examples/s]40147 examples [00:43, 863.43 examples/s]40237 examples [00:43, 873.47 examples/s]40326 examples [00:43, 869.55 examples/s]40417 examples [00:43, 880.95 examples/s]40510 examples [00:43, 893.97 examples/s]40604 examples [00:43, 907.05 examples/s]40698 examples [00:44, 914.91 examples/s]40790 examples [00:44, 913.03 examples/s]40883 examples [00:44, 916.68 examples/s]40979 examples [00:44, 928.58 examples/s]41075 examples [00:44, 937.13 examples/s]41169 examples [00:44, 911.59 examples/s]41261 examples [00:44, 899.51 examples/s]41352 examples [00:44, 872.24 examples/s]41442 examples [00:44, 878.60 examples/s]41539 examples [00:44, 903.19 examples/s]41630 examples [00:45, 903.60 examples/s]41724 examples [00:45, 913.06 examples/s]41816 examples [00:45, 910.76 examples/s]41908 examples [00:45, 905.24 examples/s]42001 examples [00:45, 911.96 examples/s]42094 examples [00:45, 915.04 examples/s]42195 examples [00:45, 940.81 examples/s]42291 examples [00:45, 944.30 examples/s]42395 examples [00:45, 970.24 examples/s]42501 examples [00:46, 993.64 examples/s]42611 examples [00:46, 1021.93 examples/s]42714 examples [00:46, 993.82 examples/s] 42814 examples [00:46, 966.26 examples/s]42917 examples [00:46, 982.13 examples/s]43018 examples [00:46, 988.12 examples/s]43124 examples [00:46, 1008.54 examples/s]43226 examples [00:46, 1011.63 examples/s]43328 examples [00:46, 988.65 examples/s] 43428 examples [00:46, 976.89 examples/s]43526 examples [00:47, 969.21 examples/s]43624 examples [00:47, 955.53 examples/s]43720 examples [00:47, 943.05 examples/s]43815 examples [00:47, 921.52 examples/s]43908 examples [00:47, 913.03 examples/s]44006 examples [00:47, 930.16 examples/s]44101 examples [00:47, 934.11 examples/s]44196 examples [00:47, 937.41 examples/s]44290 examples [00:47, 920.91 examples/s]44385 examples [00:47, 927.59 examples/s]44486 examples [00:48, 950.09 examples/s]44582 examples [00:48, 934.07 examples/s]44676 examples [00:48, 907.73 examples/s]44768 examples [00:48, 896.32 examples/s]44864 examples [00:48, 913.83 examples/s]44956 examples [00:48, 915.29 examples/s]45048 examples [00:48, 907.12 examples/s]45141 examples [00:48, 911.21 examples/s]45237 examples [00:48, 924.48 examples/s]45334 examples [00:49, 936.65 examples/s]45433 examples [00:49, 950.92 examples/s]45529 examples [00:49, 950.77 examples/s]45625 examples [00:49, 942.98 examples/s]45720 examples [00:49, 935.82 examples/s]45817 examples [00:49, 944.01 examples/s]45912 examples [00:49, 942.65 examples/s]46007 examples [00:49, 927.70 examples/s]46104 examples [00:49, 936.38 examples/s]46199 examples [00:49, 939.79 examples/s]46297 examples [00:50, 949.41 examples/s]46398 examples [00:50, 964.74 examples/s]46495 examples [00:50, 962.10 examples/s]46592 examples [00:50, 927.07 examples/s]46689 examples [00:50, 937.45 examples/s]46787 examples [00:50, 946.78 examples/s]46884 examples [00:50, 952.31 examples/s]46982 examples [00:50, 958.29 examples/s]47082 examples [00:50, 969.13 examples/s]47180 examples [00:50, 952.24 examples/s]47276 examples [00:51, 952.53 examples/s]47377 examples [00:51, 968.67 examples/s]47476 examples [00:51, 974.09 examples/s]47574 examples [00:51, 957.26 examples/s]47676 examples [00:51, 974.55 examples/s]47774 examples [00:51, 961.98 examples/s]47874 examples [00:51, 972.42 examples/s]47973 examples [00:51, 976.04 examples/s]48071 examples [00:51, 973.20 examples/s]48169 examples [00:51, 960.48 examples/s]48266 examples [00:52, 937.56 examples/s]48363 examples [00:52, 946.61 examples/s]48459 examples [00:52, 950.02 examples/s]48555 examples [00:52, 922.87 examples/s]48652 examples [00:52, 934.80 examples/s]48746 examples [00:52, 922.10 examples/s]48842 examples [00:52, 930.83 examples/s]48943 examples [00:52, 952.91 examples/s]49039 examples [00:52, 951.68 examples/s]49142 examples [00:52, 972.98 examples/s]49241 examples [00:53, 976.07 examples/s]49345 examples [00:53, 992.63 examples/s]49448 examples [00:53, 1003.53 examples/s]49549 examples [00:53, 988.04 examples/s] 49648 examples [00:53, 965.56 examples/s]49745 examples [00:53, 941.04 examples/s]49840 examples [00:53, 937.38 examples/s]49934 examples [00:53, 937.16 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7019/50000 [00:00<00:00, 70189.56 examples/s] 39%|███▉      | 19562/50000 [00:00<00:00, 80873.72 examples/s] 64%|██████▍   | 32108/50000 [00:00<00:00, 90524.66 examples/s] 89%|████████▉ | 44657/50000 [00:00<00:00, 98780.99 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 718.61 examples/s]168 examples [00:00, 775.91 examples/s]268 examples [00:00, 831.53 examples/s]366 examples [00:00, 870.94 examples/s]461 examples [00:00, 891.81 examples/s]555 examples [00:00, 903.30 examples/s]653 examples [00:00, 923.01 examples/s]749 examples [00:00, 933.65 examples/s]846 examples [00:00, 942.38 examples/s]941 examples [00:01, 943.25 examples/s]1039 examples [00:01, 952.48 examples/s]1138 examples [00:01, 962.88 examples/s]1239 examples [00:01, 975.95 examples/s]1339 examples [00:01, 980.59 examples/s]1437 examples [00:01, 785.27 examples/s]1543 examples [00:01, 850.53 examples/s]1649 examples [00:01, 903.02 examples/s]1757 examples [00:01, 948.75 examples/s]1860 examples [00:01, 969.43 examples/s]1960 examples [00:02, 972.65 examples/s]2060 examples [00:02, 964.09 examples/s]2164 examples [00:02, 985.37 examples/s]2265 examples [00:02, 992.53 examples/s]2368 examples [00:02, 1002.75 examples/s]2469 examples [00:02, 956.81 examples/s] 2570 examples [00:02, 969.80 examples/s]2668 examples [00:02, 959.10 examples/s]2767 examples [00:02, 967.65 examples/s]2872 examples [00:03, 989.63 examples/s]2972 examples [00:03, 983.18 examples/s]3071 examples [00:03, 970.25 examples/s]3170 examples [00:03, 975.88 examples/s]3268 examples [00:03, 960.77 examples/s]3365 examples [00:03, 949.44 examples/s]3461 examples [00:03, 951.67 examples/s]3558 examples [00:03, 955.37 examples/s]3657 examples [00:03, 962.78 examples/s]3757 examples [00:03, 971.96 examples/s]3855 examples [00:04, 969.66 examples/s]3953 examples [00:04, 950.43 examples/s]4049 examples [00:04, 949.02 examples/s]4145 examples [00:04, 949.65 examples/s]4242 examples [00:04, 954.08 examples/s]4338 examples [00:04, 944.34 examples/s]4433 examples [00:04, 939.62 examples/s]4533 examples [00:04, 955.10 examples/s]4632 examples [00:04, 963.40 examples/s]4736 examples [00:04, 984.07 examples/s]4835 examples [00:05, 985.33 examples/s]4934 examples [00:05, 957.53 examples/s]5031 examples [00:05, 951.82 examples/s]5127 examples [00:05, 947.17 examples/s]5224 examples [00:05, 952.50 examples/s]5320 examples [00:05, 950.00 examples/s]5416 examples [00:05, 904.95 examples/s]5511 examples [00:05, 916.12 examples/s]5603 examples [00:05, 908.30 examples/s]5701 examples [00:06, 926.10 examples/s]5798 examples [00:06, 937.94 examples/s]5894 examples [00:06, 943.45 examples/s]5994 examples [00:06, 958.10 examples/s]6090 examples [00:06, 955.72 examples/s]6191 examples [00:06, 968.52 examples/s]6292 examples [00:06, 979.03 examples/s]6391 examples [00:06, 970.49 examples/s]6493 examples [00:06, 983.43 examples/s]6594 examples [00:06, 988.49 examples/s]6693 examples [00:07, 986.28 examples/s]6792 examples [00:07, 976.40 examples/s]6890 examples [00:07, 965.22 examples/s]6987 examples [00:07, 961.38 examples/s]7084 examples [00:07, 957.95 examples/s]7182 examples [00:07, 960.69 examples/s]7279 examples [00:07, 939.81 examples/s]7374 examples [00:07, 934.83 examples/s]7470 examples [00:07, 939.92 examples/s]7565 examples [00:07, 937.62 examples/s]7659 examples [00:08, 931.29 examples/s]7753 examples [00:08, 933.74 examples/s]7847 examples [00:08, 926.93 examples/s]7947 examples [00:08, 946.14 examples/s]8044 examples [00:08, 952.94 examples/s]8140 examples [00:08, 942.13 examples/s]8238 examples [00:08, 951.47 examples/s]8334 examples [00:08, 934.21 examples/s]8429 examples [00:08, 938.79 examples/s]8526 examples [00:08, 946.10 examples/s]8621 examples [00:09, 899.29 examples/s]8712 examples [00:09, 900.42 examples/s]8806 examples [00:09, 911.12 examples/s]8906 examples [00:09, 935.36 examples/s]9009 examples [00:09, 960.21 examples/s]9106 examples [00:09, 944.05 examples/s]9205 examples [00:09, 956.55 examples/s]9301 examples [00:09, 951.42 examples/s]9398 examples [00:09, 956.76 examples/s]9494 examples [00:09, 949.60 examples/s]9590 examples [00:10, 943.88 examples/s]9686 examples [00:10, 948.15 examples/s]9781 examples [00:10, 948.30 examples/s]9883 examples [00:10, 967.76 examples/s]9983 examples [00:10, 974.68 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteHXRVO7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteHXRVO7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fefac0a5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fefac0a5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fefac0a5950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feff7a923c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fef314d36d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fefac0a5950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fefac0a5950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fefac0a5950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fefa9394080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fefa93940f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fef23ad22f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fef23ad22f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fef23ad22f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fef23ad2400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fef23ad2400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fef23ad2400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2ceb549805a57da5cf51550ab718cd22fb7f06f5441d6be7f24cc9ab2b26c74b
  Stored in directory: /tmp/pip-ephem-wheel-cache-cr8f597v/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:04:33, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:17:02, 16.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:03:08, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:02:42, 34.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.59M/862M [00:01<4:55:09, 48.5kB/s].vector_cache/glove.6B.zip:   1%|          | 7.88M/862M [00:01<3:25:40, 69.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:23:21, 98.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:01<1:39:46, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.7M/862M [00:01<1:09:27, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<48:23, 287kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<33:44, 409kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:02<23:33, 581kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<16:28, 825kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<11:47, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<10:08, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<10:56, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<08:26, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<06:05, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<08:41, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<07:45, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<05:49, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:46, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:08<04:37, 2.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<06:08, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<04:15, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<06:06, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<06:57, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<05:25, 2.42MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:12<03:57, 3.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<10:45, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<08:52, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<06:29, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<07:35, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:24, 2.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<04:48, 2.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:25, 2.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:07, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<05:37, 2.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<06:00, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<05:32, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<04:12, 3.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<05:54, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<06:43, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:21, 2.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:22<03:52, 3.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<12:23:29, 17.1kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<8:41:30, 24.4kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<6:04:36, 34.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<4:17:28, 49.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:01:25, 69.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<2:07:02, 99.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:31:41, 138kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:06:38, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<47:14, 267kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<33:04, 379kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:29:16, 16.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<8:45:32, 23.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<6:07:25, 34.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<4:19:22, 48.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<3:04:02, 67.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<2:09:19, 96.3kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:32:09, 135kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:05:33, 189kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<46:06, 268kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<35:06, 351kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<27:04, 455kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<19:28, 633kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<13:43, 894kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<20:36, 595kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:39, 783kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<11:12, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:42, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:43, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:21, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:20, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:35, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:50, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:13, 2.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<10:14, 1.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:23, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:10, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<07:07, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:10, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:36, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<06:02, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:26, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:05, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:40, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:23, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:58, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:36, 3.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<13:35, 867kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<10:41, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:46, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:10, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:05, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:09, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:26, 2.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:00, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:12, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:01, 1.93MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:56, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:13, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<05:38, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:03, 2.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:25:58, 134kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<1:01:19, 188kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<43:05, 266kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<32:46, 349kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<25:09, 455kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<18:10, 629kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<12:47, 890kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<11:07:24, 17.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<7:48:06, 24.3kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<5:27:12, 34.7kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<3:50:58, 48.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<2:43:56, 68.9kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<1:55:06, 98.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<1:20:26, 140kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:02:57, 178kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<45:11, 248kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<31:48, 352kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<24:50, 450kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<18:30, 603kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<13:10, 846kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:50, 937kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:24, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<06:51, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:23, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:24, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:38, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:06, 2.68MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:42, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:21, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:43, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:50, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:17, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:52, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:18, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:12, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:34, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:40, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:03, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:48, 2.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:11, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:43, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:31, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:59, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:39, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:50, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:31, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:19, 2.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:08, 3.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<09:25, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:40, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<05:37, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<06:21, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:34, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:15, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:43, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:33, 2.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:54, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:28, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:46, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:24, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:18, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:38, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:23, 1.88MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:06, 3.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<1:06:50, 151kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<47:47, 211kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<33:35, 299kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<25:46, 388kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<20:03, 499kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<14:31, 687kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<11:43, 847kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<09:12, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:41, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:59, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:53, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:47, 2.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<08:23, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<06:51, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:02, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:48, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:01, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:37, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:20, 2.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:46, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:09, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<05:14, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:54, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:04, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:39, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<03:23, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<06:40, 1.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:38, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:08, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:07, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:32, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:22, 2.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:01, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:55, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<02:51, 3.26MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:58, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:50, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:17, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:07, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:38, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:22, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:11, 2.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:30, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:52, 1.88MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:37, 2.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:30, 2.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:09, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:06, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<02:58, 3.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<11:44, 769kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:13, 980kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<06:38, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:35, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:36, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:07, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<03:40, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<11:03, 805kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:43, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<06:20, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<06:20, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<06:22, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:52, 1.81MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:32, 2.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:22, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:31, 2.48MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:20, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:59, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:51, 3.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<11:14, 767kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<08:49, 977kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<06:21, 1.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:17, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<06:17, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:52, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<03:29, 2.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<11:33, 734kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<09:01, 940kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:29, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:21, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:18, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:48, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<03:29, 2.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:13, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:34, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:25, 2.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:11, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:50, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:52, 2.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:28, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:30, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:37, 3.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:58, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:41, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:46, 2.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:41, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:21, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:30, 2.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<02:32, 3.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<09:23, 851kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:27, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:25, 1.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:30, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:44, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:29, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:09, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:39, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:42, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:40, 2.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<09:17, 839kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<07:21, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<05:21, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:24, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:38, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:24, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:03, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:31, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:35, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:36, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<11:01, 688kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<08:32, 887kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:08, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:55, 1.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:58, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:39, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:09, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:33, 1.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:35, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:35, 2.85MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<09:42, 760kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:34, 971kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<05:29, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:26, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:26, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:08, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:59, 2.42MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:59, 1.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:17, 1.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:09, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:46, 1.90MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:13, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:17, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:23, 2.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:05, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<04:20, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:11, 2.21MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:45, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:11, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:14, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:21, 2.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:25, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:43, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:47, 2.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:28, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:11, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:24, 2.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:09, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:57, 2.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:15, 3.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:02, 2.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:38, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:54, 2.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:07, 3.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<09:34, 697kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<07:25, 898kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:20, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:10, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:04, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:51, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:46, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:39, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:58, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:55, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:27, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:51, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:03, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:11, 2.92MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<08:23, 762kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:34, 972kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:45, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:41, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:40, 1.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:33, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:34, 2.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<04:00, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:29, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:36, 2.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:10, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:35, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<02:04, 2.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<08:27, 724kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<06:35, 928kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:44, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:36, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:33, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:31, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:31, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<09:01, 663kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:56, 860kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:00, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:48, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:37, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:29, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:31, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:49, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:18, 1.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:00, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:22, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:37, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:56, 2.96MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:00, 1.90MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:42, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:02, 2.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:41, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:07, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:26, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:46, 3.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<03:36, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<03:08, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:18, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:48, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:11, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:31, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:49, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<07:24, 732kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<06:20, 855kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:40, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:19, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:34, 960kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:29, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:17, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:26, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:34, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<02:00, 2.61MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<07:00, 745kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<05:26, 957kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:56, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:53, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:48, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:54, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:05, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<12:18, 413kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<09:09, 554kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<06:29, 778kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<05:38, 889kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:59, 1.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:45, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:40, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<16:45, 295kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<12:15, 403kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<08:40, 566kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<07:05, 687kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<06:02, 806kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<04:29, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<03:10, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:21, 653kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:40, 846kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<04:04, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:51, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:44, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:52, 1.64MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<02:03, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<06:27, 723kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<05:00, 931kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<03:35, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:31, 1.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:29, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:41, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:55, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:34, 811kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:23, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<03:09, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:12, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:13, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:29, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:46, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<05:57, 736kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<04:37, 947kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:20, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:17, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:12, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:25, 1.77MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:44, 2.44MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:15, 1.31MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:43, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:59, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:19, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:05, 2.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:33, 2.67MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:58, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:17, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:19, 3.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:49, 693kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:31, 891kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<03:14, 1.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:07, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:03, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:21, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<01:40, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<05:21, 728kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:10, 933kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<03:00, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:55, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:53, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:13, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<01:35, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<05:08, 732kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:59, 942kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:52, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:49, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:47, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:08, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:31, 2.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<04:57, 731kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:51, 936kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<02:46, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:41, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:38, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:01, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<01:26, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<11:30, 303kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<08:25, 413kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<05:56, 581kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<04:51, 703kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<04:08, 822kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:03, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<02:10, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:15, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:39, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:51, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:38, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:12, 2.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:34, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:48, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:26, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:02, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:39, 857kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:54, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<02:06, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:07, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:09, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:40, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:11, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:16, 702kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:15, 919kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:20, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:15, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:13, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:42, 1.70MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:13, 2.35MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<04:14, 673kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<03:17, 867kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:21, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:14, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:49, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:20, 2.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:58, 2.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:10, 857kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:49, 962kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:07, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:11<01:29, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<03:31, 753kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:45, 959kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:59, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:55, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:55, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:29, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<01:03, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<03:46, 665kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:54, 863kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:04, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:58, 1.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:53, 1.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:25, 1.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<01:00, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:06, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:43, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:15, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:23, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:12, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:53, 2.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:07, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:16, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:59, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:43, 3.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:10, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:04, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:47, 2.70MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:59, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:09, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:55, 2.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:39, 3.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:27, 823kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:56, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:24, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:23, 1.41MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:23, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:03, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:45, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:21, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:08, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:59, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:04, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:49, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:35, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:03, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:55, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:40, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:50, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:56, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:43, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:31, 3.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<01:07, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:57, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:42, 2.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:54, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:43, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:30, 2.91MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<02:08, 689kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:39, 887kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:10, 1.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:06, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:05, 1.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:49, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:34, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:49, 731kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:25, 936kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:00, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:57, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:57, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:43, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:30, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<01:48, 664kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:23, 858kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:58, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:54, 1.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:53, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:40, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:27, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<01:36, 658kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:13, 852kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:52, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:48, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:46, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:34, 1.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:24, 2.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:39, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:33, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:24, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:27, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:30, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:23, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:16, 2.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:37, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:31, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:22, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:26, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:20, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:09<00:13, 2.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:55, 696kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:42, 896kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:29, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:26, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:22, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:15, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:16, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:14, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:10, 2.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:14, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:10, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:11, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:08, 2.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 2.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:19, 691kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:14, 889kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:07, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:04, 1.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 1.89MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.04MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<22:41:09,  4.90it/s]  0%|          | 838/400000 [00:00<15:51:03,  7.00it/s]  0%|          | 1713/400000 [00:00<11:04:30,  9.99it/s]  1%|          | 2575/400000 [00:00<7:44:22, 14.26it/s]   1%|          | 3374/400000 [00:00<5:24:39, 20.36it/s]  1%|          | 4208/400000 [00:00<3:47:01, 29.06it/s]  1%|▏         | 5099/400000 [00:00<2:38:46, 41.45it/s]  1%|▏         | 5974/400000 [00:00<1:51:07, 59.10it/s]  2%|▏         | 6804/400000 [00:01<1:17:51, 84.17it/s]  2%|▏         | 7631/400000 [00:01<54:37, 119.72it/s]   2%|▏         | 8440/400000 [00:01<38:24, 169.88it/s]  2%|▏         | 9317/400000 [00:01<27:03, 240.68it/s]  3%|▎         | 10200/400000 [00:01<19:06, 339.86it/s]  3%|▎         | 11071/400000 [00:01<13:34, 477.53it/s]  3%|▎         | 11919/400000 [00:01<09:43, 665.22it/s]  3%|▎         | 12754/400000 [00:01<07:02, 916.50it/s]  3%|▎         | 13652/400000 [00:01<05:07, 1254.40it/s]  4%|▎         | 14530/400000 [00:01<03:48, 1688.60it/s]  4%|▍         | 15429/400000 [00:02<02:52, 2232.56it/s]  4%|▍         | 16296/400000 [00:02<02:13, 2872.23it/s]  4%|▍         | 17163/400000 [00:02<01:48, 3533.25it/s]  4%|▍         | 17999/400000 [00:02<01:29, 4265.97it/s]  5%|▍         | 18832/400000 [00:02<01:16, 4973.95it/s]  5%|▍         | 19659/400000 [00:02<01:08, 5575.76it/s]  5%|▌         | 20475/400000 [00:02<01:01, 6160.58it/s]  5%|▌         | 21288/400000 [00:02<00:58, 6516.79it/s]  6%|▌         | 22152/400000 [00:02<00:53, 7032.54it/s]  6%|▌         | 23007/400000 [00:02<00:50, 7426.09it/s]  6%|▌         | 23906/400000 [00:03<00:48, 7834.56it/s]  6%|▌         | 24814/400000 [00:03<00:45, 8169.83it/s]  6%|▋         | 25681/400000 [00:03<00:46, 8097.47it/s]  7%|▋         | 26526/400000 [00:03<00:46, 8047.45it/s]  7%|▋         | 27356/400000 [00:03<00:46, 8076.15it/s]  7%|▋         | 28203/400000 [00:03<00:45, 8187.68it/s]  7%|▋         | 29062/400000 [00:03<00:44, 8303.60it/s]  7%|▋         | 29902/400000 [00:03<00:44, 8240.45it/s]  8%|▊         | 30774/400000 [00:03<00:44, 8376.27it/s]  8%|▊         | 31666/400000 [00:03<00:43, 8531.06it/s]  8%|▊         | 32530/400000 [00:04<00:42, 8561.04it/s]  8%|▊         | 33407/400000 [00:04<00:42, 8619.05it/s]  9%|▊         | 34272/400000 [00:04<00:43, 8370.89it/s]  9%|▉         | 35117/400000 [00:04<00:43, 8394.29it/s]  9%|▉         | 36015/400000 [00:04<00:42, 8560.01it/s]  9%|▉         | 36888/400000 [00:04<00:42, 8609.09it/s]  9%|▉         | 37766/400000 [00:04<00:41, 8657.75it/s] 10%|▉         | 38634/400000 [00:04<00:42, 8412.86it/s] 10%|▉         | 39515/400000 [00:04<00:42, 8527.57it/s] 10%|█         | 40370/400000 [00:05<00:42, 8485.73it/s] 10%|█         | 41221/400000 [00:05<00:42, 8413.52it/s] 11%|█         | 42128/400000 [00:05<00:41, 8598.68it/s] 11%|█         | 42990/400000 [00:05<00:42, 8342.72it/s] 11%|█         | 43845/400000 [00:05<00:42, 8401.81it/s] 11%|█         | 44719/400000 [00:05<00:41, 8498.45it/s] 11%|█▏        | 45576/400000 [00:05<00:41, 8517.74it/s] 12%|█▏        | 46441/400000 [00:05<00:41, 8554.88it/s] 12%|█▏        | 47298/400000 [00:05<00:41, 8403.61it/s] 12%|█▏        | 48197/400000 [00:05<00:41, 8569.95it/s] 12%|█▏        | 49079/400000 [00:06<00:40, 8643.05it/s] 12%|█▏        | 49945/400000 [00:06<00:40, 8624.55it/s] 13%|█▎        | 50809/400000 [00:06<00:40, 8565.65it/s] 13%|█▎        | 51667/400000 [00:06<00:42, 8264.53it/s] 13%|█▎        | 52534/400000 [00:06<00:41, 8380.91it/s] 13%|█▎        | 53410/400000 [00:06<00:40, 8488.55it/s] 14%|█▎        | 54261/400000 [00:06<00:40, 8486.78it/s] 14%|█▍        | 55112/400000 [00:06<00:41, 8297.58it/s] 14%|█▍        | 55944/400000 [00:06<00:42, 8028.74it/s] 14%|█▍        | 56808/400000 [00:06<00:41, 8202.80it/s] 14%|█▍        | 57641/400000 [00:07<00:41, 8239.65it/s] 15%|█▍        | 58468/400000 [00:07<00:41, 8194.62it/s] 15%|█▍        | 59290/400000 [00:07<00:42, 8086.36it/s] 15%|█▌        | 60101/400000 [00:07<00:42, 8051.99it/s] 15%|█▌        | 60908/400000 [00:07<00:42, 8022.57it/s] 15%|█▌        | 61712/400000 [00:07<00:42, 7901.00it/s] 16%|█▌        | 62504/400000 [00:07<00:42, 7899.33it/s] 16%|█▌        | 63295/400000 [00:07<00:42, 7887.20it/s] 16%|█▌        | 64085/400000 [00:07<00:42, 7835.93it/s] 16%|█▌        | 64870/400000 [00:07<00:43, 7791.65it/s] 16%|█▋        | 65650/400000 [00:08<00:43, 7610.75it/s] 17%|█▋        | 66417/400000 [00:08<00:43, 7626.55it/s] 17%|█▋        | 67181/400000 [00:08<00:44, 7541.46it/s] 17%|█▋        | 67936/400000 [00:08<00:44, 7496.83it/s] 17%|█▋        | 68687/400000 [00:08<00:45, 7356.62it/s] 17%|█▋        | 69448/400000 [00:08<00:44, 7430.33it/s] 18%|█▊        | 70305/400000 [00:08<00:42, 7737.31it/s] 18%|█▊        | 71163/400000 [00:08<00:41, 7969.83it/s] 18%|█▊        | 72046/400000 [00:08<00:39, 8209.12it/s] 18%|█▊        | 72928/400000 [00:08<00:39, 8382.85it/s] 18%|█▊        | 73771/400000 [00:09<00:38, 8387.66it/s] 19%|█▊        | 74683/400000 [00:09<00:37, 8593.25it/s] 19%|█▉        | 75576/400000 [00:09<00:37, 8690.94it/s] 19%|█▉        | 76517/400000 [00:09<00:36, 8891.43it/s] 19%|█▉        | 77410/400000 [00:09<00:36, 8759.09it/s] 20%|█▉        | 78289/400000 [00:09<00:36, 8702.06it/s] 20%|█▉        | 79177/400000 [00:09<00:36, 8751.95it/s] 20%|██        | 80054/400000 [00:09<00:37, 8632.94it/s] 20%|██        | 80919/400000 [00:09<00:37, 8596.18it/s] 20%|██        | 81780/400000 [00:10<00:37, 8564.52it/s] 21%|██        | 82645/400000 [00:10<00:36, 8585.94it/s] 21%|██        | 83545/400000 [00:10<00:36, 8705.17it/s] 21%|██        | 84417/400000 [00:10<00:36, 8658.95it/s] 21%|██▏       | 85326/400000 [00:10<00:35, 8781.53it/s] 22%|██▏       | 86205/400000 [00:10<00:36, 8655.40it/s] 22%|██▏       | 87137/400000 [00:10<00:35, 8843.77it/s] 22%|██▏       | 88051/400000 [00:10<00:34, 8929.86it/s] 22%|██▏       | 88946/400000 [00:10<00:34, 8923.21it/s] 22%|██▏       | 89855/400000 [00:10<00:34, 8971.11it/s] 23%|██▎       | 90753/400000 [00:11<00:35, 8775.96it/s] 23%|██▎       | 91633/400000 [00:11<00:35, 8735.69it/s] 23%|██▎       | 92583/400000 [00:11<00:34, 8951.55it/s] 23%|██▎       | 93481/400000 [00:11<00:34, 8887.18it/s] 24%|██▎       | 94415/400000 [00:11<00:33, 9017.30it/s] 24%|██▍       | 95319/400000 [00:11<00:34, 8939.09it/s] 24%|██▍       | 96215/400000 [00:11<00:34, 8868.87it/s] 24%|██▍       | 97103/400000 [00:11<00:34, 8783.82it/s] 24%|██▍       | 97983/400000 [00:11<00:34, 8679.49it/s] 25%|██▍       | 98909/400000 [00:11<00:34, 8844.32it/s] 25%|██▍       | 99795/400000 [00:12<00:34, 8735.13it/s] 25%|██▌       | 100696/400000 [00:12<00:33, 8814.08it/s] 25%|██▌       | 101579/400000 [00:12<00:34, 8768.05it/s] 26%|██▌       | 102457/400000 [00:12<00:34, 8731.82it/s] 26%|██▌       | 103333/400000 [00:12<00:33, 8738.98it/s] 26%|██▌       | 104251/400000 [00:12<00:33, 8866.62it/s] 26%|██▋       | 105139/400000 [00:12<00:33, 8765.88it/s] 27%|██▋       | 106017/400000 [00:12<00:33, 8674.87it/s] 27%|██▋       | 106896/400000 [00:12<00:33, 8707.16it/s] 27%|██▋       | 107775/400000 [00:12<00:33, 8729.67it/s] 27%|██▋       | 108665/400000 [00:13<00:33, 8778.78it/s] 27%|██▋       | 109616/400000 [00:13<00:32, 8985.88it/s] 28%|██▊       | 110545/400000 [00:13<00:31, 9074.22it/s] 28%|██▊       | 111454/400000 [00:13<00:32, 8818.08it/s] 28%|██▊       | 112339/400000 [00:13<00:33, 8674.37it/s] 28%|██▊       | 113209/400000 [00:13<00:33, 8677.95it/s] 29%|██▊       | 114079/400000 [00:13<00:33, 8598.67it/s] 29%|██▊       | 114941/400000 [00:13<00:33, 8469.86it/s] 29%|██▉       | 115790/400000 [00:13<00:33, 8363.94it/s] 29%|██▉       | 116628/400000 [00:13<00:34, 8269.82it/s] 29%|██▉       | 117543/400000 [00:14<00:33, 8515.26it/s] 30%|██▉       | 118398/400000 [00:14<00:33, 8409.82it/s] 30%|██▉       | 119242/400000 [00:14<00:33, 8337.12it/s] 30%|███       | 120095/400000 [00:14<00:33, 8393.63it/s] 30%|███       | 121000/400000 [00:14<00:32, 8578.39it/s] 30%|███       | 121860/400000 [00:14<00:32, 8518.16it/s] 31%|███       | 122748/400000 [00:14<00:32, 8622.09it/s] 31%|███       | 123612/400000 [00:14<00:32, 8569.08it/s] 31%|███       | 124476/400000 [00:14<00:32, 8587.68it/s] 31%|███▏      | 125336/400000 [00:15<00:32, 8393.72it/s] 32%|███▏      | 126177/400000 [00:15<00:33, 8204.43it/s] 32%|███▏      | 127083/400000 [00:15<00:32, 8441.64it/s] 32%|███▏      | 128020/400000 [00:15<00:31, 8700.09it/s] 32%|███▏      | 128895/400000 [00:15<00:31, 8641.54it/s] 32%|███▏      | 129763/400000 [00:15<00:31, 8576.25it/s] 33%|███▎      | 130678/400000 [00:15<00:30, 8740.58it/s] 33%|███▎      | 131555/400000 [00:15<00:30, 8727.47it/s] 33%|███▎      | 132467/400000 [00:15<00:30, 8840.23it/s] 33%|███▎      | 133353/400000 [00:15<00:30, 8776.43it/s] 34%|███▎      | 134232/400000 [00:16<00:30, 8714.58it/s] 34%|███▍      | 135113/400000 [00:16<00:30, 8742.45it/s] 34%|███▍      | 135988/400000 [00:16<00:30, 8741.97it/s] 34%|███▍      | 136927/400000 [00:16<00:29, 8925.93it/s] 34%|███▍      | 137821/400000 [00:16<00:30, 8701.37it/s] 35%|███▍      | 138739/400000 [00:16<00:29, 8837.31it/s] 35%|███▍      | 139643/400000 [00:16<00:29, 8897.12it/s] 35%|███▌      | 140535/400000 [00:16<00:29, 8828.22it/s] 35%|███▌      | 141419/400000 [00:16<00:29, 8711.95it/s] 36%|███▌      | 142292/400000 [00:16<00:30, 8568.90it/s] 36%|███▌      | 143208/400000 [00:17<00:29, 8737.89it/s] 36%|███▌      | 144127/400000 [00:17<00:28, 8867.94it/s] 36%|███▋      | 145016/400000 [00:17<00:28, 8800.70it/s] 36%|███▋      | 145924/400000 [00:17<00:28, 8880.65it/s] 37%|███▋      | 146814/400000 [00:17<00:29, 8549.49it/s] 37%|███▋      | 147673/400000 [00:17<00:30, 8251.91it/s] 37%|███▋      | 148557/400000 [00:17<00:29, 8419.72it/s] 37%|███▋      | 149470/400000 [00:17<00:29, 8619.29it/s] 38%|███▊      | 150417/400000 [00:17<00:28, 8856.81it/s] 38%|███▊      | 151308/400000 [00:18<00:28, 8599.18it/s] 38%|███▊      | 152238/400000 [00:18<00:28, 8797.22it/s] 38%|███▊      | 153144/400000 [00:18<00:27, 8873.74it/s] 39%|███▊      | 154058/400000 [00:18<00:27, 8949.53it/s] 39%|███▊      | 154956/400000 [00:18<00:27, 8816.06it/s] 39%|███▉      | 155840/400000 [00:18<00:27, 8786.97it/s] 39%|███▉      | 156721/400000 [00:18<00:27, 8784.29it/s] 39%|███▉      | 157617/400000 [00:18<00:27, 8832.99it/s] 40%|███▉      | 158502/400000 [00:18<00:27, 8726.99it/s] 40%|███▉      | 159376/400000 [00:18<00:28, 8555.81it/s] 40%|████      | 160272/400000 [00:19<00:27, 8672.25it/s] 40%|████      | 161141/400000 [00:19<00:27, 8670.15it/s] 41%|████      | 162080/400000 [00:19<00:26, 8871.11it/s] 41%|████      | 163008/400000 [00:19<00:26, 8989.60it/s] 41%|████      | 163909/400000 [00:19<00:26, 8953.57it/s] 41%|████      | 164806/400000 [00:19<00:26, 8794.31it/s] 41%|████▏     | 165703/400000 [00:19<00:26, 8843.94it/s] 42%|████▏     | 166625/400000 [00:19<00:26, 8950.83it/s] 42%|████▏     | 167522/400000 [00:19<00:26, 8850.70it/s] 42%|████▏     | 168409/400000 [00:19<00:26, 8810.85it/s] 42%|████▏     | 169291/400000 [00:20<00:27, 8538.61it/s] 43%|████▎     | 170148/400000 [00:20<00:27, 8493.61it/s] 43%|████▎     | 171082/400000 [00:20<00:26, 8728.82it/s] 43%|████▎     | 171958/400000 [00:20<00:26, 8586.54it/s] 43%|████▎     | 172820/400000 [00:20<00:27, 8404.26it/s] 43%|████▎     | 173664/400000 [00:20<00:27, 8373.80it/s] 44%|████▎     | 174515/400000 [00:20<00:26, 8414.06it/s] 44%|████▍     | 175422/400000 [00:20<00:26, 8598.33it/s] 44%|████▍     | 176287/400000 [00:20<00:25, 8611.93it/s] 44%|████▍     | 177200/400000 [00:20<00:25, 8759.27it/s] 45%|████▍     | 178078/400000 [00:21<00:25, 8719.08it/s] 45%|████▍     | 178972/400000 [00:21<00:25, 8781.80it/s] 45%|████▍     | 179860/400000 [00:21<00:24, 8810.05it/s] 45%|████▌     | 180746/400000 [00:21<00:24, 8824.32it/s] 45%|████▌     | 181675/400000 [00:21<00:24, 8955.39it/s] 46%|████▌     | 182572/400000 [00:21<00:24, 8930.23it/s] 46%|████▌     | 183466/400000 [00:21<00:24, 8896.49it/s] 46%|████▌     | 184357/400000 [00:21<00:24, 8852.45it/s] 46%|████▋     | 185243/400000 [00:21<00:24, 8831.71it/s] 47%|████▋     | 186184/400000 [00:21<00:23, 8995.70it/s] 47%|████▋     | 187085/400000 [00:22<00:24, 8820.27it/s] 47%|████▋     | 187983/400000 [00:22<00:23, 8864.99it/s] 47%|████▋     | 188874/400000 [00:22<00:23, 8878.21it/s] 47%|████▋     | 189763/400000 [00:22<00:23, 8802.23it/s] 48%|████▊     | 190677/400000 [00:22<00:23, 8899.56it/s] 48%|████▊     | 191568/400000 [00:22<00:23, 8740.26it/s] 48%|████▊     | 192444/400000 [00:22<00:23, 8676.02it/s] 48%|████▊     | 193325/400000 [00:22<00:23, 8715.29it/s] 49%|████▊     | 194198/400000 [00:22<00:24, 8497.46it/s] 49%|████▉     | 195082/400000 [00:22<00:23, 8596.95it/s] 49%|████▉     | 195952/400000 [00:23<00:23, 8626.70it/s] 49%|████▉     | 196876/400000 [00:23<00:23, 8801.74it/s] 49%|████▉     | 197758/400000 [00:23<00:22, 8793.79it/s] 50%|████▉     | 198639/400000 [00:23<00:23, 8691.26it/s] 50%|████▉     | 199510/400000 [00:23<00:23, 8619.20it/s] 50%|█████     | 200373/400000 [00:23<00:23, 8377.91it/s] 50%|█████     | 201241/400000 [00:23<00:23, 8466.28it/s] 51%|█████     | 202140/400000 [00:23<00:22, 8615.99it/s] 51%|█████     | 203007/400000 [00:23<00:22, 8629.71it/s] 51%|█████     | 203924/400000 [00:24<00:22, 8784.15it/s] 51%|█████     | 204804/400000 [00:24<00:22, 8617.99it/s] 51%|█████▏    | 205668/400000 [00:24<00:22, 8601.25it/s] 52%|█████▏    | 206530/400000 [00:24<00:23, 8405.61it/s] 52%|█████▏    | 207409/400000 [00:24<00:22, 8516.73it/s] 52%|█████▏    | 208280/400000 [00:24<00:22, 8571.75it/s] 52%|█████▏    | 209139/400000 [00:24<00:22, 8460.05it/s] 53%|█████▎    | 210070/400000 [00:24<00:21, 8696.50it/s] 53%|█████▎    | 210943/400000 [00:24<00:22, 8362.70it/s] 53%|█████▎    | 211860/400000 [00:24<00:21, 8587.17it/s] 53%|█████▎    | 212750/400000 [00:25<00:21, 8677.04it/s] 53%|█████▎    | 213622/400000 [00:25<00:21, 8539.90it/s] 54%|█████▎    | 214529/400000 [00:25<00:21, 8689.45it/s] 54%|█████▍    | 215401/400000 [00:25<00:21, 8610.90it/s] 54%|█████▍    | 216272/400000 [00:25<00:21, 8637.01it/s] 54%|█████▍    | 217165/400000 [00:25<00:20, 8720.97it/s] 55%|█████▍    | 218039/400000 [00:25<00:21, 8532.98it/s] 55%|█████▍    | 218925/400000 [00:25<00:20, 8626.62it/s] 55%|█████▍    | 219790/400000 [00:25<00:21, 8381.66it/s] 55%|█████▌    | 220649/400000 [00:25<00:21, 8442.42it/s] 55%|█████▌    | 221496/400000 [00:26<00:21, 8421.20it/s] 56%|█████▌    | 222357/400000 [00:26<00:20, 8475.47it/s] 56%|█████▌    | 223235/400000 [00:26<00:20, 8562.96it/s] 56%|█████▌    | 224093/400000 [00:26<00:20, 8566.82it/s] 56%|█████▌    | 224982/400000 [00:26<00:20, 8661.07it/s] 56%|█████▋    | 225849/400000 [00:26<00:20, 8612.82it/s] 57%|█████▋    | 226711/400000 [00:26<00:20, 8498.80it/s] 57%|█████▋    | 227562/400000 [00:26<00:20, 8489.32it/s] 57%|█████▋    | 228412/400000 [00:26<00:20, 8352.35it/s] 57%|█████▋    | 229303/400000 [00:26<00:20, 8509.93it/s] 58%|█████▊    | 230224/400000 [00:27<00:19, 8706.67it/s] 58%|█████▊    | 231101/400000 [00:27<00:19, 8723.26it/s] 58%|█████▊    | 231975/400000 [00:27<00:19, 8584.89it/s] 58%|█████▊    | 232835/400000 [00:27<00:19, 8412.05it/s] 58%|█████▊    | 233683/400000 [00:27<00:19, 8429.54it/s] 59%|█████▊    | 234531/400000 [00:27<00:19, 8442.66it/s] 59%|█████▉    | 235400/400000 [00:27<00:19, 8513.25it/s] 59%|█████▉    | 236253/400000 [00:27<00:19, 8516.82it/s] 59%|█████▉    | 237166/400000 [00:27<00:18, 8690.08it/s] 60%|█████▉    | 238103/400000 [00:28<00:18, 8880.55it/s] 60%|█████▉    | 238993/400000 [00:28<00:18, 8758.75it/s] 60%|█████▉    | 239871/400000 [00:28<00:18, 8762.14it/s] 60%|██████    | 240749/400000 [00:28<00:18, 8612.16it/s] 60%|██████    | 241612/400000 [00:28<00:18, 8549.29it/s] 61%|██████    | 242483/400000 [00:28<00:18, 8595.78it/s] 61%|██████    | 243366/400000 [00:28<00:18, 8664.07it/s] 61%|██████    | 244234/400000 [00:28<00:18, 8544.38it/s] 61%|██████▏   | 245090/400000 [00:28<00:18, 8259.00it/s] 61%|██████▏   | 245919/400000 [00:28<00:18, 8250.91it/s] 62%|██████▏   | 246859/400000 [00:29<00:17, 8563.08it/s] 62%|██████▏   | 247720/400000 [00:29<00:17, 8525.62it/s] 62%|██████▏   | 248576/400000 [00:29<00:17, 8490.66it/s] 62%|██████▏   | 249472/400000 [00:29<00:17, 8624.05it/s] 63%|██████▎   | 250380/400000 [00:29<00:17, 8753.82it/s] 63%|██████▎   | 251258/400000 [00:29<00:16, 8755.26it/s] 63%|██████▎   | 252135/400000 [00:29<00:16, 8732.53it/s] 63%|██████▎   | 253010/400000 [00:29<00:17, 8523.24it/s] 63%|██████▎   | 253865/400000 [00:29<00:17, 8490.64it/s] 64%|██████▎   | 254800/400000 [00:29<00:16, 8728.80it/s] 64%|██████▍   | 255693/400000 [00:30<00:16, 8786.69it/s] 64%|██████▍   | 256574/400000 [00:30<00:16, 8755.92it/s] 64%|██████▍   | 257499/400000 [00:30<00:16, 8897.34it/s] 65%|██████▍   | 258402/400000 [00:30<00:15, 8934.64it/s] 65%|██████▍   | 259297/400000 [00:30<00:15, 8903.32it/s] 65%|██████▌   | 260221/400000 [00:30<00:15, 8999.75it/s] 65%|██████▌   | 261122/400000 [00:30<00:15, 8771.56it/s] 66%|██████▌   | 262001/400000 [00:30<00:15, 8683.90it/s] 66%|██████▌   | 262871/400000 [00:30<00:15, 8620.20it/s] 66%|██████▌   | 263735/400000 [00:30<00:15, 8535.69it/s] 66%|██████▌   | 264593/400000 [00:31<00:15, 8547.29it/s] 66%|██████▋   | 265449/400000 [00:31<00:16, 8361.43it/s] 67%|██████▋   | 266287/400000 [00:31<00:16, 8271.27it/s] 67%|██████▋   | 267116/400000 [00:31<00:16, 7932.95it/s] 67%|██████▋   | 267918/400000 [00:31<00:16, 7956.16it/s] 67%|██████▋   | 268777/400000 [00:31<00:16, 8135.08it/s] 67%|██████▋   | 269594/400000 [00:31<00:16, 8056.30it/s] 68%|██████▊   | 270402/400000 [00:31<00:16, 8060.33it/s] 68%|██████▊   | 271210/400000 [00:31<00:16, 8044.27it/s] 68%|██████▊   | 272047/400000 [00:32<00:15, 8138.75it/s] 68%|██████▊   | 272862/400000 [00:32<00:15, 8076.80it/s] 68%|██████▊   | 273678/400000 [00:32<00:15, 8100.50it/s] 69%|██████▊   | 274528/400000 [00:32<00:15, 8215.97it/s] 69%|██████▉   | 275351/400000 [00:32<00:15, 8108.93it/s] 69%|██████▉   | 276163/400000 [00:32<00:15, 8085.87it/s] 69%|██████▉   | 276973/400000 [00:32<00:15, 7982.22it/s] 69%|██████▉   | 277815/400000 [00:32<00:15, 8108.45it/s] 70%|██████▉   | 278720/400000 [00:32<00:14, 8368.85it/s] 70%|██████▉   | 279560/400000 [00:32<00:14, 8273.61it/s] 70%|███████   | 280429/400000 [00:33<00:14, 8391.50it/s] 70%|███████   | 281307/400000 [00:33<00:13, 8503.10it/s] 71%|███████   | 282161/400000 [00:33<00:13, 8509.75it/s] 71%|███████   | 283026/400000 [00:33<00:13, 8550.60it/s] 71%|███████   | 283882/400000 [00:33<00:13, 8345.29it/s] 71%|███████   | 284719/400000 [00:33<00:13, 8271.80it/s] 71%|███████▏  | 285596/400000 [00:33<00:13, 8414.18it/s] 72%|███████▏  | 286467/400000 [00:33<00:13, 8497.17it/s] 72%|███████▏  | 287357/400000 [00:33<00:13, 8610.77it/s] 72%|███████▏  | 288220/400000 [00:33<00:13, 8352.89it/s] 72%|███████▏  | 289128/400000 [00:34<00:12, 8556.91it/s] 73%|███████▎  | 290046/400000 [00:34<00:12, 8733.79it/s] 73%|███████▎  | 290924/400000 [00:34<00:12, 8746.77it/s] 73%|███████▎  | 291801/400000 [00:34<00:13, 8231.74it/s] 73%|███████▎  | 292632/400000 [00:34<00:13, 8009.89it/s] 73%|███████▎  | 293466/400000 [00:34<00:13, 8105.29it/s] 74%|███████▎  | 294354/400000 [00:34<00:12, 8320.10it/s] 74%|███████▍  | 295193/400000 [00:34<00:12, 8340.49it/s] 74%|███████▍  | 296033/400000 [00:34<00:12, 8355.97it/s] 74%|███████▍  | 296872/400000 [00:34<00:12, 8218.52it/s] 74%|███████▍  | 297706/400000 [00:35<00:12, 8252.90it/s] 75%|███████▍  | 298586/400000 [00:35<00:12, 8408.33it/s] 75%|███████▍  | 299429/400000 [00:35<00:11, 8413.39it/s] 75%|███████▌  | 300312/400000 [00:35<00:11, 8531.88it/s] 75%|███████▌  | 301224/400000 [00:35<00:11, 8700.19it/s] 76%|███████▌  | 302137/400000 [00:35<00:11, 8823.67it/s] 76%|███████▌  | 303052/400000 [00:35<00:10, 8917.50it/s] 76%|███████▌  | 303946/400000 [00:35<00:10, 8841.48it/s] 76%|███████▌  | 304847/400000 [00:35<00:10, 8890.88it/s] 76%|███████▋  | 305769/400000 [00:35<00:10, 8986.19it/s] 77%|███████▋  | 306669/400000 [00:36<00:10, 8952.20it/s] 77%|███████▋  | 307589/400000 [00:36<00:10, 9023.73it/s] 77%|███████▋  | 308492/400000 [00:36<00:10, 8996.81it/s] 77%|███████▋  | 309393/400000 [00:36<00:10, 8922.11it/s] 78%|███████▊  | 310300/400000 [00:36<00:10, 8964.73it/s] 78%|███████▊  | 311197/400000 [00:36<00:09, 8959.24it/s] 78%|███████▊  | 312096/400000 [00:36<00:09, 8966.98it/s] 78%|███████▊  | 312993/400000 [00:36<00:10, 8659.24it/s] 78%|███████▊  | 313862/400000 [00:36<00:10, 8580.51it/s] 79%|███████▊  | 314743/400000 [00:37<00:09, 8647.70it/s] 79%|███████▉  | 315688/400000 [00:37<00:09, 8872.07it/s] 79%|███████▉  | 316591/400000 [00:37<00:09, 8918.79it/s] 79%|███████▉  | 317485/400000 [00:37<00:09, 8608.88it/s] 80%|███████▉  | 318350/400000 [00:37<00:09, 8517.97it/s] 80%|███████▉  | 319205/400000 [00:37<00:09, 8412.32it/s] 80%|████████  | 320087/400000 [00:37<00:09, 8528.93it/s] 80%|████████  | 320970/400000 [00:37<00:09, 8616.73it/s] 80%|████████  | 321834/400000 [00:37<00:09, 8575.51it/s] 81%|████████  | 322728/400000 [00:37<00:08, 8680.42it/s] 81%|████████  | 323598/400000 [00:38<00:08, 8655.88it/s] 81%|████████  | 324546/400000 [00:38<00:08, 8886.51it/s] 81%|████████▏ | 325437/400000 [00:38<00:08, 8879.56it/s] 82%|████████▏ | 326327/400000 [00:38<00:08, 8702.20it/s] 82%|████████▏ | 327204/400000 [00:38<00:08, 8720.58it/s] 82%|████████▏ | 328096/400000 [00:38<00:08, 8775.03it/s] 82%|████████▏ | 329003/400000 [00:38<00:08, 8859.46it/s] 82%|████████▏ | 329956/400000 [00:38<00:07, 9049.78it/s] 83%|████████▎ | 330863/400000 [00:38<00:07, 9017.39it/s] 83%|████████▎ | 331788/400000 [00:38<00:07, 9084.38it/s] 83%|████████▎ | 332743/400000 [00:39<00:07, 9216.76it/s] 83%|████████▎ | 333666/400000 [00:39<00:07, 9122.03it/s] 84%|████████▎ | 334580/400000 [00:39<00:07, 9063.11it/s] 84%|████████▍ | 335520/400000 [00:39<00:07, 9159.75it/s] 84%|████████▍ | 336456/400000 [00:39<00:06, 9217.91it/s] 84%|████████▍ | 337379/400000 [00:39<00:07, 8942.73it/s] 85%|████████▍ | 338276/400000 [00:39<00:06, 8872.05it/s] 85%|████████▍ | 339196/400000 [00:39<00:06, 8965.19it/s] 85%|████████▌ | 340094/400000 [00:39<00:06, 8884.86it/s] 85%|████████▌ | 340987/400000 [00:39<00:06, 8898.27it/s] 85%|████████▌ | 341878/400000 [00:40<00:06, 8821.20it/s] 86%|████████▌ | 342800/400000 [00:40<00:06, 8934.74it/s] 86%|████████▌ | 343695/400000 [00:40<00:06, 8896.03it/s] 86%|████████▌ | 344586/400000 [00:40<00:06, 8825.06it/s] 86%|████████▋ | 345503/400000 [00:40<00:06, 8923.35it/s] 87%|████████▋ | 346425/400000 [00:40<00:05, 9009.44it/s] 87%|████████▋ | 347327/400000 [00:40<00:05, 8960.44it/s] 87%|████████▋ | 348224/400000 [00:40<00:05, 8951.51it/s] 87%|████████▋ | 349120/400000 [00:40<00:05, 8795.07it/s] 88%|████████▊ | 350046/400000 [00:40<00:05, 8928.59it/s] 88%|████████▊ | 350940/400000 [00:41<00:05, 8863.06it/s] 88%|████████▊ | 351846/400000 [00:41<00:05, 8920.18it/s] 88%|████████▊ | 352763/400000 [00:41<00:05, 8992.40it/s] 88%|████████▊ | 353663/400000 [00:41<00:05, 8983.58it/s] 89%|████████▊ | 354603/400000 [00:41<00:04, 9103.22it/s] 89%|████████▉ | 355561/400000 [00:41<00:04, 9240.18it/s] 89%|████████▉ | 356487/400000 [00:41<00:04, 9100.05it/s] 89%|████████▉ | 357399/400000 [00:41<00:04, 9022.17it/s] 90%|████████▉ | 358303/400000 [00:41<00:04, 8904.46it/s] 90%|████████▉ | 359237/400000 [00:41<00:04, 9028.55it/s] 90%|█████████ | 360152/400000 [00:42<00:04, 9064.44it/s] 90%|█████████ | 361063/400000 [00:42<00:04, 9077.91it/s] 90%|█████████ | 361972/400000 [00:42<00:04, 8840.39it/s] 91%|█████████ | 362858/400000 [00:42<00:04, 8831.56it/s] 91%|█████████ | 363757/400000 [00:42<00:04, 8876.59it/s] 91%|█████████ | 364646/400000 [00:42<00:04, 8717.09it/s] 91%|█████████▏| 365519/400000 [00:42<00:03, 8644.23it/s] 92%|█████████▏| 366416/400000 [00:42<00:03, 8737.51it/s] 92%|█████████▏| 367291/400000 [00:42<00:03, 8729.68it/s] 92%|█████████▏| 368165/400000 [00:43<00:03, 8611.05it/s] 92%|█████████▏| 369115/400000 [00:43<00:03, 8859.56it/s] 93%|█████████▎| 370004/400000 [00:43<00:03, 8793.85it/s] 93%|█████████▎| 370886/400000 [00:43<00:03, 8714.79it/s] 93%|█████████▎| 371759/400000 [00:43<00:03, 8604.05it/s] 93%|█████████▎| 372674/400000 [00:43<00:03, 8759.95it/s] 93%|█████████▎| 373563/400000 [00:43<00:03, 8797.79it/s] 94%|█████████▎| 374506/400000 [00:43<00:02, 8978.15it/s] 94%|█████████▍| 375406/400000 [00:43<00:02, 8869.54it/s] 94%|█████████▍| 376295/400000 [00:43<00:02, 8740.43it/s] 94%|█████████▍| 377171/400000 [00:44<00:02, 8648.85it/s] 95%|█████████▍| 378040/400000 [00:44<00:02, 8661.09it/s] 95%|█████████▍| 378930/400000 [00:44<00:02, 8730.05it/s] 95%|█████████▍| 379804/400000 [00:44<00:02, 8732.63it/s] 95%|█████████▌| 380686/400000 [00:44<00:02, 8755.85it/s] 95%|█████████▌| 381562/400000 [00:44<00:02, 8628.04it/s] 96%|█████████▌| 382436/400000 [00:44<00:02, 8659.12it/s] 96%|█████████▌| 383303/400000 [00:44<00:01, 8647.89it/s] 96%|█████████▌| 384199/400000 [00:44<00:01, 8737.30it/s] 96%|█████████▋| 385074/400000 [00:44<00:01, 8657.11it/s] 96%|█████████▋| 385941/400000 [00:45<00:01, 8547.75it/s] 97%|█████████▋| 386829/400000 [00:45<00:01, 8644.60it/s] 97%|█████████▋| 387734/400000 [00:45<00:01, 8759.95it/s] 97%|█████████▋| 388667/400000 [00:45<00:01, 8921.77it/s] 97%|█████████▋| 389561/400000 [00:45<00:01, 8912.76it/s] 98%|█████████▊| 390488/400000 [00:45<00:01, 9016.72it/s] 98%|█████████▊| 391391/400000 [00:45<00:00, 8891.69it/s] 98%|█████████▊| 392306/400000 [00:45<00:00, 8967.18it/s] 98%|█████████▊| 393222/400000 [00:45<00:00, 9023.56it/s] 99%|█████████▊| 394126/400000 [00:45<00:00, 8822.71it/s] 99%|█████████▉| 395029/400000 [00:46<00:00, 8883.65it/s] 99%|█████████▉| 395919/400000 [00:46<00:00, 8800.18it/s] 99%|█████████▉| 396836/400000 [00:46<00:00, 8905.54it/s] 99%|█████████▉| 397747/400000 [00:46<00:00, 8963.56it/s]100%|█████████▉| 398645/400000 [00:46<00:00, 8776.85it/s]100%|█████████▉| 399593/400000 [00:46<00:00, 8974.29it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8577.92it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff0182d90f0>, <torchtext.data.dataset.TabularDataset object at 0x7ff0182d9240>, <torchtext.vocab.Vocab object at 0x7ff0182d9160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.80 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 20.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 14.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.33 file/s]2020-06-12 00:18:03.476741: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-12 00:18:03.480821: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-12 00:18:03.481028: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561d00232030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-12 00:18:03.481048: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144645.79it/s] 74%|███████▍  | 7315456/9912422 [00:00<00:12, 206456.09it/s]9920512it [00:00, 42592028.30it/s]                           
0it [00:00, ?it/s]32768it [00:00, 610031.84it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 4388893.97it/s]          
0it [00:00, ?it/s]8192it [00:00, 212140.36it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
