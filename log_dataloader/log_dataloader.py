
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fdb03f51620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fdb03f51620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fdb6f5180d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fdb6f5180d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fdb89245ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fdb89245ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdb1cd95950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdb1cd95950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdb1cd95950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 141565.01it/s] 54%|█████▍    | 5398528/9912422 [00:00<00:22, 202007.45it/s]9920512it [00:00, 34213588.99it/s]                           
0it [00:00, ?it/s]32768it [00:00, 485236.79it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 144349.24it/s]1654784it [00:00, 10835802.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 167715.65it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdb1a3f5b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdb1a3eada0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fdb1cd95598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fdb1cd95598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fdb1cd95598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  4.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:16,  7.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 51.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 68.81 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.54s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.81 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.54s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.54s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.54s/ url]
0 examples [00:00, ? examples/s]2020-06-08 00:24:22.378315: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-08 00:24:22.392299: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-08 00:24:22.392472: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56059cb5c4a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 00:24:22.392508: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
11 examples [00:00, 109.06 examples/s]107 examples [00:00, 148.57 examples/s]205 examples [00:00, 199.25 examples/s]299 examples [00:00, 260.80 examples/s]391 examples [00:00, 331.81 examples/s]482 examples [00:00, 409.85 examples/s]574 examples [00:00, 491.12 examples/s]666 examples [00:00, 570.02 examples/s]755 examples [00:00, 637.83 examples/s]843 examples [00:01, 693.98 examples/s]937 examples [00:01, 751.42 examples/s]1032 examples [00:01, 800.70 examples/s]1128 examples [00:01, 842.64 examples/s]1220 examples [00:01, 840.35 examples/s]1315 examples [00:01, 870.29 examples/s]1412 examples [00:01, 896.07 examples/s]1505 examples [00:01, 902.89 examples/s]1598 examples [00:01, 890.81 examples/s]1689 examples [00:01, 891.71 examples/s]1780 examples [00:02, 892.39 examples/s]1876 examples [00:02, 910.38 examples/s]1968 examples [00:02, 905.90 examples/s]2063 examples [00:02, 916.55 examples/s]2155 examples [00:02, 911.64 examples/s]2257 examples [00:02, 940.78 examples/s]2354 examples [00:02, 946.54 examples/s]2449 examples [00:02, 928.42 examples/s]2543 examples [00:02, 900.54 examples/s]2638 examples [00:02, 910.48 examples/s]2730 examples [00:03, 912.95 examples/s]2826 examples [00:03, 926.37 examples/s]2919 examples [00:03, 916.09 examples/s]3011 examples [00:03, 914.31 examples/s]3103 examples [00:03, 894.47 examples/s]3195 examples [00:03, 899.78 examples/s]3287 examples [00:03, 903.12 examples/s]3382 examples [00:03, 915.13 examples/s]3474 examples [00:03, 905.21 examples/s]3565 examples [00:03, 895.60 examples/s]3656 examples [00:04, 899.06 examples/s]3746 examples [00:04, 877.65 examples/s]3834 examples [00:04, 872.53 examples/s]3922 examples [00:04, 870.04 examples/s]4010 examples [00:04, 870.75 examples/s]4102 examples [00:04, 881.93 examples/s]4197 examples [00:04, 901.21 examples/s]4288 examples [00:04, 903.06 examples/s]4379 examples [00:04, 898.35 examples/s]4469 examples [00:05, 891.40 examples/s]4559 examples [00:05, 887.95 examples/s]4648 examples [00:05, 881.25 examples/s]4737 examples [00:05, 875.31 examples/s]4827 examples [00:05, 882.56 examples/s]4917 examples [00:05, 885.91 examples/s]5007 examples [00:05, 890.05 examples/s]5101 examples [00:05, 904.03 examples/s]5192 examples [00:05, 899.85 examples/s]5283 examples [00:05, 900.76 examples/s]5374 examples [00:06, 897.01 examples/s]5464 examples [00:06, 897.89 examples/s]5554 examples [00:06, 895.88 examples/s]5644 examples [00:06, 893.58 examples/s]5735 examples [00:06, 896.68 examples/s]5825 examples [00:06, 883.13 examples/s]5914 examples [00:06, 854.28 examples/s]6000 examples [00:06, 852.78 examples/s]6087 examples [00:06, 856.78 examples/s]6175 examples [00:06, 863.29 examples/s]6263 examples [00:07, 866.59 examples/s]6352 examples [00:07, 872.08 examples/s]6440 examples [00:07, 869.53 examples/s]6527 examples [00:07, 868.80 examples/s]6614 examples [00:07, 842.42 examples/s]6699 examples [00:07, 830.85 examples/s]6786 examples [00:07, 840.38 examples/s]6876 examples [00:07, 857.05 examples/s]6962 examples [00:07, 857.42 examples/s]7048 examples [00:07, 855.11 examples/s]7134 examples [00:08, 822.43 examples/s]7229 examples [00:08, 856.35 examples/s]7322 examples [00:08, 875.15 examples/s]7411 examples [00:08, 851.37 examples/s]7504 examples [00:08, 871.47 examples/s]7603 examples [00:08, 901.86 examples/s]7694 examples [00:08, 884.24 examples/s]7783 examples [00:08, 879.13 examples/s]7878 examples [00:08, 899.06 examples/s]7969 examples [00:09, 898.41 examples/s]8065 examples [00:09, 914.49 examples/s]8157 examples [00:09, 907.27 examples/s]8252 examples [00:09, 917.76 examples/s]8344 examples [00:09, 917.98 examples/s]8441 examples [00:09, 932.64 examples/s]8535 examples [00:09, 922.91 examples/s]8628 examples [00:09, 895.74 examples/s]8719 examples [00:09, 897.86 examples/s]8812 examples [00:09, 906.00 examples/s]8909 examples [00:10, 922.57 examples/s]9002 examples [00:10, 924.77 examples/s]9095 examples [00:10, 922.80 examples/s]9189 examples [00:10, 925.66 examples/s]9282 examples [00:10, 888.10 examples/s]9372 examples [00:10, 881.43 examples/s]9463 examples [00:10, 889.51 examples/s]9553 examples [00:10, 888.68 examples/s]9643 examples [00:10, 889.70 examples/s]9733 examples [00:10, 890.95 examples/s]9823 examples [00:11, 880.18 examples/s]9912 examples [00:11, 880.89 examples/s]10001 examples [00:11, 820.99 examples/s]10084 examples [00:11, 815.54 examples/s]10171 examples [00:11, 830.29 examples/s]10258 examples [00:11, 840.44 examples/s]10345 examples [00:11, 847.29 examples/s]10435 examples [00:11, 861.06 examples/s]10523 examples [00:11, 866.58 examples/s]10610 examples [00:11, 842.83 examples/s]10695 examples [00:12, 837.25 examples/s]10787 examples [00:12, 858.93 examples/s]10881 examples [00:12, 881.63 examples/s]10974 examples [00:12, 893.38 examples/s]11067 examples [00:12, 903.87 examples/s]11158 examples [00:12, 905.34 examples/s]11251 examples [00:12, 911.85 examples/s]11344 examples [00:12, 914.01 examples/s]11437 examples [00:12, 916.90 examples/s]11530 examples [00:13, 918.30 examples/s]11622 examples [00:13, 905.33 examples/s]11713 examples [00:13, 873.48 examples/s]11806 examples [00:13, 887.31 examples/s]11895 examples [00:13, 870.08 examples/s]11983 examples [00:13, 871.29 examples/s]12072 examples [00:13, 876.04 examples/s]12160 examples [00:13, 863.88 examples/s]12248 examples [00:13, 867.37 examples/s]12335 examples [00:13, 853.54 examples/s]12423 examples [00:14, 854.31 examples/s]12510 examples [00:14, 857.56 examples/s]12600 examples [00:14, 867.90 examples/s]12694 examples [00:14, 887.98 examples/s]12784 examples [00:14, 889.76 examples/s]12876 examples [00:14, 895.57 examples/s]12966 examples [00:14, 889.98 examples/s]13056 examples [00:14, 882.56 examples/s]13150 examples [00:14, 898.93 examples/s]13241 examples [00:14, 899.84 examples/s]13332 examples [00:15, 892.21 examples/s]13422 examples [00:15, 887.91 examples/s]13521 examples [00:15, 913.97 examples/s]13619 examples [00:15, 932.46 examples/s]13717 examples [00:15, 944.95 examples/s]13815 examples [00:15, 954.97 examples/s]13911 examples [00:15, 950.66 examples/s]14007 examples [00:15, 945.63 examples/s]14102 examples [00:15, 933.66 examples/s]14200 examples [00:15, 944.19 examples/s]14295 examples [00:16, 934.61 examples/s]14389 examples [00:16, 916.84 examples/s]14481 examples [00:16, 915.10 examples/s]14573 examples [00:16, 895.97 examples/s]14667 examples [00:16, 907.81 examples/s]14762 examples [00:16, 919.88 examples/s]14855 examples [00:16, 914.13 examples/s]14947 examples [00:16, 870.61 examples/s]15039 examples [00:16, 882.12 examples/s]15129 examples [00:17, 886.06 examples/s]15224 examples [00:17, 902.79 examples/s]15323 examples [00:17, 925.62 examples/s]15428 examples [00:17, 957.43 examples/s]15526 examples [00:17, 962.74 examples/s]15623 examples [00:17, 928.59 examples/s]15717 examples [00:17, 928.48 examples/s]15811 examples [00:17, 918.84 examples/s]15904 examples [00:17, 916.46 examples/s]15996 examples [00:17, 911.86 examples/s]16088 examples [00:18, 911.74 examples/s]16184 examples [00:18, 923.33 examples/s]16277 examples [00:18, 908.67 examples/s]16368 examples [00:18, 902.53 examples/s]16459 examples [00:18, 890.03 examples/s]16552 examples [00:18, 901.06 examples/s]16643 examples [00:18, 896.33 examples/s]16733 examples [00:18, 891.05 examples/s]16823 examples [00:18, 891.83 examples/s]16913 examples [00:18, 883.56 examples/s]17002 examples [00:19, 879.90 examples/s]17091 examples [00:19, 880.10 examples/s]17180 examples [00:19, 873.30 examples/s]17268 examples [00:19, 849.65 examples/s]17354 examples [00:19, 848.47 examples/s]17444 examples [00:19, 861.08 examples/s]17531 examples [00:19, 842.60 examples/s]17619 examples [00:19, 851.70 examples/s]17705 examples [00:19, 851.27 examples/s]17796 examples [00:19, 866.86 examples/s]17883 examples [00:20, 843.40 examples/s]17973 examples [00:20, 859.41 examples/s]18061 examples [00:20, 863.84 examples/s]18157 examples [00:20, 889.80 examples/s]18247 examples [00:20, 888.00 examples/s]18337 examples [00:20, 890.09 examples/s]18427 examples [00:20, 889.59 examples/s]18517 examples [00:20, 888.60 examples/s]18618 examples [00:20, 919.25 examples/s]18711 examples [00:21, 912.39 examples/s]18804 examples [00:21, 915.14 examples/s]18896 examples [00:21, 915.59 examples/s]18989 examples [00:21, 916.88 examples/s]19085 examples [00:21, 928.16 examples/s]19181 examples [00:21, 935.25 examples/s]19275 examples [00:21, 913.51 examples/s]19367 examples [00:21, 905.92 examples/s]19458 examples [00:21, 900.29 examples/s]19549 examples [00:21, 902.07 examples/s]19640 examples [00:22, 903.28 examples/s]19731 examples [00:22, 901.93 examples/s]19823 examples [00:22, 905.77 examples/s]19914 examples [00:22, 901.57 examples/s]20005 examples [00:22, 797.81 examples/s]20093 examples [00:22, 819.92 examples/s]20183 examples [00:22, 840.12 examples/s]20273 examples [00:22, 857.18 examples/s]20366 examples [00:22, 877.30 examples/s]20461 examples [00:22, 895.68 examples/s]20563 examples [00:23, 928.05 examples/s]20660 examples [00:23, 940.24 examples/s]20755 examples [00:23, 934.62 examples/s]20849 examples [00:23, 935.69 examples/s]20946 examples [00:23, 944.78 examples/s]21046 examples [00:23, 959.55 examples/s]21143 examples [00:23, 958.28 examples/s]21239 examples [00:23, 951.35 examples/s]21335 examples [00:23, 945.42 examples/s]21432 examples [00:24, 951.03 examples/s]21528 examples [00:24, 939.22 examples/s]21626 examples [00:24, 950.99 examples/s]21722 examples [00:24, 940.46 examples/s]21820 examples [00:24, 950.22 examples/s]21916 examples [00:24, 938.40 examples/s]22018 examples [00:24, 959.72 examples/s]22115 examples [00:24, 954.39 examples/s]22211 examples [00:24, 928.46 examples/s]22305 examples [00:24, 909.80 examples/s]22397 examples [00:25, 891.35 examples/s]22487 examples [00:25, 879.16 examples/s]22580 examples [00:25, 891.31 examples/s]22670 examples [00:25, 889.22 examples/s]22760 examples [00:25, 887.39 examples/s]22852 examples [00:25, 896.58 examples/s]22946 examples [00:25, 907.63 examples/s]23043 examples [00:25, 923.16 examples/s]23137 examples [00:25, 927.90 examples/s]23236 examples [00:25, 945.19 examples/s]23334 examples [00:26, 954.86 examples/s]23435 examples [00:26, 969.43 examples/s]23542 examples [00:26, 994.89 examples/s]23642 examples [00:26, 980.88 examples/s]23742 examples [00:26, 985.30 examples/s]23844 examples [00:26, 993.53 examples/s]23944 examples [00:26, 979.74 examples/s]24043 examples [00:26, 967.28 examples/s]24140 examples [00:26, 963.69 examples/s]24237 examples [00:26, 944.08 examples/s]24336 examples [00:27, 956.61 examples/s]24432 examples [00:27, 945.62 examples/s]24527 examples [00:27, 941.80 examples/s]24622 examples [00:27, 940.77 examples/s]24722 examples [00:27, 955.99 examples/s]24819 examples [00:27, 959.77 examples/s]24920 examples [00:27, 972.98 examples/s]25018 examples [00:27, 961.20 examples/s]25115 examples [00:27, 933.17 examples/s]25209 examples [00:28, 933.72 examples/s]25303 examples [00:28, 924.69 examples/s]25409 examples [00:28, 960.75 examples/s]25513 examples [00:28, 981.83 examples/s]25612 examples [00:28, 946.40 examples/s]25708 examples [00:28, 929.30 examples/s]25802 examples [00:28, 929.23 examples/s]25896 examples [00:28, 925.20 examples/s]25989 examples [00:28, 917.18 examples/s]26081 examples [00:28, 892.92 examples/s]26174 examples [00:29, 903.04 examples/s]26274 examples [00:29, 929.48 examples/s]26377 examples [00:29, 957.48 examples/s]26474 examples [00:29, 940.06 examples/s]26569 examples [00:29, 918.35 examples/s]26664 examples [00:29, 925.63 examples/s]26757 examples [00:29, 925.37 examples/s]26850 examples [00:29, 914.99 examples/s]26942 examples [00:29, 903.06 examples/s]27033 examples [00:29, 903.88 examples/s]27124 examples [00:30, 901.15 examples/s]27215 examples [00:30, 896.76 examples/s]27305 examples [00:30, 893.33 examples/s]27395 examples [00:30, 887.54 examples/s]27485 examples [00:30, 888.81 examples/s]27575 examples [00:30, 891.55 examples/s]27666 examples [00:30, 896.83 examples/s]27758 examples [00:30, 901.85 examples/s]27849 examples [00:30, 899.41 examples/s]27942 examples [00:30, 907.58 examples/s]28034 examples [00:31, 910.04 examples/s]28128 examples [00:31, 916.34 examples/s]28224 examples [00:31, 928.33 examples/s]28317 examples [00:31, 918.62 examples/s]28409 examples [00:31, 906.28 examples/s]28500 examples [00:31, 882.38 examples/s]28589 examples [00:31, 882.66 examples/s]28678 examples [00:31, 881.60 examples/s]28767 examples [00:31, 879.42 examples/s]28856 examples [00:32, 858.41 examples/s]28947 examples [00:32, 872.29 examples/s]29038 examples [00:32, 881.74 examples/s]29128 examples [00:32, 886.69 examples/s]29217 examples [00:32, 872.40 examples/s]29308 examples [00:32, 879.92 examples/s]29397 examples [00:32, 867.38 examples/s]29484 examples [00:32, 836.61 examples/s]29568 examples [00:32, 807.49 examples/s]29650 examples [00:32, 792.95 examples/s]29748 examples [00:33, 839.99 examples/s]29845 examples [00:33, 873.79 examples/s]29940 examples [00:33, 893.32 examples/s]30031 examples [00:33, 854.07 examples/s]30118 examples [00:33, 851.51 examples/s]30208 examples [00:33, 863.69 examples/s]30295 examples [00:33, 855.55 examples/s]30386 examples [00:33, 869.44 examples/s]30484 examples [00:33, 897.63 examples/s]30576 examples [00:34, 901.40 examples/s]30673 examples [00:34, 920.09 examples/s]30767 examples [00:34, 922.63 examples/s]30860 examples [00:34, 907.43 examples/s]30951 examples [00:34, 902.59 examples/s]31042 examples [00:34, 877.13 examples/s]31134 examples [00:34, 888.66 examples/s]31227 examples [00:34, 897.57 examples/s]31319 examples [00:34, 904.04 examples/s]31410 examples [00:34, 902.44 examples/s]31501 examples [00:35, 898.88 examples/s]31591 examples [00:35, 898.47 examples/s]31685 examples [00:35, 908.11 examples/s]31787 examples [00:35, 934.94 examples/s]31881 examples [00:35, 916.37 examples/s]31973 examples [00:35, 902.50 examples/s]32072 examples [00:35, 925.31 examples/s]32173 examples [00:35, 947.39 examples/s]32269 examples [00:35, 945.27 examples/s]32364 examples [00:35, 900.39 examples/s]32455 examples [00:36, 897.13 examples/s]32549 examples [00:36, 908.16 examples/s]32645 examples [00:36, 922.80 examples/s]32747 examples [00:36, 949.44 examples/s]32843 examples [00:36, 949.91 examples/s]32939 examples [00:36, 944.58 examples/s]33034 examples [00:36, 942.61 examples/s]33129 examples [00:36, 928.75 examples/s]33230 examples [00:36, 949.27 examples/s]33329 examples [00:36, 960.02 examples/s]33427 examples [00:37, 965.50 examples/s]33527 examples [00:37, 973.32 examples/s]33633 examples [00:37, 995.95 examples/s]33737 examples [00:37, 1008.36 examples/s]33839 examples [00:37, 933.79 examples/s] 33934 examples [00:37, 931.46 examples/s]34029 examples [00:37, 927.24 examples/s]34123 examples [00:37, 911.85 examples/s]34215 examples [00:37, 904.41 examples/s]34306 examples [00:38, 891.07 examples/s]34396 examples [00:38, 884.45 examples/s]34485 examples [00:38, 881.38 examples/s]34574 examples [00:38, 880.67 examples/s]34666 examples [00:38, 889.12 examples/s]34756 examples [00:38, 884.15 examples/s]34845 examples [00:38, 882.44 examples/s]34934 examples [00:38, 879.18 examples/s]35024 examples [00:38, 884.60 examples/s]35119 examples [00:38, 900.15 examples/s]35210 examples [00:39, 896.91 examples/s]35300 examples [00:39, 892.30 examples/s]35397 examples [00:39, 913.89 examples/s]35499 examples [00:39, 941.17 examples/s]35598 examples [00:39, 952.33 examples/s]35694 examples [00:39, 936.48 examples/s]35788 examples [00:39, 923.43 examples/s]35884 examples [00:39, 931.45 examples/s]35978 examples [00:39, 924.80 examples/s]36071 examples [00:39, 915.61 examples/s]36163 examples [00:40, 893.81 examples/s]36253 examples [00:40, 872.55 examples/s]36342 examples [00:40, 876.91 examples/s]36435 examples [00:40, 892.05 examples/s]36525 examples [00:40, 871.82 examples/s]36613 examples [00:40, 867.97 examples/s]36703 examples [00:40, 876.99 examples/s]36796 examples [00:40, 891.01 examples/s]36893 examples [00:40, 913.30 examples/s]36988 examples [00:41, 923.23 examples/s]37081 examples [00:41, 911.60 examples/s]37173 examples [00:41, 900.90 examples/s]37264 examples [00:41, 897.07 examples/s]37354 examples [00:41, 894.17 examples/s]37444 examples [00:41, 883.71 examples/s]37533 examples [00:41, 873.20 examples/s]37622 examples [00:41, 876.04 examples/s]37710 examples [00:41, 867.81 examples/s]37799 examples [00:41, 872.21 examples/s]37887 examples [00:42, 871.48 examples/s]37975 examples [00:42, 849.66 examples/s]38066 examples [00:42, 865.43 examples/s]38157 examples [00:42, 876.56 examples/s]38248 examples [00:42, 884.04 examples/s]38337 examples [00:42, 882.92 examples/s]38427 examples [00:42, 885.57 examples/s]38523 examples [00:42, 906.10 examples/s]38618 examples [00:42, 918.37 examples/s]38710 examples [00:42, 918.47 examples/s]38804 examples [00:43, 922.58 examples/s]38897 examples [00:43, 924.43 examples/s]38990 examples [00:43, 924.94 examples/s]39083 examples [00:43, 890.86 examples/s]39173 examples [00:43, 878.87 examples/s]39262 examples [00:43, 867.00 examples/s]39359 examples [00:43, 894.66 examples/s]39459 examples [00:43, 922.78 examples/s]39556 examples [00:43, 934.92 examples/s]39659 examples [00:43, 959.42 examples/s]39756 examples [00:44, 960.72 examples/s]39853 examples [00:44, 944.89 examples/s]39948 examples [00:44, 933.38 examples/s]40042 examples [00:44, 887.86 examples/s]40138 examples [00:44, 905.12 examples/s]40231 examples [00:44, 911.21 examples/s]40323 examples [00:44, 886.36 examples/s]40418 examples [00:44, 902.36 examples/s]40509 examples [00:44, 896.39 examples/s]40600 examples [00:45, 898.21 examples/s]40691 examples [00:45, 881.29 examples/s]40780 examples [00:45, 854.60 examples/s]40869 examples [00:45, 864.59 examples/s]40956 examples [00:45, 861.53 examples/s]41043 examples [00:45, 854.96 examples/s]41131 examples [00:45, 859.70 examples/s]41220 examples [00:45, 867.23 examples/s]41312 examples [00:45, 882.34 examples/s]41404 examples [00:45, 893.23 examples/s]41496 examples [00:46, 899.41 examples/s]41587 examples [00:46, 901.93 examples/s]41680 examples [00:46, 907.96 examples/s]41774 examples [00:46, 916.68 examples/s]41866 examples [00:46, 874.44 examples/s]41960 examples [00:46, 891.29 examples/s]42050 examples [00:46, 886.48 examples/s]42145 examples [00:46, 904.40 examples/s]42237 examples [00:46, 906.58 examples/s]42333 examples [00:46, 920.71 examples/s]42426 examples [00:47, 921.13 examples/s]42519 examples [00:47, 902.00 examples/s]42611 examples [00:47, 904.58 examples/s]42704 examples [00:47, 910.22 examples/s]42796 examples [00:47, 911.18 examples/s]42888 examples [00:47, 912.73 examples/s]42980 examples [00:47, 899.09 examples/s]43071 examples [00:47, 901.42 examples/s]43162 examples [00:47, 900.43 examples/s]43257 examples [00:47, 913.46 examples/s]43351 examples [00:48, 920.88 examples/s]43444 examples [00:48, 920.59 examples/s]43537 examples [00:48, 879.23 examples/s]43628 examples [00:48, 886.13 examples/s]43717 examples [00:48, 878.71 examples/s]43806 examples [00:48, 857.38 examples/s]43893 examples [00:48, 849.76 examples/s]43980 examples [00:48, 854.82 examples/s]44066 examples [00:48, 836.39 examples/s]44158 examples [00:49, 857.88 examples/s]44247 examples [00:49, 865.20 examples/s]44334 examples [00:49, 842.19 examples/s]44421 examples [00:49, 849.11 examples/s]44513 examples [00:49, 867.41 examples/s]44600 examples [00:49, 825.68 examples/s]44686 examples [00:49, 833.46 examples/s]44770 examples [00:49, 827.60 examples/s]44854 examples [00:49, 825.61 examples/s]44939 examples [00:49, 831.10 examples/s]45029 examples [00:50, 849.94 examples/s]45115 examples [00:50, 849.14 examples/s]45205 examples [00:50, 861.50 examples/s]45304 examples [00:50, 894.63 examples/s]45401 examples [00:50, 913.48 examples/s]45493 examples [00:50, 912.51 examples/s]45585 examples [00:50, 911.64 examples/s]45677 examples [00:50, 909.37 examples/s]45769 examples [00:50, 886.50 examples/s]45858 examples [00:50, 880.71 examples/s]45947 examples [00:51, 878.49 examples/s]46036 examples [00:51, 879.88 examples/s]46125 examples [00:51, 874.62 examples/s]46213 examples [00:51, 850.77 examples/s]46305 examples [00:51, 869.67 examples/s]46396 examples [00:51, 879.06 examples/s]46486 examples [00:51, 883.42 examples/s]46575 examples [00:51, 853.51 examples/s]46665 examples [00:51, 864.38 examples/s]46759 examples [00:52, 885.36 examples/s]46851 examples [00:52, 895.26 examples/s]46944 examples [00:52, 903.21 examples/s]47035 examples [00:52, 902.48 examples/s]47126 examples [00:52, 901.90 examples/s]47217 examples [00:52, 891.78 examples/s]47307 examples [00:52, 887.84 examples/s]47396 examples [00:52, 877.11 examples/s]47484 examples [00:52, 866.74 examples/s]47571 examples [00:52, 861.97 examples/s]47663 examples [00:53, 876.59 examples/s]47751 examples [00:53, 875.73 examples/s]47841 examples [00:53, 880.62 examples/s]47930 examples [00:53, 867.24 examples/s]48017 examples [00:53, 866.44 examples/s]48104 examples [00:53, 863.38 examples/s]48194 examples [00:53, 871.45 examples/s]48285 examples [00:53, 881.59 examples/s]48375 examples [00:53, 884.29 examples/s]48467 examples [00:53, 892.41 examples/s]48557 examples [00:54, 892.86 examples/s]48647 examples [00:54, 874.10 examples/s]48737 examples [00:54, 880.30 examples/s]48826 examples [00:54, 878.65 examples/s]48914 examples [00:54, 862.63 examples/s]49004 examples [00:54, 872.47 examples/s]49092 examples [00:54, 870.77 examples/s]49180 examples [00:54, 863.13 examples/s]49267 examples [00:54, 850.37 examples/s]49353 examples [00:54, 852.91 examples/s]49445 examples [00:55, 869.60 examples/s]49541 examples [00:55, 893.46 examples/s]49632 examples [00:55, 897.08 examples/s]49722 examples [00:55, 882.99 examples/s]49811 examples [00:55, 869.81 examples/s]49899 examples [00:55, 854.12 examples/s]49988 examples [00:55, 863.45 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6651/50000 [00:00<00:00, 66507.84 examples/s] 37%|███▋      | 18353/50000 [00:00<00:00, 76401.32 examples/s] 60%|██████    | 30107/50000 [00:00<00:00, 85363.88 examples/s] 84%|████████▍ | 41900/50000 [00:00<00:00, 93071.98 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 717.30 examples/s]159 examples [00:00, 755.95 examples/s]254 examples [00:00, 803.67 examples/s]345 examples [00:00, 831.13 examples/s]439 examples [00:00, 860.56 examples/s]532 examples [00:00, 880.00 examples/s]613 examples [00:00, 855.78 examples/s]697 examples [00:00, 684.67 examples/s]789 examples [00:00, 739.91 examples/s]886 examples [00:01, 796.05 examples/s]980 examples [00:01, 833.54 examples/s]1075 examples [00:01, 863.56 examples/s]1167 examples [00:01, 877.22 examples/s]1257 examples [00:01, 878.90 examples/s]1351 examples [00:01, 893.98 examples/s]1443 examples [00:01, 900.71 examples/s]1535 examples [00:01, 905.86 examples/s]1630 examples [00:01, 916.20 examples/s]1722 examples [00:01, 903.95 examples/s]1821 examples [00:02, 927.56 examples/s]1916 examples [00:02, 931.84 examples/s]2010 examples [00:02, 895.72 examples/s]2102 examples [00:02, 901.78 examples/s]2194 examples [00:02, 904.90 examples/s]2285 examples [00:02, 899.37 examples/s]2376 examples [00:02, 897.25 examples/s]2466 examples [00:02, 896.94 examples/s]2560 examples [00:02, 908.69 examples/s]2653 examples [00:03, 913.86 examples/s]2746 examples [00:03, 917.50 examples/s]2838 examples [00:03, 908.33 examples/s]2929 examples [00:03, 903.64 examples/s]3021 examples [00:03, 907.47 examples/s]3112 examples [00:03, 881.96 examples/s]3202 examples [00:03, 886.25 examples/s]3291 examples [00:03, 886.41 examples/s]3387 examples [00:03, 905.19 examples/s]3480 examples [00:03, 911.49 examples/s]3575 examples [00:04, 919.92 examples/s]3668 examples [00:04, 907.27 examples/s]3759 examples [00:04, 887.52 examples/s]3852 examples [00:04, 897.43 examples/s]3942 examples [00:04, 886.14 examples/s]4035 examples [00:04, 895.56 examples/s]4128 examples [00:04, 905.39 examples/s]4219 examples [00:04, 905.65 examples/s]4316 examples [00:04, 923.13 examples/s]4411 examples [00:04, 930.74 examples/s]4505 examples [00:05, 918.59 examples/s]4600 examples [00:05, 926.09 examples/s]4693 examples [00:05, 905.42 examples/s]4788 examples [00:05, 917.23 examples/s]4884 examples [00:05, 927.84 examples/s]4977 examples [00:05, 921.89 examples/s]5070 examples [00:05, 908.48 examples/s]5161 examples [00:05, 888.91 examples/s]5251 examples [00:05, 891.99 examples/s]5343 examples [00:05, 898.17 examples/s]5433 examples [00:06, 882.87 examples/s]5525 examples [00:06, 891.82 examples/s]5620 examples [00:06, 908.43 examples/s]5713 examples [00:06, 913.37 examples/s]5805 examples [00:06, 907.65 examples/s]5896 examples [00:06, 905.24 examples/s]5987 examples [00:06, 900.02 examples/s]6078 examples [00:06, 888.16 examples/s]6168 examples [00:06, 891.14 examples/s]6264 examples [00:07, 907.93 examples/s]6355 examples [00:07, 902.37 examples/s]6446 examples [00:07, 893.42 examples/s]6536 examples [00:07, 889.37 examples/s]6625 examples [00:07, 869.19 examples/s]6713 examples [00:07, 868.09 examples/s]6808 examples [00:07, 890.42 examples/s]6903 examples [00:07, 905.73 examples/s]6994 examples [00:07, 875.64 examples/s]7082 examples [00:07, 876.29 examples/s]7177 examples [00:08, 896.96 examples/s]7267 examples [00:08, 890.35 examples/s]7357 examples [00:08, 875.18 examples/s]7445 examples [00:08, 841.83 examples/s]7533 examples [00:08, 852.72 examples/s]7622 examples [00:08, 863.48 examples/s]7709 examples [00:08, 854.70 examples/s]7799 examples [00:08, 866.46 examples/s]7886 examples [00:08, 851.38 examples/s]7976 examples [00:08, 862.82 examples/s]8063 examples [00:09, 864.59 examples/s]8153 examples [00:09, 873.34 examples/s]8243 examples [00:09, 880.91 examples/s]8332 examples [00:09, 878.88 examples/s]8423 examples [00:09, 885.44 examples/s]8514 examples [00:09, 890.35 examples/s]8604 examples [00:09, 892.19 examples/s]8694 examples [00:09, 877.55 examples/s]8784 examples [00:09, 883.97 examples/s]8878 examples [00:09, 898.04 examples/s]8968 examples [00:10, 879.72 examples/s]9059 examples [00:10, 885.71 examples/s]9149 examples [00:10, 888.21 examples/s]9238 examples [00:10, 884.25 examples/s]9327 examples [00:10, 865.96 examples/s]9421 examples [00:10, 886.57 examples/s]9514 examples [00:10, 897.51 examples/s]9604 examples [00:10, 897.63 examples/s]9694 examples [00:10, 896.66 examples/s]9785 examples [00:11, 900.47 examples/s]9878 examples [00:11, 907.01 examples/s]9970 examples [00:11, 909.15 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI7J3FJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteI7J3FJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdb1cd95950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdb1cd95950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdb1cd95950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdaa61d2390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdaa620cf28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fdb1cd95950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fdb1cd95950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fdb1cd95950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdb1cd80ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fdafa98f128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fda94763268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fda94763268> 

  function with postional parmater data_info <function split_train_valid at 0x7fda94763268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fda94763378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fda94763378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fda94763378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2a0604a0aa4f1027e7def81307ceb95f4c9828002c96055afbe510ca1bb1c7e3
  Stored in directory: /tmp/pip-ephem-wheel-cache-4dwnn9o2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:17:17, 9.86kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:14:02, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:07:00, 19.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:29:21, 28.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<5:55:36, 40.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.27M/862M [00:01<4:07:39, 57.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:52:34, 82.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<2:00:04, 117kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<1:23:53, 167kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<58:29, 238kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.0M/862M [00:01<40:51, 339kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.3M/862M [00:02<28:48, 479kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<20:09, 682kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<14:07, 968kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<09:56, 1.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<07:02, 1.92MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<05:22, 2.51MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:40, 2.37MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:36, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<06:07, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<04:32, 2.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<06:39, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<06:00, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<04:33, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<06:18, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<10:15, 1.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<08:13, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<06:10, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:09<04:31, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<12:15, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<11:22, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<08:25, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<06:13, 2.12MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<07:50, 1.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<10:21, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<08:42, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<07:46, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<07:52, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<06:01, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<04:33, 2.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:15<03:26, 3.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<24:12, 538kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<22:35, 576kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<17:10, 758kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:17<12:36, 1.03MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<08:59, 1.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<15:17, 848kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<12:57, 999kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<09:42, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<06:59, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<14:19, 899kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<14:12, 907kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<11:01, 1.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.4M/862M [00:21<07:58, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<09:01, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<10:28, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<08:24, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<06:09, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:43, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<09:33, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:45, 1.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<05:41, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:30, 1.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:24, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:28, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:27, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:50, 981kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:23, 942kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:26, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<07:34, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:34, 1.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:22, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:17, 1.51MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<06:03, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:36, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:22, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:26, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:28, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:14, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:06, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:14, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<05:23, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<03:57, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<15:20, 804kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<14:44, 836kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:07, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<07:59, 1.54MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:52, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<21:33, 569kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<18:50, 651kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:05, 869kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:04, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:38, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:56, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:34, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<06:13, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:51, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:58, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:09, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:16, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:21, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<06:34, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:51, 2.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:04, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:50, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:08, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:13, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:56, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:55, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:14, 1.91MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:34, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:34, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:11, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:26, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:42, 2.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:21, 1.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:02, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:08, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:12, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:28, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:46, 2.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:20, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:08, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:20, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:36, 2.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:58, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:01, 1.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:15, 1.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:34, 2.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:34, 1.75MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:02, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:28, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:59, 2.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<13:25, 851kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<12:10, 938kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<09:08, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<06:34, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<10:17, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:57, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:36, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<05:29, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:31, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:53, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:09, 1.83MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:33, 2.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:38, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:42, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:55, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:36, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<09:16, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<11:55, 936kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<09:42, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:06, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:40, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<12:54, 859kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<10:05, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:24, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<05:22, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<09:41, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<11:39, 944kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:13, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<04:57, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<13:23, 817kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<14:12, 770kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<11:06, 985kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:02, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<17:01, 639kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<16:43, 650kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<12:41, 856kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<09:17, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:43, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:32, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<10:21, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<08:20, 1.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:05, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<04:26, 2.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<12:10, 883kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<12:52, 835kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<09:55, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<07:14, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<05:16, 2.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<24:50, 430kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<21:43, 491kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<16:04, 663kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<11:31, 924kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<08:17, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<09:44, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<11:09, 950kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:50, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:28, 1.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:47, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<08:43, 1.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:57, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:05, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:55, 2.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<06:40, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<08:35, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:01, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:12, 2.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:51, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<08:00, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:27, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:48, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:31, 2.93MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<15:02, 687kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<14:23, 717kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<10:55, 945kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<07:54, 1.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:42, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<9:13:47, 18.5kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<6:31:29, 26.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<4:34:47, 37.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<3:12:11, 53.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<2:14:13, 75.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<3:49:47, 44.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<2:44:38, 61.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<1:56:00, 87.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:21:16, 125kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<59:00, 171kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<45:05, 224kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<32:27, 311kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<22:56, 440kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<18:15, 550kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<16:51, 596kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<12:40, 792kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<09:10, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:30, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<09:42, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:35, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:34, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:04, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<1:31:35, 108kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<1:07:49, 146kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<48:13, 205kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<33:55, 291kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<23:55, 412kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<20:49, 473kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<18:15, 539kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<13:41, 718kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:49, 999kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:02, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:59, 977kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:48, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:43, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:07, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:54, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:19, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:40, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:28, 2.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:09, 1.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:37, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<06:56, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:06, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:41, 1.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:34, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:11, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<04:32, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:19, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:01, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:46, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:15, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:07, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:53, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:41, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:10, 2.25MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:02, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<4:19:13, 36.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<3:04:56, 50.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<2:10:12, 71.8kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<1:31:06, 102kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<1:05:39, 141kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<49:10, 189kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<35:12, 264kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<24:46, 373kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<19:27, 474kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<16:49, 548kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<12:28, 739kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<08:53, 1.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:29, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<08:59, 1.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:31, 961kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<07:18, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<05:16, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<09:41, 937kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<09:43, 933kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:32, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<05:26, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:11, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:15, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:48, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:13, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:22, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:28, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:12, 1.71MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:48, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:08, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:10, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:51, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:34, 2.47MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<02:37, 3.35MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<32:50, 268kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<25:31, 345kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<18:22, 479kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<13:00, 674kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<11:44, 744kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<10:42, 815kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<08:01, 1.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<05:46, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<06:47, 1.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:13, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:39, 1.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<04:06, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:48, 1.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<06:08, 1.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:50, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:32, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:43, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:18, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:00, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:39, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:28, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:47, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<04:29, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<03:16, 2.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:08, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:27, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:59, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:35, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:29, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:58, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:43, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:27, 2.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:34, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:34, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:19, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:07, 2.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<08:55, 917kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:53, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:53, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:13, 1.92MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:42, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:46, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:11, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:44, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:51, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:40, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:05, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:40, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:06, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:22, 1.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:45, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:41<03:23, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<19:05, 414kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<15:11, 520kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:02, 714kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<07:48, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<10:55, 717kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<08:55, 877kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:33, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<05:55, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:47, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:26, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:10, 2.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<09:39, 797kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<07:55, 969kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:48, 1.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:25, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:20, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:04, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:54, 2.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<15:33, 486kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<12:25, 608kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<09:01, 835kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<06:21, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<20:47, 360kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<16:04, 466kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<11:34, 645kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<08:08, 911kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<2:28:06, 50.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<1:45:07, 70.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<1:13:47, 100kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<51:27, 143kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<41:08, 179kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<30:16, 243kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<21:29, 341kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<15:02, 485kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<17:26, 418kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<13:40, 532kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<09:53, 735kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:58, 1.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<09:03, 797kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:48, 924kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:48, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:12, 1.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<05:05, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:52, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:47, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:57, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:34, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:12, 1.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:00, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<08:28, 828kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<07:20, 953kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:26, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:52, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:10, 849kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:07, 974kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:18, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:46, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<1:00:32, 114kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<43:45, 157kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<30:53, 222kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<21:32, 316kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<1:04:10, 106kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<46:18, 147kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<32:38, 208kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<22:50, 296kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<18:47, 358kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<14:01, 480kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<09:59, 672kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:14, 809kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:15, 807kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<06:19, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:34, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:49, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:33, 1.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:29, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:31, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:38, 1.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:50, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:03, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:18, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:36, 2.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:53, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:10, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:30, 2.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:48, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:06, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:27, 2.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:45, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:03, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:23, 2.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<01:45, 3.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:14, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:07, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:08, 1.97MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:23, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:38, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:51, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:15, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:28, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:01, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:12, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:28, 2.41MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:47, 3.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:19, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:58, 987kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:39, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:36, 1.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:47, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:54, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:07, 1.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:24, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:45, 3.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:44, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:23, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:15, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:20, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:43, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:55, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:17, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:30, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:45, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:09, 2.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:24, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:03, 2.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:30, 3.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:37, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<04:03, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:01, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:16, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:58, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<03:02, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:21, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<02:31, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:35, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<03:00, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:12, 2.36MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:47, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:55, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:14, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:37, 3.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<05:23, 942kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:41, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:28, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:27, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<20:15, 247kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<15:04, 332kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<10:43, 466kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<07:29, 661kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<4:57:29, 16.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<3:28:59, 23.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<2:26:05, 33.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<1:42:02, 47.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<1:12:15, 67.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<50:37, 96.0kB/s]  .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<35:14, 137kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<27:12, 177kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<19:53, 242kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<14:06, 340kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<10:34, 449kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<08:14, 574kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:58, 791kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:55, 949kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:18, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:15, 2.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<24:16, 190kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<17:51, 258kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<12:38, 363kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<08:52, 514kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<07:48, 580kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<06:18, 718kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:36, 980kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<03:15, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<25:15, 177kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<18:49, 237kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<13:25, 331kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<09:22, 470kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<08:43, 504kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<07:14, 607kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<05:20, 821kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:51, 890kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:27, 968kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:23, 1.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:24, 1.77MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:09, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:56, 1.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:59, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:45, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:42, 1.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:49, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:00, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:05, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:07, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:26, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:44, 2.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:39, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:27, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:38, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:52, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<04:07, 965kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:45, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:50, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:29, 870kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:58, 984kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:57, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<02:05, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:28, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:22, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:33, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:50, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:36, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:45, 1.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:07, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:32, 2.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:13, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:25, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:52, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:21, 2.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:11, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:24, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:51, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:19, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:11, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:07, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:21, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:05, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:16, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:47, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:17, 2.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:07, 1.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:02, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:18, 1.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:38, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:37, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:26, 2.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:07, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:07, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:38, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:10, 2.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:08, 626kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:13, 761kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:05, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<02:11, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<05:26, 578kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<04:26, 707kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:15, 961kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<02:17, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<05:35, 550kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:30, 681kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:17, 928kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:18, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<10:46, 279kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<08:11, 367kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<05:51, 511kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<04:05, 723kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<04:06, 717kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:26, 854kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:32, 1.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<01:47, 1.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<07:21, 390kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<05:49, 493kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<04:14, 675kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<02:57, 952kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:47, 738kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:18, 845kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:26, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:25, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:19, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:47, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:32, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:17, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:43, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<04:35, 566kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:52, 668kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:51, 901kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<02:00, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:50, 886kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:36, 967kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:57, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:23, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:50, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:45, 1.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:20, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:57, 2.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<04:05, 585kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:19, 718kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:25, 975kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<01:42, 1.37MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<04:11, 552kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<03:24, 679kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<02:28, 933kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:44, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:58, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:54, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:27, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:01, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<02:01, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:56, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:28, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:02, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:32, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:37, 1.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:15, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:53, 2.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:21, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:21, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:02, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:44, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:21, 588kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<02:50, 691kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:05, 937kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:28, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:35, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:34, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:12, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:51, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:41, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:31, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:08, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:47, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:59, 592kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:25, 725kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:45, 991kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:13, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<03:08, 541kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<02:30, 675kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:48, 927kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:16, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:25, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:18, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:58, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:40, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:32, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:21, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:01, 1.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:42, 2.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<11:11, 133kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<08:05, 184kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<05:40, 259kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<03:52, 368kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<06:47, 209kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<05:05, 279kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<03:37, 389kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<02:28, 551kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:36, 519kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<02:07, 637kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:32, 868kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:03, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:24, 907kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:16, 1.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:56, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:39, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:57, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:57, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:30, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:45, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:47, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:37, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:25, 2.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:53, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:48, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:36, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:24, 2.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:55, 522kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:32, 647kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:06, 890kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:46, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:50, 1.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:47, 1.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:35, 1.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:09, 744kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:59, 868kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:43, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:42, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:40, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:30, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:03, 693kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:50, 856kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:36, 1.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:30, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:29, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:22, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:35, 989kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:32, 1.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:23, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:12<00:15, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<03:55, 133kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<02:50, 184kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:57, 259kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<01:15, 369kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:28, 309kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<01:07, 404kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:46, 560kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:29, 792kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<03:06, 125kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<02:13, 172kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:31, 243kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:56, 345kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<01:09, 276kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:50, 374kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:33, 526kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:22, 658kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:18, 799kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:12, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:08, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:05, 1.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 2.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<29:43:31,  3.74it/s]  0%|          | 718/400000 [00:00<20:46:30,  5.34it/s]  0%|          | 1473/400000 [00:00<14:31:10,  7.62it/s]  1%|          | 2190/400000 [00:00<10:08:59, 10.89it/s]  1%|          | 2887/400000 [00:00<7:05:50, 15.54it/s]   1%|          | 3640/400000 [00:00<4:57:47, 22.18it/s]  1%|          | 4422/400000 [00:00<3:28:17, 31.65it/s]  1%|▏         | 5222/400000 [00:00<2:25:45, 45.14it/s]  2%|▏         | 6022/400000 [00:01<1:42:04, 64.33it/s]  2%|▏         | 6811/400000 [00:01<1:11:33, 91.58it/s]  2%|▏         | 7572/400000 [00:01<50:14, 130.16it/s]   2%|▏         | 8324/400000 [00:01<35:24, 184.35it/s]  2%|▏         | 9092/400000 [00:01<24:59, 260.68it/s]  2%|▏         | 9827/400000 [00:01<17:43, 366.75it/s]  3%|▎         | 10572/400000 [00:01<12:38, 513.10it/s]  3%|▎         | 11308/400000 [00:01<09:07, 710.04it/s]  3%|▎         | 12027/400000 [00:01<06:38, 972.92it/s]  3%|▎         | 12755/400000 [00:01<04:54, 1314.55it/s]  3%|▎         | 13522/400000 [00:02<03:40, 1749.35it/s]  4%|▎         | 14271/400000 [00:02<02:49, 2271.43it/s]  4%|▍         | 15010/400000 [00:02<02:14, 2866.49it/s]  4%|▍         | 15763/400000 [00:02<01:49, 3520.13it/s]  4%|▍         | 16506/400000 [00:02<01:32, 4161.68it/s]  4%|▍         | 17243/400000 [00:02<01:20, 4780.48it/s]  4%|▍         | 17979/400000 [00:02<01:11, 5340.81it/s]  5%|▍         | 18715/400000 [00:02<01:05, 5819.11it/s]  5%|▍         | 19451/400000 [00:02<01:01, 6153.28it/s]  5%|▌         | 20179/400000 [00:03<00:59, 6339.69it/s]  5%|▌         | 20894/400000 [00:03<00:58, 6524.52it/s]  5%|▌         | 21625/400000 [00:03<00:56, 6740.80it/s]  6%|▌         | 22348/400000 [00:03<00:54, 6880.17it/s]  6%|▌         | 23087/400000 [00:03<00:53, 7024.62it/s]  6%|▌         | 23825/400000 [00:03<00:52, 7127.18it/s]  6%|▌         | 24554/400000 [00:03<00:52, 7102.30it/s]  6%|▋         | 25276/400000 [00:03<00:52, 7086.33it/s]  7%|▋         | 26006/400000 [00:03<00:52, 7148.33it/s]  7%|▋         | 26746/400000 [00:03<00:51, 7219.54it/s]  7%|▋         | 27490/400000 [00:04<00:51, 7282.19it/s]  7%|▋         | 28258/400000 [00:04<00:50, 7394.44it/s]  7%|▋         | 29023/400000 [00:04<00:49, 7467.25it/s]  7%|▋         | 29793/400000 [00:04<00:49, 7534.50it/s]  8%|▊         | 30561/400000 [00:04<00:48, 7575.93it/s]  8%|▊         | 31320/400000 [00:04<00:49, 7479.62it/s]  8%|▊         | 32083/400000 [00:04<00:48, 7521.16it/s]  8%|▊         | 32838/400000 [00:04<00:48, 7528.34it/s]  8%|▊         | 33592/400000 [00:04<00:49, 7444.36it/s]  9%|▊         | 34338/400000 [00:04<00:49, 7382.30it/s]  9%|▉         | 35077/400000 [00:05<00:49, 7344.83it/s]  9%|▉         | 35818/400000 [00:05<00:49, 7364.21it/s]  9%|▉         | 36573/400000 [00:05<00:48, 7418.46it/s]  9%|▉         | 37320/400000 [00:05<00:48, 7431.04it/s] 10%|▉         | 38064/400000 [00:05<00:49, 7321.72it/s] 10%|▉         | 38797/400000 [00:05<00:49, 7290.41it/s] 10%|▉         | 39584/400000 [00:05<00:48, 7453.89it/s] 10%|█         | 40396/400000 [00:05<00:47, 7639.80it/s] 10%|█         | 41162/400000 [00:05<00:46, 7636.94it/s] 10%|█         | 41966/400000 [00:05<00:46, 7749.68it/s] 11%|█         | 42743/400000 [00:06<00:46, 7682.66it/s] 11%|█         | 43587/400000 [00:06<00:45, 7894.45it/s] 11%|█         | 44433/400000 [00:06<00:44, 8053.99it/s] 11%|█▏        | 45241/400000 [00:06<00:44, 8045.19it/s] 12%|█▏        | 46072/400000 [00:06<00:43, 8120.71it/s] 12%|█▏        | 46886/400000 [00:06<00:44, 7873.81it/s] 12%|█▏        | 47676/400000 [00:06<00:45, 7781.52it/s] 12%|█▏        | 48457/400000 [00:06<00:45, 7782.44it/s] 12%|█▏        | 49237/400000 [00:06<00:45, 7689.95it/s] 13%|█▎        | 50008/400000 [00:06<00:45, 7655.31it/s] 13%|█▎        | 50799/400000 [00:07<00:45, 7728.29it/s] 13%|█▎        | 51614/400000 [00:07<00:44, 7849.14it/s] 13%|█▎        | 52400/400000 [00:07<00:44, 7730.56it/s] 13%|█▎        | 53175/400000 [00:07<00:44, 7720.67it/s] 13%|█▎        | 53987/400000 [00:07<00:44, 7836.07it/s] 14%|█▎        | 54772/400000 [00:07<00:44, 7791.68it/s] 14%|█▍        | 55552/400000 [00:07<00:45, 7609.08it/s] 14%|█▍        | 56317/400000 [00:07<00:45, 7619.71it/s] 14%|█▍        | 57080/400000 [00:07<00:45, 7503.65it/s] 14%|█▍        | 57832/400000 [00:07<00:45, 7483.20it/s] 15%|█▍        | 58588/400000 [00:08<00:45, 7505.30it/s] 15%|█▍        | 59352/400000 [00:08<00:45, 7543.57it/s] 15%|█▌        | 60107/400000 [00:08<00:45, 7545.06it/s] 15%|█▌        | 60862/400000 [00:08<00:45, 7531.94it/s] 15%|█▌        | 61691/400000 [00:08<00:43, 7743.58it/s] 16%|█▌        | 62467/400000 [00:08<00:44, 7541.06it/s] 16%|█▌        | 63267/400000 [00:08<00:43, 7672.58it/s] 16%|█▌        | 64122/400000 [00:08<00:42, 7915.49it/s] 16%|█▌        | 64918/400000 [00:08<00:42, 7800.15it/s] 16%|█▋        | 65701/400000 [00:09<00:42, 7795.18it/s] 17%|█▋        | 66530/400000 [00:09<00:42, 7936.07it/s] 17%|█▋        | 67326/400000 [00:09<00:41, 7920.99it/s] 17%|█▋        | 68120/400000 [00:09<00:42, 7806.43it/s] 17%|█▋        | 68902/400000 [00:09<00:43, 7607.18it/s] 17%|█▋        | 69694/400000 [00:09<00:42, 7697.73it/s] 18%|█▊        | 70468/400000 [00:09<00:42, 7707.95it/s] 18%|█▊        | 71240/400000 [00:09<00:43, 7600.01it/s] 18%|█▊        | 72027/400000 [00:09<00:42, 7677.24it/s] 18%|█▊        | 72851/400000 [00:09<00:41, 7835.94it/s] 18%|█▊        | 73712/400000 [00:10<00:40, 8051.39it/s] 19%|█▊        | 74572/400000 [00:10<00:39, 8207.16it/s] 19%|█▉        | 75469/400000 [00:10<00:38, 8419.00it/s] 19%|█▉        | 76323/400000 [00:10<00:38, 8454.89it/s] 19%|█▉        | 77171/400000 [00:10<00:39, 8179.07it/s] 20%|█▉        | 78037/400000 [00:10<00:38, 8316.94it/s] 20%|█▉        | 78872/400000 [00:10<00:38, 8249.31it/s] 20%|█▉        | 79700/400000 [00:10<00:39, 8208.12it/s] 20%|██        | 80565/400000 [00:10<00:38, 8332.10it/s] 20%|██        | 81400/400000 [00:10<00:38, 8205.54it/s] 21%|██        | 82223/400000 [00:11<00:39, 8037.10it/s] 21%|██        | 83035/400000 [00:11<00:39, 8058.25it/s] 21%|██        | 83891/400000 [00:11<00:38, 8201.64it/s] 21%|██        | 84716/400000 [00:11<00:38, 8216.08it/s] 21%|██▏       | 85539/400000 [00:11<00:38, 8170.63it/s] 22%|██▏       | 86357/400000 [00:11<00:38, 8112.11it/s] 22%|██▏       | 87193/400000 [00:11<00:38, 8182.56it/s] 22%|██▏       | 88012/400000 [00:11<00:39, 7833.84it/s] 22%|██▏       | 88799/400000 [00:11<00:40, 7621.26it/s] 22%|██▏       | 89565/400000 [00:11<00:40, 7606.57it/s] 23%|██▎       | 90374/400000 [00:12<00:39, 7744.87it/s] 23%|██▎       | 91151/400000 [00:12<00:40, 7681.90it/s] 23%|██▎       | 92001/400000 [00:12<00:38, 7908.82it/s] 23%|██▎       | 92795/400000 [00:12<00:39, 7812.03it/s] 23%|██▎       | 93579/400000 [00:12<00:39, 7672.47it/s] 24%|██▎       | 94380/400000 [00:12<00:39, 7770.38it/s] 24%|██▍       | 95159/400000 [00:12<00:40, 7578.69it/s] 24%|██▍       | 95974/400000 [00:12<00:39, 7741.16it/s] 24%|██▍       | 96751/400000 [00:12<00:39, 7634.29it/s] 24%|██▍       | 97517/400000 [00:13<00:40, 7543.96it/s] 25%|██▍       | 98343/400000 [00:13<00:38, 7744.93it/s] 25%|██▍       | 99148/400000 [00:13<00:38, 7831.89it/s] 25%|██▍       | 99943/400000 [00:13<00:38, 7865.67it/s] 25%|██▌       | 100746/400000 [00:13<00:37, 7913.54it/s] 25%|██▌       | 101558/400000 [00:13<00:37, 7972.37it/s] 26%|██▌       | 102357/400000 [00:13<00:37, 7875.78it/s] 26%|██▌       | 103146/400000 [00:13<00:37, 7821.80it/s] 26%|██▌       | 103984/400000 [00:13<00:37, 7978.92it/s] 26%|██▌       | 104784/400000 [00:13<00:37, 7945.19it/s] 26%|██▋       | 105580/400000 [00:14<00:37, 7874.13it/s] 27%|██▋       | 106379/400000 [00:14<00:37, 7906.77it/s] 27%|██▋       | 107181/400000 [00:14<00:36, 7939.89it/s] 27%|██▋       | 107976/400000 [00:14<00:36, 7899.62it/s] 27%|██▋       | 108767/400000 [00:14<00:36, 7880.88it/s] 27%|██▋       | 109556/400000 [00:14<00:37, 7841.11it/s] 28%|██▊       | 110359/400000 [00:14<00:36, 7892.26it/s] 28%|██▊       | 111149/400000 [00:14<00:36, 7839.21it/s] 28%|██▊       | 111945/400000 [00:14<00:36, 7872.67it/s] 28%|██▊       | 112733/400000 [00:14<00:37, 7728.38it/s] 28%|██▊       | 113507/400000 [00:15<00:37, 7588.40it/s] 29%|██▊       | 114293/400000 [00:15<00:37, 7665.50it/s] 29%|██▉       | 115088/400000 [00:15<00:36, 7747.46it/s] 29%|██▉       | 115864/400000 [00:15<00:36, 7737.13it/s] 29%|██▉       | 116639/400000 [00:15<00:37, 7494.20it/s] 29%|██▉       | 117391/400000 [00:15<00:37, 7484.52it/s] 30%|██▉       | 118141/400000 [00:15<00:38, 7271.06it/s] 30%|██▉       | 118880/400000 [00:15<00:38, 7303.60it/s] 30%|██▉       | 119680/400000 [00:15<00:37, 7498.72it/s] 30%|███       | 120457/400000 [00:15<00:36, 7575.96it/s] 30%|███       | 121270/400000 [00:16<00:36, 7730.54it/s] 31%|███       | 122137/400000 [00:16<00:34, 7990.10it/s] 31%|███       | 122967/400000 [00:16<00:34, 8079.11it/s] 31%|███       | 123793/400000 [00:16<00:33, 8131.05it/s] 31%|███       | 124609/400000 [00:16<00:34, 8004.02it/s] 31%|███▏      | 125412/400000 [00:16<00:35, 7652.17it/s] 32%|███▏      | 126203/400000 [00:16<00:35, 7727.45it/s] 32%|███▏      | 126980/400000 [00:16<00:35, 7697.76it/s] 32%|███▏      | 127753/400000 [00:16<00:35, 7666.43it/s] 32%|███▏      | 128522/400000 [00:17<00:36, 7494.49it/s] 32%|███▏      | 129274/400000 [00:17<00:36, 7473.25it/s] 33%|███▎      | 130089/400000 [00:17<00:35, 7662.57it/s] 33%|███▎      | 130858/400000 [00:17<00:35, 7617.48it/s] 33%|███▎      | 131622/400000 [00:17<00:36, 7369.40it/s] 33%|███▎      | 132362/400000 [00:17<00:36, 7368.64it/s] 33%|███▎      | 133108/400000 [00:17<00:36, 7395.53it/s] 33%|███▎      | 133849/400000 [00:17<00:36, 7354.53it/s] 34%|███▎      | 134663/400000 [00:17<00:35, 7572.32it/s] 34%|███▍      | 135442/400000 [00:17<00:34, 7634.50it/s] 34%|███▍      | 136239/400000 [00:18<00:34, 7724.55it/s] 34%|███▍      | 137026/400000 [00:18<00:33, 7767.18it/s] 34%|███▍      | 137848/400000 [00:18<00:33, 7895.71it/s] 35%|███▍      | 138740/400000 [00:18<00:31, 8174.97it/s] 35%|███▍      | 139607/400000 [00:18<00:31, 8316.14it/s] 35%|███▌      | 140442/400000 [00:18<00:31, 8160.71it/s] 35%|███▌      | 141261/400000 [00:18<00:32, 8014.83it/s] 36%|███▌      | 142065/400000 [00:18<00:32, 7852.39it/s] 36%|███▌      | 142878/400000 [00:18<00:32, 7931.22it/s] 36%|███▌      | 143720/400000 [00:18<00:31, 8070.62it/s] 36%|███▌      | 144530/400000 [00:19<00:31, 8065.01it/s] 36%|███▋      | 145371/400000 [00:19<00:31, 8163.48it/s] 37%|███▋      | 146253/400000 [00:19<00:30, 8348.13it/s] 37%|███▋      | 147145/400000 [00:19<00:29, 8511.32it/s] 37%|███▋      | 148039/400000 [00:19<00:29, 8634.25it/s] 37%|███▋      | 148905/400000 [00:19<00:29, 8434.38it/s] 37%|███▋      | 149771/400000 [00:19<00:29, 8497.41it/s] 38%|███▊      | 150623/400000 [00:19<00:30, 8267.31it/s] 38%|███▊      | 151467/400000 [00:19<00:29, 8315.30it/s] 38%|███▊      | 152326/400000 [00:19<00:29, 8394.16it/s] 38%|███▊      | 153167/400000 [00:20<00:30, 8180.71it/s] 39%|███▊      | 154030/400000 [00:20<00:29, 8308.99it/s] 39%|███▊      | 154915/400000 [00:20<00:28, 8462.21it/s] 39%|███▉      | 155778/400000 [00:20<00:28, 8509.60it/s] 39%|███▉      | 156677/400000 [00:20<00:28, 8647.96it/s] 39%|███▉      | 157544/400000 [00:20<00:29, 8333.17it/s] 40%|███▉      | 158381/400000 [00:20<00:29, 8182.20it/s] 40%|███▉      | 159203/400000 [00:20<00:30, 8007.78it/s] 40%|████      | 160007/400000 [00:20<00:30, 7820.51it/s] 40%|████      | 160793/400000 [00:21<00:31, 7604.73it/s] 40%|████      | 161557/400000 [00:21<00:31, 7480.45it/s] 41%|████      | 162308/400000 [00:21<00:31, 7448.33it/s] 41%|████      | 163095/400000 [00:21<00:31, 7566.54it/s] 41%|████      | 163854/400000 [00:21<00:32, 7278.74it/s] 41%|████      | 164626/400000 [00:21<00:31, 7404.54it/s] 41%|████▏     | 165451/400000 [00:21<00:30, 7638.36it/s] 42%|████▏     | 166219/400000 [00:21<00:30, 7582.80it/s] 42%|████▏     | 167109/400000 [00:21<00:29, 7933.79it/s] 42%|████▏     | 167909/400000 [00:21<00:29, 7908.70it/s] 42%|████▏     | 168772/400000 [00:22<00:28, 8111.57it/s] 42%|████▏     | 169588/400000 [00:22<00:28, 8032.28it/s] 43%|████▎     | 170431/400000 [00:22<00:28, 8146.88it/s] 43%|████▎     | 171345/400000 [00:22<00:27, 8420.68it/s] 43%|████▎     | 172199/400000 [00:22<00:26, 8452.78it/s] 43%|████▎     | 173049/400000 [00:22<00:26, 8466.10it/s] 43%|████▎     | 173902/400000 [00:22<00:26, 8483.05it/s] 44%|████▎     | 174824/400000 [00:22<00:25, 8691.25it/s] 44%|████▍     | 175696/400000 [00:22<00:27, 8288.79it/s] 44%|████▍     | 176531/400000 [00:22<00:27, 8007.80it/s] 44%|████▍     | 177338/400000 [00:23<00:28, 7765.70it/s] 45%|████▍     | 178143/400000 [00:23<00:28, 7845.51it/s] 45%|████▍     | 178932/400000 [00:23<00:28, 7685.05it/s] 45%|████▍     | 179705/400000 [00:23<00:28, 7652.56it/s] 45%|████▌     | 180473/400000 [00:23<00:28, 7617.19it/s] 45%|████▌     | 181237/400000 [00:23<00:29, 7512.91it/s] 46%|████▌     | 182086/400000 [00:23<00:28, 7771.16it/s] 46%|████▌     | 182935/400000 [00:23<00:27, 7973.03it/s] 46%|████▌     | 183803/400000 [00:23<00:26, 8170.62it/s] 46%|████▌     | 184640/400000 [00:24<00:26, 8228.52it/s] 46%|████▋     | 185466/400000 [00:24<00:27, 7912.66it/s] 47%|████▋     | 186262/400000 [00:24<00:27, 7834.30it/s] 47%|████▋     | 187049/400000 [00:24<00:27, 7746.36it/s] 47%|████▋     | 187827/400000 [00:24<00:27, 7626.20it/s] 47%|████▋     | 188592/400000 [00:24<00:28, 7502.68it/s] 47%|████▋     | 189345/400000 [00:24<00:28, 7465.71it/s] 48%|████▊     | 190094/400000 [00:24<00:28, 7439.21it/s] 48%|████▊     | 190848/400000 [00:24<00:28, 7466.55it/s] 48%|████▊     | 191655/400000 [00:24<00:27, 7637.74it/s] 48%|████▊     | 192421/400000 [00:25<00:27, 7499.97it/s] 48%|████▊     | 193173/400000 [00:25<00:27, 7460.35it/s] 48%|████▊     | 193921/400000 [00:25<00:27, 7410.95it/s] 49%|████▊     | 194698/400000 [00:25<00:27, 7513.97it/s] 49%|████▉     | 195469/400000 [00:25<00:27, 7571.24it/s] 49%|████▉     | 196256/400000 [00:25<00:26, 7657.98it/s] 49%|████▉     | 197069/400000 [00:25<00:26, 7791.69it/s] 49%|████▉     | 197850/400000 [00:25<00:26, 7652.85it/s] 50%|████▉     | 198617/400000 [00:25<00:26, 7577.15it/s] 50%|████▉     | 199402/400000 [00:25<00:26, 7655.76it/s] 50%|█████     | 200169/400000 [00:26<00:26, 7513.69it/s] 50%|█████     | 200927/400000 [00:26<00:26, 7531.41it/s] 50%|█████     | 201692/400000 [00:26<00:26, 7565.10it/s] 51%|█████     | 202450/400000 [00:26<00:26, 7448.89it/s] 51%|█████     | 203203/400000 [00:26<00:26, 7472.12it/s] 51%|█████     | 204016/400000 [00:26<00:25, 7657.00it/s] 51%|█████     | 204860/400000 [00:26<00:24, 7874.77it/s] 51%|█████▏    | 205736/400000 [00:26<00:23, 8119.02it/s] 52%|█████▏    | 206552/400000 [00:26<00:23, 8101.61it/s] 52%|█████▏    | 207365/400000 [00:26<00:24, 7865.01it/s] 52%|█████▏    | 208155/400000 [00:27<00:24, 7747.85it/s] 52%|█████▏    | 208933/400000 [00:27<00:25, 7542.62it/s] 52%|█████▏    | 209706/400000 [00:27<00:25, 7596.13it/s] 53%|█████▎    | 210472/400000 [00:27<00:24, 7613.09it/s] 53%|█████▎    | 211235/400000 [00:27<00:25, 7542.56it/s] 53%|█████▎    | 211993/400000 [00:27<00:24, 7543.92it/s] 53%|█████▎    | 212749/400000 [00:27<00:25, 7362.70it/s] 53%|█████▎    | 213517/400000 [00:27<00:25, 7452.23it/s] 54%|█████▎    | 214264/400000 [00:27<00:25, 7403.39it/s] 54%|█████▍    | 215008/400000 [00:28<00:24, 7414.27it/s] 54%|█████▍    | 215755/400000 [00:28<00:24, 7428.64it/s] 54%|█████▍    | 216499/400000 [00:28<00:24, 7404.34it/s] 54%|█████▍    | 217240/400000 [00:28<00:24, 7368.86it/s] 54%|█████▍    | 217978/400000 [00:28<00:24, 7342.30it/s] 55%|█████▍    | 218713/400000 [00:28<00:24, 7302.44it/s] 55%|█████▍    | 219444/400000 [00:28<00:25, 7135.50it/s] 55%|█████▌    | 220159/400000 [00:28<00:25, 7131.39it/s] 55%|█████▌    | 220905/400000 [00:28<00:24, 7224.24it/s] 55%|█████▌    | 221639/400000 [00:28<00:24, 7257.99it/s] 56%|█████▌    | 222394/400000 [00:29<00:24, 7342.70it/s] 56%|█████▌    | 223137/400000 [00:29<00:24, 7367.50it/s] 56%|█████▌    | 223875/400000 [00:29<00:24, 7246.56it/s] 56%|█████▌    | 224617/400000 [00:29<00:24, 7293.69it/s] 56%|█████▋    | 225348/400000 [00:29<00:23, 7298.21it/s] 57%|█████▋    | 226085/400000 [00:29<00:23, 7316.92it/s] 57%|█████▋    | 226822/400000 [00:29<00:23, 7330.89it/s] 57%|█████▋    | 227556/400000 [00:29<00:23, 7269.13it/s] 57%|█████▋    | 228284/400000 [00:29<00:23, 7215.51it/s] 57%|█████▋    | 229011/400000 [00:29<00:23, 7229.23it/s] 57%|█████▋    | 229735/400000 [00:30<00:23, 7153.31it/s] 58%|█████▊    | 230465/400000 [00:30<00:23, 7193.29it/s] 58%|█████▊    | 231185/400000 [00:30<00:23, 7184.93it/s] 58%|█████▊    | 231917/400000 [00:30<00:23, 7224.27it/s] 58%|█████▊    | 232640/400000 [00:30<00:23, 7222.96it/s] 58%|█████▊    | 233363/400000 [00:30<00:23, 7138.17it/s] 59%|█████▊    | 234154/400000 [00:30<00:22, 7351.04it/s] 59%|█████▊    | 234891/400000 [00:30<00:22, 7215.01it/s] 59%|█████▉    | 235687/400000 [00:30<00:22, 7421.78it/s] 59%|█████▉    | 236471/400000 [00:30<00:21, 7541.77it/s] 59%|█████▉    | 237348/400000 [00:31<00:20, 7871.73it/s] 60%|█████▉    | 238159/400000 [00:31<00:20, 7939.98it/s] 60%|█████▉    | 238961/400000 [00:31<00:20, 7963.22it/s] 60%|█████▉    | 239761/400000 [00:31<00:20, 7952.34it/s] 60%|██████    | 240559/400000 [00:31<00:20, 7906.19it/s] 60%|██████    | 241389/400000 [00:31<00:19, 8019.33it/s] 61%|██████    | 242213/400000 [00:31<00:19, 8081.52it/s] 61%|██████    | 243023/400000 [00:31<00:19, 7918.79it/s] 61%|██████    | 243849/400000 [00:31<00:19, 8017.38it/s] 61%|██████    | 244666/400000 [00:31<00:19, 8062.11it/s] 61%|██████▏   | 245474/400000 [00:32<00:19, 7987.19it/s] 62%|██████▏   | 246274/400000 [00:32<00:19, 7885.29it/s] 62%|██████▏   | 247064/400000 [00:32<00:19, 7854.02it/s] 62%|██████▏   | 247851/400000 [00:32<00:19, 7693.07it/s] 62%|██████▏   | 248622/400000 [00:32<00:19, 7632.97it/s] 62%|██████▏   | 249387/400000 [00:32<00:20, 7492.88it/s] 63%|██████▎   | 250138/400000 [00:32<00:20, 7446.35it/s] 63%|██████▎   | 250884/400000 [00:32<00:20, 7416.55it/s] 63%|██████▎   | 251645/400000 [00:32<00:19, 7470.95it/s] 63%|██████▎   | 252413/400000 [00:33<00:19, 7529.96it/s] 63%|██████▎   | 253167/400000 [00:33<00:19, 7522.53it/s] 63%|██████▎   | 253925/400000 [00:33<00:19, 7539.59it/s] 64%|██████▎   | 254704/400000 [00:33<00:19, 7610.72it/s] 64%|██████▍   | 255492/400000 [00:33<00:18, 7688.29it/s] 64%|██████▍   | 256294/400000 [00:33<00:18, 7780.81it/s] 64%|██████▍   | 257073/400000 [00:33<00:18, 7760.29it/s] 64%|██████▍   | 257850/400000 [00:33<00:18, 7570.87it/s] 65%|██████▍   | 258609/400000 [00:33<00:19, 7409.68it/s] 65%|██████▍   | 259360/400000 [00:33<00:18, 7438.61it/s] 65%|██████▌   | 260106/400000 [00:34<00:18, 7403.95it/s] 65%|██████▌   | 260884/400000 [00:34<00:18, 7510.90it/s] 65%|██████▌   | 261757/400000 [00:34<00:17, 7836.89it/s] 66%|██████▌   | 262546/400000 [00:34<00:17, 7756.80it/s] 66%|██████▌   | 263375/400000 [00:34<00:17, 7907.55it/s] 66%|██████▌   | 264169/400000 [00:34<00:17, 7887.69it/s] 66%|██████▌   | 264960/400000 [00:34<00:17, 7713.17it/s] 66%|██████▋   | 265734/400000 [00:34<00:17, 7582.03it/s] 67%|██████▋   | 266495/400000 [00:34<00:17, 7564.92it/s] 67%|██████▋   | 267274/400000 [00:34<00:17, 7630.70it/s] 67%|██████▋   | 268045/400000 [00:35<00:17, 7654.30it/s] 67%|██████▋   | 268878/400000 [00:35<00:16, 7844.30it/s] 67%|██████▋   | 269682/400000 [00:35<00:16, 7901.85it/s] 68%|██████▊   | 270474/400000 [00:35<00:16, 7793.89it/s] 68%|██████▊   | 271255/400000 [00:35<00:16, 7647.84it/s] 68%|██████▊   | 272022/400000 [00:35<00:17, 7273.29it/s] 68%|██████▊   | 272783/400000 [00:35<00:17, 7369.51it/s] 68%|██████▊   | 273591/400000 [00:35<00:16, 7569.11it/s] 69%|██████▊   | 274382/400000 [00:35<00:16, 7667.42it/s] 69%|██████▉   | 275168/400000 [00:35<00:16, 7722.94it/s] 69%|██████▉   | 275943/400000 [00:36<00:16, 7464.67it/s] 69%|██████▉   | 276693/400000 [00:36<00:16, 7423.02it/s] 69%|██████▉   | 277438/400000 [00:36<00:16, 7368.08it/s] 70%|██████▉   | 278177/400000 [00:36<00:16, 7337.95it/s] 70%|██████▉   | 278913/400000 [00:36<00:16, 7341.20it/s] 70%|██████▉   | 279656/400000 [00:36<00:16, 7366.10it/s] 70%|███████   | 280438/400000 [00:36<00:15, 7479.09it/s] 70%|███████   | 281187/400000 [00:36<00:15, 7432.56it/s] 70%|███████   | 281931/400000 [00:36<00:15, 7395.18it/s] 71%|███████   | 282717/400000 [00:36<00:15, 7527.50it/s] 71%|███████   | 283471/400000 [00:37<00:15, 7500.24it/s] 71%|███████   | 284222/400000 [00:37<00:15, 7412.52it/s] 71%|███████   | 284966/400000 [00:37<00:15, 7418.07it/s] 71%|███████▏  | 285709/400000 [00:37<00:15, 7407.99it/s] 72%|███████▏  | 286451/400000 [00:37<00:15, 7363.48it/s] 72%|███████▏  | 287188/400000 [00:37<00:15, 7280.88it/s] 72%|███████▏  | 287917/400000 [00:37<00:15, 7256.56it/s] 72%|███████▏  | 288674/400000 [00:37<00:15, 7346.27it/s] 72%|███████▏  | 289410/400000 [00:37<00:15, 7313.20it/s] 73%|███████▎  | 290159/400000 [00:38<00:14, 7365.06it/s] 73%|███████▎  | 290901/400000 [00:38<00:14, 7378.49it/s] 73%|███████▎  | 291640/400000 [00:38<00:14, 7286.98it/s] 73%|███████▎  | 292370/400000 [00:38<00:14, 7276.70it/s] 73%|███████▎  | 293098/400000 [00:38<00:14, 7213.80it/s] 73%|███████▎  | 293820/400000 [00:38<00:14, 7177.97it/s] 74%|███████▎  | 294630/400000 [00:38<00:14, 7430.60it/s] 74%|███████▍  | 295376/400000 [00:38<00:14, 7395.19it/s] 74%|███████▍  | 296171/400000 [00:38<00:13, 7551.98it/s] 74%|███████▍  | 296929/400000 [00:38<00:13, 7521.32it/s] 74%|███████▍  | 297818/400000 [00:39<00:12, 7884.80it/s] 75%|███████▍  | 298613/400000 [00:39<00:12, 7891.91it/s] 75%|███████▍  | 299407/400000 [00:39<00:12, 7813.05it/s] 75%|███████▌  | 300192/400000 [00:39<00:12, 7769.20it/s] 75%|███████▌  | 300972/400000 [00:39<00:13, 7580.43it/s] 75%|███████▌  | 301733/400000 [00:39<00:12, 7576.26it/s] 76%|███████▌  | 302495/400000 [00:39<00:12, 7588.67it/s] 76%|███████▌  | 303256/400000 [00:39<00:13, 7223.91it/s] 76%|███████▌  | 303983/400000 [00:39<00:13, 6915.79it/s] 76%|███████▌  | 304757/400000 [00:39<00:13, 7143.77it/s] 76%|███████▋  | 305541/400000 [00:40<00:12, 7338.59it/s] 77%|███████▋  | 306363/400000 [00:40<00:12, 7581.40it/s] 77%|███████▋  | 307128/400000 [00:40<00:12, 7328.09it/s] 77%|███████▋  | 307907/400000 [00:40<00:12, 7459.44it/s] 77%|███████▋  | 308676/400000 [00:40<00:12, 7526.76it/s] 77%|███████▋  | 309433/400000 [00:40<00:12, 7536.70it/s] 78%|███████▊  | 310203/400000 [00:40<00:11, 7582.77it/s] 78%|███████▊  | 310985/400000 [00:40<00:11, 7650.35it/s] 78%|███████▊  | 311851/400000 [00:40<00:11, 7926.42it/s] 78%|███████▊  | 312648/400000 [00:40<00:11, 7866.56it/s] 78%|███████▊  | 313443/400000 [00:41<00:10, 7888.63it/s] 79%|███████▊  | 314252/400000 [00:41<00:10, 7947.52it/s] 79%|███████▉  | 315049/400000 [00:41<00:10, 7885.62it/s] 79%|███████▉  | 315839/400000 [00:41<00:10, 7863.81it/s] 79%|███████▉  | 316627/400000 [00:41<00:10, 7838.03it/s] 79%|███████▉  | 317412/400000 [00:41<00:10, 7783.73it/s] 80%|███████▉  | 318191/400000 [00:41<00:10, 7737.58it/s] 80%|███████▉  | 318966/400000 [00:41<00:10, 7659.14it/s] 80%|███████▉  | 319733/400000 [00:41<00:10, 7516.25it/s] 80%|████████  | 320486/400000 [00:42<00:10, 7469.44it/s] 80%|████████  | 321251/400000 [00:42<00:10, 7500.13it/s] 81%|████████  | 322002/400000 [00:42<00:10, 7339.88it/s] 81%|████████  | 322845/400000 [00:42<00:10, 7634.57it/s] 81%|████████  | 323613/400000 [00:42<00:10, 7616.84it/s] 81%|████████  | 324378/400000 [00:42<00:10, 7435.73it/s] 81%|████████▏ | 325137/400000 [00:42<00:10, 7478.81it/s] 81%|████████▏ | 325919/400000 [00:42<00:09, 7575.69it/s] 82%|████████▏ | 326689/400000 [00:42<00:09, 7611.28it/s] 82%|████████▏ | 327452/400000 [00:42<00:09, 7595.50it/s] 82%|████████▏ | 328213/400000 [00:43<00:09, 7569.52it/s] 82%|████████▏ | 329076/400000 [00:43<00:09, 7857.88it/s] 82%|████████▏ | 329884/400000 [00:43<00:08, 7921.49it/s] 83%|████████▎ | 330679/400000 [00:43<00:08, 7794.22it/s] 83%|████████▎ | 331461/400000 [00:43<00:08, 7667.85it/s] 83%|████████▎ | 332230/400000 [00:43<00:08, 7603.25it/s] 83%|████████▎ | 332992/400000 [00:43<00:08, 7596.37it/s] 83%|████████▎ | 333756/400000 [00:43<00:08, 7605.17it/s] 84%|████████▎ | 334541/400000 [00:43<00:08, 7676.49it/s] 84%|████████▍ | 335339/400000 [00:43<00:08, 7765.02it/s] 84%|████████▍ | 336117/400000 [00:44<00:08, 7742.49it/s] 84%|████████▍ | 336997/400000 [00:44<00:07, 8031.06it/s] 84%|████████▍ | 337840/400000 [00:44<00:07, 8146.62it/s] 85%|████████▍ | 338661/400000 [00:44<00:07, 8162.68it/s] 85%|████████▍ | 339480/400000 [00:44<00:07, 7952.83it/s] 85%|████████▌ | 340278/400000 [00:44<00:07, 7770.37it/s] 85%|████████▌ | 341058/400000 [00:44<00:07, 7709.44it/s] 85%|████████▌ | 341832/400000 [00:44<00:07, 7718.37it/s] 86%|████████▌ | 342628/400000 [00:44<00:07, 7785.72it/s] 86%|████████▌ | 343408/400000 [00:44<00:07, 7707.04it/s] 86%|████████▌ | 344180/400000 [00:45<00:07, 7691.75it/s] 86%|████████▋ | 345020/400000 [00:45<00:06, 7890.18it/s] 86%|████████▋ | 345875/400000 [00:45<00:06, 8075.19it/s] 87%|████████▋ | 346685/400000 [00:45<00:06, 7979.75it/s] 87%|████████▋ | 347485/400000 [00:45<00:06, 7980.57it/s] 87%|████████▋ | 348285/400000 [00:45<00:06, 7814.44it/s] 87%|████████▋ | 349069/400000 [00:45<00:06, 7739.44it/s] 87%|████████▋ | 349859/400000 [00:45<00:06, 7785.38it/s] 88%|████████▊ | 350699/400000 [00:45<00:06, 7958.92it/s] 88%|████████▊ | 351547/400000 [00:45<00:05, 8107.17it/s] 88%|████████▊ | 352360/400000 [00:46<00:05, 7950.53it/s] 88%|████████▊ | 353175/400000 [00:46<00:05, 8008.90it/s] 88%|████████▊ | 353978/400000 [00:46<00:05, 7940.71it/s] 89%|████████▊ | 354774/400000 [00:46<00:05, 7809.95it/s] 89%|████████▉ | 355557/400000 [00:46<00:05, 7642.77it/s] 89%|████████▉ | 356323/400000 [00:46<00:05, 7465.79it/s] 89%|████████▉ | 357085/400000 [00:46<00:05, 7509.01it/s] 89%|████████▉ | 357896/400000 [00:46<00:05, 7677.47it/s] 90%|████████▉ | 358699/400000 [00:46<00:05, 7778.66it/s] 90%|████████▉ | 359481/400000 [00:47<00:05, 7789.69it/s] 90%|█████████ | 360262/400000 [00:47<00:05, 7658.08it/s] 90%|█████████ | 361030/400000 [00:47<00:05, 7537.59it/s] 90%|█████████ | 361796/400000 [00:47<00:05, 7573.77it/s] 91%|█████████ | 362555/400000 [00:47<00:05, 7411.98it/s] 91%|█████████ | 363298/400000 [00:47<00:05, 7333.25it/s] 91%|█████████ | 364033/400000 [00:47<00:04, 7320.24it/s] 91%|█████████ | 364768/400000 [00:47<00:04, 7328.69it/s] 91%|█████████▏| 365514/400000 [00:47<00:04, 7364.46it/s] 92%|█████████▏| 366251/400000 [00:47<00:04, 7337.72it/s] 92%|█████████▏| 366986/400000 [00:48<00:04, 7323.42it/s] 92%|█████████▏| 367722/400000 [00:48<00:04, 7332.49it/s] 92%|█████████▏| 368456/400000 [00:48<00:04, 7114.31it/s] 92%|█████████▏| 369169/400000 [00:48<00:04, 7104.73it/s] 92%|█████████▏| 369905/400000 [00:48<00:04, 7179.29it/s] 93%|█████████▎| 370662/400000 [00:48<00:04, 7290.84it/s] 93%|█████████▎| 371408/400000 [00:48<00:03, 7339.89it/s] 93%|█████████▎| 372182/400000 [00:48<00:03, 7453.92it/s] 93%|█████████▎| 372929/400000 [00:48<00:03, 7341.49it/s] 93%|█████████▎| 373668/400000 [00:48<00:03, 7355.82it/s] 94%|█████████▎| 374405/400000 [00:49<00:03, 7236.52it/s] 94%|█████████▍| 375130/400000 [00:49<00:03, 7080.89it/s] 94%|█████████▍| 375843/400000 [00:49<00:03, 7093.01it/s] 94%|█████████▍| 376607/400000 [00:49<00:03, 7246.63it/s] 94%|█████████▍| 377334/400000 [00:49<00:03, 7033.21it/s] 95%|█████████▍| 378040/400000 [00:49<00:03, 6968.10it/s] 95%|█████████▍| 378755/400000 [00:49<00:03, 7021.13it/s] 95%|█████████▍| 379473/400000 [00:49<00:02, 7067.16it/s] 95%|█████████▌| 380227/400000 [00:49<00:02, 7200.35it/s] 95%|█████████▌| 380953/400000 [00:49<00:02, 7216.19it/s] 95%|█████████▌| 381676/400000 [00:50<00:02, 7136.28it/s] 96%|█████████▌| 382411/400000 [00:50<00:02, 7198.84it/s] 96%|█████████▌| 383154/400000 [00:50<00:02, 7264.25it/s] 96%|█████████▌| 383882/400000 [00:50<00:02, 7234.07it/s] 96%|█████████▌| 384606/400000 [00:50<00:02, 7215.06it/s] 96%|█████████▋| 385328/400000 [00:50<00:02, 7187.27it/s] 97%|█████████▋| 386047/400000 [00:50<00:01, 7154.70it/s] 97%|█████████▋| 386790/400000 [00:50<00:01, 7233.32it/s] 97%|█████████▋| 387586/400000 [00:50<00:01, 7435.48it/s] 97%|█████████▋| 388360/400000 [00:51<00:01, 7522.61it/s] 97%|█████████▋| 389143/400000 [00:51<00:01, 7611.98it/s] 97%|█████████▋| 389906/400000 [00:51<00:01, 7578.83it/s] 98%|█████████▊| 390665/400000 [00:51<00:01, 7556.02it/s] 98%|█████████▊| 391446/400000 [00:51<00:01, 7629.42it/s] 98%|█████████▊| 392210/400000 [00:51<00:01, 7404.66it/s] 98%|█████████▊| 392968/400000 [00:51<00:00, 7454.14it/s] 98%|█████████▊| 393720/400000 [00:51<00:00, 7472.10it/s] 99%|█████████▊| 394469/400000 [00:51<00:00, 7382.78it/s] 99%|█████████▉| 395238/400000 [00:51<00:00, 7469.75it/s] 99%|█████████▉| 396014/400000 [00:52<00:00, 7552.29it/s] 99%|█████████▉| 396771/400000 [00:52<00:00, 7504.69it/s] 99%|█████████▉| 397523/400000 [00:52<00:00, 7486.58it/s]100%|█████████▉| 398273/400000 [00:52<00:00, 7383.07it/s]100%|█████████▉| 399048/400000 [00:52<00:00, 7487.36it/s]100%|█████████▉| 399798/400000 [00:52<00:00, 7394.70it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7610.48it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fdb88fc70f0>, <torchtext.data.dataset.TabularDataset object at 0x7fdb88fc7240>, <torchtext.vocab.Vocab object at 0x7fdb88fc7160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.25 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.26 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.26 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.03 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.03 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.54 file/s]2020-06-08 00:33:47.115219: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-08 00:33:47.119558: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-08 00:33:47.119716: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a587590a40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 00:33:47.119731: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147284.64it/s] 76%|███████▋  | 7561216/9912422 [00:00<00:11, 210226.58it/s]9920512it [00:00, 42546865.52it/s]                           
0it [00:00, ?it/s]32768it [00:00, 571065.70it/s]
0it [00:00, ?it/s]1654784it [00:00, 22290022.32it/s]
0it [00:00, ?it/s]8192it [00:00, 150799.82it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
