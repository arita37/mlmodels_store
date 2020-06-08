
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f743677e6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f743677e6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f74a1d460d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f74a1d460d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f74bba73ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f74bba73ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f744f5c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f744f5c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f744f5c1950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 469134.70it/s] 84%|████████▍ | 8364032/9912422 [00:00<00:02, 668561.71it/s]9920512it [00:00, 45152585.97it/s]                           
0it [00:00, ?it/s]32768it [00:00, 657036.78it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160611.31it/s]1654784it [00:00, 11574877.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 149829.23it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f744cc22b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f744cc17da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f744f5c1598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f744f5c1598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f744f5c1598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:43,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:42,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:41,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:29,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:19,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.32 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:12, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 13.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 18.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:00<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:00<00:03, 24.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:00<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 46.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 68.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 72.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 72.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 73.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.79 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.79 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.79 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ url]
0 examples [00:00, ? examples/s]2020-06-08 18:08:54.780938: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-08 18:08:54.796702: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095199999 Hz
2020-06-08 18:08:54.797206: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bfa26465a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 18:08:54.797224: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 179.83 examples/s]117 examples [00:00, 238.27 examples/s]233 examples [00:00, 312.79 examples/s]332 examples [00:00, 393.41 examples/s]444 examples [00:00, 488.19 examples/s]552 examples [00:00, 583.47 examples/s]668 examples [00:00, 684.75 examples/s]774 examples [00:00, 764.62 examples/s]879 examples [00:00, 831.30 examples/s]995 examples [00:01, 907.86 examples/s]1106 examples [00:01, 959.21 examples/s]1219 examples [00:01, 1004.22 examples/s]1333 examples [00:01, 1039.91 examples/s]1445 examples [00:01, 1061.45 examples/s]1562 examples [00:01, 1089.75 examples/s]1678 examples [00:01, 1108.62 examples/s]1796 examples [00:01, 1126.83 examples/s]1911 examples [00:01, 1132.94 examples/s]2026 examples [00:01, 1123.39 examples/s]2141 examples [00:02, 1130.01 examples/s]2255 examples [00:02, 1119.29 examples/s]2370 examples [00:02, 1126.45 examples/s]2483 examples [00:02, 1125.10 examples/s]2596 examples [00:02, 1103.57 examples/s]2712 examples [00:02, 1119.02 examples/s]2825 examples [00:02, 1119.69 examples/s]2938 examples [00:02, 1100.76 examples/s]3055 examples [00:02, 1117.85 examples/s]3167 examples [00:02, 1100.57 examples/s]3284 examples [00:03, 1118.56 examples/s]3397 examples [00:03, 1118.84 examples/s]3510 examples [00:03, 1121.30 examples/s]3623 examples [00:03, 1084.51 examples/s]3732 examples [00:03, 1083.14 examples/s]3847 examples [00:03, 1102.13 examples/s]3962 examples [00:03, 1114.36 examples/s]4077 examples [00:03, 1123.66 examples/s]4191 examples [00:03, 1127.15 examples/s]4304 examples [00:03, 1127.33 examples/s]4417 examples [00:04, 1096.45 examples/s]4529 examples [00:04, 1100.96 examples/s]4641 examples [00:04, 1103.98 examples/s]4755 examples [00:04, 1114.24 examples/s]4868 examples [00:04, 1116.66 examples/s]4980 examples [00:04, 1113.76 examples/s]5097 examples [00:04, 1129.21 examples/s]5211 examples [00:04, 1126.36 examples/s]5324 examples [00:04, 1113.21 examples/s]5436 examples [00:04, 1095.50 examples/s]5546 examples [00:05, 1084.11 examples/s]5665 examples [00:05, 1112.79 examples/s]5777 examples [00:05, 1083.14 examples/s]5889 examples [00:05, 1091.74 examples/s]5999 examples [00:05, 1087.78 examples/s]6113 examples [00:05, 1100.82 examples/s]6226 examples [00:05, 1107.35 examples/s]6337 examples [00:05, 1106.11 examples/s]6448 examples [00:05, 1102.57 examples/s]6563 examples [00:06, 1114.97 examples/s]6676 examples [00:06, 1117.38 examples/s]6794 examples [00:06, 1133.55 examples/s]6908 examples [00:06, 1134.74 examples/s]7022 examples [00:06, 1088.23 examples/s]7132 examples [00:06, 1088.12 examples/s]7248 examples [00:06, 1107.07 examples/s]7360 examples [00:06, 1033.75 examples/s]7465 examples [00:06, 1020.64 examples/s]7568 examples [00:06, 1005.52 examples/s]7670 examples [00:07, 978.28 examples/s] 7777 examples [00:07, 1003.42 examples/s]7878 examples [00:07, 970.17 examples/s] 7976 examples [00:07, 960.79 examples/s]8073 examples [00:07, 929.86 examples/s]8170 examples [00:07, 939.41 examples/s]8266 examples [00:07, 943.65 examples/s]8361 examples [00:07, 941.89 examples/s]8456 examples [00:07, 929.21 examples/s]8550 examples [00:08, 909.80 examples/s]8666 examples [00:08, 971.49 examples/s]8781 examples [00:08, 1018.13 examples/s]8894 examples [00:08, 1047.19 examples/s]9007 examples [00:08, 1070.15 examples/s]9117 examples [00:08, 1078.00 examples/s]9228 examples [00:08, 1086.60 examples/s]9338 examples [00:08, 1071.60 examples/s]9448 examples [00:08, 1077.57 examples/s]9559 examples [00:08, 1086.15 examples/s]9668 examples [00:09, 1043.35 examples/s]9781 examples [00:09, 1066.10 examples/s]9892 examples [00:09, 1078.89 examples/s]10001 examples [00:09, 1038.67 examples/s]10106 examples [00:09, 995.36 examples/s] 10218 examples [00:09, 1028.54 examples/s]10332 examples [00:09, 1058.57 examples/s]10447 examples [00:09, 1084.11 examples/s]10561 examples [00:09, 1097.71 examples/s]10676 examples [00:09, 1111.00 examples/s]10788 examples [00:10, 1078.41 examples/s]10899 examples [00:10, 1085.56 examples/s]11012 examples [00:10, 1097.17 examples/s]11128 examples [00:10, 1114.77 examples/s]11240 examples [00:10, 1104.83 examples/s]11353 examples [00:10, 1110.59 examples/s]11471 examples [00:10, 1127.90 examples/s]11584 examples [00:10, 1093.95 examples/s]11696 examples [00:10, 1101.23 examples/s]11807 examples [00:10, 1095.77 examples/s]11922 examples [00:11, 1108.74 examples/s]12042 examples [00:11, 1133.06 examples/s]12156 examples [00:11, 1111.66 examples/s]12271 examples [00:11, 1121.88 examples/s]12384 examples [00:11, 1114.91 examples/s]12496 examples [00:11, 1101.25 examples/s]12612 examples [00:11, 1116.40 examples/s]12726 examples [00:11, 1120.94 examples/s]12841 examples [00:11, 1128.95 examples/s]12955 examples [00:12, 1131.29 examples/s]13069 examples [00:12, 1119.67 examples/s]13182 examples [00:12, 1120.48 examples/s]13295 examples [00:12, 1122.21 examples/s]13408 examples [00:12, 1102.43 examples/s]13519 examples [00:12, 1079.86 examples/s]13630 examples [00:12, 1088.14 examples/s]13739 examples [00:12, 1081.92 examples/s]13850 examples [00:12, 1087.27 examples/s]13965 examples [00:12, 1103.29 examples/s]14078 examples [00:13, 1108.80 examples/s]14191 examples [00:13, 1113.21 examples/s]14309 examples [00:13, 1130.16 examples/s]14423 examples [00:13, 1130.75 examples/s]14539 examples [00:13, 1138.77 examples/s]14653 examples [00:13, 1127.30 examples/s]14766 examples [00:13, 1101.09 examples/s]14883 examples [00:13, 1120.54 examples/s]15003 examples [00:13, 1141.39 examples/s]15121 examples [00:13, 1151.35 examples/s]15237 examples [00:14, 1150.75 examples/s]15353 examples [00:14, 1144.70 examples/s]15468 examples [00:14, 1138.70 examples/s]15583 examples [00:14, 1141.23 examples/s]15698 examples [00:14, 1141.99 examples/s]15813 examples [00:14, 1132.99 examples/s]15927 examples [00:14, 1120.86 examples/s]16041 examples [00:14, 1126.36 examples/s]16154 examples [00:14, 1122.24 examples/s]16267 examples [00:14, 1116.95 examples/s]16379 examples [00:15, 1098.28 examples/s]16491 examples [00:15, 1103.96 examples/s]16605 examples [00:15, 1113.42 examples/s]16718 examples [00:15, 1117.99 examples/s]16834 examples [00:15, 1128.67 examples/s]16947 examples [00:15, 1087.45 examples/s]17057 examples [00:15, 1077.69 examples/s]17173 examples [00:15, 1100.73 examples/s]17287 examples [00:15, 1112.11 examples/s]17402 examples [00:15, 1122.05 examples/s]17515 examples [00:16, 1120.54 examples/s]17631 examples [00:16, 1130.92 examples/s]17745 examples [00:16, 1129.47 examples/s]17860 examples [00:16, 1134.23 examples/s]17977 examples [00:16, 1144.40 examples/s]18092 examples [00:16, 1136.96 examples/s]18206 examples [00:16, 1136.04 examples/s]18320 examples [00:16, 1113.05 examples/s]18432 examples [00:16, 1106.84 examples/s]18544 examples [00:17, 1109.97 examples/s]18656 examples [00:17, 1111.06 examples/s]18772 examples [00:17, 1123.00 examples/s]18885 examples [00:17, 1115.00 examples/s]18997 examples [00:17, 1112.10 examples/s]19113 examples [00:17, 1123.58 examples/s]19226 examples [00:17, 1116.56 examples/s]19338 examples [00:17, 1098.45 examples/s]19449 examples [00:17, 1099.45 examples/s]19562 examples [00:17, 1106.10 examples/s]19676 examples [00:18, 1113.71 examples/s]19788 examples [00:18, 1115.20 examples/s]19900 examples [00:18, 1113.83 examples/s]20012 examples [00:18, 1058.14 examples/s]20123 examples [00:18, 1072.70 examples/s]20236 examples [00:18, 1087.76 examples/s]20346 examples [00:18, 1042.61 examples/s]20452 examples [00:18, 1045.54 examples/s]20564 examples [00:18, 1065.12 examples/s]20680 examples [00:18, 1089.34 examples/s]20793 examples [00:19, 1100.40 examples/s]20904 examples [00:19, 1089.05 examples/s]21016 examples [00:19, 1097.54 examples/s]21129 examples [00:19, 1106.17 examples/s]21240 examples [00:19, 1098.22 examples/s]21350 examples [00:19, 1087.00 examples/s]21463 examples [00:19, 1098.15 examples/s]21573 examples [00:19, 1091.43 examples/s]21683 examples [00:19, 1079.61 examples/s]21794 examples [00:19, 1088.33 examples/s]21906 examples [00:20, 1097.08 examples/s]22018 examples [00:20, 1101.06 examples/s]22129 examples [00:20, 1072.50 examples/s]22238 examples [00:20, 1077.07 examples/s]22348 examples [00:20, 1083.25 examples/s]22459 examples [00:20, 1089.91 examples/s]22569 examples [00:20, 1091.99 examples/s]22679 examples [00:20, 1067.93 examples/s]22786 examples [00:20, 1062.56 examples/s]22896 examples [00:21, 1073.35 examples/s]23004 examples [00:21, 1072.99 examples/s]23112 examples [00:21, 1060.84 examples/s]23223 examples [00:21, 1073.45 examples/s]23335 examples [00:21, 1085.67 examples/s]23446 examples [00:21, 1092.17 examples/s]23556 examples [00:21, 1093.20 examples/s]23666 examples [00:21, 1050.72 examples/s]23772 examples [00:21, 1048.48 examples/s]23880 examples [00:21, 1056.07 examples/s]23990 examples [00:22, 1067.72 examples/s]24097 examples [00:22, 1056.72 examples/s]24208 examples [00:22, 1069.42 examples/s]24322 examples [00:22, 1088.49 examples/s]24432 examples [00:22, 1081.28 examples/s]24542 examples [00:22, 1086.02 examples/s]24654 examples [00:22, 1094.80 examples/s]24766 examples [00:22, 1099.17 examples/s]24879 examples [00:22, 1107.63 examples/s]24991 examples [00:22, 1110.55 examples/s]25103 examples [00:23, 1090.15 examples/s]25216 examples [00:23, 1100.24 examples/s]25327 examples [00:23, 1101.14 examples/s]25438 examples [00:23, 1098.76 examples/s]25548 examples [00:23, 1093.04 examples/s]25658 examples [00:23, 1090.53 examples/s]25773 examples [00:23, 1106.46 examples/s]25887 examples [00:23, 1115.56 examples/s]25999 examples [00:23, 1108.38 examples/s]26113 examples [00:23, 1115.02 examples/s]26225 examples [00:24, 1084.79 examples/s]26344 examples [00:24, 1112.00 examples/s]26460 examples [00:24, 1122.89 examples/s]26573 examples [00:24, 1116.17 examples/s]26685 examples [00:24, 1114.41 examples/s]26801 examples [00:24, 1126.68 examples/s]26915 examples [00:24, 1127.81 examples/s]27028 examples [00:24, 1088.14 examples/s]27144 examples [00:24, 1108.20 examples/s]27257 examples [00:24, 1113.39 examples/s]27369 examples [00:25, 1108.53 examples/s]27481 examples [00:25, 1100.65 examples/s]27592 examples [00:25, 1097.33 examples/s]27706 examples [00:25, 1108.11 examples/s]27817 examples [00:25, 1092.49 examples/s]27927 examples [00:25, 1094.62 examples/s]28045 examples [00:25, 1118.40 examples/s]28158 examples [00:25, 1119.18 examples/s]28274 examples [00:25, 1130.73 examples/s]28388 examples [00:26, 1127.00 examples/s]28501 examples [00:26, 1111.95 examples/s]28614 examples [00:26, 1115.93 examples/s]28726 examples [00:26, 1107.40 examples/s]28838 examples [00:26, 1109.66 examples/s]28953 examples [00:26, 1119.45 examples/s]29066 examples [00:26, 1121.34 examples/s]29179 examples [00:26, 1117.62 examples/s]29291 examples [00:26, 1101.87 examples/s]29402 examples [00:26, 1099.35 examples/s]29516 examples [00:27, 1110.70 examples/s]29628 examples [00:27, 1107.89 examples/s]29739 examples [00:27, 1099.95 examples/s]29850 examples [00:27, 1084.61 examples/s]29959 examples [00:27, 1062.36 examples/s]30066 examples [00:27, 997.55 examples/s] 30170 examples [00:27, 1007.48 examples/s]30273 examples [00:27, 1012.01 examples/s]30375 examples [00:27, 941.62 examples/s] 30482 examples [00:27, 975.90 examples/s]30592 examples [00:28, 1004.77 examples/s]30700 examples [00:28, 1023.18 examples/s]30804 examples [00:28, 973.96 examples/s] 30914 examples [00:28, 1005.16 examples/s]31022 examples [00:28, 1025.14 examples/s]31134 examples [00:28, 1051.01 examples/s]31246 examples [00:28, 1070.11 examples/s]31354 examples [00:28, 1071.85 examples/s]31462 examples [00:28, 1044.08 examples/s]31567 examples [00:29, 1045.26 examples/s]31672 examples [00:29, 1045.88 examples/s]31781 examples [00:29, 1056.12 examples/s]31887 examples [00:29, 1036.11 examples/s]31999 examples [00:29, 1058.69 examples/s]32106 examples [00:29, 1048.51 examples/s]32219 examples [00:29, 1069.37 examples/s]32334 examples [00:29, 1091.93 examples/s]32445 examples [00:29, 1096.66 examples/s]32557 examples [00:29, 1103.36 examples/s]32668 examples [00:30, 1095.75 examples/s]32778 examples [00:30, 1041.19 examples/s]32888 examples [00:30, 1057.26 examples/s]32999 examples [00:30, 1072.42 examples/s]33107 examples [00:30, 1069.82 examples/s]33223 examples [00:30, 1093.67 examples/s]33338 examples [00:30, 1107.12 examples/s]33450 examples [00:30, 1108.21 examples/s]33562 examples [00:30, 1050.03 examples/s]33668 examples [00:30, 1047.77 examples/s]33774 examples [00:31, 1042.40 examples/s]33886 examples [00:31, 1062.32 examples/s]33997 examples [00:31, 1072.59 examples/s]34107 examples [00:31, 1079.99 examples/s]34217 examples [00:31, 1085.38 examples/s]34326 examples [00:31, 1086.32 examples/s]34436 examples [00:31, 1088.27 examples/s]34549 examples [00:31, 1100.36 examples/s]34661 examples [00:31, 1105.23 examples/s]34772 examples [00:32, 1048.14 examples/s]34884 examples [00:32, 1067.27 examples/s]34992 examples [00:32, 1063.29 examples/s]35101 examples [00:32, 1070.21 examples/s]35209 examples [00:32, 1069.12 examples/s]35317 examples [00:32, 1064.42 examples/s]35424 examples [00:32, 1057.87 examples/s]35537 examples [00:32, 1076.48 examples/s]35649 examples [00:32, 1086.55 examples/s]35758 examples [00:32, 1087.03 examples/s]35870 examples [00:33, 1096.46 examples/s]35983 examples [00:33, 1105.53 examples/s]36097 examples [00:33, 1114.12 examples/s]36209 examples [00:33, 1102.88 examples/s]36320 examples [00:33, 1102.49 examples/s]36434 examples [00:33, 1112.27 examples/s]36548 examples [00:33, 1119.70 examples/s]36666 examples [00:33, 1135.22 examples/s]36782 examples [00:33, 1140.87 examples/s]36897 examples [00:33, 1131.70 examples/s]37012 examples [00:34, 1134.79 examples/s]37126 examples [00:34, 1129.12 examples/s]37240 examples [00:34, 1130.12 examples/s]37354 examples [00:34, 1103.96 examples/s]37465 examples [00:34, 1103.80 examples/s]37577 examples [00:34, 1106.18 examples/s]37688 examples [00:34, 1083.49 examples/s]37797 examples [00:34, 1081.70 examples/s]37909 examples [00:34, 1092.78 examples/s]38019 examples [00:34, 1093.66 examples/s]38131 examples [00:35, 1098.56 examples/s]38242 examples [00:35, 1100.46 examples/s]38353 examples [00:35, 1053.66 examples/s]38464 examples [00:35, 1069.08 examples/s]38574 examples [00:35, 1078.12 examples/s]38684 examples [00:35, 1081.67 examples/s]38793 examples [00:35, 1075.09 examples/s]38905 examples [00:35, 1087.47 examples/s]39018 examples [00:35, 1098.16 examples/s]39129 examples [00:35, 1100.29 examples/s]39242 examples [00:36, 1108.14 examples/s]39353 examples [00:36, 1105.53 examples/s]39467 examples [00:36, 1114.63 examples/s]39579 examples [00:36, 1113.41 examples/s]39691 examples [00:36, 1104.59 examples/s]39802 examples [00:36, 1098.55 examples/s]39912 examples [00:36, 1093.96 examples/s]40022 examples [00:36, 1049.57 examples/s]40132 examples [00:36, 1063.67 examples/s]40243 examples [00:37, 1075.95 examples/s]40356 examples [00:37, 1089.16 examples/s]40469 examples [00:37, 1098.96 examples/s]40580 examples [00:37, 1099.18 examples/s]40692 examples [00:37, 1104.77 examples/s]40803 examples [00:37, 1077.32 examples/s]40913 examples [00:37, 1083.60 examples/s]41026 examples [00:37, 1096.48 examples/s]41139 examples [00:37, 1106.02 examples/s]41252 examples [00:37, 1111.51 examples/s]41364 examples [00:38, 1111.62 examples/s]41476 examples [00:38, 1102.75 examples/s]41587 examples [00:38, 1104.71 examples/s]41698 examples [00:38, 1050.13 examples/s]41809 examples [00:38, 1065.17 examples/s]41921 examples [00:38, 1079.77 examples/s]42030 examples [00:38, 1082.51 examples/s]42145 examples [00:38, 1100.17 examples/s]42256 examples [00:38, 1099.37 examples/s]42367 examples [00:38, 1061.97 examples/s]42477 examples [00:39, 1070.81 examples/s]42588 examples [00:39, 1080.35 examples/s]42702 examples [00:39, 1097.36 examples/s]42812 examples [00:39, 1095.96 examples/s]42922 examples [00:39, 1095.38 examples/s]43033 examples [00:39, 1099.56 examples/s]43144 examples [00:39, 1079.89 examples/s]43253 examples [00:39, 1066.75 examples/s]43364 examples [00:39, 1077.95 examples/s]43472 examples [00:39, 1066.20 examples/s]43581 examples [00:40, 1072.52 examples/s]43689 examples [00:40, 1025.64 examples/s]43798 examples [00:40, 1044.05 examples/s]43911 examples [00:40, 1066.70 examples/s]44019 examples [00:40, 1047.40 examples/s]44125 examples [00:40, 1049.54 examples/s]44231 examples [00:40, 1040.80 examples/s]44340 examples [00:40, 1053.25 examples/s]44446 examples [00:40, 1042.54 examples/s]44551 examples [00:41, 1017.65 examples/s]44653 examples [00:41, 1007.79 examples/s]44761 examples [00:41, 1028.25 examples/s]44867 examples [00:41, 1036.92 examples/s]44971 examples [00:41, 1033.37 examples/s]45085 examples [00:41, 1060.77 examples/s]45198 examples [00:41, 1079.37 examples/s]45313 examples [00:41, 1096.99 examples/s]45429 examples [00:41, 1113.02 examples/s]45544 examples [00:41, 1122.98 examples/s]45658 examples [00:42, 1126.95 examples/s]45771 examples [00:42, 1126.29 examples/s]45885 examples [00:42, 1128.38 examples/s]45998 examples [00:42, 1097.68 examples/s]46108 examples [00:42, 1095.39 examples/s]46218 examples [00:42, 1090.35 examples/s]46329 examples [00:42, 1093.61 examples/s]46439 examples [00:42, 1095.16 examples/s]46549 examples [00:42, 1088.69 examples/s]46658 examples [00:42, 1081.59 examples/s]46767 examples [00:43, 1082.70 examples/s]46882 examples [00:43, 1100.36 examples/s]46993 examples [00:43, 1065.97 examples/s]47107 examples [00:43, 1084.70 examples/s]47222 examples [00:43, 1100.95 examples/s]47333 examples [00:43, 1098.69 examples/s]47444 examples [00:43, 1097.28 examples/s]47557 examples [00:43, 1105.80 examples/s]47671 examples [00:43, 1113.34 examples/s]47784 examples [00:43, 1118.27 examples/s]47896 examples [00:44, 1117.14 examples/s]48010 examples [00:44, 1121.92 examples/s]48123 examples [00:44, 1111.00 examples/s]48235 examples [00:44, 1089.10 examples/s]48345 examples [00:44, 1075.98 examples/s]48454 examples [00:44, 1079.50 examples/s]48563 examples [00:44, 1071.09 examples/s]48677 examples [00:44, 1089.09 examples/s]48787 examples [00:44, 1064.10 examples/s]48898 examples [00:44, 1077.28 examples/s]49006 examples [00:45, 1073.49 examples/s]49119 examples [00:45, 1089.53 examples/s]49229 examples [00:45, 1073.21 examples/s]49339 examples [00:45, 1081.10 examples/s]49454 examples [00:45, 1099.72 examples/s]49566 examples [00:45, 1105.53 examples/s]49678 examples [00:45, 1109.48 examples/s]49791 examples [00:45, 1115.36 examples/s]49905 examples [00:45, 1121.66 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7105/50000 [00:00<00:00, 71048.71 examples/s] 41%|████      | 20503/50000 [00:00<00:00, 82702.24 examples/s] 67%|██████▋   | 33492/50000 [00:00<00:00, 92817.44 examples/s] 94%|█████████▍| 47123/50000 [00:00<00:00, 102641.65 examples/s]                                                                0 examples [00:00, ? examples/s]97 examples [00:00, 963.89 examples/s]209 examples [00:00, 1003.60 examples/s]322 examples [00:00, 1037.82 examples/s]436 examples [00:00, 1066.39 examples/s]543 examples [00:00, 1067.23 examples/s]658 examples [00:00, 1090.03 examples/s]757 examples [00:00, 858.27 examples/s] 873 examples [00:00, 928.96 examples/s]985 examples [00:00, 976.99 examples/s]1085 examples [00:01, 972.41 examples/s]1197 examples [00:01, 1010.40 examples/s]1310 examples [00:01, 1041.74 examples/s]1421 examples [00:01, 1060.52 examples/s]1532 examples [00:01, 1073.88 examples/s]1644 examples [00:01, 1086.43 examples/s]1757 examples [00:01, 1097.64 examples/s]1870 examples [00:01, 1104.76 examples/s]1982 examples [00:01, 1105.92 examples/s]2093 examples [00:01, 1092.03 examples/s]2203 examples [00:02, 1093.27 examples/s]2318 examples [00:02, 1107.44 examples/s]2431 examples [00:02, 1111.80 examples/s]2545 examples [00:02, 1118.53 examples/s]2657 examples [00:02, 1110.02 examples/s]2769 examples [00:02, 1063.74 examples/s]2880 examples [00:02, 1074.60 examples/s]2995 examples [00:02, 1096.05 examples/s]3105 examples [00:02, 1095.78 examples/s]3218 examples [00:03, 1105.06 examples/s]3329 examples [00:03, 1097.26 examples/s]3439 examples [00:03, 1080.76 examples/s]3549 examples [00:03, 1084.76 examples/s]3661 examples [00:03, 1094.01 examples/s]3773 examples [00:03, 1100.47 examples/s]3884 examples [00:03, 1093.42 examples/s]3995 examples [00:03, 1096.36 examples/s]4108 examples [00:03, 1105.46 examples/s]4223 examples [00:03, 1117.67 examples/s]4335 examples [00:04, 1114.79 examples/s]4447 examples [00:04, 1070.73 examples/s]4558 examples [00:04, 1080.22 examples/s]4667 examples [00:04, 1079.40 examples/s]4780 examples [00:04, 1092.20 examples/s]4890 examples [00:04, 1086.48 examples/s]5001 examples [00:04, 1092.21 examples/s]5112 examples [00:04, 1096.73 examples/s]5223 examples [00:04, 1100.66 examples/s]5334 examples [00:04, 1096.20 examples/s]5444 examples [00:05, 1086.91 examples/s]5553 examples [00:05, 1072.13 examples/s]5662 examples [00:05, 1075.25 examples/s]5773 examples [00:05, 1083.26 examples/s]5888 examples [00:05, 1101.71 examples/s]5999 examples [00:05, 1104.06 examples/s]6110 examples [00:05, 1104.23 examples/s]6222 examples [00:05, 1104.98 examples/s]6334 examples [00:05, 1108.69 examples/s]6445 examples [00:05, 1108.06 examples/s]6556 examples [00:06, 1091.74 examples/s]6673 examples [00:06, 1111.37 examples/s]6785 examples [00:06, 1106.37 examples/s]6896 examples [00:06, 1105.05 examples/s]7007 examples [00:06, 1104.75 examples/s]7118 examples [00:06, 1086.85 examples/s]7229 examples [00:06, 1093.00 examples/s]7340 examples [00:06, 1096.71 examples/s]7450 examples [00:06, 1088.32 examples/s]7560 examples [00:06, 1089.00 examples/s]7672 examples [00:07, 1095.73 examples/s]7782 examples [00:07, 1050.54 examples/s]7890 examples [00:07, 1056.39 examples/s]7996 examples [00:07, 1053.42 examples/s]8106 examples [00:07, 1066.51 examples/s]8213 examples [00:07, 1050.17 examples/s]8319 examples [00:07, 1040.83 examples/s]8430 examples [00:07, 1059.93 examples/s]8541 examples [00:07, 1073.54 examples/s]8658 examples [00:08, 1097.83 examples/s]8775 examples [00:08, 1117.51 examples/s]8888 examples [00:08, 1091.44 examples/s]9001 examples [00:08, 1101.90 examples/s]9115 examples [00:08, 1111.45 examples/s]9227 examples [00:08, 1096.09 examples/s]9342 examples [00:08, 1110.56 examples/s]9457 examples [00:08, 1121.17 examples/s]9570 examples [00:08, 1113.59 examples/s]9684 examples [00:08, 1120.43 examples/s]9798 examples [00:09, 1125.97 examples/s]9912 examples [00:09, 1128.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW6ZJCE/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW6ZJCE/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f744f5c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f744f5c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f744f5c1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f73d5b29cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f749afae3c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f744f5c1950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f744f5c1950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f744f5c1950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f73d4a055f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7435980128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f73c6fb32f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f73c6fb32f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f73c6fb32f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f73c6fb3400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f73c6fb3400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f73c6fb3400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c76e3000a61114f1b5c7c3e54e584f2bd4e19d603ea7824b0ae40c754beeb25c
  Stored in directory: /tmp/pip-ephem-wheel-cache-k_irbaem/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<131:10:55, 1.83kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<92:03:28, 2.60kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<64:29:02, 3.71kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:04<45:07:02, 5.30kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:05<31:29:10, 7.57kB/s].vector_cache/glove.6B.zip:   1%|          | 8.25M/862M [00:05<21:55:21, 10.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:05<15:16:43, 15.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.5M/862M [00:05<10:38:26, 22.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:05<7:24:51, 31.5kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.1M/862M [00:05<5:11:10, 45.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.8M/862M [00:05<3:36:25, 64.3kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.5M/862M [00:05<2:30:56, 91.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.4M/862M [00:05<1:45:07, 131kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:05<1:13:19, 187kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:06<51:02, 266kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 49.1M/862M [00:06<35:45, 379kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:06<25:30, 529kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:08<19:41, 682kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:08<15:41, 856kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:08<11:21, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:08<08:04, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:10<13:22:15, 16.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:10<9:23:00, 23.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:10<6:33:48, 33.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:12<4:37:32, 47.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:12<3:17:06, 67.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:12<2:18:27, 95.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.3M/862M [00:12<1:36:54, 137kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:14<1:12:30, 182kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:14<52:05, 254kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:14<36:43, 359kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:16<28:45, 458kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:16<22:49, 576kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:16<16:38, 790kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:18<13:43, 953kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:18<10:58, 1.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:18<08:00, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:20<08:37, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:20<07:20, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:20<05:27, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:22<06:52, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<06:08, 2.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:22<04:34, 2.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:24<06:15, 2.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:24<07:01, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:24<05:34, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:26<05:58, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<05:30, 2.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:26<04:08, 3.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:28<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:28<06:36, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:28<05:16, 2.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:44, 2.21MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:21, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<04:03, 3.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<05:47, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<06:45, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<05:16, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<03:52, 3.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<08:02, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<06:56, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<05:10, 2.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<06:30, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<05:50, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<04:23, 2.83MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<05:58, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<03:57, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<02:55, 4.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<51:43, 238kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<37:27, 329kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<26:28, 465kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<21:22, 574kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<16:13, 756kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<11:38, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<11:01, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<08:57, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<06:34, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<07:27, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<07:36, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:50, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<04:15, 2.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<07:58, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<06:50, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:48<05:05, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<06:20, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<06:54, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<05:21, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<03:54, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<08:33, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<07:12, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<05:18, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<06:29, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<05:44, 2.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<04:16, 2.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<05:47, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<05:15, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<03:55, 2.99MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<05:32, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<06:17, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<04:54, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<03:35, 3.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<07:36, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<06:31, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<04:51, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<06:07, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<06:39, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:01<05:15, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<03:49, 3.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<1:24:42, 136kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<1:00:26, 190kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<42:30, 270kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<32:20, 354kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<25:04, 456kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:05<18:03, 633kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<12:44, 894kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<16:17, 698kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<12:34, 903kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<09:05, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<08:57, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<07:27, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<05:29, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<06:28, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:42, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<04:14, 2.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:34, 2.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<05:02, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:13<03:48, 2.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<05:17, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<04:49, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<03:38, 3.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<05:09, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<05:56, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<04:35, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<03:32, 3.10MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<04:58, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<03:30, 3.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:21<05:00, 2.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<05:44, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<03:16, 3.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<09:19, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<07:27, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<05:25, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<03:56, 2.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<46:01, 234kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<34:24, 312kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<24:36, 437kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<18:52, 566kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<14:19, 746kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<10:17, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<09:38, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<09:00, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<06:51, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<41:31, 254kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<30:09, 350kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<21:19, 493kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<17:20, 605kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<13:13, 792kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<09:30, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<09:04, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<07:25, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<05:26, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<06:15, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:24, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:02, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:38<05:15, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:38<05:47, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<04:29, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<06:19, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<05:31, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<04:05, 2.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<05:08, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<05:46, 1.75MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<04:30, 2.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<03:17, 3.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<06:58, 1.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<05:57, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<04:25, 2.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<05:20, 1.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<04:48, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<03:34, 2.78MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<04:45, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<05:27, 1.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<04:20, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<03:09, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<33:23, 295kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<24:24, 404kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<17:15, 570kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<14:14, 687kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:52<11:00, 888kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<07:57, 1.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<07:44, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<07:29, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<05:41, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<06:19, 1.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<05:27, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<04:03, 2.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<05:04, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<05:35, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<04:21, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<03:10, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:00<06:40, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<05:41, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<04:11, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:04, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:34, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<04:23, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<03:11, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<30:21, 309kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<22:14, 421kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<15:46, 593kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<13:05, 711kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<11:09, 834kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<08:14, 1.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:06<05:53, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<07:49, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<06:28, 1.43MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:08<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<05:22, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<05:45, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<04:26, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<03:13, 2.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<06:54, 1.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<05:48, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<04:18, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<05:02, 1.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<05:27, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<04:13, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<03:05, 2.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<05:42, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<04:58, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<03:40, 2.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<04:33, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<05:05, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<03:57, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<02:54, 3.05MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<05:19, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<04:40, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<03:29, 2.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<04:25, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<04:57, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<03:51, 2.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<02:49, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<05:36, 1.55MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<03:37, 2.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:28, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:58, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:26<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<02:48, 3.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<10:38, 803kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<08:22, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<06:08, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<06:06, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<04:38, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<03:22, 2.50MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<05:27, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:32<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<03:31, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<04:20, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<04:49, 1.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<03:44, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<02:44, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<04:58, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<04:21, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<03:14, 2.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<04:06, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<04:38, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<03:37, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<02:37, 3.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<06:00, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<05:03, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<03:43, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<04:24, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<04:49, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<02:44, 2.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<04:33, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<04:02, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<03:02, 2.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<03:54, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<04:25, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<03:31, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<02:32, 3.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<20:41, 380kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:48<15:18, 513kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:48<10:51, 721kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<09:20, 835kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<08:11, 950kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<06:04, 1.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<04:21, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<06:26, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<05:19, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:52<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<04:27, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:45, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<03:39, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<02:41, 2.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:55<04:29, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<03:57, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<02:55, 2.58MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<03:45, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:26, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<02:35, 2.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<03:29, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<04:02, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<02:20, 3.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<23:46, 310kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<18:27, 400kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<13:22, 551kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<09:59, 737kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<07:40, 959kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<06:01, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<04:51, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<04:02, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:26, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<03:01, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<14:06, 518kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<11:31, 634kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<08:35, 850kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<06:28, 1.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<05:19, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<03:33, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:04<03:02, 2.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<02:39, 2.72MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<02:24, 3.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<26:33, 273kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<20:00, 362kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<14:41, 492kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<10:47, 669kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<08:03, 894kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<06:09, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<04:45, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<03:49, 1.87MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<03:10, 2.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<22:01, 326kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<18:08, 395kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<13:18, 538kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<09:48, 729kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<07:26, 960kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<05:36, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<04:25, 1.61MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<03:29, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:08<03:05, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<06:47, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<07:26, 954kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<05:58, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:10<04:37, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<03:47, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<03:03, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<02:37, 2.68MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<02:19, 3.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<02:06, 3.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<11:42, 601kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<10:51, 648kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<08:15, 850kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<06:22, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:12<04:56, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<03:51, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<03:11, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<02:47, 2.50MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<02:26, 2.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<11:32, 603kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<10:42, 650kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<08:12, 848kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<06:13, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<04:49, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<03:51, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:09, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<02:37, 2.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<02:21, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<20:24, 338kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<16:54, 408kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<12:32, 550kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<09:13, 746kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<06:50, 1.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<05:18, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<04:05, 1.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<03:18, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<05:59, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:17<06:37, 1.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:17<05:18, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<02:43, 2.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<02:19, 2.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<02:06, 3.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<06:41, 1.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<06:55, 975kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<05:27, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<04:12, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<03:20, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<02:42, 2.48MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<02:15, 2.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<01:56, 3.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<2:07:14, 52.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<1:31:06, 73.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:21<1:04:16, 104kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<45:15, 147kB/s]  .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:22<31:57, 208kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<22:36, 294kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<16:07, 411kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<17:47, 372kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<16:05, 411kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<12:07, 546kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<08:49, 748kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<06:24, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<04:46, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<03:37, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<11:08, 589kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<10:43, 611kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<08:14, 795kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<06:03, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<04:29, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<03:22, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<05:49, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<06:43, 963kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<05:22, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<04:00, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<03:01, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<02:17, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<10:59, 583kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<09:56, 645kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<07:30, 853kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<05:25, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<05:21, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<07:39, 829kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<06:21, 997kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<04:44, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<03:30, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<02:37, 2.40MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<13:18, 472kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<11:08, 564kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<08:12, 763kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<05:53, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<04:14, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<11:03, 562kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<10:48, 574kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<08:19, 745kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<06:02, 1.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<04:20, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<06:23, 961kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<07:14, 848kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<05:45, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<04:11, 1.46MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<06:36, 918kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<07:07, 852kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<05:29, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<04:00, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<02:52, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<13:45, 436kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<11:44, 511kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<08:39, 692kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<06:09, 969kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<04:27, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<05:47, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<06:00, 988kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<04:42, 1.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<03:23, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<03:48, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<04:28, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:34, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<02:35, 2.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:26, 1.69MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<04:05, 1.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<03:17, 1.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<02:24, 2.38MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<03:21, 1.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<03:50, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<03:03, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:49<02:14, 2.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<03:38, 1.55MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<04:06, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<03:13, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<02:22, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<02:59, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:53<03:21, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<02:39, 2.10MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:53<01:55, 2.87MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<04:19, 1.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<04:12, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<03:11, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<02:17, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:57<04:29, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<04:13, 1.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:57<02:18, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:59<07:14, 743kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<06:08, 876kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<04:30, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<06:54, 768kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<06:05, 872kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<04:34, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<03:15, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<05:30, 952kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<04:59, 1.05MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<03:43, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<02:39, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<04:24, 1.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<03:55, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<02:57, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<02:56, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:07<03:12, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<02:28, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<01:48, 2.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<03:01, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:09<02:51, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<02:09, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:24, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:11<02:40, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:11<02:06, 2.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:35, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:02, 2.38MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:12, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:29, 1.93MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:15<01:56, 2.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<01:24, 3.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:43, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:34, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:17<02:44, 1.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:19<02:12, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:16, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:30, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<01:56, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<01:24, 3.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<04:23, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<04:00, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<03:01, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<02:49, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<02:51, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:10, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<01:34, 2.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:10, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<03:07, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<02:22, 1.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:42, 2.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<04:36, 944kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<04:04, 1.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<03:01, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<02:09, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<04:08, 1.03MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<03:47, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<02:51, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<02:39, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<02:41, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<02:05, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:07, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:17, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<01:48, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<01:55, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<02:10, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<01:43, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<01:50, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<01:57, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<01:34, 2.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<01:08, 3.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<02:53, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<02:48, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<02:09, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<01:33, 2.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<07:07, 542kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<06:06, 633kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:29, 858kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<03:11, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<03:20, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<03:22, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<02:34, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:51, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:19, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:38, 1.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:05, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<01:31, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:39, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:48, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:10, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<01:33, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:11, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:25, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:52, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<01:22, 2.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<01:56, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<02:13, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<01:46, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<01:17, 2.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:38, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:44, 1.25MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<02:09, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<01:32, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<02:33, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<02:39, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<02:04, 1.62MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<01:28, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<02:35, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:33, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<01:56, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<01:25, 2.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<01:50, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<02:00, 1.61MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<01:08, 2.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:36, 1.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:32, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<01:23, 2.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:53, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<02:01, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<01:35, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:04<01:08, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<02:37, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<02:31, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:55, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<01:22, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<02:45, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<02:35, 1.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:56, 1.52MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<02:08, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<02:14, 1.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:44, 1.65MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:10<01:15, 2.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<02:12, 1.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<02:19, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:48, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<01:17, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<02:06, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<02:11, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:42, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:14<01:12, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<02:06, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<02:08, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:39, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<01:11, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<02:07, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<02:02, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:34, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:06, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<02:31, 1.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:20<02:19, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:45, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<01:14, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:22<02:40, 928kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<02:24, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<01:48, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:22<01:16, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<02:40, 902kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:27, 978kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<01:52, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:19, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:25<02:02, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<02:00, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<01:32, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:26<01:05, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:56, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:28<01:50, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:23, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<01:00, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:19, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:30<01:29, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:11, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:30<00:50, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:36, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<01:39, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<00:54, 2.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:34<01:25, 1.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<01:06, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<00:47, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:34, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<01:33, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<01:12, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:36<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:37, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:35, 1.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<01:11, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<00:51, 2.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:16, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:20, 1.38MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:40<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:49, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<00:38, 2.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:40<00:31, 3.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:45, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<02:01, 881kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:42<01:35, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<00:52, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<00:39, 2.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<00:33, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<04:19, 397kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<03:43, 460kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<02:47, 614kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:59, 851kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:26, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<01:03, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<01:55, 857kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<02:01, 815kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<01:32, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:46<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:50, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<00:38, 2.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:52, 845kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:54, 829kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<01:27, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<01:04, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<00:36, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:49<01:58, 762kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:49<01:57, 771kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<01:30, 993kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<01:06, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<00:48, 1.81MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:37, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<02:06, 685kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<02:01, 714kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<01:31, 942kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<01:06, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:48, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:37, 2.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<02:21, 582kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:53<02:10, 632kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:53<01:38, 833kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<01:11, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:51, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<00:38, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<02:01, 645kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:54, 683kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<01:25, 906kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<01:01, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:46, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:27, 2.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:47, 691kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:57<01:42, 719kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:57<01:18, 936kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:57, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:41, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:31, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<01:51, 626kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<01:44, 668kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<01:19, 875kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:56, 1.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:41, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:30, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<01:50, 598kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<01:41, 645kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<01:17, 847kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:55, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:40, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:51, 553kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:41, 608kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:15, 811kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<00:54, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:39, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<01:37, 589kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<01:30, 638kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<01:07, 848kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:48, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:35, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:25, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<02:10, 409kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<01:49, 485kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<01:21, 651kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:57, 903kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:41, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<03:00, 274kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<02:22, 344kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<01:42, 474kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:09<01:12, 657kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:09<00:50, 916kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:50, 896kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:49, 902kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:37, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<00:27, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:19, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:15, 2.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:45, 909kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<01:00, 676kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:48, 834kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:13<00:25, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:18, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:45, 802kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:43, 847kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:32, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:23, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:16, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:26, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:38, 846kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:31, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:22, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:17<00:16, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:19, 1.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:29, 981kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:24, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:17, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:11, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:17, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:24, 1.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:19, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:14, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:09, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:16, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:21, 937kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:17, 1.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<00:11, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:15, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:18, 878kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:14, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:25<00:09, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:25<00:06, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:11, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:13, 912kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:09, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<00:06, 1.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:04, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:05, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:06, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:05, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:02, 1.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:01, 2.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<03:15, 18.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<02:13, 26.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<01:22, 38.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:29, 54.3kB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<24:00:22,  4.63it/s]  0%|          | 846/400000 [00:00<16:46:21,  6.61it/s]  0%|          | 1646/400000 [00:00<11:43:17,  9.44it/s]  1%|          | 2553/400000 [00:00<8:11:24, 13.48it/s]   1%|          | 3405/400000 [00:00<5:43:28, 19.24it/s]  1%|          | 4260/400000 [00:00<4:00:08, 27.47it/s]  1%|▏         | 5132/400000 [00:00<2:47:57, 39.18it/s]  2%|▏         | 6033/400000 [00:00<1:57:31, 55.87it/s]  2%|▏         | 6903/400000 [00:01<1:22:18, 79.60it/s]  2%|▏         | 7767/400000 [00:01<57:43, 113.26it/s]   2%|▏         | 8662/400000 [00:01<40:31, 160.93it/s]  2%|▏         | 9603/400000 [00:01<28:30, 228.23it/s]  3%|▎         | 10522/400000 [00:01<20:07, 322.60it/s]  3%|▎         | 11418/400000 [00:01<14:16, 453.85it/s]  3%|▎         | 12311/400000 [00:01<10:11, 633.73it/s]  3%|▎         | 13244/400000 [00:01<07:19, 879.71it/s]  4%|▎         | 14183/400000 [00:01<05:19, 1208.17it/s]  4%|▍         | 15109/400000 [00:01<03:55, 1634.48it/s]  4%|▍         | 16043/400000 [00:02<02:56, 2171.94it/s]  4%|▍         | 16962/400000 [00:02<02:17, 2780.14it/s]  4%|▍         | 17891/400000 [00:02<01:48, 3520.06it/s]  5%|▍         | 18789/400000 [00:02<01:28, 4294.97it/s]  5%|▍         | 19694/400000 [00:02<01:14, 5097.88it/s]  5%|▌         | 20591/400000 [00:02<01:05, 5835.59it/s]  5%|▌         | 21483/400000 [00:02<00:58, 6470.27it/s]  6%|▌         | 22378/400000 [00:02<00:53, 7055.93it/s]  6%|▌         | 23318/400000 [00:02<00:49, 7624.57it/s]  6%|▌         | 24260/400000 [00:02<00:46, 8085.40it/s]  6%|▋         | 25175/400000 [00:03<00:44, 8343.52it/s]  7%|▋         | 26086/400000 [00:03<00:44, 8472.64it/s]  7%|▋         | 27010/400000 [00:03<00:42, 8687.98it/s]  7%|▋         | 27918/400000 [00:03<00:42, 8784.67it/s]  7%|▋         | 28825/400000 [00:03<00:42, 8823.94it/s]  7%|▋         | 29727/400000 [00:03<00:43, 8481.68it/s]  8%|▊         | 30592/400000 [00:03<00:44, 8301.51it/s]  8%|▊         | 31511/400000 [00:03<00:43, 8547.61it/s]  8%|▊         | 32377/400000 [00:03<00:43, 8537.81it/s]  8%|▊         | 33322/400000 [00:03<00:41, 8790.89it/s]  9%|▊         | 34209/400000 [00:04<00:42, 8534.96it/s]  9%|▉         | 35108/400000 [00:04<00:42, 8665.76it/s]  9%|▉         | 35996/400000 [00:04<00:41, 8728.22it/s]  9%|▉         | 36892/400000 [00:04<00:41, 8795.70it/s]  9%|▉         | 37790/400000 [00:04<00:40, 8849.12it/s] 10%|▉         | 38677/400000 [00:04<00:41, 8785.37it/s] 10%|▉         | 39558/400000 [00:04<00:41, 8782.38it/s] 10%|█         | 40484/400000 [00:04<00:40, 8915.97it/s] 10%|█         | 41379/400000 [00:04<00:40, 8925.07it/s] 11%|█         | 42298/400000 [00:04<00:39, 9000.96it/s] 11%|█         | 43243/400000 [00:05<00:39, 9129.00it/s] 11%|█         | 44157/400000 [00:05<00:39, 9119.51it/s] 11%|█▏        | 45106/400000 [00:05<00:38, 9227.17it/s] 12%|█▏        | 46030/400000 [00:05<00:38, 9119.96it/s] 12%|█▏        | 46943/400000 [00:05<00:38, 9057.65it/s] 12%|█▏        | 47865/400000 [00:05<00:38, 9103.11it/s] 12%|█▏        | 48776/400000 [00:05<00:39, 8859.72it/s] 12%|█▏        | 49672/400000 [00:05<00:39, 8884.20it/s] 13%|█▎        | 50597/400000 [00:05<00:38, 8988.37it/s] 13%|█▎        | 51498/400000 [00:06<00:39, 8933.05it/s] 13%|█▎        | 52393/400000 [00:06<00:39, 8866.55it/s] 13%|█▎        | 53281/400000 [00:06<00:39, 8826.85it/s] 14%|█▎        | 54170/400000 [00:06<00:39, 8845.40it/s] 14%|█▍        | 55086/400000 [00:06<00:38, 8935.60it/s] 14%|█▍        | 55991/400000 [00:06<00:38, 8968.66it/s] 14%|█▍        | 56906/400000 [00:06<00:38, 9019.85it/s] 14%|█▍        | 57809/400000 [00:06<00:38, 8815.83it/s] 15%|█▍        | 58692/400000 [00:06<00:39, 8739.16it/s] 15%|█▍        | 59567/400000 [00:06<00:39, 8693.37it/s] 15%|█▌        | 60438/400000 [00:07<00:39, 8621.09it/s] 15%|█▌        | 61301/400000 [00:07<00:39, 8586.22it/s] 16%|█▌        | 62161/400000 [00:07<00:39, 8548.44it/s] 16%|█▌        | 63041/400000 [00:07<00:39, 8621.05it/s] 16%|█▌        | 63917/400000 [00:07<00:38, 8662.14it/s] 16%|█▌        | 64784/400000 [00:07<00:38, 8621.25it/s] 16%|█▋        | 65647/400000 [00:07<00:38, 8610.41it/s] 17%|█▋        | 66541/400000 [00:07<00:38, 8705.62it/s] 17%|█▋        | 67415/400000 [00:07<00:38, 8713.51it/s] 17%|█▋        | 68287/400000 [00:07<00:38, 8713.69it/s] 17%|█▋        | 69159/400000 [00:08<00:38, 8625.91it/s] 18%|█▊        | 70022/400000 [00:08<00:38, 8595.94it/s] 18%|█▊        | 70884/400000 [00:08<00:38, 8601.10it/s] 18%|█▊        | 71745/400000 [00:08<00:38, 8590.79it/s] 18%|█▊        | 72605/400000 [00:08<00:38, 8573.34it/s] 18%|█▊        | 73463/400000 [00:08<00:38, 8497.33it/s] 19%|█▊        | 74313/400000 [00:08<00:38, 8443.89it/s] 19%|█▉        | 75158/400000 [00:08<00:38, 8372.19it/s] 19%|█▉        | 76009/400000 [00:08<00:38, 8411.25it/s] 19%|█▉        | 76851/400000 [00:08<00:39, 8187.01it/s] 19%|█▉        | 77681/400000 [00:09<00:39, 8218.20it/s] 20%|█▉        | 78537/400000 [00:09<00:38, 8317.16it/s] 20%|█▉        | 79370/400000 [00:09<00:38, 8292.16it/s] 20%|██        | 80233/400000 [00:09<00:38, 8390.60it/s] 20%|██        | 81112/400000 [00:09<00:37, 8505.13it/s] 20%|██        | 81964/400000 [00:09<00:37, 8464.91it/s] 21%|██        | 82847/400000 [00:09<00:37, 8570.44it/s] 21%|██        | 83705/400000 [00:09<00:36, 8570.03it/s] 21%|██        | 84580/400000 [00:09<00:36, 8622.27it/s] 21%|██▏       | 85471/400000 [00:09<00:36, 8705.04it/s] 22%|██▏       | 86343/400000 [00:10<00:36, 8578.69it/s] 22%|██▏       | 87207/400000 [00:10<00:36, 8596.00it/s] 22%|██▏       | 88074/400000 [00:10<00:36, 8615.93it/s] 22%|██▏       | 88936/400000 [00:10<00:36, 8526.70it/s] 22%|██▏       | 89790/400000 [00:10<00:36, 8453.82it/s] 23%|██▎       | 90636/400000 [00:10<00:36, 8431.50it/s] 23%|██▎       | 91512/400000 [00:10<00:36, 8527.04it/s] 23%|██▎       | 92366/400000 [00:10<00:36, 8506.57it/s] 23%|██▎       | 93256/400000 [00:10<00:35, 8619.36it/s] 24%|██▎       | 94144/400000 [00:10<00:35, 8695.23it/s] 24%|██▍       | 95015/400000 [00:11<00:35, 8607.50it/s] 24%|██▍       | 95877/400000 [00:11<00:35, 8579.91it/s] 24%|██▍       | 96741/400000 [00:11<00:35, 8596.02it/s] 24%|██▍       | 97601/400000 [00:11<00:35, 8582.70it/s] 25%|██▍       | 98469/400000 [00:11<00:35, 8611.41it/s] 25%|██▍       | 99331/400000 [00:11<00:35, 8558.18it/s] 25%|██▌       | 100194/400000 [00:11<00:34, 8579.20it/s] 25%|██▌       | 101053/400000 [00:11<00:34, 8579.09it/s] 25%|██▌       | 101916/400000 [00:11<00:34, 8594.29it/s] 26%|██▌       | 102795/400000 [00:11<00:34, 8650.48it/s] 26%|██▌       | 103664/400000 [00:12<00:34, 8660.84it/s] 26%|██▌       | 104575/400000 [00:12<00:33, 8789.98it/s] 26%|██▋       | 105481/400000 [00:12<00:33, 8866.90it/s] 27%|██▋       | 106369/400000 [00:12<00:33, 8725.39it/s] 27%|██▋       | 107289/400000 [00:12<00:33, 8861.65it/s] 27%|██▋       | 108194/400000 [00:12<00:32, 8914.92it/s] 27%|██▋       | 109087/400000 [00:12<00:32, 8886.71it/s] 27%|██▋       | 109977/400000 [00:12<00:32, 8869.92it/s] 28%|██▊       | 110892/400000 [00:12<00:32, 8951.31it/s] 28%|██▊       | 111793/400000 [00:12<00:32, 8967.11it/s] 28%|██▊       | 112745/400000 [00:13<00:31, 9126.09it/s] 28%|██▊       | 113678/400000 [00:13<00:31, 9185.34it/s] 29%|██▊       | 114607/400000 [00:13<00:30, 9214.84it/s] 29%|██▉       | 115530/400000 [00:13<00:32, 8869.16it/s] 29%|██▉       | 116431/400000 [00:13<00:31, 8908.12it/s] 29%|██▉       | 117393/400000 [00:13<00:31, 9107.78it/s] 30%|██▉       | 118349/400000 [00:13<00:30, 9238.60it/s] 30%|██▉       | 119276/400000 [00:13<00:30, 9134.00it/s] 30%|███       | 120192/400000 [00:13<00:30, 9085.65it/s] 30%|███       | 121102/400000 [00:14<00:31, 8754.29it/s] 30%|███       | 121981/400000 [00:14<00:31, 8735.37it/s] 31%|███       | 122885/400000 [00:14<00:31, 8824.54it/s] 31%|███       | 123770/400000 [00:14<00:31, 8710.98it/s] 31%|███       | 124649/400000 [00:14<00:31, 8733.63it/s] 31%|███▏      | 125524/400000 [00:14<00:31, 8665.58it/s] 32%|███▏      | 126431/400000 [00:14<00:31, 8781.12it/s] 32%|███▏      | 127326/400000 [00:14<00:30, 8829.27it/s] 32%|███▏      | 128223/400000 [00:14<00:30, 8868.85it/s] 32%|███▏      | 129111/400000 [00:14<00:30, 8841.60it/s] 33%|███▎      | 130052/400000 [00:15<00:29, 9003.27it/s] 33%|███▎      | 130958/400000 [00:15<00:29, 9017.27it/s] 33%|███▎      | 131907/400000 [00:15<00:29, 9152.34it/s] 33%|███▎      | 132824/400000 [00:15<00:29, 9044.56it/s] 33%|███▎      | 133730/400000 [00:15<00:29, 8952.26it/s] 34%|███▎      | 134635/400000 [00:15<00:29, 8979.31it/s] 34%|███▍      | 135534/400000 [00:15<00:29, 8922.79it/s] 34%|███▍      | 136481/400000 [00:15<00:29, 9079.24it/s] 34%|███▍      | 137390/400000 [00:15<00:29, 9044.64it/s] 35%|███▍      | 138296/400000 [00:15<00:28, 9040.30it/s] 35%|███▍      | 139201/400000 [00:16<00:29, 8988.84it/s] 35%|███▌      | 140101/400000 [00:16<00:28, 8966.71it/s] 35%|███▌      | 141048/400000 [00:16<00:28, 9110.13it/s] 35%|███▌      | 141960/400000 [00:16<00:28, 8985.36it/s] 36%|███▌      | 142860/400000 [00:16<00:28, 8936.02it/s] 36%|███▌      | 143755/400000 [00:16<00:28, 8892.70it/s] 36%|███▌      | 144656/400000 [00:16<00:28, 8926.21it/s] 36%|███▋      | 145603/400000 [00:16<00:28, 9082.39it/s] 37%|███▋      | 146513/400000 [00:16<00:28, 9029.59it/s] 37%|███▋      | 147417/400000 [00:16<00:28, 9017.97it/s] 37%|███▋      | 148320/400000 [00:17<00:27, 9002.54it/s] 37%|███▋      | 149268/400000 [00:17<00:27, 9139.32it/s] 38%|███▊      | 150183/400000 [00:17<00:27, 9033.32it/s] 38%|███▊      | 151088/400000 [00:17<00:27, 8934.34it/s] 38%|███▊      | 151983/400000 [00:17<00:27, 8881.37it/s] 38%|███▊      | 152900/400000 [00:17<00:27, 8965.20it/s] 38%|███▊      | 153832/400000 [00:17<00:27, 9065.62it/s] 39%|███▊      | 154772/400000 [00:17<00:26, 9160.57it/s] 39%|███▉      | 155689/400000 [00:17<00:26, 9072.45it/s] 39%|███▉      | 156597/400000 [00:17<00:26, 9069.90it/s] 39%|███▉      | 157541/400000 [00:18<00:26, 9176.39it/s] 40%|███▉      | 158486/400000 [00:18<00:26, 9254.93it/s] 40%|███▉      | 159414/400000 [00:18<00:25, 9261.70it/s] 40%|████      | 160341/400000 [00:18<00:25, 9239.31it/s] 40%|████      | 161266/400000 [00:18<00:26, 9138.54it/s] 41%|████      | 162207/400000 [00:18<00:25, 9216.75it/s] 41%|████      | 163132/400000 [00:18<00:25, 9225.78it/s] 41%|████      | 164055/400000 [00:18<00:25, 9199.78it/s] 41%|████      | 164976/400000 [00:18<00:26, 8969.45it/s] 41%|████▏     | 165875/400000 [00:19<00:27, 8612.55it/s] 42%|████▏     | 166772/400000 [00:19<00:26, 8715.13it/s] 42%|████▏     | 167684/400000 [00:19<00:26, 8830.78it/s] 42%|████▏     | 168575/400000 [00:19<00:26, 8854.05it/s] 42%|████▏     | 169464/400000 [00:19<00:26, 8864.29it/s] 43%|████▎     | 170352/400000 [00:19<00:25, 8861.07it/s] 43%|████▎     | 171298/400000 [00:19<00:25, 9030.75it/s] 43%|████▎     | 172220/400000 [00:19<00:25, 9085.33it/s] 43%|████▎     | 173178/400000 [00:19<00:24, 9226.25it/s] 44%|████▎     | 174102/400000 [00:19<00:24, 9195.16it/s] 44%|████▍     | 175023/400000 [00:20<00:24, 9141.20it/s] 44%|████▍     | 175958/400000 [00:20<00:24, 9200.81it/s] 44%|████▍     | 176901/400000 [00:20<00:24, 9266.97it/s] 44%|████▍     | 177837/400000 [00:20<00:23, 9293.31it/s] 45%|████▍     | 178767/400000 [00:20<00:24, 9148.67it/s] 45%|████▍     | 179683/400000 [00:20<00:24, 9068.97it/s] 45%|████▌     | 180596/400000 [00:20<00:24, 9084.93it/s] 45%|████▌     | 181530/400000 [00:20<00:23, 9159.56it/s] 46%|████▌     | 182451/400000 [00:20<00:23, 9172.50it/s] 46%|████▌     | 183369/400000 [00:20<00:23, 9069.18it/s] 46%|████▌     | 184277/400000 [00:21<00:23, 8993.42it/s] 46%|████▋     | 185195/400000 [00:21<00:23, 9047.44it/s] 47%|████▋     | 186101/400000 [00:21<00:23, 8997.61it/s] 47%|████▋     | 187029/400000 [00:21<00:23, 9079.53it/s] 47%|████▋     | 187944/400000 [00:21<00:23, 9098.99it/s] 47%|████▋     | 188855/400000 [00:21<00:23, 9053.11it/s] 47%|████▋     | 189773/400000 [00:21<00:23, 9089.12it/s] 48%|████▊     | 190688/400000 [00:21<00:22, 9105.33it/s] 48%|████▊     | 191599/400000 [00:21<00:22, 9095.94it/s] 48%|████▊     | 192509/400000 [00:21<00:22, 9031.52it/s] 48%|████▊     | 193413/400000 [00:22<00:22, 8986.12it/s] 49%|████▊     | 194346/400000 [00:22<00:22, 9085.49it/s] 49%|████▉     | 195255/400000 [00:22<00:23, 8900.38it/s] 49%|████▉     | 196147/400000 [00:22<00:23, 8836.62it/s] 49%|████▉     | 197060/400000 [00:22<00:22, 8921.04it/s] 49%|████▉     | 197953/400000 [00:22<00:23, 8693.13it/s] 50%|████▉     | 198881/400000 [00:22<00:22, 8858.33it/s] 50%|████▉     | 199769/400000 [00:22<00:22, 8822.98it/s] 50%|█████     | 200685/400000 [00:22<00:22, 8920.07it/s] 50%|█████     | 201585/400000 [00:22<00:22, 8942.65it/s] 51%|█████     | 202481/400000 [00:23<00:22, 8864.77it/s] 51%|█████     | 203453/400000 [00:23<00:21, 9103.64it/s] 51%|█████     | 204402/400000 [00:23<00:21, 9214.21it/s] 51%|█████▏    | 205326/400000 [00:23<00:21, 9220.72it/s] 52%|█████▏    | 206250/400000 [00:23<00:21, 9195.27it/s] 52%|█████▏    | 207171/400000 [00:23<00:21, 9015.67it/s] 52%|█████▏    | 208074/400000 [00:23<00:21, 8808.38it/s] 52%|█████▏    | 208957/400000 [00:23<00:21, 8768.43it/s] 52%|█████▏    | 209836/400000 [00:23<00:21, 8743.15it/s] 53%|█████▎    | 210712/400000 [00:23<00:22, 8573.96it/s] 53%|█████▎    | 211625/400000 [00:24<00:21, 8730.79it/s] 53%|█████▎    | 212539/400000 [00:24<00:21, 8846.94it/s] 53%|█████▎    | 213426/400000 [00:24<00:21, 8826.62it/s] 54%|█████▎    | 214310/400000 [00:24<00:21, 8784.89it/s] 54%|█████▍    | 215190/400000 [00:24<00:21, 8703.94it/s] 54%|█████▍    | 216062/400000 [00:24<00:21, 8644.03it/s] 54%|█████▍    | 216965/400000 [00:24<00:20, 8754.54it/s] 54%|█████▍    | 217881/400000 [00:24<00:20, 8872.35it/s] 55%|█████▍    | 218777/400000 [00:24<00:20, 8896.73it/s] 55%|█████▍    | 219668/400000 [00:24<00:20, 8844.50it/s] 55%|█████▌    | 220598/400000 [00:25<00:19, 8973.70it/s] 55%|█████▌    | 221541/400000 [00:25<00:19, 9104.82it/s] 56%|█████▌    | 222454/400000 [00:25<00:19, 9110.40it/s] 56%|█████▌    | 223366/400000 [00:25<00:19, 9015.09it/s] 56%|█████▌    | 224269/400000 [00:25<00:20, 8687.20it/s] 56%|█████▋    | 225163/400000 [00:25<00:19, 8760.60it/s] 57%|█████▋    | 226113/400000 [00:25<00:19, 8969.13it/s] 57%|█████▋    | 227047/400000 [00:25<00:19, 9076.66it/s] 57%|█████▋    | 227957/400000 [00:25<00:18, 9072.92it/s] 57%|█████▋    | 228870/400000 [00:26<00:18, 9086.82it/s] 57%|█████▋    | 229799/400000 [00:26<00:18, 9146.25it/s] 58%|█████▊    | 230750/400000 [00:26<00:18, 9250.10it/s] 58%|█████▊    | 231701/400000 [00:26<00:18, 9324.10it/s] 58%|█████▊    | 232635/400000 [00:26<00:18, 9189.80it/s] 58%|█████▊    | 233555/400000 [00:26<00:18, 9102.45it/s] 59%|█████▊    | 234484/400000 [00:26<00:18, 9157.25it/s] 59%|█████▉    | 235401/400000 [00:26<00:18, 9096.89it/s] 59%|█████▉    | 236312/400000 [00:26<00:18, 9038.43it/s] 59%|█████▉    | 237243/400000 [00:26<00:17, 9116.54it/s] 60%|█████▉    | 238156/400000 [00:27<00:18, 8853.27it/s] 60%|█████▉    | 239060/400000 [00:27<00:18, 8908.09it/s] 60%|██████    | 240002/400000 [00:27<00:17, 9053.77it/s] 60%|██████    | 240909/400000 [00:27<00:17, 8968.17it/s] 60%|██████    | 241819/400000 [00:27<00:17, 9005.70it/s] 61%|██████    | 242721/400000 [00:27<00:17, 8916.59it/s] 61%|██████    | 243622/400000 [00:27<00:17, 8942.01it/s] 61%|██████    | 244559/400000 [00:27<00:17, 9064.33it/s] 61%|██████▏   | 245467/400000 [00:27<00:17, 8906.90it/s] 62%|██████▏   | 246417/400000 [00:27<00:16, 9075.75it/s] 62%|██████▏   | 247327/400000 [00:28<00:16, 9050.84it/s] 62%|██████▏   | 248265/400000 [00:28<00:16, 9146.22it/s] 62%|██████▏   | 249188/400000 [00:28<00:16, 9170.67it/s] 63%|██████▎   | 250106/400000 [00:28<00:16, 9049.27it/s] 63%|██████▎   | 251024/400000 [00:28<00:16, 9087.97it/s] 63%|██████▎   | 251935/400000 [00:28<00:16, 9092.43it/s] 63%|██████▎   | 252845/400000 [00:28<00:16, 8697.72it/s] 63%|██████▎   | 253722/400000 [00:28<00:16, 8716.76it/s] 64%|██████▎   | 254621/400000 [00:28<00:16, 8794.63it/s] 64%|██████▍   | 255533/400000 [00:28<00:16, 8888.08it/s] 64%|██████▍   | 256433/400000 [00:29<00:16, 8919.81it/s] 64%|██████▍   | 257362/400000 [00:29<00:15, 9025.76it/s] 65%|██████▍   | 258288/400000 [00:29<00:15, 9092.71it/s] 65%|██████▍   | 259199/400000 [00:29<00:15, 9075.51it/s] 65%|██████▌   | 260108/400000 [00:29<00:15, 8988.02it/s] 65%|██████▌   | 261008/400000 [00:29<00:15, 8935.30it/s] 65%|██████▌   | 261903/400000 [00:29<00:15, 8901.71it/s] 66%|██████▌   | 262850/400000 [00:29<00:15, 9061.83it/s] 66%|██████▌   | 263764/400000 [00:29<00:14, 9084.33it/s] 66%|██████▌   | 264683/400000 [00:29<00:14, 9113.51it/s] 66%|██████▋   | 265602/400000 [00:30<00:14, 9133.87it/s] 67%|██████▋   | 266524/400000 [00:30<00:14, 9156.93it/s] 67%|██████▋   | 267440/400000 [00:30<00:14, 9157.36it/s] 67%|██████▋   | 268356/400000 [00:30<00:14, 8968.55it/s] 67%|██████▋   | 269272/400000 [00:30<00:14, 9023.85it/s] 68%|██████▊   | 270201/400000 [00:30<00:14, 9100.99it/s] 68%|██████▊   | 271112/400000 [00:30<00:14, 9073.12it/s] 68%|██████▊   | 272028/400000 [00:30<00:14, 9096.72it/s] 68%|██████▊   | 272960/400000 [00:30<00:13, 9159.77it/s] 68%|██████▊   | 273893/400000 [00:30<00:13, 9207.34it/s] 69%|██████▊   | 274815/400000 [00:31<00:13, 9000.62it/s] 69%|██████▉   | 275750/400000 [00:31<00:13, 9100.71it/s] 69%|██████▉   | 276688/400000 [00:31<00:13, 9180.51it/s] 69%|██████▉   | 277608/400000 [00:31<00:13, 9079.46it/s] 70%|██████▉   | 278522/400000 [00:31<00:13, 9094.81it/s] 70%|██████▉   | 279433/400000 [00:31<00:13, 8896.23it/s] 70%|███████   | 280325/400000 [00:31<00:13, 8709.60it/s] 70%|███████   | 281217/400000 [00:31<00:13, 8770.06it/s] 71%|███████   | 282127/400000 [00:31<00:13, 8865.92it/s] 71%|███████   | 283066/400000 [00:32<00:12, 9015.03it/s] 71%|███████   | 283970/400000 [00:32<00:12, 9021.64it/s] 71%|███████   | 284894/400000 [00:32<00:12, 9083.20it/s] 71%|███████▏  | 285837/400000 [00:32<00:12, 9184.07it/s] 72%|███████▏  | 286757/400000 [00:32<00:12, 9025.14it/s] 72%|███████▏  | 287661/400000 [00:32<00:12, 8997.88it/s] 72%|███████▏  | 288562/400000 [00:32<00:12, 8956.49it/s] 72%|███████▏  | 289476/400000 [00:32<00:12, 9008.77it/s] 73%|███████▎  | 290378/400000 [00:32<00:12, 8889.98it/s] 73%|███████▎  | 291276/400000 [00:32<00:12, 8915.58it/s] 73%|███████▎  | 292174/400000 [00:33<00:12, 8933.19it/s] 73%|███████▎  | 293069/400000 [00:33<00:11, 8936.53it/s] 73%|███████▎  | 293977/400000 [00:33<00:11, 8976.04it/s] 74%|███████▎  | 294875/400000 [00:33<00:11, 8952.03it/s] 74%|███████▍  | 295798/400000 [00:33<00:11, 9033.13it/s] 74%|███████▍  | 296702/400000 [00:33<00:12, 8489.59it/s] 74%|███████▍  | 297572/400000 [00:33<00:11, 8549.18it/s] 75%|███████▍  | 298464/400000 [00:33<00:11, 8655.74it/s] 75%|███████▍  | 299391/400000 [00:33<00:11, 8828.42it/s] 75%|███████▌  | 300308/400000 [00:33<00:11, 8925.23it/s] 75%|███████▌  | 301213/400000 [00:34<00:11, 8961.97it/s] 76%|███████▌  | 302134/400000 [00:34<00:10, 9033.02it/s] 76%|███████▌  | 303081/400000 [00:34<00:10, 9157.66it/s] 76%|███████▌  | 303999/400000 [00:34<00:10, 9160.33it/s] 76%|███████▌  | 304917/400000 [00:34<00:10, 9037.59it/s] 76%|███████▋  | 305822/400000 [00:34<00:10, 8849.08it/s] 77%|███████▋  | 306709/400000 [00:34<00:10, 8626.37it/s] 77%|███████▋  | 307577/400000 [00:34<00:10, 8640.28it/s] 77%|███████▋  | 308493/400000 [00:34<00:10, 8789.44it/s] 77%|███████▋  | 309413/400000 [00:34<00:10, 8906.98it/s] 78%|███████▊  | 310327/400000 [00:35<00:09, 8974.08it/s] 78%|███████▊  | 311260/400000 [00:35<00:09, 9075.19it/s] 78%|███████▊  | 312207/400000 [00:35<00:09, 9189.69it/s] 78%|███████▊  | 313132/400000 [00:35<00:09, 9205.57it/s] 79%|███████▊  | 314076/400000 [00:35<00:09, 9273.71it/s] 79%|███████▉  | 315005/400000 [00:35<00:09, 9230.52it/s] 79%|███████▉  | 315929/400000 [00:35<00:09, 9167.53it/s] 79%|███████▉  | 316869/400000 [00:35<00:09, 9233.62it/s] 79%|███████▉  | 317793/400000 [00:35<00:08, 9135.51it/s] 80%|███████▉  | 318714/400000 [00:35<00:08, 9155.56it/s] 80%|███████▉  | 319630/400000 [00:36<00:08, 9029.00it/s] 80%|████████  | 320534/400000 [00:36<00:08, 8910.60it/s] 80%|████████  | 321447/400000 [00:36<00:08, 8974.81it/s] 81%|████████  | 322356/400000 [00:36<00:08, 9008.71it/s] 81%|████████  | 323280/400000 [00:36<00:08, 9076.13it/s] 81%|████████  | 324194/400000 [00:36<00:08, 9093.63it/s] 81%|████████▏ | 325104/400000 [00:36<00:08, 9026.03it/s] 82%|████████▏ | 326007/400000 [00:36<00:08, 8883.06it/s] 82%|████████▏ | 326933/400000 [00:36<00:08, 8991.03it/s] 82%|████████▏ | 327859/400000 [00:36<00:07, 9068.35it/s] 82%|████████▏ | 328767/400000 [00:37<00:07, 8991.55it/s] 82%|████████▏ | 329670/400000 [00:37<00:07, 9002.49it/s] 83%|████████▎ | 330576/400000 [00:37<00:07, 9016.95it/s] 83%|████████▎ | 331503/400000 [00:37<00:07, 9088.76it/s] 83%|████████▎ | 332415/400000 [00:37<00:07, 9097.63it/s] 83%|████████▎ | 333326/400000 [00:37<00:07, 8964.51it/s] 84%|████████▎ | 334235/400000 [00:37<00:07, 9001.70it/s] 84%|████████▍ | 335162/400000 [00:37<00:07, 9078.11it/s] 84%|████████▍ | 336071/400000 [00:37<00:07, 8994.77it/s] 84%|████████▍ | 336988/400000 [00:38<00:06, 9045.84it/s] 84%|████████▍ | 337895/400000 [00:38<00:06, 9050.62it/s] 85%|████████▍ | 338801/400000 [00:38<00:06, 8946.93it/s] 85%|████████▍ | 339698/400000 [00:38<00:06, 8952.17it/s] 85%|████████▌ | 340594/400000 [00:38<00:06, 8952.00it/s] 85%|████████▌ | 341499/400000 [00:38<00:06, 8979.32it/s] 86%|████████▌ | 342423/400000 [00:38<00:06, 9055.42it/s] 86%|████████▌ | 343329/400000 [00:38<00:06, 8984.47it/s] 86%|████████▌ | 344305/400000 [00:38<00:06, 9202.18it/s] 86%|████████▋ | 345234/400000 [00:38<00:05, 9226.68it/s] 87%|████████▋ | 346158/400000 [00:39<00:05, 9218.12it/s] 87%|████████▋ | 347100/400000 [00:39<00:05, 9277.63it/s] 87%|████████▋ | 348029/400000 [00:39<00:05, 9246.64it/s] 87%|████████▋ | 348967/400000 [00:39<00:05, 9283.46it/s] 87%|████████▋ | 349896/400000 [00:39<00:05, 9266.97it/s] 88%|████████▊ | 350823/400000 [00:39<00:05, 9239.20it/s] 88%|████████▊ | 351748/400000 [00:39<00:05, 9233.44it/s] 88%|████████▊ | 352672/400000 [00:39<00:05, 9182.50it/s] 88%|████████▊ | 353591/400000 [00:39<00:05, 8980.41it/s] 89%|████████▊ | 354515/400000 [00:39<00:05, 9054.45it/s] 89%|████████▉ | 355422/400000 [00:40<00:04, 9054.81it/s] 89%|████████▉ | 356366/400000 [00:40<00:04, 9165.69it/s] 89%|████████▉ | 357290/400000 [00:40<00:04, 9187.44it/s] 90%|████████▉ | 358215/400000 [00:40<00:04, 9204.86it/s] 90%|████████▉ | 359136/400000 [00:40<00:04, 9108.53it/s] 90%|█████████ | 360048/400000 [00:40<00:04, 9008.41it/s] 90%|█████████ | 360994/400000 [00:40<00:04, 9137.52it/s] 90%|█████████ | 361909/400000 [00:40<00:04, 9081.23it/s] 91%|█████████ | 362835/400000 [00:40<00:04, 9132.79it/s] 91%|█████████ | 363763/400000 [00:40<00:03, 9174.38it/s] 91%|█████████ | 364682/400000 [00:41<00:03, 9178.60it/s] 91%|█████████▏| 365601/400000 [00:41<00:03, 9155.05it/s] 92%|█████████▏| 366517/400000 [00:41<00:03, 9086.63it/s] 92%|█████████▏| 367426/400000 [00:41<00:03, 9079.87it/s] 92%|█████████▏| 368346/400000 [00:41<00:03, 9113.06it/s] 92%|█████████▏| 369273/400000 [00:41<00:03, 9157.87it/s] 93%|█████████▎| 370212/400000 [00:41<00:03, 9224.60it/s] 93%|█████████▎| 371135/400000 [00:41<00:03, 9087.72it/s] 93%|█████████▎| 372046/400000 [00:41<00:03, 9092.78it/s] 93%|█████████▎| 372956/400000 [00:41<00:02, 9031.27it/s] 93%|█████████▎| 373872/400000 [00:42<00:02, 9067.73it/s] 94%|█████████▎| 374833/400000 [00:42<00:02, 9221.23it/s] 94%|█████████▍| 375756/400000 [00:42<00:02, 9172.82it/s] 94%|█████████▍| 376699/400000 [00:42<00:02, 9248.37it/s] 94%|█████████▍| 377625/400000 [00:42<00:02, 9217.94it/s] 95%|█████████▍| 378548/400000 [00:42<00:02, 9048.24it/s] 95%|█████████▍| 379480/400000 [00:42<00:02, 9127.20it/s] 95%|█████████▌| 380398/400000 [00:42<00:02, 9142.19it/s] 95%|█████████▌| 381313/400000 [00:42<00:02, 9043.26it/s] 96%|█████████▌| 382227/400000 [00:42<00:01, 9070.95it/s] 96%|█████████▌| 383138/400000 [00:43<00:01, 9081.73it/s] 96%|█████████▌| 384092/400000 [00:43<00:01, 9212.16it/s] 96%|█████████▋| 385026/400000 [00:43<00:01, 9247.79it/s] 96%|█████████▋| 385952/400000 [00:43<00:01, 8996.55it/s] 97%|█████████▋| 386854/400000 [00:43<00:01, 8960.75it/s] 97%|█████████▋| 387752/400000 [00:43<00:01, 8876.41it/s] 97%|█████████▋| 388679/400000 [00:43<00:01, 8988.39it/s] 97%|█████████▋| 389579/400000 [00:43<00:01, 8927.59it/s] 98%|█████████▊| 390475/400000 [00:43<00:01, 8935.02it/s] 98%|█████████▊| 391370/400000 [00:43<00:00, 8868.81it/s] 98%|█████████▊| 392286/400000 [00:44<00:00, 8952.24it/s] 98%|█████████▊| 393222/400000 [00:44<00:00, 9069.32it/s] 99%|█████████▊| 394130/400000 [00:44<00:00, 8974.47it/s] 99%|█████████▉| 395029/400000 [00:44<00:00, 8947.85it/s] 99%|█████████▉| 395925/400000 [00:44<00:00, 8766.41it/s] 99%|█████████▉| 396823/400000 [00:44<00:00, 8827.35it/s] 99%|█████████▉| 397707/400000 [00:44<00:00, 8806.56it/s]100%|█████████▉| 398589/400000 [00:44<00:00, 8720.36it/s]100%|█████████▉| 399499/400000 [00:44<00:00, 8828.32it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8897.03it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f74bb7f50f0>, <torchtext.data.dataset.TabularDataset object at 0x7f74bb7f5240>, <torchtext.vocab.Vocab object at 0x7f74bb7f5160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 30.87 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 59.88 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 31.44 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.82 file/s]2020-06-08 18:17:59.144680: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-08 18:17:59.148783: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095199999 Hz
2020-06-08 18:17:59.148935: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fcd455f8b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-08 18:17:59.148949: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 145511.34it/s] 64%|██████▍   | 6356992/9912422 [00:00<00:17, 207625.26it/s]9920512it [00:00, 39922019.63it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 261855.82it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 6277976.51it/s]          
0it [00:00, ?it/s]8192it [00:00, 191089.14it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
