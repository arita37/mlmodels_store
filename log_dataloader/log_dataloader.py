
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f36e22b46a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f36e22b46a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f374d87c0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f374d87c0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f37675a8ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f37675a8ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f36fb0f8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f36fb0f8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f36fb0f8950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|███       | 2998272/9912422 [00:00<00:00, 29977674.05it/s]9920512it [00:00, 33907323.40it/s]                             
0it [00:00, ?it/s]32768it [00:00, 710300.85it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163441.91it/s]1654784it [00:00, 11481400.18it/s]                         
0it [00:00, ?it/s]8192it [00:00, 248086.54it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36e14cda90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36e14d7d30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f36fb0f8598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f36fb0f8598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f36fb0f8598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:21,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:19,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:53,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:16,  7.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 70.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.19 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.60s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.19 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.25 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.87s/ url]
0 examples [00:00, ? examples/s]2020-06-11 18:10:00.967204: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-11 18:10:00.981009: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-11 18:10:00.981312: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5599c0b2f780 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-11 18:10:00.981333: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
15 examples [00:00, 149.57 examples/s]102 examples [00:00, 198.98 examples/s]188 examples [00:00, 258.39 examples/s]281 examples [00:00, 329.66 examples/s]374 examples [00:00, 408.51 examples/s]465 examples [00:00, 489.08 examples/s]556 examples [00:00, 567.87 examples/s]638 examples [00:00, 624.21 examples/s]722 examples [00:00, 675.87 examples/s]805 examples [00:01, 714.72 examples/s]891 examples [00:01, 752.49 examples/s]982 examples [00:01, 792.51 examples/s]1071 examples [00:01, 818.74 examples/s]1161 examples [00:01, 836.44 examples/s]1255 examples [00:01, 863.22 examples/s]1349 examples [00:01, 882.92 examples/s]1441 examples [00:01, 892.00 examples/s]1533 examples [00:01, 898.60 examples/s]1624 examples [00:01, 885.17 examples/s]1715 examples [00:02, 891.13 examples/s]1805 examples [00:02, 871.60 examples/s]1893 examples [00:02, 870.52 examples/s]1981 examples [00:02, 867.74 examples/s]2071 examples [00:02, 875.14 examples/s]2159 examples [00:02, 869.90 examples/s]2247 examples [00:02, 849.53 examples/s]2333 examples [00:02, 838.51 examples/s]2420 examples [00:02, 846.56 examples/s]2505 examples [00:02, 843.90 examples/s]2590 examples [00:03, 827.20 examples/s]2673 examples [00:03, 825.95 examples/s]2759 examples [00:03, 834.99 examples/s]2843 examples [00:03, 826.63 examples/s]2933 examples [00:03, 846.79 examples/s]3025 examples [00:03, 866.68 examples/s]3120 examples [00:03, 887.62 examples/s]3210 examples [00:03, 877.32 examples/s]3298 examples [00:03, 868.74 examples/s]3388 examples [00:03, 876.61 examples/s]3480 examples [00:04, 887.05 examples/s]3572 examples [00:04, 896.42 examples/s]3662 examples [00:04, 887.67 examples/s]3751 examples [00:04, 877.70 examples/s]3841 examples [00:04, 883.13 examples/s]3934 examples [00:04, 894.40 examples/s]4026 examples [00:04, 899.77 examples/s]4117 examples [00:04, 889.80 examples/s]4207 examples [00:04, 883.15 examples/s]4296 examples [00:04, 875.76 examples/s]4384 examples [00:05, 856.12 examples/s]4471 examples [00:05, 858.72 examples/s]4557 examples [00:05, 847.50 examples/s]4642 examples [00:05, 810.80 examples/s]4732 examples [00:05, 835.07 examples/s]4818 examples [00:05, 842.00 examples/s]4906 examples [00:05, 851.26 examples/s]4993 examples [00:05, 855.45 examples/s]5079 examples [00:05, 854.68 examples/s]5167 examples [00:06, 859.03 examples/s]5254 examples [00:06, 837.25 examples/s]5338 examples [00:06, 826.74 examples/s]5421 examples [00:06, 795.90 examples/s]5502 examples [00:06, 798.25 examples/s]5592 examples [00:06, 824.35 examples/s]5682 examples [00:06, 844.18 examples/s]5775 examples [00:06, 866.82 examples/s]5864 examples [00:06, 871.82 examples/s]5952 examples [00:06, 869.07 examples/s]6040 examples [00:07, 859.75 examples/s]6130 examples [00:07, 870.40 examples/s]6224 examples [00:07, 887.70 examples/s]6316 examples [00:07, 896.42 examples/s]6406 examples [00:07, 888.83 examples/s]6496 examples [00:07, 890.04 examples/s]6586 examples [00:07, 876.19 examples/s]6674 examples [00:07, 873.42 examples/s]6762 examples [00:07, 874.92 examples/s]6852 examples [00:07, 880.37 examples/s]6944 examples [00:08, 891.62 examples/s]7034 examples [00:08, 883.33 examples/s]7123 examples [00:08, 881.04 examples/s]7215 examples [00:08, 892.05 examples/s]7305 examples [00:08, 878.53 examples/s]7393 examples [00:08, 875.22 examples/s]7486 examples [00:08, 889.25 examples/s]7576 examples [00:08, 891.18 examples/s]7667 examples [00:08, 895.38 examples/s]7757 examples [00:09, 888.87 examples/s]7851 examples [00:09, 901.50 examples/s]7942 examples [00:09, 892.23 examples/s]8032 examples [00:09, 878.72 examples/s]8120 examples [00:09, 822.17 examples/s]8208 examples [00:09, 828.17 examples/s]8301 examples [00:09, 855.58 examples/s]8398 examples [00:09, 886.08 examples/s]8494 examples [00:09, 906.60 examples/s]8586 examples [00:09, 906.72 examples/s]8678 examples [00:10, 910.41 examples/s]8770 examples [00:10, 909.75 examples/s]8862 examples [00:10, 911.99 examples/s]8954 examples [00:10, 910.59 examples/s]9046 examples [00:10, 897.04 examples/s]9136 examples [00:10, 876.01 examples/s]9228 examples [00:10, 886.76 examples/s]9322 examples [00:10, 900.44 examples/s]9414 examples [00:10, 905.81 examples/s]9505 examples [00:10, 879.59 examples/s]9594 examples [00:11, 854.69 examples/s]9686 examples [00:11, 872.84 examples/s]9778 examples [00:11, 885.89 examples/s]9872 examples [00:11, 899.42 examples/s]9963 examples [00:11, 884.04 examples/s]10052 examples [00:11, 828.25 examples/s]10142 examples [00:11, 847.02 examples/s]10229 examples [00:11, 853.50 examples/s]10319 examples [00:11, 864.30 examples/s]10408 examples [00:12, 869.72 examples/s]10501 examples [00:12, 886.32 examples/s]10596 examples [00:12, 903.85 examples/s]10695 examples [00:12, 927.05 examples/s]10789 examples [00:12, 911.02 examples/s]10881 examples [00:12, 866.17 examples/s]10972 examples [00:12, 876.21 examples/s]11061 examples [00:12, 847.57 examples/s]11147 examples [00:12, 834.88 examples/s]11240 examples [00:12, 860.69 examples/s]11331 examples [00:13, 874.15 examples/s]11423 examples [00:13, 886.96 examples/s]11513 examples [00:13, 887.80 examples/s]11603 examples [00:13, 889.82 examples/s]11693 examples [00:13, 877.85 examples/s]11781 examples [00:13, 873.98 examples/s]11869 examples [00:13, 873.21 examples/s]11959 examples [00:13, 878.72 examples/s]12051 examples [00:13, 889.98 examples/s]12141 examples [00:13, 879.01 examples/s]12229 examples [00:14, 842.40 examples/s]12318 examples [00:14, 855.40 examples/s]12405 examples [00:14, 858.30 examples/s]12492 examples [00:14, 859.66 examples/s]12579 examples [00:14, 862.18 examples/s]12666 examples [00:14, 855.19 examples/s]12756 examples [00:14, 865.02 examples/s]12843 examples [00:14, 862.38 examples/s]12932 examples [00:14, 868.78 examples/s]13019 examples [00:15, 868.08 examples/s]13106 examples [00:15, 867.35 examples/s]13194 examples [00:15, 870.91 examples/s]13282 examples [00:15, 868.75 examples/s]13369 examples [00:15, 868.08 examples/s]13462 examples [00:15, 884.51 examples/s]13551 examples [00:15, 873.17 examples/s]13647 examples [00:15, 896.98 examples/s]13737 examples [00:15, 880.98 examples/s]13826 examples [00:15, 871.83 examples/s]13914 examples [00:16, 854.99 examples/s]14004 examples [00:16, 866.10 examples/s]14091 examples [00:16, 846.26 examples/s]14186 examples [00:16, 874.01 examples/s]14280 examples [00:16, 891.73 examples/s]14370 examples [00:16, 890.98 examples/s]14463 examples [00:16, 900.86 examples/s]14557 examples [00:16, 912.01 examples/s]14649 examples [00:16, 912.38 examples/s]14744 examples [00:16, 921.44 examples/s]14838 examples [00:17, 926.71 examples/s]14931 examples [00:17, 917.50 examples/s]15030 examples [00:17, 936.76 examples/s]15125 examples [00:17, 938.11 examples/s]15219 examples [00:17, 932.26 examples/s]15313 examples [00:17, 919.83 examples/s]15406 examples [00:17, 910.48 examples/s]15498 examples [00:17, 881.66 examples/s]15590 examples [00:17, 892.81 examples/s]15680 examples [00:17, 886.02 examples/s]15769 examples [00:18, 875.39 examples/s]15859 examples [00:18, 880.37 examples/s]15948 examples [00:18, 870.55 examples/s]16036 examples [00:18, 867.15 examples/s]16123 examples [00:18, 856.60 examples/s]16209 examples [00:18, 821.47 examples/s]16296 examples [00:18, 833.55 examples/s]16382 examples [00:18, 839.58 examples/s]16473 examples [00:18, 859.46 examples/s]16560 examples [00:19, 849.72 examples/s]16646 examples [00:19, 849.19 examples/s]16732 examples [00:19, 839.23 examples/s]16817 examples [00:19, 826.12 examples/s]16906 examples [00:19, 841.63 examples/s]16997 examples [00:19, 859.18 examples/s]17086 examples [00:19, 867.39 examples/s]17183 examples [00:19, 892.52 examples/s]17273 examples [00:19, 880.71 examples/s]17362 examples [00:19, 880.67 examples/s]17455 examples [00:20, 892.88 examples/s]17545 examples [00:20, 877.16 examples/s]17633 examples [00:20, 868.03 examples/s]17721 examples [00:20, 869.61 examples/s]17809 examples [00:20, 857.49 examples/s]17895 examples [00:20, 847.27 examples/s]17981 examples [00:20, 848.66 examples/s]18074 examples [00:20, 870.16 examples/s]18166 examples [00:20, 884.46 examples/s]18255 examples [00:20, 881.60 examples/s]18344 examples [00:21, 871.16 examples/s]18432 examples [00:21, 848.99 examples/s]18524 examples [00:21, 869.04 examples/s]18618 examples [00:21, 889.03 examples/s]18710 examples [00:21, 894.63 examples/s]18804 examples [00:21, 904.95 examples/s]18895 examples [00:21, 885.57 examples/s]18992 examples [00:21, 907.52 examples/s]19087 examples [00:21, 918.84 examples/s]19180 examples [00:21, 921.45 examples/s]19273 examples [00:22, 906.16 examples/s]19364 examples [00:22, 852.83 examples/s]19451 examples [00:22, 853.93 examples/s]19541 examples [00:22, 865.96 examples/s]19629 examples [00:22, 856.55 examples/s]19721 examples [00:22, 872.86 examples/s]19809 examples [00:22, 834.39 examples/s]19897 examples [00:22, 847.34 examples/s]19991 examples [00:22, 871.51 examples/s]20079 examples [00:23, 840.19 examples/s]20171 examples [00:23, 860.39 examples/s]20262 examples [00:23, 874.65 examples/s]20354 examples [00:23, 885.94 examples/s]20449 examples [00:23, 902.38 examples/s]20540 examples [00:23, 889.17 examples/s]20630 examples [00:23, 886.80 examples/s]20719 examples [00:23, 865.06 examples/s]20806 examples [00:23, 858.60 examples/s]20893 examples [00:23, 844.19 examples/s]20978 examples [00:24, 844.11 examples/s]21064 examples [00:24, 846.27 examples/s]21149 examples [00:24, 842.06 examples/s]21234 examples [00:24, 829.93 examples/s]21318 examples [00:24, 812.11 examples/s]21406 examples [00:24, 829.96 examples/s]21490 examples [00:24, 821.72 examples/s]21573 examples [00:24, 802.15 examples/s]21655 examples [00:24, 806.44 examples/s]21739 examples [00:25, 814.84 examples/s]21827 examples [00:25, 830.83 examples/s]21911 examples [00:25, 813.30 examples/s]21995 examples [00:25, 820.46 examples/s]22081 examples [00:25, 830.36 examples/s]22165 examples [00:25, 819.80 examples/s]22248 examples [00:25, 814.08 examples/s]22341 examples [00:25, 843.88 examples/s]22432 examples [00:25, 862.01 examples/s]22524 examples [00:25, 875.74 examples/s]22612 examples [00:26, 867.37 examples/s]22699 examples [00:26, 843.85 examples/s]22788 examples [00:26, 856.82 examples/s]22875 examples [00:26, 858.24 examples/s]22965 examples [00:26, 870.01 examples/s]23054 examples [00:26, 874.83 examples/s]23144 examples [00:26, 881.72 examples/s]23233 examples [00:26, 876.81 examples/s]23322 examples [00:26, 879.24 examples/s]23417 examples [00:26, 898.68 examples/s]23508 examples [00:27, 901.78 examples/s]23599 examples [00:27, 871.15 examples/s]23689 examples [00:27, 878.44 examples/s]23784 examples [00:27, 896.78 examples/s]23880 examples [00:27, 913.81 examples/s]23975 examples [00:27, 923.41 examples/s]24068 examples [00:27, 919.03 examples/s]24161 examples [00:27, 903.33 examples/s]24254 examples [00:27, 910.11 examples/s]24346 examples [00:27, 885.45 examples/s]24435 examples [00:28, 867.29 examples/s]24522 examples [00:28, 844.34 examples/s]24607 examples [00:28, 834.75 examples/s]24697 examples [00:28, 849.77 examples/s]24783 examples [00:28, 851.79 examples/s]24869 examples [00:28, 839.39 examples/s]24960 examples [00:28, 857.56 examples/s]25046 examples [00:28, 832.52 examples/s]25131 examples [00:28, 836.83 examples/s]25215 examples [00:29, 832.31 examples/s]25300 examples [00:29, 836.21 examples/s]25391 examples [00:29, 856.66 examples/s]25477 examples [00:29, 849.20 examples/s]25564 examples [00:29, 854.56 examples/s]25656 examples [00:29, 871.54 examples/s]25747 examples [00:29, 882.49 examples/s]25836 examples [00:29, 866.66 examples/s]25923 examples [00:29, 858.86 examples/s]26010 examples [00:29, 856.92 examples/s]26096 examples [00:30, 855.68 examples/s]26182 examples [00:30, 832.22 examples/s]26266 examples [00:30, 819.59 examples/s]26353 examples [00:30, 832.99 examples/s]26440 examples [00:30, 843.02 examples/s]26525 examples [00:30, 834.16 examples/s]26614 examples [00:30, 849.34 examples/s]26702 examples [00:30, 856.96 examples/s]26788 examples [00:30, 853.57 examples/s]26875 examples [00:30, 857.78 examples/s]26961 examples [00:31, 854.09 examples/s]27047 examples [00:31, 821.89 examples/s]27130 examples [00:31, 798.85 examples/s]27219 examples [00:31, 822.70 examples/s]27304 examples [00:31, 828.50 examples/s]27388 examples [00:31, 816.72 examples/s]27474 examples [00:31, 825.75 examples/s]27561 examples [00:31, 836.33 examples/s]27651 examples [00:31, 853.91 examples/s]27741 examples [00:32, 865.74 examples/s]27831 examples [00:32, 873.43 examples/s]27919 examples [00:32, 874.64 examples/s]28007 examples [00:32, 871.47 examples/s]28095 examples [00:32, 865.59 examples/s]28182 examples [00:32, 865.08 examples/s]28274 examples [00:32, 880.10 examples/s]28364 examples [00:32, 885.25 examples/s]28453 examples [00:32, 880.28 examples/s]28543 examples [00:32, 884.75 examples/s]28636 examples [00:33, 895.74 examples/s]28726 examples [00:33, 895.19 examples/s]28817 examples [00:33, 896.72 examples/s]28907 examples [00:33, 881.43 examples/s]28996 examples [00:33, 875.57 examples/s]29089 examples [00:33, 889.46 examples/s]29179 examples [00:33, 892.48 examples/s]29269 examples [00:33, 887.61 examples/s]29358 examples [00:33, 878.21 examples/s]29446 examples [00:33, 875.93 examples/s]29534 examples [00:34, 864.80 examples/s]29621 examples [00:34, 856.72 examples/s]29707 examples [00:34, 853.51 examples/s]29793 examples [00:34, 833.34 examples/s]29877 examples [00:34, 832.59 examples/s]29966 examples [00:34, 847.11 examples/s]30051 examples [00:34, 769.16 examples/s]30140 examples [00:34, 800.57 examples/s]30222 examples [00:34, 801.00 examples/s]30309 examples [00:35, 818.28 examples/s]30395 examples [00:35, 828.79 examples/s]30479 examples [00:35, 819.97 examples/s]30568 examples [00:35, 838.06 examples/s]30656 examples [00:35, 849.89 examples/s]30746 examples [00:35, 861.80 examples/s]30833 examples [00:35, 831.06 examples/s]30925 examples [00:35, 854.95 examples/s]31018 examples [00:35, 874.67 examples/s]31107 examples [00:35, 878.15 examples/s]31196 examples [00:36, 855.74 examples/s]31282 examples [00:36, 856.77 examples/s]31368 examples [00:36, 845.34 examples/s]31454 examples [00:36, 848.96 examples/s]31540 examples [00:36, 844.64 examples/s]31625 examples [00:36, 846.22 examples/s]31711 examples [00:36, 850.02 examples/s]31797 examples [00:36, 822.81 examples/s]31880 examples [00:36, 804.24 examples/s]31961 examples [00:36, 793.52 examples/s]32041 examples [00:37, 765.86 examples/s]32125 examples [00:37, 784.29 examples/s]32212 examples [00:37, 807.59 examples/s]32299 examples [00:37, 823.55 examples/s]32384 examples [00:37, 829.50 examples/s]32471 examples [00:37, 839.64 examples/s]32556 examples [00:37, 824.23 examples/s]32640 examples [00:37, 825.99 examples/s]32724 examples [00:37, 827.20 examples/s]32811 examples [00:38, 838.42 examples/s]32900 examples [00:38, 850.96 examples/s]32987 examples [00:38, 854.48 examples/s]33073 examples [00:38, 850.56 examples/s]33159 examples [00:38, 834.98 examples/s]33246 examples [00:38, 844.26 examples/s]33331 examples [00:38, 845.20 examples/s]33417 examples [00:38, 849.28 examples/s]33502 examples [00:38, 848.99 examples/s]33587 examples [00:38, 844.06 examples/s]33672 examples [00:39, 822.87 examples/s]33757 examples [00:39, 830.51 examples/s]33843 examples [00:39, 836.84 examples/s]33929 examples [00:39, 841.01 examples/s]34014 examples [00:39, 823.59 examples/s]34097 examples [00:39, 803.34 examples/s]34182 examples [00:39, 815.71 examples/s]34270 examples [00:39, 833.68 examples/s]34362 examples [00:39, 855.87 examples/s]34453 examples [00:39, 871.06 examples/s]34545 examples [00:40, 882.41 examples/s]34634 examples [00:40, 883.43 examples/s]34723 examples [00:40, 873.17 examples/s]34813 examples [00:40, 879.50 examples/s]34904 examples [00:40, 885.99 examples/s]34993 examples [00:40, 885.90 examples/s]35082 examples [00:40, 861.64 examples/s]35169 examples [00:40, 846.41 examples/s]35256 examples [00:40, 852.89 examples/s]35342 examples [00:40, 854.18 examples/s]35428 examples [00:41, 837.47 examples/s]35512 examples [00:41, 821.05 examples/s]35602 examples [00:41, 841.86 examples/s]35691 examples [00:41, 854.65 examples/s]35779 examples [00:41, 861.35 examples/s]35869 examples [00:41, 870.66 examples/s]35957 examples [00:41, 871.43 examples/s]36045 examples [00:41, 867.92 examples/s]36134 examples [00:41, 874.02 examples/s]36222 examples [00:42, 830.57 examples/s]36306 examples [00:42, 830.53 examples/s]36391 examples [00:42, 835.16 examples/s]36475 examples [00:42, 810.66 examples/s]36562 examples [00:42, 827.12 examples/s]36646 examples [00:42, 828.56 examples/s]36733 examples [00:42, 838.84 examples/s]36819 examples [00:42, 843.17 examples/s]36904 examples [00:42, 837.94 examples/s]36996 examples [00:42, 858.71 examples/s]37086 examples [00:43, 868.23 examples/s]37176 examples [00:43, 875.39 examples/s]37265 examples [00:43, 879.33 examples/s]37354 examples [00:43, 864.71 examples/s]37441 examples [00:43, 860.89 examples/s]37528 examples [00:43, 852.51 examples/s]37617 examples [00:43, 863.13 examples/s]37704 examples [00:43, 853.77 examples/s]37790 examples [00:43, 835.44 examples/s]37876 examples [00:43, 840.27 examples/s]37961 examples [00:44, 825.94 examples/s]38044 examples [00:44, 820.96 examples/s]38131 examples [00:44, 834.98 examples/s]38216 examples [00:44, 838.80 examples/s]38300 examples [00:44, 838.03 examples/s]38384 examples [00:44, 810.84 examples/s]38467 examples [00:44, 815.61 examples/s]38549 examples [00:44, 816.92 examples/s]38635 examples [00:44, 828.50 examples/s]38727 examples [00:44, 852.19 examples/s]38813 examples [00:45, 850.32 examples/s]38901 examples [00:45, 858.67 examples/s]38988 examples [00:45, 846.19 examples/s]39078 examples [00:45, 861.02 examples/s]39165 examples [00:45, 851.79 examples/s]39251 examples [00:45, 845.03 examples/s]39336 examples [00:45, 832.57 examples/s]39424 examples [00:45, 845.58 examples/s]39511 examples [00:45, 851.03 examples/s]39597 examples [00:46, 840.71 examples/s]39682 examples [00:46, 826.14 examples/s]39765 examples [00:46, 805.88 examples/s]39846 examples [00:46, 798.96 examples/s]39933 examples [00:46, 816.72 examples/s]40015 examples [00:46, 771.19 examples/s]40101 examples [00:46, 794.05 examples/s]40191 examples [00:46, 820.81 examples/s]40274 examples [00:46, 801.75 examples/s]40356 examples [00:46, 804.59 examples/s]40438 examples [00:47, 807.42 examples/s]40526 examples [00:47, 827.65 examples/s]40610 examples [00:47, 829.44 examples/s]40696 examples [00:47, 837.37 examples/s]40780 examples [00:47, 832.36 examples/s]40864 examples [00:47, 829.66 examples/s]40950 examples [00:47, 836.06 examples/s]41040 examples [00:47, 852.66 examples/s]41126 examples [00:47, 824.95 examples/s]41209 examples [00:47, 809.67 examples/s]41291 examples [00:48, 805.72 examples/s]41375 examples [00:48, 814.08 examples/s]41460 examples [00:48, 822.50 examples/s]41545 examples [00:48, 830.30 examples/s]41630 examples [00:48, 834.93 examples/s]41715 examples [00:48, 836.85 examples/s]41801 examples [00:48, 840.87 examples/s]41887 examples [00:48, 845.27 examples/s]41972 examples [00:48, 842.96 examples/s]42057 examples [00:49, 828.28 examples/s]42140 examples [00:49, 823.52 examples/s]42223 examples [00:49, 824.22 examples/s]42306 examples [00:49, 810.93 examples/s]42388 examples [00:49, 804.87 examples/s]42469 examples [00:49, 795.51 examples/s]42555 examples [00:49, 812.37 examples/s]42647 examples [00:49, 840.10 examples/s]42736 examples [00:49, 852.63 examples/s]42822 examples [00:49, 828.19 examples/s]42915 examples [00:50, 854.30 examples/s]43002 examples [00:50, 857.16 examples/s]43093 examples [00:50, 871.45 examples/s]43182 examples [00:50, 874.29 examples/s]43275 examples [00:50, 889.24 examples/s]43365 examples [00:50, 890.82 examples/s]43455 examples [00:50, 881.63 examples/s]43544 examples [00:50, 879.80 examples/s]43633 examples [00:50, 876.39 examples/s]43721 examples [00:50, 875.08 examples/s]43810 examples [00:51, 878.11 examples/s]43898 examples [00:51, 851.04 examples/s]43984 examples [00:51, 821.92 examples/s]44069 examples [00:51, 828.53 examples/s]44158 examples [00:51, 843.91 examples/s]44244 examples [00:51, 848.29 examples/s]44330 examples [00:51, 848.66 examples/s]44418 examples [00:51, 856.13 examples/s]44508 examples [00:51, 867.81 examples/s]44597 examples [00:51, 873.31 examples/s]44688 examples [00:52, 883.69 examples/s]44777 examples [00:52, 875.42 examples/s]44867 examples [00:52, 882.01 examples/s]44956 examples [00:52, 874.96 examples/s]45044 examples [00:52, 860.43 examples/s]45131 examples [00:52, 856.97 examples/s]45217 examples [00:52, 855.54 examples/s]45303 examples [00:52, 852.90 examples/s]45389 examples [00:52, 814.39 examples/s]45471 examples [00:53, 784.98 examples/s]45559 examples [00:53, 809.52 examples/s]45644 examples [00:53, 819.59 examples/s]45730 examples [00:53, 829.97 examples/s]45819 examples [00:53, 844.84 examples/s]45908 examples [00:53, 857.64 examples/s]45995 examples [00:53, 861.23 examples/s]46086 examples [00:53, 873.40 examples/s]46178 examples [00:53, 886.30 examples/s]46270 examples [00:53, 895.11 examples/s]46361 examples [00:54, 898.71 examples/s]46451 examples [00:54, 894.77 examples/s]46542 examples [00:54, 897.64 examples/s]46632 examples [00:54, 866.96 examples/s]46719 examples [00:54, 867.69 examples/s]46806 examples [00:54, 854.62 examples/s]46892 examples [00:54, 855.50 examples/s]46980 examples [00:54, 860.67 examples/s]47069 examples [00:54, 868.45 examples/s]47156 examples [00:54, 858.89 examples/s]47242 examples [00:55, 852.53 examples/s]47328 examples [00:55, 842.69 examples/s]47416 examples [00:55, 851.10 examples/s]47502 examples [00:55, 850.27 examples/s]47589 examples [00:55, 854.98 examples/s]47676 examples [00:55, 857.54 examples/s]47762 examples [00:55, 848.84 examples/s]47847 examples [00:55, 808.42 examples/s]47938 examples [00:55, 834.08 examples/s]48025 examples [00:55, 844.14 examples/s]48110 examples [00:56, 843.29 examples/s]48195 examples [00:56, 818.69 examples/s]48278 examples [00:56, 790.74 examples/s]48362 examples [00:56, 803.79 examples/s]48448 examples [00:56, 819.67 examples/s]48534 examples [00:56, 830.08 examples/s]48618 examples [00:56, 832.68 examples/s]48702 examples [00:56, 824.58 examples/s]48787 examples [00:56, 830.01 examples/s]48872 examples [00:57, 833.14 examples/s]48956 examples [00:57, 830.03 examples/s]49040 examples [00:57, 806.58 examples/s]49124 examples [00:57, 815.72 examples/s]49206 examples [00:57, 809.84 examples/s]49288 examples [00:57, 808.75 examples/s]49369 examples [00:57, 808.30 examples/s]49450 examples [00:57, 807.66 examples/s]49535 examples [00:57, 816.97 examples/s]49617 examples [00:57, 810.75 examples/s]49701 examples [00:58, 818.52 examples/s]49783 examples [00:58, 802.59 examples/s]49864 examples [00:58, 790.24 examples/s]49954 examples [00:58, 818.22 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6201/50000 [00:00<00:00, 62005.18 examples/s] 35%|███▌      | 17626/50000 [00:00<00:00, 71863.10 examples/s] 58%|█████▊    | 29042/50000 [00:00<00:00, 80849.32 examples/s] 81%|████████  | 40529/50000 [00:00<00:00, 88733.05 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 697.60 examples/s]148 examples [00:00, 718.62 examples/s]237 examples [00:00, 762.46 examples/s]324 examples [00:00, 791.73 examples/s]410 examples [00:00, 808.92 examples/s]495 examples [00:00, 819.74 examples/s]581 examples [00:00, 831.08 examples/s]668 examples [00:00, 840.74 examples/s]749 examples [00:00, 649.55 examples/s]838 examples [00:01, 705.80 examples/s]930 examples [00:01, 758.68 examples/s]1023 examples [00:01, 802.95 examples/s]1111 examples [00:01, 823.81 examples/s]1200 examples [00:01, 840.38 examples/s]1288 examples [00:01, 849.45 examples/s]1378 examples [00:01, 862.12 examples/s]1467 examples [00:01, 867.76 examples/s]1555 examples [00:01, 866.83 examples/s]1643 examples [00:02, 859.23 examples/s]1732 examples [00:02, 866.87 examples/s]1825 examples [00:02, 884.16 examples/s]1914 examples [00:02, 872.63 examples/s]2002 examples [00:02, 864.87 examples/s]2094 examples [00:02, 880.07 examples/s]2185 examples [00:02, 888.09 examples/s]2275 examples [00:02, 891.16 examples/s]2368 examples [00:02, 901.22 examples/s]2459 examples [00:02, 895.03 examples/s]2553 examples [00:03, 904.78 examples/s]2645 examples [00:03, 905.40 examples/s]2736 examples [00:03, 868.27 examples/s]2824 examples [00:03, 853.65 examples/s]2910 examples [00:03, 845.77 examples/s]2995 examples [00:03, 843.52 examples/s]3081 examples [00:03, 846.42 examples/s]3170 examples [00:03, 858.61 examples/s]3261 examples [00:03, 870.37 examples/s]3349 examples [00:03, 858.23 examples/s]3435 examples [00:04, 833.93 examples/s]3519 examples [00:04, 832.21 examples/s]3606 examples [00:04, 841.74 examples/s]3693 examples [00:04, 849.20 examples/s]3780 examples [00:04, 853.85 examples/s]3866 examples [00:04, 851.19 examples/s]3954 examples [00:04, 858.99 examples/s]4042 examples [00:04, 864.53 examples/s]4129 examples [00:04, 860.77 examples/s]4218 examples [00:04, 868.56 examples/s]4305 examples [00:05, 868.80 examples/s]4392 examples [00:05, 861.54 examples/s]4481 examples [00:05, 868.28 examples/s]4569 examples [00:05, 870.23 examples/s]4657 examples [00:05, 862.63 examples/s]4748 examples [00:05, 874.92 examples/s]4837 examples [00:05, 877.11 examples/s]4925 examples [00:05, 871.75 examples/s]5014 examples [00:05, 876.35 examples/s]5102 examples [00:05, 864.82 examples/s]5189 examples [00:06, 852.55 examples/s]5275 examples [00:06, 843.42 examples/s]5360 examples [00:06, 835.62 examples/s]5444 examples [00:06, 834.00 examples/s]5528 examples [00:06, 815.37 examples/s]5613 examples [00:06, 823.43 examples/s]5696 examples [00:06, 807.06 examples/s]5782 examples [00:06, 821.13 examples/s]5872 examples [00:06, 842.71 examples/s]5963 examples [00:07, 860.43 examples/s]6053 examples [00:07, 871.22 examples/s]6141 examples [00:07, 862.86 examples/s]6228 examples [00:07, 848.12 examples/s]6313 examples [00:07, 795.22 examples/s]6394 examples [00:07, 786.45 examples/s]6477 examples [00:07, 786.60 examples/s]6563 examples [00:07, 805.11 examples/s]6649 examples [00:07, 818.50 examples/s]6732 examples [00:07, 814.27 examples/s]6817 examples [00:08, 824.23 examples/s]6904 examples [00:08, 835.23 examples/s]6988 examples [00:08, 835.01 examples/s]7072 examples [00:08, 826.13 examples/s]7160 examples [00:08, 840.54 examples/s]7250 examples [00:08, 855.25 examples/s]7337 examples [00:08, 858.32 examples/s]7425 examples [00:08, 859.23 examples/s]7516 examples [00:08, 872.65 examples/s]7607 examples [00:08, 881.10 examples/s]7696 examples [00:09, 879.98 examples/s]7785 examples [00:09, 863.50 examples/s]7872 examples [00:09, 843.97 examples/s]7957 examples [00:09, 827.17 examples/s]8044 examples [00:09, 836.90 examples/s]8134 examples [00:09, 852.76 examples/s]8224 examples [00:09, 865.06 examples/s]8314 examples [00:09, 872.60 examples/s]8406 examples [00:09, 884.73 examples/s]8495 examples [00:10, 877.09 examples/s]8586 examples [00:10, 884.58 examples/s]8675 examples [00:10, 862.25 examples/s]8762 examples [00:10, 864.09 examples/s]8850 examples [00:10, 866.15 examples/s]8937 examples [00:10, 848.40 examples/s]9023 examples [00:10, 850.02 examples/s]9109 examples [00:10, 828.17 examples/s]9193 examples [00:10, 825.62 examples/s]9283 examples [00:10, 844.73 examples/s]9370 examples [00:11, 850.17 examples/s]9459 examples [00:11, 861.34 examples/s]9549 examples [00:11, 871.99 examples/s]9643 examples [00:11, 889.05 examples/s]9736 examples [00:11, 899.87 examples/s]9827 examples [00:11, 896.41 examples/s]9919 examples [00:11, 901.82 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 98%|█████████▊| 9802/10000 [00:00<00:00, 98011.45 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWSQD5X/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWSQD5X/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f36fb0f8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f36fb0f8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f36fb0f8950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36e598c9b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f368059de80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f36fb0f8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f36fb0f8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f36fb0f8950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36f83fb0f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36f83fb0b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f367c102268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f367c102268> 

  function with postional parmater data_info <function split_train_valid at 0x7f367c102268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f367c102378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f367c102378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f367c102378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=34c1a1a5c2b00905425a0c9399065bfcfa70edb7cf29024e39dc7f24ea13985c
  Stored in directory: /tmp/pip-ephem-wheel-cache-g0h8za1m/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<25:38:28, 9.34kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<18:11:08, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<12:46:36, 18.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<8:57:06, 26.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<6:15:00, 38.2kB/s].vector_cache/glove.6B.zip:   1%|          | 7.68M/862M [00:01<4:21:21, 54.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<3:02:05, 77.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.3M/862M [00:01<2:06:57, 111kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.6M/862M [00:01<1:28:23, 158kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.9M/862M [00:01<1:01:46, 226kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:02<43:04, 322kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.5M/862M [00:02<30:08, 458kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:02<21:03, 652kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.9M/862M [00:02<14:47, 924kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<10:23, 1.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:02<07:20, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<06:12, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<06:14, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<08:48, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<07:15, 1.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:05<05:20, 2.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<07:48, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<07:17, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:07<05:28, 2.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:07<03:59, 3.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<21:15, 626kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<16:29, 806kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<11:55, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<11:02, 1.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<09:08, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<06:41, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<07:43, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:47, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<05:04, 2.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<05:59, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:15<04:29, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<06:13, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<05:43, 2.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:17<04:16, 3.04MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<06:05, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<05:21, 2.41MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<04:26, 2.92MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<03:13, 3.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<19:53, 648kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<9:33:21, 22.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<6:41:40, 32.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<4:40:16, 45.8kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<3:31:03, 60.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<2:30:20, 85.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<1:45:41, 121kB/s] .vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:22<1:13:58, 173kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<57:30, 222kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<41:34, 307kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<29:22, 433kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<23:28, 540kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<17:44, 715kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<12:42, 995kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<11:51, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:35, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:58, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:50, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:06, 1.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:18, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:28, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:12, 1.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:07, 3.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<45:10, 275kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<32:41, 379kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<23:06, 536kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<16:18, 757kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<57:52, 213kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<41:45, 295kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<29:29, 417kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<23:28, 523kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<17:41, 693kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<12:37, 969kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<11:43, 1.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:28, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:55, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:42, 1.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:39, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:54, 2.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<06:17, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:38, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:15, 2.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<05:48, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:17, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:00, 2.99MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:37, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:08, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:53, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<05:31, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:05, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:48, 3.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:28, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:49, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:00, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:45, 3.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:24, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:46, 3.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:57, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:42, 3.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:18, 2.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:53, 2.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:42, 3.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:53, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:42, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<03:38, 3.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:50, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:49, 2.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:35, 3.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:08, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:54, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:36, 2.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:22, 3.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:32, 3.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<8:12:42, 22.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<5:45:05, 32.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<4:00:35, 45.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<3:08:25, 58.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:14:08, 82.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:34:20, 117kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<1:07:30, 162kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<48:25, 226kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<34:06, 321kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<26:18, 414kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<20:38, 528kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<14:59, 726kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<12:12, 888kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:41, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:03, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:24, 1.45MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:17, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:40, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:47, 1.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:09, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:52, 2.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:12, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:44, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:35, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:59, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:34, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:25, 3.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:52, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:24, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:50, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:27, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:22, 3.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:25, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:21, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:46, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:18, 3.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:43, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:40, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:18, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:13, 3.11MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:16, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<03:11, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:36, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:14, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:13, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:34, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<04:13, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:12, 3.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:08, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:29, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:08, 2.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:08, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:28, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:07, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:00, 2.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<02:55, 3.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:23, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:30, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:23, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:43, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:30, 2.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:42, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:13, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:56, 3.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<19:53, 472kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<14:52, 631kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<10:37, 880kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<07:56, 1.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<7:18:55, 21.2kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<5:07:18, 30.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<3:34:02, 43.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<2:47:53, 55.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<1:59:23, 77.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<1:23:54, 110kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<59:54, 153kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<42:53, 214kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<30:10, 303kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<23:05, 395kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<17:08, 532kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<12:12, 744kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<10:33, 857kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:14, 978kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:56, 1.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:10<04:56, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<1:07:25, 133kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<48:05, 187kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<33:47, 265kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<25:38, 347kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<18:52, 472kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<13:24, 662kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<11:24, 775kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:54, 991kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<06:25, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:31, 1.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:28, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:52, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:18, 2.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:11, 2.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:16, 2.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:56, 2.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:03, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:44, 2.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:47, 3.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:57, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:39, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:46, 3.05MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<03:53, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:35, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:43, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:51, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:33, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<02:41, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:50, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:32, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:40, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:48, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:21, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:28, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:44, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:28, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:36, 3.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:41, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:15, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:22, 2.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:39, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:23, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:32, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:38, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<04:10, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:19, 2.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:35, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:12, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<02:33, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<01:52, 4.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<16:30, 473kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<13:12, 591kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<09:33, 816kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:53, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<06:37, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:26, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:59, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:34, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:41, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:41, 2.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:39, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:03, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:59, 2.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:35, 2.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<5:35:22, 22.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<3:54:44, 32.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<2:43:17, 45.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<2:08:49, 58.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<1:31:40, 81.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:04:26, 116kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<45:59, 161kB/s]  .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<32:58, 224kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<23:09, 318kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<17:49, 411kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<13:13, 554kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<09:24, 776kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<08:16, 878kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:32, 1.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:44, 1.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:00, 1.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:14, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:08, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:52, 1.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:26, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:35, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:28, 2.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:09, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:22, 2.95MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:18, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:02, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:17, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:14, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:58, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:13, 3.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:12, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:39, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:50, 2.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:04, 3.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:01, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:13, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:05, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:44, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:18, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:28, 2.70MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:17, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:40, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:54, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:05, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:49, 2.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:08, 3.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:00, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:46, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:06, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:59, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:45, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:04, 3.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:57, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:03, 3.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:55, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:41, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:02, 3.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:53, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:39, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<01:59, 3.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:32, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<01:55, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:46, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:34, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<01:56, 3.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:46, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:33, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:56, 3.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<02:45, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:09, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:30, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:41, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:30, 2.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:52, 3.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:40, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:22, 2.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:48, 3.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:36, 3.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<4:30:11, 21.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<3:09:00, 30.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<2:11:23, 43.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<1:35:35, 59.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<1:08:02, 83.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<47:48, 119kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<34:04, 165kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<24:24, 230kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<17:09, 326kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<13:13, 420kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<09:48, 565kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<06:58, 790kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:09, 891kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:51, 1.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:31, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<03:44, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:04, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:18, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<01:40, 3.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<21:17, 251kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<15:59, 334kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<11:26, 466kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:47, 600kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:42, 786kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:58<04:47, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:33, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:15, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:11, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:18, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:28, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:57, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:11, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:43, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:25, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:48, 2.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:26, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:46, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:11, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:35, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<36:14, 136kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<25:45, 191kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<18:06, 271kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<12:37, 386kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<27:11, 179kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<19:30, 249kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<13:41, 353kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<10:38, 450kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:56, 603kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<05:38, 843kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<05:02, 938kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:00, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:54, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:07, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:39, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:58, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:27, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:38, 2.78MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:12, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:28, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:57, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:55, 2.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:27, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<01:52, 2.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:24, 3.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:00, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:17, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:49, 2.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:18, 3.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<4:06:00, 17.3kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<2:52:23, 24.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<2:00:02, 35.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<1:24:16, 49.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<59:19, 70.3kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<41:22, 100kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<29:41, 138kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:10, 194kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<14:49, 275kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<11:14, 360kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:41, 464kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<06:14, 644kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<04:24, 906kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:22, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<3:04:53, 21.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<2:09:12, 30.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<1:29:18, 43.8kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<1:11:24, 54.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<50:45, 76.9kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<35:36, 109kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<25:12, 152kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<18:01, 212kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<12:37, 301kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<09:36, 392kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<07:06, 530kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<05:02, 742kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:21, 847kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:48, 969kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:51, 1.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:33, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:09, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:35, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:07, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:38, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:11, 2.95MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:18, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:59, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:28, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:49, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:00, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:34, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:07, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<22:14, 151kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<15:53, 211kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<11:07, 299kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<08:28, 388kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<06:36, 497kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<04:45, 689kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<03:19, 971kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:55, 819kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:05, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:13, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:16, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:55, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:24, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:42, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:29, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:07, 2.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:29, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:00, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:23, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:16, 2.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:57, 3.05MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<00:41, 4.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<28:58, 99.3kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<20:32, 140kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<14:19, 199kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<09:55, 283kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<27:47, 101kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<20:01, 140kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<14:04, 198kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<09:51, 281kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<07:27, 367kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<05:29, 498kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<03:52, 698kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:18, 809kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<02:34, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:51, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:53, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:35, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:10, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:24, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:30, 1.69MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:09, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:49, 2.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:46, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:29, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:06, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:19, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:10, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:51, 2.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:08, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:02, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:46, 2.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:04, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:58, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:24<00:43, 3.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:37, 3.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:43:06, 21.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:11:51, 30.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:26<49:06, 43.3kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<39:51, 53.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<28:17, 74.9kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<19:47, 106kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<13:50, 148kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<09:52, 207kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<06:52, 294kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<05:11, 382kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<04:02, 491kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:54, 677kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:17, 836kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:47, 1.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:17, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:19, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:06, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:48, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:58, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:51, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:38, 2.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:50, 2.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:45, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:34, 2.95MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:46, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:42, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:31, 3.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:44, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:40, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:29, 3.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:41, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:38, 2.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:28, 3.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:35, 2.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:27, 3.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:19, 4.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<05:23, 254kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<03:53, 349kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<02:42, 494kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:09, 604kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:37, 793kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:09, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:04, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:42, 1.65MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:26, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:21, 2.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:33, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:18, 3.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:41, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:25, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:31, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:24, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:16, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<03:20, 246kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<02:24, 339kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<01:38, 478kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:16, 589kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:57, 771kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:40, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:36, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:33, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:24, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:17, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:15, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:18, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:14, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:09, 3.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:10, 2.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.98MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:05, 4.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:42, 486kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:30, 648kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:20, 908kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:16, 990kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:08, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.83MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.98MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:12:24,  7.31it/s]  0%|          | 699/400000 [00:00<10:37:51, 10.43it/s]  0%|          | 1493/400000 [00:00<7:25:51, 14.90it/s]  1%|          | 2305/400000 [00:00<5:11:42, 21.26it/s]  1%|          | 3124/400000 [00:00<3:37:59, 30.34it/s]  1%|          | 3887/400000 [00:00<2:32:33, 43.27it/s]  1%|          | 4720/400000 [00:00<1:46:48, 61.68it/s]  1%|▏         | 5522/400000 [00:00<1:14:51, 87.83it/s]  2%|▏         | 6315/400000 [00:00<52:32, 124.88it/s]   2%|▏         | 7091/400000 [00:01<36:57, 177.17it/s]  2%|▏         | 7851/400000 [00:01<26:05, 250.54it/s]  2%|▏         | 8659/400000 [00:01<18:27, 353.22it/s]  2%|▏         | 9430/400000 [00:01<13:10, 493.94it/s]  3%|▎         | 10217/400000 [00:01<09:27, 687.08it/s]  3%|▎         | 10978/400000 [00:01<06:52, 944.13it/s]  3%|▎         | 11779/400000 [00:01<05:02, 1283.86it/s]  3%|▎         | 12548/400000 [00:01<03:46, 1709.21it/s]  3%|▎         | 13312/400000 [00:01<02:53, 2227.08it/s]  4%|▎         | 14075/400000 [00:01<02:17, 2815.50it/s]  4%|▎         | 14830/400000 [00:02<01:51, 3455.67it/s]  4%|▍         | 15584/400000 [00:02<01:33, 4125.78it/s]  4%|▍         | 16335/400000 [00:02<01:21, 4734.78it/s]  4%|▍         | 17077/400000 [00:02<01:12, 5275.54it/s]  4%|▍         | 17854/400000 [00:02<01:05, 5836.98it/s]  5%|▍         | 18602/400000 [00:02<01:01, 6215.55it/s]  5%|▍         | 19347/400000 [00:02<00:58, 6539.40it/s]  5%|▌         | 20211/400000 [00:02<00:53, 7053.12it/s]  5%|▌         | 21130/400000 [00:02<00:49, 7581.49it/s]  5%|▌         | 21999/400000 [00:02<00:47, 7882.98it/s]  6%|▌         | 22847/400000 [00:03<00:46, 8052.24it/s]  6%|▌         | 23732/400000 [00:03<00:45, 8274.53it/s]  6%|▌         | 24599/400000 [00:03<00:44, 8387.36it/s]  6%|▋         | 25456/400000 [00:03<00:46, 8087.83it/s]  7%|▋         | 26295/400000 [00:03<00:45, 8172.51it/s]  7%|▋         | 27138/400000 [00:03<00:45, 8247.45it/s]  7%|▋         | 28050/400000 [00:03<00:43, 8490.44it/s]  7%|▋         | 28947/400000 [00:03<00:43, 8628.16it/s]  7%|▋         | 29816/400000 [00:03<00:43, 8447.55it/s]  8%|▊         | 30666/400000 [00:04<00:47, 7723.03it/s]  8%|▊         | 31454/400000 [00:04<00:47, 7739.08it/s]  8%|▊         | 32250/400000 [00:04<00:47, 7802.30it/s]  8%|▊         | 33043/400000 [00:04<00:46, 7838.24it/s]  8%|▊         | 33833/400000 [00:04<00:46, 7842.88it/s]  9%|▊         | 34622/400000 [00:04<00:46, 7780.72it/s]  9%|▉         | 35460/400000 [00:04<00:45, 7951.28it/s]  9%|▉         | 36258/400000 [00:04<00:45, 7951.65it/s]  9%|▉         | 37064/400000 [00:04<00:45, 7983.58it/s]  9%|▉         | 37891/400000 [00:04<00:44, 8066.16it/s] 10%|▉         | 38699/400000 [00:05<00:45, 7955.93it/s] 10%|▉         | 39507/400000 [00:05<00:45, 7991.44it/s] 10%|█         | 40308/400000 [00:05<00:46, 7817.31it/s] 10%|█         | 41092/400000 [00:05<00:46, 7780.37it/s] 10%|█         | 41907/400000 [00:05<00:45, 7882.83it/s] 11%|█         | 42697/400000 [00:05<00:46, 7747.87it/s] 11%|█         | 43474/400000 [00:05<00:48, 7339.41it/s] 11%|█         | 44214/400000 [00:05<00:49, 7206.95it/s] 11%|█▏        | 45039/400000 [00:05<00:47, 7490.64it/s] 11%|█▏        | 45956/400000 [00:05<00:44, 7924.95it/s] 12%|█▏        | 46786/400000 [00:06<00:43, 8032.97it/s] 12%|█▏        | 47598/400000 [00:06<00:44, 7985.41it/s] 12%|█▏        | 48403/400000 [00:06<00:44, 7845.77it/s] 12%|█▏        | 49193/400000 [00:06<00:47, 7402.09it/s] 12%|█▏        | 49946/400000 [00:06<00:47, 7439.44it/s] 13%|█▎        | 50709/400000 [00:06<00:46, 7494.63it/s] 13%|█▎        | 51503/400000 [00:06<00:45, 7622.19it/s] 13%|█▎        | 52290/400000 [00:06<00:45, 7692.61it/s] 13%|█▎        | 53062/400000 [00:06<00:45, 7627.31it/s] 13%|█▎        | 53834/400000 [00:07<00:45, 7653.98it/s] 14%|█▎        | 54620/400000 [00:07<00:44, 7713.92it/s] 14%|█▍        | 55500/400000 [00:07<00:43, 8009.76it/s] 14%|█▍        | 56330/400000 [00:07<00:42, 8093.71it/s] 14%|█▍        | 57143/400000 [00:07<00:42, 8076.10it/s] 15%|█▍        | 58008/400000 [00:07<00:41, 8238.06it/s] 15%|█▍        | 58866/400000 [00:07<00:40, 8337.72it/s] 15%|█▍        | 59773/400000 [00:07<00:39, 8543.95it/s] 15%|█▌        | 60630/400000 [00:07<00:40, 8308.51it/s] 15%|█▌        | 61475/400000 [00:07<00:40, 8350.15it/s] 16%|█▌        | 62313/400000 [00:08<00:41, 8129.56it/s] 16%|█▌        | 63129/400000 [00:08<00:42, 7865.50it/s] 16%|█▌        | 63920/400000 [00:08<00:43, 7785.26it/s] 16%|█▌        | 64702/400000 [00:08<00:44, 7520.02it/s] 16%|█▋        | 65487/400000 [00:08<00:43, 7614.75it/s] 17%|█▋        | 66273/400000 [00:08<00:43, 7686.09it/s] 17%|█▋        | 67044/400000 [00:08<00:44, 7425.20it/s] 17%|█▋        | 67791/400000 [00:08<00:45, 7361.79it/s] 17%|█▋        | 68530/400000 [00:08<00:45, 7212.64it/s] 17%|█▋        | 69254/400000 [00:08<00:46, 7049.15it/s] 17%|█▋        | 69977/400000 [00:09<00:46, 7102.35it/s] 18%|█▊        | 70727/400000 [00:09<00:45, 7217.04it/s] 18%|█▊        | 71463/400000 [00:09<00:45, 7244.36it/s] 18%|█▊        | 72198/400000 [00:09<00:45, 7273.24it/s] 18%|█▊        | 72941/400000 [00:09<00:44, 7318.05it/s] 18%|█▊        | 73678/400000 [00:09<00:44, 7331.55it/s] 19%|█▊        | 74416/400000 [00:09<00:44, 7343.67it/s] 19%|█▉        | 75156/400000 [00:09<00:44, 7358.41it/s] 19%|█▉        | 75893/400000 [00:09<00:44, 7207.14it/s] 19%|█▉        | 76615/400000 [00:10<00:45, 7054.46it/s] 19%|█▉        | 77322/400000 [00:10<00:46, 6929.68it/s] 20%|█▉        | 78017/400000 [00:10<00:47, 6817.73it/s] 20%|█▉        | 78704/400000 [00:10<00:47, 6830.80it/s] 20%|█▉        | 79428/400000 [00:10<00:46, 6947.70it/s] 20%|██        | 80173/400000 [00:10<00:45, 7089.48it/s] 20%|██        | 80922/400000 [00:10<00:44, 7203.17it/s] 20%|██        | 81644/400000 [00:10<00:44, 7186.47it/s] 21%|██        | 82413/400000 [00:10<00:43, 7329.94it/s] 21%|██        | 83185/400000 [00:10<00:42, 7440.33it/s] 21%|██        | 83937/400000 [00:11<00:42, 7463.89it/s] 21%|██        | 84711/400000 [00:11<00:41, 7544.50it/s] 21%|██▏       | 85494/400000 [00:11<00:41, 7625.93it/s] 22%|██▏       | 86268/400000 [00:11<00:40, 7659.60it/s] 22%|██▏       | 87035/400000 [00:11<00:41, 7618.68it/s] 22%|██▏       | 87798/400000 [00:11<00:41, 7528.54it/s] 22%|██▏       | 88552/400000 [00:11<00:42, 7402.78it/s] 22%|██▏       | 89294/400000 [00:11<00:43, 7159.35it/s] 23%|██▎       | 90013/400000 [00:11<00:44, 7014.76it/s] 23%|██▎       | 90717/400000 [00:11<00:44, 6961.10it/s] 23%|██▎       | 91415/400000 [00:12<00:44, 6949.26it/s] 23%|██▎       | 92112/400000 [00:12<00:44, 6902.99it/s] 23%|██▎       | 92814/400000 [00:12<00:44, 6934.89it/s] 23%|██▎       | 93509/400000 [00:12<00:44, 6929.19it/s] 24%|██▎       | 94238/400000 [00:12<00:43, 7032.71it/s] 24%|██▎       | 94976/400000 [00:12<00:42, 7131.74it/s] 24%|██▍       | 95694/400000 [00:12<00:42, 7143.21it/s] 24%|██▍       | 96409/400000 [00:12<00:43, 7014.54it/s] 24%|██▍       | 97112/400000 [00:12<00:43, 6926.93it/s] 24%|██▍       | 97840/400000 [00:12<00:43, 7026.77it/s] 25%|██▍       | 98544/400000 [00:13<00:43, 6899.93it/s] 25%|██▍       | 99236/400000 [00:13<00:44, 6820.65it/s] 25%|██▍       | 99920/400000 [00:13<00:44, 6776.56it/s] 25%|██▌       | 100599/400000 [00:13<00:44, 6677.71it/s] 25%|██▌       | 101268/400000 [00:13<00:44, 6667.18it/s] 25%|██▌       | 101945/400000 [00:13<00:44, 6695.37it/s] 26%|██▌       | 102615/400000 [00:13<00:44, 6650.42it/s] 26%|██▌       | 103281/400000 [00:13<00:44, 6635.86it/s] 26%|██▌       | 104025/400000 [00:13<00:43, 6856.02it/s] 26%|██▌       | 104780/400000 [00:13<00:41, 7048.27it/s] 26%|██▋       | 105506/400000 [00:14<00:41, 7107.83it/s] 27%|██▋       | 106252/400000 [00:14<00:40, 7209.48it/s] 27%|██▋       | 106975/400000 [00:14<00:40, 7203.48it/s] 27%|██▋       | 107697/400000 [00:14<00:41, 7068.29it/s] 27%|██▋       | 108406/400000 [00:14<00:41, 6982.84it/s] 27%|██▋       | 109106/400000 [00:14<00:42, 6879.83it/s] 27%|██▋       | 109796/400000 [00:14<00:42, 6875.46it/s] 28%|██▊       | 110494/400000 [00:14<00:41, 6904.31it/s] 28%|██▊       | 111186/400000 [00:14<00:42, 6874.60it/s] 28%|██▊       | 111874/400000 [00:15<00:42, 6703.96it/s] 28%|██▊       | 112553/400000 [00:15<00:42, 6728.90it/s] 28%|██▊       | 113227/400000 [00:15<00:42, 6673.59it/s] 28%|██▊       | 113896/400000 [00:15<00:42, 6677.45it/s] 29%|██▊       | 114586/400000 [00:15<00:42, 6741.92it/s] 29%|██▉       | 115276/400000 [00:15<00:41, 6779.59it/s] 29%|██▉       | 115961/400000 [00:15<00:41, 6797.75it/s] 29%|██▉       | 116642/400000 [00:15<00:42, 6593.85it/s] 29%|██▉       | 117303/400000 [00:15<00:43, 6564.03it/s] 30%|██▉       | 118009/400000 [00:15<00:42, 6704.13it/s] 30%|██▉       | 118700/400000 [00:16<00:41, 6762.41it/s] 30%|██▉       | 119386/400000 [00:16<00:41, 6789.77it/s] 30%|███       | 120087/400000 [00:16<00:40, 6853.67it/s] 30%|███       | 120818/400000 [00:16<00:39, 6984.38it/s] 30%|███       | 121593/400000 [00:16<00:38, 7196.84it/s] 31%|███       | 122320/400000 [00:16<00:38, 7215.71it/s] 31%|███       | 123044/400000 [00:16<00:38, 7139.29it/s] 31%|███       | 123760/400000 [00:16<00:39, 7060.55it/s] 31%|███       | 124468/400000 [00:16<00:39, 7023.10it/s] 31%|███▏      | 125172/400000 [00:16<00:39, 6952.09it/s] 31%|███▏      | 125877/400000 [00:17<00:39, 6978.49it/s] 32%|███▏      | 126577/400000 [00:17<00:39, 6981.93it/s] 32%|███▏      | 127276/400000 [00:17<00:39, 6904.33it/s] 32%|███▏      | 127974/400000 [00:17<00:39, 6925.00it/s] 32%|███▏      | 128667/400000 [00:17<00:39, 6919.50it/s] 32%|███▏      | 129360/400000 [00:17<00:39, 6767.05it/s] 33%|███▎      | 130061/400000 [00:17<00:39, 6835.94it/s] 33%|███▎      | 130747/400000 [00:17<00:39, 6840.57it/s] 33%|███▎      | 131433/400000 [00:17<00:39, 6845.24it/s] 33%|███▎      | 132150/400000 [00:17<00:38, 6937.00it/s] 33%|███▎      | 132903/400000 [00:18<00:37, 7104.50it/s] 33%|███▎      | 133639/400000 [00:18<00:37, 7177.32it/s] 34%|███▎      | 134389/400000 [00:18<00:36, 7270.78it/s] 34%|███▍      | 135137/400000 [00:18<00:36, 7330.39it/s] 34%|███▍      | 135903/400000 [00:18<00:35, 7425.55it/s] 34%|███▍      | 136653/400000 [00:18<00:35, 7446.03it/s] 34%|███▍      | 137404/400000 [00:18<00:35, 7463.97it/s] 35%|███▍      | 138190/400000 [00:18<00:34, 7578.48it/s] 35%|███▍      | 138995/400000 [00:18<00:33, 7713.72it/s] 35%|███▍      | 139825/400000 [00:18<00:33, 7880.60it/s] 35%|███▌      | 140650/400000 [00:19<00:32, 7986.78it/s] 35%|███▌      | 141451/400000 [00:19<00:32, 7962.25it/s] 36%|███▌      | 142249/400000 [00:19<00:32, 7862.07it/s] 36%|███▌      | 143037/400000 [00:19<00:34, 7516.51it/s] 36%|███▌      | 143793/400000 [00:19<00:34, 7402.73it/s] 36%|███▌      | 144537/400000 [00:19<00:35, 7283.37it/s] 36%|███▋      | 145268/400000 [00:19<00:35, 7193.21it/s] 36%|███▋      | 145990/400000 [00:19<00:36, 7024.93it/s] 37%|███▋      | 146695/400000 [00:19<00:36, 6965.54it/s] 37%|███▋      | 147413/400000 [00:20<00:35, 7025.83it/s] 37%|███▋      | 148117/400000 [00:20<00:36, 6981.80it/s] 37%|███▋      | 148824/400000 [00:20<00:35, 7007.53it/s] 37%|███▋      | 149526/400000 [00:20<00:35, 6982.70it/s] 38%|███▊      | 150225/400000 [00:20<00:36, 6905.48it/s] 38%|███▊      | 150917/400000 [00:20<00:36, 6903.50it/s] 38%|███▊      | 151657/400000 [00:20<00:35, 7043.14it/s] 38%|███▊      | 152429/400000 [00:20<00:34, 7232.06it/s] 38%|███▊      | 153171/400000 [00:20<00:33, 7285.55it/s] 38%|███▊      | 153945/400000 [00:20<00:33, 7415.30it/s] 39%|███▊      | 154750/400000 [00:21<00:32, 7594.03it/s] 39%|███▉      | 155552/400000 [00:21<00:31, 7714.56it/s] 39%|███▉      | 156374/400000 [00:21<00:31, 7858.21it/s] 39%|███▉      | 157162/400000 [00:21<00:31, 7625.52it/s] 39%|███▉      | 157928/400000 [00:21<00:32, 7536.89it/s] 40%|███▉      | 158684/400000 [00:21<00:32, 7443.07it/s] 40%|███▉      | 159431/400000 [00:21<00:32, 7340.73it/s] 40%|████      | 160167/400000 [00:21<00:33, 7264.02it/s] 40%|████      | 160906/400000 [00:21<00:32, 7299.70it/s] 40%|████      | 161750/400000 [00:21<00:31, 7607.59it/s] 41%|████      | 162574/400000 [00:22<00:30, 7785.08it/s] 41%|████      | 163376/400000 [00:22<00:30, 7853.85it/s] 41%|████      | 164165/400000 [00:22<00:30, 7832.80it/s] 41%|████      | 164951/400000 [00:22<00:30, 7736.71it/s] 41%|████▏     | 165796/400000 [00:22<00:29, 7934.22it/s] 42%|████▏     | 166592/400000 [00:22<00:30, 7537.06it/s] 42%|████▏     | 167363/400000 [00:22<00:30, 7585.59it/s] 42%|████▏     | 168126/400000 [00:22<00:31, 7460.31it/s] 42%|████▏     | 168903/400000 [00:22<00:30, 7550.59it/s] 42%|████▏     | 169705/400000 [00:22<00:29, 7684.55it/s] 43%|████▎     | 170476/400000 [00:23<00:29, 7656.92it/s] 43%|████▎     | 171263/400000 [00:23<00:29, 7718.83it/s] 43%|████▎     | 172037/400000 [00:23<00:29, 7667.02it/s] 43%|████▎     | 172805/400000 [00:23<00:29, 7652.84it/s] 43%|████▎     | 173591/400000 [00:23<00:29, 7711.53it/s] 44%|████▎     | 174399/400000 [00:23<00:28, 7816.82it/s] 44%|████▍     | 175182/400000 [00:23<00:28, 7810.12it/s] 44%|████▍     | 175970/400000 [00:23<00:28, 7830.91it/s] 44%|████▍     | 176754/400000 [00:23<00:30, 7366.29it/s] 44%|████▍     | 177497/400000 [00:24<00:30, 7243.21it/s] 45%|████▍     | 178226/400000 [00:24<00:30, 7160.81it/s] 45%|████▍     | 178946/400000 [00:24<00:32, 6813.20it/s] 45%|████▍     | 179636/400000 [00:24<00:32, 6838.20it/s] 45%|████▌     | 180325/400000 [00:24<00:32, 6831.72it/s] 45%|████▌     | 181054/400000 [00:24<00:31, 6962.55it/s] 45%|████▌     | 181852/400000 [00:24<00:30, 7237.44it/s] 46%|████▌     | 182628/400000 [00:24<00:29, 7386.39it/s] 46%|████▌     | 183371/400000 [00:24<00:30, 7193.85it/s] 46%|████▌     | 184095/400000 [00:24<00:30, 7157.36it/s] 46%|████▌     | 184814/400000 [00:25<00:30, 7149.95it/s] 46%|████▋     | 185531/400000 [00:25<00:29, 7149.05it/s] 47%|████▋     | 186248/400000 [00:25<00:30, 7099.90it/s] 47%|████▋     | 186960/400000 [00:25<00:30, 6980.18it/s] 47%|████▋     | 187717/400000 [00:25<00:29, 7144.90it/s] 47%|████▋     | 188509/400000 [00:25<00:28, 7359.04it/s] 47%|████▋     | 189263/400000 [00:25<00:28, 7411.53it/s] 48%|████▊     | 190007/400000 [00:25<00:28, 7362.50it/s] 48%|████▊     | 190745/400000 [00:25<00:29, 7039.14it/s] 48%|████▊     | 191453/400000 [00:25<00:30, 6742.95it/s] 48%|████▊     | 192142/400000 [00:26<00:30, 6785.39it/s] 48%|████▊     | 192833/400000 [00:26<00:30, 6820.32it/s] 48%|████▊     | 193608/400000 [00:26<00:29, 7073.00it/s] 49%|████▊     | 194372/400000 [00:26<00:28, 7232.82it/s] 49%|████▉     | 195196/400000 [00:26<00:27, 7507.76it/s] 49%|████▉     | 196071/400000 [00:26<00:26, 7840.34it/s] 49%|████▉     | 196920/400000 [00:26<00:25, 8022.46it/s] 49%|████▉     | 197730/400000 [00:26<00:25, 8045.41it/s] 50%|████▉     | 198540/400000 [00:26<00:25, 7847.85it/s] 50%|████▉     | 199329/400000 [00:27<00:26, 7691.54it/s] 50%|█████     | 200102/400000 [00:27<00:26, 7659.05it/s] 50%|█████     | 200871/400000 [00:27<00:26, 7621.32it/s] 50%|█████     | 201635/400000 [00:27<00:26, 7582.99it/s] 51%|█████     | 202395/400000 [00:27<00:26, 7460.90it/s] 51%|█████     | 203143/400000 [00:27<00:26, 7434.94it/s] 51%|█████     | 203930/400000 [00:27<00:25, 7560.26it/s] 51%|█████     | 204812/400000 [00:27<00:24, 7897.58it/s] 51%|█████▏    | 205686/400000 [00:27<00:23, 8132.30it/s] 52%|█████▏    | 206505/400000 [00:27<00:23, 8144.13it/s] 52%|█████▏    | 207373/400000 [00:28<00:23, 8296.01it/s] 52%|█████▏    | 208234/400000 [00:28<00:22, 8385.63it/s] 52%|█████▏    | 209107/400000 [00:28<00:22, 8485.11it/s] 52%|█████▏    | 209958/400000 [00:28<00:22, 8391.57it/s] 53%|█████▎    | 210799/400000 [00:28<00:23, 7927.45it/s] 53%|█████▎    | 211599/400000 [00:28<00:24, 7728.66it/s] 53%|█████▎    | 212378/400000 [00:28<00:24, 7742.93it/s] 53%|█████▎    | 213185/400000 [00:28<00:23, 7835.50it/s] 53%|█████▎    | 213981/400000 [00:28<00:23, 7869.63it/s] 54%|█████▎    | 214771/400000 [00:28<00:23, 7790.53it/s] 54%|█████▍    | 215552/400000 [00:29<00:23, 7719.57it/s] 54%|█████▍    | 216326/400000 [00:29<00:24, 7467.70it/s] 54%|█████▍    | 217076/400000 [00:29<00:25, 7198.73it/s] 54%|█████▍    | 217826/400000 [00:29<00:25, 7286.40it/s] 55%|█████▍    | 218643/400000 [00:29<00:24, 7529.48it/s] 55%|█████▍    | 219453/400000 [00:29<00:23, 7690.64it/s] 55%|█████▌    | 220354/400000 [00:29<00:22, 8042.96it/s] 55%|█████▌    | 221252/400000 [00:29<00:21, 8301.83it/s] 56%|█████▌    | 222119/400000 [00:29<00:21, 8405.10it/s] 56%|█████▌    | 222982/400000 [00:29<00:20, 8469.70it/s] 56%|█████▌    | 223833/400000 [00:30<00:20, 8436.54it/s] 56%|█████▌    | 224705/400000 [00:30<00:20, 8517.21it/s] 56%|█████▋    | 225559/400000 [00:30<00:21, 8272.72it/s] 57%|█████▋    | 226390/400000 [00:30<00:21, 8118.27it/s] 57%|█████▋    | 227205/400000 [00:30<00:21, 8059.87it/s] 57%|█████▋    | 228013/400000 [00:30<00:21, 7943.90it/s] 57%|█████▋    | 228810/400000 [00:30<00:21, 7915.80it/s] 57%|█████▋    | 229603/400000 [00:30<00:21, 7903.71it/s] 58%|█████▊    | 230395/400000 [00:30<00:22, 7641.03it/s] 58%|█████▊    | 231230/400000 [00:31<00:21, 7840.48it/s] 58%|█████▊    | 232093/400000 [00:31<00:20, 8059.56it/s] 58%|█████▊    | 232956/400000 [00:31<00:20, 8222.52it/s] 58%|█████▊    | 233785/400000 [00:31<00:20, 8240.65it/s] 59%|█████▊    | 234612/400000 [00:31<00:20, 8114.15it/s] 59%|█████▉    | 235465/400000 [00:31<00:19, 8231.90it/s] 59%|█████▉    | 236291/400000 [00:31<00:20, 8077.26it/s] 59%|█████▉    | 237101/400000 [00:31<00:20, 7965.16it/s] 59%|█████▉    | 237900/400000 [00:31<00:20, 7843.87it/s] 60%|█████▉    | 238687/400000 [00:31<00:21, 7648.50it/s] 60%|█████▉    | 239455/400000 [00:32<00:21, 7601.82it/s] 60%|██████    | 240221/400000 [00:32<00:20, 7613.47it/s] 60%|██████    | 240984/400000 [00:32<00:21, 7545.62it/s] 60%|██████    | 241740/400000 [00:32<00:20, 7538.00it/s] 61%|██████    | 242495/400000 [00:32<00:21, 7303.83it/s] 61%|██████    | 243271/400000 [00:32<00:21, 7434.17it/s] 61%|██████    | 244079/400000 [00:32<00:20, 7615.99it/s] 61%|██████    | 244886/400000 [00:32<00:20, 7744.95it/s] 61%|██████▏   | 245663/400000 [00:32<00:20, 7694.99it/s] 62%|██████▏   | 246435/400000 [00:32<00:20, 7546.71it/s] 62%|██████▏   | 247233/400000 [00:33<00:19, 7670.92it/s] 62%|██████▏   | 248002/400000 [00:33<00:19, 7643.42it/s] 62%|██████▏   | 248797/400000 [00:33<00:19, 7705.41it/s] 62%|██████▏   | 249569/400000 [00:33<00:19, 7642.71it/s] 63%|██████▎   | 250335/400000 [00:33<00:19, 7580.36it/s] 63%|██████▎   | 251134/400000 [00:33<00:19, 7698.39it/s] 63%|██████▎   | 251922/400000 [00:33<00:19, 7751.32it/s] 63%|██████▎   | 252732/400000 [00:33<00:18, 7852.72it/s] 63%|██████▎   | 253527/400000 [00:33<00:18, 7880.45it/s] 64%|██████▎   | 254341/400000 [00:33<00:18, 7955.67it/s] 64%|██████▍   | 255219/400000 [00:34<00:17, 8184.59it/s] 64%|██████▍   | 256040/400000 [00:34<00:17, 8027.89it/s] 64%|██████▍   | 256845/400000 [00:34<00:18, 7908.75it/s] 64%|██████▍   | 257638/400000 [00:34<00:18, 7691.36it/s] 65%|██████▍   | 258410/400000 [00:34<00:18, 7482.79it/s] 65%|██████▍   | 259226/400000 [00:34<00:18, 7673.22it/s] 65%|██████▌   | 260092/400000 [00:34<00:17, 7943.81it/s] 65%|██████▌   | 260944/400000 [00:34<00:17, 8106.44it/s] 65%|██████▌   | 261804/400000 [00:34<00:16, 8245.92it/s] 66%|██████▌   | 262633/400000 [00:35<00:16, 8144.94it/s] 66%|██████▌   | 263497/400000 [00:35<00:16, 8277.13it/s] 66%|██████▌   | 264328/400000 [00:35<00:16, 8275.66it/s] 66%|██████▋   | 265158/400000 [00:35<00:16, 8281.81it/s] 66%|██████▋   | 265996/400000 [00:35<00:16, 8310.73it/s] 67%|██████▋   | 266828/400000 [00:35<00:16, 7927.17it/s] 67%|██████▋   | 267626/400000 [00:35<00:16, 7798.43it/s] 67%|██████▋   | 268410/400000 [00:35<00:17, 7690.75it/s] 67%|██████▋   | 269182/400000 [00:35<00:17, 7458.29it/s] 67%|██████▋   | 269932/400000 [00:35<00:17, 7259.32it/s] 68%|██████▊   | 270662/400000 [00:36<00:17, 7245.00it/s] 68%|██████▊   | 271396/400000 [00:36<00:17, 7270.73it/s] 68%|██████▊   | 272125/400000 [00:36<00:17, 7271.50it/s] 68%|██████▊   | 272854/400000 [00:36<00:17, 7166.15it/s] 68%|██████▊   | 273582/400000 [00:36<00:17, 7199.59it/s] 69%|██████▊   | 274303/400000 [00:36<00:17, 7194.45it/s] 69%|██████▉   | 275039/400000 [00:36<00:17, 7241.51it/s] 69%|██████▉   | 275771/400000 [00:36<00:17, 7261.71it/s] 69%|██████▉   | 276514/400000 [00:36<00:16, 7310.51it/s] 69%|██████▉   | 277246/400000 [00:36<00:17, 7177.89it/s] 70%|██████▉   | 278004/400000 [00:37<00:16, 7292.22it/s] 70%|██████▉   | 278755/400000 [00:37<00:16, 7355.21it/s] 70%|██████▉   | 279492/400000 [00:37<00:16, 7270.11it/s] 70%|███████   | 280327/400000 [00:37<00:15, 7560.75it/s] 70%|███████   | 281139/400000 [00:37<00:15, 7718.91it/s] 70%|███████   | 281934/400000 [00:37<00:15, 7783.96it/s] 71%|███████   | 282715/400000 [00:37<00:15, 7718.43it/s] 71%|███████   | 283489/400000 [00:37<00:15, 7575.92it/s] 71%|███████   | 284249/400000 [00:37<00:15, 7538.01it/s] 71%|███████▏  | 285013/400000 [00:38<00:15, 7567.00it/s] 71%|███████▏  | 285781/400000 [00:38<00:15, 7598.76it/s] 72%|███████▏  | 286592/400000 [00:38<00:14, 7743.30it/s] 72%|███████▏  | 287368/400000 [00:38<00:14, 7735.83it/s] 72%|███████▏  | 288143/400000 [00:38<00:14, 7602.20it/s] 72%|███████▏  | 288933/400000 [00:38<00:14, 7688.35it/s] 72%|███████▏  | 289776/400000 [00:38<00:13, 7896.09it/s] 73%|███████▎  | 290651/400000 [00:38<00:13, 8133.88it/s] 73%|███████▎  | 291557/400000 [00:38<00:12, 8390.23it/s] 73%|███████▎  | 292405/400000 [00:38<00:12, 8415.79it/s] 73%|███████▎  | 293250/400000 [00:39<00:12, 8255.77it/s] 74%|███████▎  | 294105/400000 [00:39<00:12, 8340.91it/s] 74%|███████▎  | 294982/400000 [00:39<00:12, 8461.67it/s] 74%|███████▍  | 295831/400000 [00:39<00:12, 8304.00it/s] 74%|███████▍  | 296664/400000 [00:39<00:12, 8092.68it/s] 74%|███████▍  | 297476/400000 [00:39<00:13, 7857.18it/s] 75%|███████▍  | 298289/400000 [00:39<00:12, 7935.24it/s] 75%|███████▍  | 299134/400000 [00:39<00:12, 8081.63it/s] 75%|███████▌  | 300015/400000 [00:39<00:12, 8285.25it/s] 75%|███████▌  | 300852/400000 [00:39<00:11, 8310.13it/s] 75%|███████▌  | 301688/400000 [00:40<00:11, 8323.40it/s] 76%|███████▌  | 302522/400000 [00:40<00:11, 8273.27it/s] 76%|███████▌  | 303375/400000 [00:40<00:11, 8346.24it/s] 76%|███████▌  | 304211/400000 [00:40<00:11, 8100.02it/s] 76%|███████▋  | 305024/400000 [00:40<00:11, 7968.80it/s] 76%|███████▋  | 305823/400000 [00:40<00:12, 7590.82it/s] 77%|███████▋  | 306588/400000 [00:40<00:12, 7547.99it/s] 77%|███████▋  | 307435/400000 [00:40<00:11, 7801.63it/s] 77%|███████▋  | 308220/400000 [00:40<00:12, 7553.52it/s] 77%|███████▋  | 308981/400000 [00:41<00:12, 7543.88it/s] 77%|███████▋  | 309788/400000 [00:41<00:11, 7693.08it/s] 78%|███████▊  | 310653/400000 [00:41<00:11, 7956.75it/s] 78%|███████▊  | 311488/400000 [00:41<00:10, 8069.41it/s] 78%|███████▊  | 312327/400000 [00:41<00:10, 8162.62it/s] 78%|███████▊  | 313152/400000 [00:41<00:10, 8186.49it/s] 78%|███████▊  | 313973/400000 [00:41<00:10, 8083.62it/s] 79%|███████▊  | 314784/400000 [00:41<00:10, 7941.27it/s] 79%|███████▉  | 315580/400000 [00:41<00:10, 7877.83it/s] 79%|███████▉  | 316370/400000 [00:41<00:10, 7811.32it/s] 79%|███████▉  | 317153/400000 [00:42<00:10, 7556.04it/s] 79%|███████▉  | 317912/400000 [00:42<00:11, 7397.27it/s] 80%|███████▉  | 318707/400000 [00:42<00:10, 7552.81it/s] 80%|███████▉  | 319483/400000 [00:42<00:10, 7611.96it/s] 80%|████████  | 320247/400000 [00:42<00:10, 7594.05it/s] 80%|████████  | 321018/400000 [00:42<00:10, 7628.18it/s] 80%|████████  | 321782/400000 [00:42<00:10, 7306.94it/s] 81%|████████  | 322532/400000 [00:42<00:10, 7362.82it/s] 81%|████████  | 323314/400000 [00:42<00:10, 7494.21it/s] 81%|████████  | 324109/400000 [00:42<00:09, 7625.10it/s] 81%|████████  | 324892/400000 [00:43<00:09, 7683.39it/s] 81%|████████▏ | 325663/400000 [00:43<00:09, 7557.28it/s] 82%|████████▏ | 326446/400000 [00:43<00:09, 7636.59it/s] 82%|████████▏ | 327211/400000 [00:43<00:09, 7564.04it/s] 82%|████████▏ | 328018/400000 [00:43<00:09, 7707.60it/s] 82%|████████▏ | 328858/400000 [00:43<00:09, 7900.87it/s] 82%|████████▏ | 329717/400000 [00:43<00:08, 8093.92it/s] 83%|████████▎ | 330530/400000 [00:43<00:08, 7963.14it/s] 83%|████████▎ | 331349/400000 [00:43<00:08, 8028.19it/s] 83%|████████▎ | 332185/400000 [00:43<00:08, 8122.02it/s] 83%|████████▎ | 332999/400000 [00:44<00:08, 7983.18it/s] 83%|████████▎ | 333799/400000 [00:44<00:08, 7938.57it/s] 84%|████████▎ | 334595/400000 [00:44<00:08, 7876.37it/s] 84%|████████▍ | 335384/400000 [00:44<00:08, 7381.71it/s] 84%|████████▍ | 336140/400000 [00:44<00:08, 7433.30it/s] 84%|████████▍ | 336896/400000 [00:44<00:08, 7468.59it/s] 84%|████████▍ | 337744/400000 [00:44<00:08, 7744.41it/s] 85%|████████▍ | 338605/400000 [00:44<00:07, 7983.89it/s] 85%|████████▍ | 339421/400000 [00:44<00:07, 8034.21it/s] 85%|████████▌ | 340303/400000 [00:45<00:07, 8253.17it/s] 85%|████████▌ | 341133/400000 [00:45<00:07, 8168.56it/s] 85%|████████▌ | 341973/400000 [00:45<00:07, 8230.27it/s] 86%|████████▌ | 342828/400000 [00:45<00:06, 8321.56it/s] 86%|████████▌ | 343677/400000 [00:45<00:06, 8369.18it/s] 86%|████████▌ | 344553/400000 [00:45<00:06, 8482.13it/s] 86%|████████▋ | 345403/400000 [00:45<00:06, 8442.83it/s] 87%|████████▋ | 346249/400000 [00:45<00:06, 8328.14it/s] 87%|████████▋ | 347083/400000 [00:45<00:06, 8094.60it/s] 87%|████████▋ | 347895/400000 [00:45<00:06, 7920.63it/s] 87%|████████▋ | 348690/400000 [00:46<00:06, 7543.35it/s] 87%|████████▋ | 349450/400000 [00:46<00:07, 7189.39it/s] 88%|████████▊ | 350177/400000 [00:46<00:07, 7093.51it/s] 88%|████████▊ | 350947/400000 [00:46<00:06, 7263.70it/s] 88%|████████▊ | 351679/400000 [00:46<00:06, 7111.84it/s] 88%|████████▊ | 352395/400000 [00:46<00:06, 7115.12it/s] 88%|████████▊ | 353110/400000 [00:46<00:06, 6960.38it/s] 88%|████████▊ | 353809/400000 [00:46<00:06, 6955.20it/s] 89%|████████▊ | 354620/400000 [00:46<00:06, 7265.52it/s] 89%|████████▉ | 355352/400000 [00:46<00:06, 7229.68it/s] 89%|████████▉ | 356079/400000 [00:47<00:06, 7181.18it/s] 89%|████████▉ | 356800/400000 [00:47<00:06, 7114.28it/s] 89%|████████▉ | 357521/400000 [00:47<00:05, 7141.57it/s] 90%|████████▉ | 358265/400000 [00:47<00:05, 7226.33it/s] 90%|████████▉ | 358989/400000 [00:47<00:05, 7202.06it/s] 90%|████████▉ | 359725/400000 [00:47<00:05, 7248.04it/s] 90%|█████████ | 360474/400000 [00:47<00:05, 7318.95it/s] 90%|█████████ | 361287/400000 [00:47<00:05, 7543.74it/s] 91%|█████████ | 362106/400000 [00:47<00:04, 7723.64it/s] 91%|█████████ | 362961/400000 [00:48<00:04, 7953.87it/s] 91%|█████████ | 363760/400000 [00:48<00:04, 7918.92it/s] 91%|█████████ | 364555/400000 [00:48<00:04, 7639.72it/s] 91%|█████████▏| 365323/400000 [00:48<00:04, 7642.85it/s] 92%|█████████▏| 366090/400000 [00:48<00:04, 7605.81it/s] 92%|█████████▏| 366853/400000 [00:48<00:04, 7599.10it/s] 92%|█████████▏| 367635/400000 [00:48<00:04, 7662.38it/s] 92%|█████████▏| 368403/400000 [00:48<00:04, 7349.24it/s] 92%|█████████▏| 369169/400000 [00:48<00:04, 7437.20it/s] 92%|█████████▏| 369951/400000 [00:48<00:03, 7547.72it/s] 93%|█████████▎| 370808/400000 [00:49<00:03, 7824.84it/s] 93%|█████████▎| 371675/400000 [00:49<00:03, 8058.56it/s] 93%|█████████▎| 372486/400000 [00:49<00:03, 7893.89it/s] 93%|█████████▎| 373280/400000 [00:49<00:03, 7736.68it/s] 94%|█████████▎| 374058/400000 [00:49<00:03, 7554.46it/s] 94%|█████████▎| 374817/400000 [00:49<00:03, 7418.62it/s] 94%|█████████▍| 375579/400000 [00:49<00:03, 7474.01it/s] 94%|█████████▍| 376329/400000 [00:49<00:03, 7355.23it/s] 94%|█████████▍| 377067/400000 [00:49<00:03, 7338.24it/s] 94%|█████████▍| 377834/400000 [00:49<00:02, 7434.44it/s] 95%|█████████▍| 378579/400000 [00:50<00:02, 7269.80it/s] 95%|█████████▍| 379308/400000 [00:50<00:02, 7146.31it/s] 95%|█████████▌| 380025/400000 [00:50<00:02, 7014.51it/s] 95%|█████████▌| 380732/400000 [00:50<00:02, 7029.32it/s] 95%|█████████▌| 381437/400000 [00:50<00:02, 6929.77it/s] 96%|█████████▌| 382240/400000 [00:50<00:02, 7226.42it/s] 96%|█████████▌| 383065/400000 [00:50<00:02, 7505.71it/s] 96%|█████████▌| 383863/400000 [00:50<00:02, 7639.43it/s] 96%|█████████▌| 384723/400000 [00:50<00:01, 7902.25it/s] 96%|█████████▋| 385564/400000 [00:50<00:01, 8044.70it/s] 97%|█████████▋| 386389/400000 [00:51<00:01, 8104.16it/s] 97%|█████████▋| 387203/400000 [00:51<00:01, 7787.54it/s] 97%|█████████▋| 387987/400000 [00:51<00:01, 7703.43it/s] 97%|█████████▋| 388761/400000 [00:51<00:01, 7592.96it/s] 97%|█████████▋| 389532/400000 [00:51<00:01, 7627.11it/s] 98%|█████████▊| 390297/400000 [00:51<00:01, 7544.69it/s] 98%|█████████▊| 391054/400000 [00:51<00:01, 7516.61it/s] 98%|█████████▊| 391809/400000 [00:51<00:01, 7525.47it/s] 98%|█████████▊| 392611/400000 [00:51<00:00, 7665.97it/s] 98%|█████████▊| 393440/400000 [00:52<00:00, 7840.85it/s] 99%|█████████▊| 394241/400000 [00:52<00:00, 7889.76it/s] 99%|█████████▉| 395090/400000 [00:52<00:00, 8057.80it/s] 99%|█████████▉| 395898/400000 [00:52<00:00, 8029.33it/s] 99%|█████████▉| 396748/400000 [00:52<00:00, 8162.60it/s] 99%|█████████▉| 397614/400000 [00:52<00:00, 8305.11it/s]100%|█████████▉| 398448/400000 [00:52<00:00, 8313.53it/s]100%|█████████▉| 399292/400000 [00:52<00:00, 8350.68it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7570.36it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f376732a0f0>, <torchtext.data.dataset.TabularDataset object at 0x7f3767612748>, <torchtext.vocab.Vocab object at 0x7f376732a160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.56 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.56 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.64 file/s]2020-06-11 18:19:27.138015: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-11 18:19:27.142491: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-11 18:19:27.142667: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ded7294f00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-11 18:19:27.142683: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  7%|▋         | 688128/9912422 [00:00<00:01, 6879941.51it/s]9920512it [00:00, 33578802.45it/s]                           
0it [00:00, ?it/s]32768it [00:00, 440580.07it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 490230.11it/s]1654784it [00:00, 12299539.70it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202094.71it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
