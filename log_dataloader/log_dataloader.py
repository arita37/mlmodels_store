
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7ce5bab620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7ce5bab620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7d511720d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7d511720d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7d6ae9fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7d6ae9fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cfe9ed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cfe9ed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cfe9ed950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144015.57it/s] 72%|███████▏  | 7159808/9912422 [00:00<00:13, 205558.84it/s]9920512it [00:00, 40391437.41it/s]                           
0it [00:00, ?it/s]32768it [00:00, 500620.88it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 140730.07it/s]1654784it [00:00, 9536896.48it/s]                          
0it [00:00, ?it/s]8192it [00:00, 204907.67it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cfc04fb00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cfc044da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7cfe9ed598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7cfe9ed598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7cfe9ed598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  7.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 47.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.52 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.52 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.97 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ url]
0 examples [00:00, ? examples/s]2020-06-09 18:09:09.125408: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-09 18:09:09.141124: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-09 18:09:09.141311: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557a8bd33150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 18:09:09.141327: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 199.14 examples/s]132 examples [00:00, 264.30 examples/s]250 examples [00:00, 344.25 examples/s]356 examples [00:00, 431.48 examples/s]473 examples [00:00, 531.98 examples/s]586 examples [00:00, 631.81 examples/s]688 examples [00:00, 712.28 examples/s]793 examples [00:00, 787.07 examples/s]902 examples [00:00, 857.32 examples/s]1010 examples [00:01, 912.10 examples/s]1120 examples [00:01, 960.45 examples/s]1228 examples [00:01, 992.31 examples/s]1334 examples [00:01, 999.14 examples/s]1458 examples [00:01, 1060.34 examples/s]1572 examples [00:01, 1082.31 examples/s]1694 examples [00:01, 1118.97 examples/s]1809 examples [00:01, 1097.02 examples/s]1921 examples [00:01, 1097.05 examples/s]2035 examples [00:01, 1107.39 examples/s]2148 examples [00:02, 1113.48 examples/s]2262 examples [00:02, 1120.03 examples/s]2381 examples [00:02, 1139.52 examples/s]2499 examples [00:02, 1148.93 examples/s]2615 examples [00:02, 1144.70 examples/s]2736 examples [00:02, 1161.94 examples/s]2853 examples [00:02, 1092.92 examples/s]2964 examples [00:02, 1095.22 examples/s]3077 examples [00:02, 1103.58 examples/s]3198 examples [00:02, 1132.71 examples/s]3314 examples [00:03, 1138.85 examples/s]3436 examples [00:03, 1159.61 examples/s]3560 examples [00:03, 1181.54 examples/s]3681 examples [00:03, 1189.30 examples/s]3801 examples [00:03, 1173.52 examples/s]3920 examples [00:03, 1177.15 examples/s]4041 examples [00:03, 1184.06 examples/s]4160 examples [00:03, 1157.17 examples/s]4276 examples [00:03, 1149.99 examples/s]4394 examples [00:03, 1158.71 examples/s]4523 examples [00:04, 1195.17 examples/s]4648 examples [00:04, 1208.29 examples/s]4770 examples [00:04, 1182.40 examples/s]4889 examples [00:04, 1129.86 examples/s]5009 examples [00:04, 1148.08 examples/s]5132 examples [00:04, 1170.49 examples/s]5250 examples [00:04, 1150.35 examples/s]5371 examples [00:04, 1165.15 examples/s]5488 examples [00:04, 1146.41 examples/s]5603 examples [00:05, 1120.65 examples/s]5716 examples [00:05, 1104.35 examples/s]5833 examples [00:05, 1120.54 examples/s]5954 examples [00:05, 1145.63 examples/s]6069 examples [00:05, 1130.08 examples/s]6187 examples [00:05, 1142.31 examples/s]6302 examples [00:05, 1131.43 examples/s]6416 examples [00:05, 1120.17 examples/s]6532 examples [00:05, 1131.29 examples/s]6646 examples [00:05, 1127.35 examples/s]6759 examples [00:06, 1123.81 examples/s]6872 examples [00:06, 1096.37 examples/s]6992 examples [00:06, 1125.26 examples/s]7114 examples [00:06, 1150.26 examples/s]7235 examples [00:06, 1165.85 examples/s]7357 examples [00:06, 1179.06 examples/s]7478 examples [00:06, 1186.83 examples/s]7597 examples [00:06, 1166.27 examples/s]7714 examples [00:06, 1132.02 examples/s]7828 examples [00:06, 1126.19 examples/s]7946 examples [00:07, 1139.12 examples/s]8066 examples [00:07, 1155.28 examples/s]8192 examples [00:07, 1183.97 examples/s]8311 examples [00:07, 1151.80 examples/s]8427 examples [00:07, 1135.55 examples/s]8548 examples [00:07, 1154.48 examples/s]8664 examples [00:07, 1138.39 examples/s]8779 examples [00:07, 1115.93 examples/s]8891 examples [00:07, 1112.61 examples/s]9003 examples [00:08, 1112.83 examples/s]9119 examples [00:08, 1125.75 examples/s]9239 examples [00:08, 1145.88 examples/s]9357 examples [00:08, 1153.46 examples/s]9473 examples [00:08, 1093.66 examples/s]9584 examples [00:08, 1087.38 examples/s]9708 examples [00:08, 1128.45 examples/s]9822 examples [00:08, 1102.21 examples/s]9935 examples [00:08, 1110.27 examples/s]10047 examples [00:08, 1067.99 examples/s]10164 examples [00:09, 1096.57 examples/s]10288 examples [00:09, 1135.62 examples/s]10415 examples [00:09, 1169.83 examples/s]10533 examples [00:09, 1144.75 examples/s]10649 examples [00:09, 1127.42 examples/s]10774 examples [00:09, 1159.92 examples/s]10895 examples [00:09, 1172.18 examples/s]11013 examples [00:09, 1133.82 examples/s]11133 examples [00:09, 1151.48 examples/s]11249 examples [00:09, 1117.49 examples/s]11362 examples [00:10, 1109.58 examples/s]11474 examples [00:10, 1101.52 examples/s]11585 examples [00:10, 1101.87 examples/s]11696 examples [00:10, 1088.80 examples/s]11806 examples [00:10, 1061.75 examples/s]11917 examples [00:10, 1075.06 examples/s]12026 examples [00:10, 1077.79 examples/s]12136 examples [00:10, 1083.26 examples/s]12247 examples [00:10, 1090.59 examples/s]12357 examples [00:11, 1089.41 examples/s]12468 examples [00:11, 1093.31 examples/s]12582 examples [00:11, 1106.73 examples/s]12712 examples [00:11, 1157.56 examples/s]12829 examples [00:11, 1151.94 examples/s]12948 examples [00:11, 1161.26 examples/s]13065 examples [00:11, 1147.04 examples/s]13180 examples [00:11, 1125.39 examples/s]13293 examples [00:11, 1109.00 examples/s]13405 examples [00:11, 1109.17 examples/s]13526 examples [00:12, 1136.81 examples/s]13643 examples [00:12, 1144.33 examples/s]13761 examples [00:12, 1154.29 examples/s]13877 examples [00:12, 1113.82 examples/s]13994 examples [00:12, 1129.67 examples/s]14108 examples [00:12, 1130.38 examples/s]14222 examples [00:12, 1120.27 examples/s]14344 examples [00:12, 1146.26 examples/s]14459 examples [00:12, 1143.20 examples/s]14574 examples [00:12, 1128.18 examples/s]14689 examples [00:13, 1133.64 examples/s]14811 examples [00:13, 1156.64 examples/s]14927 examples [00:13, 1143.12 examples/s]15042 examples [00:13, 1142.12 examples/s]15157 examples [00:13, 1116.79 examples/s]15269 examples [00:13, 1099.38 examples/s]15380 examples [00:13, 1036.77 examples/s]15486 examples [00:13, 1042.29 examples/s]15594 examples [00:13, 1050.73 examples/s]15709 examples [00:14, 1076.68 examples/s]15831 examples [00:14, 1114.75 examples/s]15951 examples [00:14, 1138.50 examples/s]16067 examples [00:14, 1142.83 examples/s]16182 examples [00:14, 1133.44 examples/s]16302 examples [00:14, 1152.42 examples/s]16425 examples [00:14, 1174.53 examples/s]16552 examples [00:14, 1199.06 examples/s]16673 examples [00:14, 1200.62 examples/s]16794 examples [00:14, 1200.66 examples/s]16915 examples [00:15, 1175.18 examples/s]17033 examples [00:15, 1153.11 examples/s]17149 examples [00:15, 1116.61 examples/s]17265 examples [00:15, 1127.13 examples/s]17379 examples [00:15, 1123.46 examples/s]17492 examples [00:15, 1112.79 examples/s]17604 examples [00:15, 1106.06 examples/s]17715 examples [00:15, 1084.31 examples/s]17826 examples [00:15, 1091.08 examples/s]17936 examples [00:15, 1075.07 examples/s]18048 examples [00:16, 1084.88 examples/s]18157 examples [00:16, 1058.24 examples/s]18275 examples [00:16, 1091.22 examples/s]18398 examples [00:16, 1127.69 examples/s]18522 examples [00:16, 1158.57 examples/s]18641 examples [00:16, 1167.43 examples/s]18759 examples [00:16, 1127.89 examples/s]18873 examples [00:16, 1109.14 examples/s]18985 examples [00:16, 1054.95 examples/s]19092 examples [00:17, 1055.62 examples/s]19216 examples [00:17, 1104.53 examples/s]19328 examples [00:17, 1106.21 examples/s]19453 examples [00:17, 1144.76 examples/s]19573 examples [00:17, 1158.67 examples/s]19690 examples [00:17, 1160.27 examples/s]19807 examples [00:17, 1153.37 examples/s]19923 examples [00:17, 1150.68 examples/s]20039 examples [00:17, 1096.81 examples/s]20150 examples [00:17, 1083.53 examples/s]20266 examples [00:18, 1104.66 examples/s]20394 examples [00:18, 1150.15 examples/s]20523 examples [00:18, 1186.63 examples/s]20646 examples [00:18, 1198.05 examples/s]20767 examples [00:18, 1174.33 examples/s]20885 examples [00:18, 1166.44 examples/s]21003 examples [00:18, 1151.46 examples/s]21119 examples [00:18, 1105.43 examples/s]21231 examples [00:18, 1104.13 examples/s]21343 examples [00:18, 1108.24 examples/s]21460 examples [00:19, 1125.97 examples/s]21587 examples [00:19, 1163.52 examples/s]21704 examples [00:19, 1163.28 examples/s]21821 examples [00:19, 1146.67 examples/s]21943 examples [00:19, 1164.62 examples/s]22066 examples [00:19, 1183.20 examples/s]22185 examples [00:19, 1155.67 examples/s]22312 examples [00:19, 1185.48 examples/s]22440 examples [00:19, 1210.00 examples/s]22562 examples [00:20, 1139.02 examples/s]22678 examples [00:20, 1111.75 examples/s]22791 examples [00:20, 1101.32 examples/s]22902 examples [00:20, 1102.64 examples/s]23013 examples [00:20, 1104.57 examples/s]23129 examples [00:20, 1118.29 examples/s]23242 examples [00:20, 1116.05 examples/s]23354 examples [00:20, 1107.79 examples/s]23471 examples [00:20, 1125.25 examples/s]23584 examples [00:20, 1118.30 examples/s]23696 examples [00:21, 1105.47 examples/s]23809 examples [00:21, 1110.58 examples/s]23921 examples [00:21, 1090.33 examples/s]24054 examples [00:21, 1150.20 examples/s]24179 examples [00:21, 1176.58 examples/s]24298 examples [00:21, 1156.85 examples/s]24417 examples [00:21, 1166.01 examples/s]24535 examples [00:21, 1152.21 examples/s]24651 examples [00:21, 1138.83 examples/s]24766 examples [00:21, 1074.37 examples/s]24883 examples [00:22, 1100.05 examples/s]25007 examples [00:22, 1136.55 examples/s]25128 examples [00:22, 1155.80 examples/s]25253 examples [00:22, 1181.78 examples/s]25381 examples [00:22, 1209.50 examples/s]25503 examples [00:22, 1193.71 examples/s]25623 examples [00:22, 1158.43 examples/s]25740 examples [00:22, 1151.91 examples/s]25856 examples [00:22, 1147.54 examples/s]25972 examples [00:23, 1130.80 examples/s]26097 examples [00:23, 1161.96 examples/s]26215 examples [00:23, 1165.39 examples/s]26333 examples [00:23, 1168.36 examples/s]26451 examples [00:23, 1157.62 examples/s]26573 examples [00:23, 1175.31 examples/s]26697 examples [00:23, 1193.07 examples/s]26817 examples [00:23, 1178.09 examples/s]26935 examples [00:23, 1153.97 examples/s]27059 examples [00:23, 1178.00 examples/s]27179 examples [00:24, 1184.32 examples/s]27303 examples [00:24, 1200.08 examples/s]27426 examples [00:24, 1208.41 examples/s]27547 examples [00:24, 1180.70 examples/s]27666 examples [00:24, 1179.09 examples/s]27785 examples [00:24, 1180.55 examples/s]27905 examples [00:24, 1182.66 examples/s]28032 examples [00:24, 1206.37 examples/s]28153 examples [00:24, 1202.17 examples/s]28275 examples [00:24, 1205.33 examples/s]28396 examples [00:25, 1158.62 examples/s]28513 examples [00:25, 1117.82 examples/s]28641 examples [00:25, 1161.86 examples/s]28759 examples [00:25, 1127.24 examples/s]28873 examples [00:25, 1122.74 examples/s]28986 examples [00:25, 1118.20 examples/s]29099 examples [00:25, 1096.29 examples/s]29210 examples [00:25, 1060.03 examples/s]29320 examples [00:25, 1069.22 examples/s]29431 examples [00:26, 1080.84 examples/s]29540 examples [00:26, 1081.75 examples/s]29651 examples [00:26, 1088.02 examples/s]29768 examples [00:26, 1110.67 examples/s]29889 examples [00:26, 1138.55 examples/s]30004 examples [00:26, 1105.02 examples/s]30116 examples [00:26, 1108.18 examples/s]30229 examples [00:26, 1112.86 examples/s]30341 examples [00:26, 1102.60 examples/s]30452 examples [00:26, 1100.82 examples/s]30566 examples [00:27, 1111.59 examples/s]30678 examples [00:27, 1108.73 examples/s]30789 examples [00:27, 1108.20 examples/s]30900 examples [00:27, 1101.13 examples/s]31020 examples [00:27, 1126.45 examples/s]31145 examples [00:27, 1159.48 examples/s]31262 examples [00:27, 1133.70 examples/s]31383 examples [00:27, 1154.20 examples/s]31506 examples [00:27, 1174.79 examples/s]31624 examples [00:27, 1164.77 examples/s]31746 examples [00:28, 1179.53 examples/s]31865 examples [00:28, 1142.28 examples/s]31992 examples [00:28, 1176.75 examples/s]32121 examples [00:28, 1206.06 examples/s]32243 examples [00:28, 1184.43 examples/s]32362 examples [00:28, 1180.16 examples/s]32481 examples [00:28, 1176.31 examples/s]32599 examples [00:28, 1156.49 examples/s]32715 examples [00:28, 1121.81 examples/s]32828 examples [00:28, 1114.20 examples/s]32940 examples [00:29, 1107.15 examples/s]33052 examples [00:29, 1110.47 examples/s]33164 examples [00:29, 1071.31 examples/s]33272 examples [00:29, 1070.19 examples/s]33390 examples [00:29, 1100.19 examples/s]33503 examples [00:29, 1108.42 examples/s]33621 examples [00:29, 1127.21 examples/s]33746 examples [00:29, 1161.21 examples/s]33872 examples [00:29, 1187.77 examples/s]33992 examples [00:30, 1166.13 examples/s]34110 examples [00:30, 1124.12 examples/s]34224 examples [00:30, 1099.45 examples/s]34339 examples [00:30, 1112.95 examples/s]34454 examples [00:30, 1121.21 examples/s]34572 examples [00:30, 1135.52 examples/s]34686 examples [00:30, 1101.73 examples/s]34797 examples [00:30, 1093.01 examples/s]34913 examples [00:30, 1110.96 examples/s]35025 examples [00:30, 1091.45 examples/s]35135 examples [00:31, 1084.19 examples/s]35246 examples [00:31, 1089.99 examples/s]35363 examples [00:31, 1111.50 examples/s]35475 examples [00:31, 1096.35 examples/s]35591 examples [00:31, 1113.73 examples/s]35717 examples [00:31, 1152.63 examples/s]35833 examples [00:31, 1121.07 examples/s]35946 examples [00:31, 1101.82 examples/s]36057 examples [00:31, 1098.61 examples/s]36168 examples [00:32, 1080.95 examples/s]36292 examples [00:32, 1122.58 examples/s]36411 examples [00:32, 1139.43 examples/s]36530 examples [00:32, 1152.17 examples/s]36655 examples [00:32, 1177.93 examples/s]36774 examples [00:32, 1165.68 examples/s]36891 examples [00:32, 1130.61 examples/s]37005 examples [00:32, 1133.11 examples/s]37119 examples [00:32, 1133.96 examples/s]37241 examples [00:32, 1156.55 examples/s]37357 examples [00:33, 1136.72 examples/s]37473 examples [00:33, 1141.86 examples/s]37596 examples [00:33, 1163.81 examples/s]37717 examples [00:33, 1176.86 examples/s]37843 examples [00:33, 1198.70 examples/s]37970 examples [00:33, 1218.95 examples/s]38093 examples [00:33, 1213.51 examples/s]38215 examples [00:33, 1167.81 examples/s]38333 examples [00:33, 1135.43 examples/s]38449 examples [00:33, 1141.60 examples/s]38568 examples [00:34, 1153.53 examples/s]38684 examples [00:34, 1152.69 examples/s]38807 examples [00:34, 1172.88 examples/s]38925 examples [00:34, 1161.07 examples/s]39044 examples [00:34, 1167.44 examples/s]39161 examples [00:34, 1164.89 examples/s]39278 examples [00:34, 1164.51 examples/s]39405 examples [00:34, 1193.97 examples/s]39533 examples [00:34, 1217.40 examples/s]39656 examples [00:34, 1180.99 examples/s]39775 examples [00:35, 1149.06 examples/s]39891 examples [00:35, 1127.32 examples/s]40005 examples [00:35, 1084.37 examples/s]40118 examples [00:35, 1097.10 examples/s]40236 examples [00:35, 1118.91 examples/s]40352 examples [00:35, 1128.90 examples/s]40466 examples [00:35, 1108.90 examples/s]40578 examples [00:35, 1107.37 examples/s]40689 examples [00:35, 1094.75 examples/s]40799 examples [00:36, 1079.47 examples/s]40908 examples [00:36, 1081.49 examples/s]41017 examples [00:36, 1077.57 examples/s]41135 examples [00:36, 1105.65 examples/s]41254 examples [00:36, 1129.26 examples/s]41385 examples [00:36, 1176.85 examples/s]41504 examples [00:36, 1166.80 examples/s]41622 examples [00:36, 1145.59 examples/s]41738 examples [00:36, 1141.37 examples/s]41853 examples [00:36, 1134.71 examples/s]41973 examples [00:37, 1153.35 examples/s]42098 examples [00:37, 1178.43 examples/s]42219 examples [00:37, 1186.03 examples/s]42338 examples [00:37, 1148.18 examples/s]42460 examples [00:37, 1168.20 examples/s]42578 examples [00:37, 1146.32 examples/s]42693 examples [00:37, 1134.77 examples/s]42807 examples [00:37, 1127.82 examples/s]42927 examples [00:37, 1147.89 examples/s]43052 examples [00:37, 1173.66 examples/s]43170 examples [00:38, 1123.87 examples/s]43291 examples [00:38, 1148.10 examples/s]43408 examples [00:38, 1153.77 examples/s]43524 examples [00:38, 1154.36 examples/s]43640 examples [00:38, 1151.21 examples/s]43756 examples [00:38, 1133.05 examples/s]43870 examples [00:38, 1122.62 examples/s]43983 examples [00:38, 1097.13 examples/s]44095 examples [00:38, 1103.19 examples/s]44206 examples [00:39, 1102.55 examples/s]44320 examples [00:39, 1113.37 examples/s]44433 examples [00:39, 1114.46 examples/s]44545 examples [00:39, 1115.97 examples/s]44657 examples [00:39, 1114.40 examples/s]44776 examples [00:39, 1133.37 examples/s]44895 examples [00:39, 1149.70 examples/s]45013 examples [00:39, 1155.79 examples/s]45136 examples [00:39, 1175.65 examples/s]45258 examples [00:39, 1188.52 examples/s]45377 examples [00:40, 1159.95 examples/s]45494 examples [00:40, 1144.04 examples/s]45609 examples [00:40, 1119.54 examples/s]45729 examples [00:40, 1140.91 examples/s]45844 examples [00:40, 1140.62 examples/s]45959 examples [00:40, 1116.85 examples/s]46071 examples [00:40, 1113.38 examples/s]46183 examples [00:40, 1109.70 examples/s]46295 examples [00:40, 1066.28 examples/s]46403 examples [00:40, 1068.00 examples/s]46511 examples [00:41, 1049.70 examples/s]46617 examples [00:41, 1006.71 examples/s]46732 examples [00:41, 1044.53 examples/s]46846 examples [00:41, 1070.31 examples/s]46963 examples [00:41, 1098.38 examples/s]47088 examples [00:41, 1138.20 examples/s]47204 examples [00:41, 1143.33 examples/s]47319 examples [00:41, 1119.90 examples/s]47432 examples [00:41, 1096.72 examples/s]47543 examples [00:42, 1080.95 examples/s]47663 examples [00:42, 1113.42 examples/s]47787 examples [00:42, 1148.22 examples/s]47903 examples [00:42, 1134.28 examples/s]48020 examples [00:42, 1142.78 examples/s]48135 examples [00:42, 1121.38 examples/s]48248 examples [00:42, 1122.56 examples/s]48373 examples [00:42, 1156.82 examples/s]48490 examples [00:42, 1154.41 examples/s]48616 examples [00:42, 1182.73 examples/s]48736 examples [00:43, 1186.26 examples/s]48855 examples [00:43, 1144.32 examples/s]48970 examples [00:43, 1117.91 examples/s]49083 examples [00:43, 1119.86 examples/s]49199 examples [00:43, 1128.50 examples/s]49325 examples [00:43, 1162.64 examples/s]49450 examples [00:43, 1185.64 examples/s]49569 examples [00:43, 1175.24 examples/s]49687 examples [00:43, 1148.47 examples/s]49804 examples [00:43, 1154.43 examples/s]49929 examples [00:44, 1180.42 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6225/50000 [00:00<00:00, 62249.02 examples/s] 39%|███▉      | 19641/50000 [00:00<00:00, 74176.67 examples/s] 68%|██████▊   | 34233/50000 [00:00<00:00, 87010.00 examples/s] 99%|█████████▉| 49450/50000 [00:00<00:00, 99834.65 examples/s]                                                               0 examples [00:00, ? examples/s]104 examples [00:00, 1039.34 examples/s]231 examples [00:00, 1097.63 examples/s]346 examples [00:00, 1110.85 examples/s]451 examples [00:00, 1091.55 examples/s]562 examples [00:00, 1095.42 examples/s]676 examples [00:00, 1107.72 examples/s]781 examples [00:00, 1087.46 examples/s]891 examples [00:00, 1089.36 examples/s]999 examples [00:00, 1086.33 examples/s]1112 examples [00:01, 1097.82 examples/s]1219 examples [00:01, 1079.80 examples/s]1326 examples [00:01, 1075.21 examples/s]1433 examples [00:01, 838.13 examples/s] 1542 examples [00:01, 898.78 examples/s]1644 examples [00:01, 930.16 examples/s]1744 examples [00:01, 949.62 examples/s]1857 examples [00:01, 996.78 examples/s]1982 examples [00:01, 1059.61 examples/s]2110 examples [00:02, 1117.07 examples/s]2236 examples [00:02, 1154.33 examples/s]2354 examples [00:02, 1135.00 examples/s]2470 examples [00:02, 1113.47 examples/s]2583 examples [00:02, 1108.13 examples/s]2698 examples [00:02, 1117.21 examples/s]2811 examples [00:02, 1094.81 examples/s]2922 examples [00:02, 1086.91 examples/s]3035 examples [00:02, 1098.62 examples/s]3146 examples [00:02, 1083.09 examples/s]3255 examples [00:03, 1048.19 examples/s]3367 examples [00:03, 1067.92 examples/s]3478 examples [00:03, 1079.03 examples/s]3588 examples [00:03, 1084.43 examples/s]3701 examples [00:03, 1095.75 examples/s]3811 examples [00:03, 1091.69 examples/s]3928 examples [00:03, 1111.72 examples/s]4041 examples [00:03, 1115.14 examples/s]4153 examples [00:03, 1106.35 examples/s]4268 examples [00:03, 1117.73 examples/s]4382 examples [00:04, 1122.63 examples/s]4495 examples [00:04, 1096.11 examples/s]4605 examples [00:04, 1070.40 examples/s]4715 examples [00:04, 1077.47 examples/s]4829 examples [00:04, 1093.24 examples/s]4943 examples [00:04, 1105.02 examples/s]5064 examples [00:04, 1133.49 examples/s]5178 examples [00:04, 1130.48 examples/s]5295 examples [00:04, 1141.48 examples/s]5415 examples [00:04, 1156.12 examples/s]5531 examples [00:05, 1130.31 examples/s]5645 examples [00:05, 1117.85 examples/s]5757 examples [00:05, 1092.29 examples/s]5875 examples [00:05, 1115.52 examples/s]5988 examples [00:05, 1118.34 examples/s]6101 examples [00:05, 1120.77 examples/s]6214 examples [00:05, 1108.92 examples/s]6326 examples [00:05, 1088.71 examples/s]6436 examples [00:05, 1091.58 examples/s]6558 examples [00:06, 1125.71 examples/s]6671 examples [00:06, 1117.69 examples/s]6784 examples [00:06, 1116.76 examples/s]6901 examples [00:06, 1129.78 examples/s]7015 examples [00:06, 1112.53 examples/s]7127 examples [00:06, 1102.16 examples/s]7239 examples [00:06, 1106.01 examples/s]7362 examples [00:06, 1138.29 examples/s]7483 examples [00:06, 1157.83 examples/s]7600 examples [00:06, 1154.32 examples/s]7716 examples [00:07, 1154.47 examples/s]7838 examples [00:07, 1171.34 examples/s]7956 examples [00:07, 1146.35 examples/s]8071 examples [00:07, 1124.48 examples/s]8184 examples [00:07, 1095.41 examples/s]8299 examples [00:07, 1110.89 examples/s]8424 examples [00:07, 1148.11 examples/s]8554 examples [00:07, 1188.29 examples/s]8674 examples [00:07, 1163.42 examples/s]8792 examples [00:07, 1165.97 examples/s]8910 examples [00:08, 1155.77 examples/s]9028 examples [00:08, 1160.37 examples/s]9150 examples [00:08, 1176.37 examples/s]9268 examples [00:08, 1164.82 examples/s]9385 examples [00:08, 1155.51 examples/s]9501 examples [00:08, 1156.32 examples/s]9618 examples [00:08, 1158.90 examples/s]9743 examples [00:08, 1183.92 examples/s]9862 examples [00:08, 1132.87 examples/s]9980 examples [00:09, 1146.26 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJQ3VVC/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJQ3VVC/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cfe9ed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cfe9ed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cfe9ed950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c88f3a860>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c88bcf198>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cfe9ed950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cfe9ed950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cfe9ed950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7cfe9dac50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7ce4dac128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7c763b9268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7c763b9268> 

  function with postional parmater data_info <function split_train_valid at 0x7f7c763b9268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7c763b9378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7c763b9378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7c763b9378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7e838dea0fbd4bf80fb084ba81e017148ca0fbb7f513bf519e1433c1c738ac41
  Stored in directory: /tmp/pip-ephem-wheel-cache-re4kt5eh/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<128:12:42, 1.87kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<89:58:35, 2.66kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:04<63:01:36, 3.80kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:04<44:05:53, 5.43kB/s].vector_cache/glove.6B.zip:   0%|          | 3.63M/862M [00:04<30:46:26, 7.75kB/s].vector_cache/glove.6B.zip:   1%|          | 6.39M/862M [00:05<21:28:34, 11.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:05<14:56:01, 15.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:05<10:24:41, 22.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.4M/862M [00:05<7:14:58, 32.3kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.8M/862M [00:05<5:03:00, 46.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:05<3:31:00, 65.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.3M/862M [00:05<2:27:08, 93.9kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:05<1:42:30, 134kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 41.9M/862M [00:05<1:11:31, 191kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:05<49:50, 273kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:06<34:53, 388kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:06<24:57, 541kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:08<19:18, 696kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:08<17:11, 781kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:08<12:49, 1.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:08<09:07, 1.47MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:10<11:58, 1.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:10<09:53, 1.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:10<07:17, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:12<07:54, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:12<08:22, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:12<06:27, 2.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:12<04:40, 2.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:14<10:47, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:14<08:54, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:14<06:33, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:16<07:40, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:16<08:03, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:16<06:13, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:16<04:29, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:18<12:55, 1.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:18<10:22, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:18<07:35, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:20<08:21, 1.56MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:20<08:29, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:20<06:36, 1.97MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:20<04:45, 2.72MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:22<53:19, 243kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:22<38:37, 335kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:22<27:18, 473kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:24<22:05, 583kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:24<18:04, 713kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:24<13:11, 976kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:24<09:22, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:26<13:20, 960kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<10:40, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:26<07:46, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:28<08:25, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:28<08:36, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:28<06:40, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<04:49, 2.63MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<1:34:04, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<1:07:08, 189kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<47:10, 268kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<35:54, 351kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<27:42, 455kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<20:00, 630kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<15:59, 785kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<13:44, 913kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<10:09, 1.23MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<07:16, 1.72MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<10:43, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<08:48, 1.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<06:25, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<07:23, 1.68MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<07:40, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<05:54, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<04:16, 2.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<11:53, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<09:36, 1.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<07:01, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<07:46, 1.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<07:55, 1.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<06:07, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<04:25, 2.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<1:31:03, 134kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<1:04:58, 188kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<45:41, 266kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<34:42, 350kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<26:45, 453kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<19:19, 627kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:47<15:25, 782kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<12:01, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:47<08:42, 1.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:49<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<07:25, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<06:39, 1.79MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<07:04, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<05:29, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<03:58, 2.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:53<1:28:29, 134kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<1:03:07, 188kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<44:20, 267kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<33:43, 350kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<25:59, 453kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<18:42, 630kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<13:11, 889kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<16:05, 729kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<12:28, 938kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<09:01, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<09:00, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<07:29, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<05:31, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<06:34, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<06:51, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<05:17, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:01<03:50, 3.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<09:11, 1.25MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<07:38, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<05:36, 2.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:05<06:35, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<06:56, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<05:26, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:07<05:38, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:06, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<03:51, 2.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:02, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<04:43, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<03:26, 3.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<07:54, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<06:41, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<04:57, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<06:04, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<06:37, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:12, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<03:46, 2.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<1:22:06, 135kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<58:35, 189kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<41:12, 269kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<31:19, 352kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<24:09, 457kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<17:27, 631kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<13:56, 787kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:19<10:51, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<07:52, 1.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<08:03, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<06:44, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<04:58, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<06:02, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<06:26, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<04:59, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<03:36, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<10:34, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<08:31, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<06:11, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<06:49, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<05:51, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<04:19, 2.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<05:33, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<04:57, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<03:44, 2.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<05:06, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<04:27, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<03:13, 3.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<10:59, 953kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:32<08:45, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<06:20, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:34<06:52, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<06:55, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<05:22, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<04:50, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<03:36, 2.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:38<04:56, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:38<04:29, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<03:23, 3.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<04:47, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<05:25, 1.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<04:18, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:39, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<04:07, 2.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<03:08, 3.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<04:35, 2.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:44<05:15, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:44<04:10, 2.41MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<04:32, 2.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<04:12, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<03:09, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<04:10, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<03:07, 3.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<04:32, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<05:11, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<04:07, 2.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<04:27, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<04:07, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<03:06, 3.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<03:07, 3.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<04:27, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<05:05, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<03:58, 2.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<02:54, 3.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<06:48, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<05:44, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<04:15, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<05:12, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<04:36, 2.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<03:27, 2.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<04:38, 2.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:09, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<04:00, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<02:54, 3.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<09:09, 1.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<07:22, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<05:23, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<05:56, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<06:04, 1.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<04:43, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<03:23, 2.73MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<8:57:26, 17.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<6:16:52, 24.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<4:23:18, 35.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<3:05:45, 49.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<2:11:50, 69.6kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<1:32:34, 99.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<1:04:38, 141kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<50:28, 181kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<36:14, 251kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<25:29, 356kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<19:54, 454kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<14:51, 609kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<10:33, 854kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:15<09:30, 944kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<08:28, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<06:18, 1.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<04:31, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<08:22, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<06:46, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:19<05:30, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<04:45, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:20<03:32, 2.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:33, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<04:03, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:22<03:03, 2.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:11, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:42, 1.85MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<03:39, 2.38MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<02:40, 3.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<06:18, 1.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<05:17, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<03:54, 2.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<04:44, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<05:03, 1.69MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<03:54, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<02:49, 3.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:29<06:31, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<05:25, 1.57MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<03:58, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<05:02, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<03:53, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:31<02:49, 2.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<07:13, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<05:55, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:18, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<04:57, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<05:09, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<04:01, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<04:08, 1.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:37<03:43, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<02:47, 2.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<03:51, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<04:22, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<03:23, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<02:29, 3.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<05:15, 1.54MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:41<04:30, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<03:21, 2.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<04:13, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<04:39, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:43<03:39, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<02:39, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<58:41, 135kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<41:52, 190kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<29:25, 269kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<22:21, 353kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<17:14, 457kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<12:26, 632kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<09:55, 787kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:49<07:45, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<05:36, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<05:43, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:51<04:47, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<03:32, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<04:33, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:53<03:30, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<02:33, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<05:21, 1.42MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<04:33, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<03:20, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<04:06, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<04:23, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<03:27, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<02:29, 3.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<7:13:57, 17.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<5:04:15, 24.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<3:32:25, 35.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<2:29:43, 49.4kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:01<1:46:24, 69.5kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<1:14:46, 98.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<52:05, 141kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<44:17, 166kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<31:47, 230kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<22:21, 327kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<17:09, 423kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:05<13:36, 533kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:05<09:51, 735kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<06:58, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<07:48, 921kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<06:14, 1.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<04:31, 1.58MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<04:43, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:09<04:52, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:48, 1.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<02:44, 2.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<10:27, 674kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<08:05, 871kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<05:49, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<05:34, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:13<04:39, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<03:26, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<02:35, 2.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<03:19, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<03:50, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<03:00, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<02:13, 3.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<03:34, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<03:15, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<02:27, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<03:10, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<02:58, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<02:15, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<03:01, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<03:35, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<02:52, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<02:04, 3.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<08:29, 773kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<06:39, 986kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<04:47, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<04:46, 1.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<04:02, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<02:58, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<03:28, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<03:51, 1.66MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<03:03, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<02:12, 2.89MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<08:29, 749kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<06:37, 959kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<04:47, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<04:43, 1.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<02:38, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<08:27, 736kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<06:35, 942kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<04:46, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<04:40, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<04:37, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<03:31, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<02:31, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<04:46, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<03:59, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<02:55, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:21, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:40, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<02:54, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<02:05, 2.84MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<07:47, 762kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<06:06, 971kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<04:25, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<04:21, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<04:21, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<02:25, 2.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<08:43, 665kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<06:44, 859kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:47<04:51, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:38, 1.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:30, 1.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<03:25, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<02:28, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<03:49, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<03:17, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:25, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:55, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<03:13, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<02:31, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<01:49, 3.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<05:07, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<04:11, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:54<03:02, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<03:19, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<03:32, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<02:42, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<01:57, 2.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<03:37, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:58<03:07, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:19, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:47, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<03:11, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<02:31, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<01:48, 2.89MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<06:53, 762kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<05:23, 971kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<03:54, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:50, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<03:50, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:58, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:07, 2.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:06<06:57, 733kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<05:25, 941kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<03:53, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<03:49, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<03:44, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:52, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<02:03, 2.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<09:03, 548kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<06:53, 721kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<04:55, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<04:30, 1.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<04:13, 1.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<03:13, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:17, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<06:45, 714kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<05:14, 921kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<03:45, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:41, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<03:35, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:43, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<01:59, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:50, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:29, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<01:51, 2.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:18, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:07, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:35, 2.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:05, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<02:25, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:56, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<01:23, 3.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<11:44, 382kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<08:40, 516kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:24<06:08, 725kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<05:16, 837kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<04:37, 955kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:26<03:27, 1.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<14:46, 294kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<10:47, 402kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<07:37, 566kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<06:15, 684kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<05:19, 803kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<03:55, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<02:46, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<04:04, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<03:19, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:25, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<02:43, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<02:05, 1.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:30, 2.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<02:43, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<02:19, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:42, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<02:19, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<01:49, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<01:18, 2.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<10:19, 380kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<07:38, 514kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<05:24, 720kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:37, 834kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<04:03, 952kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<02:59, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:42<02:08, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<03:13, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<02:39, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:56, 1.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:11, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<02:21, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:46<01:49, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<01:18, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<03:00, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:48<02:30, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<01:50, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:03, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<01:15, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<04:13, 832kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<03:19, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<02:23, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<02:25, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<02:27, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:54, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<01:21, 2.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<04:34, 739kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<03:33, 947kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<02:32, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:29, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<01:54, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<04:24, 734kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:00<03:26, 939kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<02:28, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:24, 1.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<02:22, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<01:50, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<01:18, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<04:13, 732kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:04<03:18, 936kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<02:22, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<02:18, 1.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<01:56, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:38, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:48, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:08<01:23, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<01:00, 2.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:29, 1.92MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<01:07, 2.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:23, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:36, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:12<01:16, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<00:54, 3.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<03:58, 693kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<03:03, 895kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<02:11, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<02:07, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<02:03, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:33, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:06, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<02:11, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<01:49, 1.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:18<01:19, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:28, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:34, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:20<01:13, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<00:52, 2.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<05:53, 419kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:22<04:23, 562kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<03:06, 785kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:40, 900kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<02:22, 1.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:24<01:45, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:24<01:15, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<01:56, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:26<01:36, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<01:10, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:18, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<01:09, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<00:52, 2.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:04, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<00:59, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<00:42, 3.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<03:03, 694kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:32<02:21, 896kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:41, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<01:36, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:34<01:35, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:13, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<00:51, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<02:46, 717kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:36<02:09, 921kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<01:32, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:28, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<01:26, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<01:05, 1.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<00:46, 2.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<04:29, 412kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<03:17, 559kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<02:20, 778kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<01:37, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<03:16, 543kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<02:41, 662kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<01:57, 898kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<01:21, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<02:46, 615kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<02:07, 802kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:30, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<01:23, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<01:08, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:49, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:55, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<00:36, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:44, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:40, 2.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:50<00:30, 2.92MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:40, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:51<00:48, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<00:38, 2.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:27, 3.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:53<01:57, 696kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<01:30, 900kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<01:03, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<01:01, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<00:59, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:45, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<00:31, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:57<01:04, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:52, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<00:38, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:41, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:59<00:36, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:26, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:32, 2.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:36, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:29, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:20, 3.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<01:19, 769kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<01:01, 979kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:03<00:43, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:41, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:05<00:42, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<00:32, 1.72MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:22, 2.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<01:19, 664kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<01:00, 858kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:07<00:42, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:39, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:09<00:38, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:09<00:28, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:19, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<01:07, 658kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:51, 853kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:11<00:36, 1.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:32, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:31, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<00:23, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:15, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:49, 724kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:38, 927kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:15<00:26, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:24, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:23, 1.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<00:17, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:11, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:24, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:19<00:20, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<00:14, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:14, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:21<00:14, 1.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<00:11, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:07, 2.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:25, 752kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:19, 965kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:23<00:13, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:11, 1.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:09, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:25<00:06, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:06, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<00:05, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:27<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:03, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<00:02, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:01, 2.92MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:31<00:01, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:31<00:01, 1.88MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:31<00:00, 2.39MB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<22:01:39,  5.04it/s]  0%|          | 863/400000 [00:00<15:23:23,  7.20it/s]  0%|          | 1757/400000 [00:00<10:45:09, 10.29it/s]  1%|          | 2716/400000 [00:00<7:30:43, 14.69it/s]   1%|          | 3698/400000 [00:00<5:14:55, 20.97it/s]  1%|          | 4625/400000 [00:00<3:40:08, 29.93it/s]  1%|▏         | 5506/400000 [00:00<2:33:59, 42.70it/s]  2%|▏         | 6430/400000 [00:00<1:47:45, 60.88it/s]  2%|▏         | 7341/400000 [00:00<1:15:27, 86.72it/s]  2%|▏         | 8239/400000 [00:01<52:55, 123.37it/s]   2%|▏         | 9138/400000 [00:01<37:10, 175.22it/s]  3%|▎         | 10055/400000 [00:01<26:10, 248.27it/s]  3%|▎         | 10984/400000 [00:01<18:29, 350.66it/s]  3%|▎         | 11906/400000 [00:01<13:07, 492.90it/s]  3%|▎         | 12815/400000 [00:01<09:23, 687.06it/s]  3%|▎         | 13706/400000 [00:01<06:46, 949.22it/s]  4%|▎         | 14619/400000 [00:01<04:56, 1298.13it/s]  4%|▍         | 15594/400000 [00:01<03:39, 1754.26it/s]  4%|▍         | 16527/400000 [00:02<02:45, 2319.04it/s]  4%|▍         | 17449/400000 [00:02<02:10, 2929.89it/s]  5%|▍         | 18325/400000 [00:02<01:44, 3660.38it/s]  5%|▍         | 19211/400000 [00:02<01:25, 4442.04it/s]  5%|▌         | 20197/400000 [00:02<01:11, 5318.69it/s]  5%|▌         | 21141/400000 [00:02<01:01, 6119.36it/s]  6%|▌         | 22066/400000 [00:02<00:55, 6810.69it/s]  6%|▌         | 22988/400000 [00:02<00:51, 7318.62it/s]  6%|▌         | 23899/400000 [00:02<00:48, 7707.28it/s]  6%|▌         | 24801/400000 [00:02<00:46, 8022.80it/s]  6%|▋         | 25764/400000 [00:03<00:44, 8444.13it/s]  7%|▋         | 26760/400000 [00:03<00:42, 8847.71it/s]  7%|▋         | 27704/400000 [00:03<00:41, 9016.68it/s]  7%|▋         | 28646/400000 [00:03<00:41, 9026.66it/s]  7%|▋         | 29577/400000 [00:03<00:41, 9029.37it/s]  8%|▊         | 30548/400000 [00:03<00:40, 9221.55it/s]  8%|▊         | 31485/400000 [00:03<00:40, 9202.62it/s]  8%|▊         | 32444/400000 [00:03<00:39, 9314.99it/s]  8%|▊         | 33384/400000 [00:03<00:39, 9328.00it/s]  9%|▊         | 34323/400000 [00:03<00:40, 9091.70it/s]  9%|▉         | 35238/400000 [00:04<00:40, 9051.84it/s]  9%|▉         | 36147/400000 [00:04<00:40, 9010.41it/s]  9%|▉         | 37062/400000 [00:04<00:40, 9049.85it/s] 10%|▉         | 38078/400000 [00:04<00:38, 9355.61it/s] 10%|▉         | 39018/400000 [00:04<00:38, 9290.04it/s] 10%|▉         | 39950/400000 [00:04<00:39, 9210.46it/s] 10%|█         | 40880/400000 [00:04<00:38, 9235.45it/s] 10%|█         | 41806/400000 [00:04<00:39, 9141.97it/s] 11%|█         | 42722/400000 [00:04<00:39, 9135.69it/s] 11%|█         | 43659/400000 [00:04<00:38, 9202.88it/s] 11%|█         | 44581/400000 [00:05<00:39, 9037.78it/s] 11%|█▏        | 45486/400000 [00:05<00:39, 8948.37it/s] 12%|█▏        | 46382/400000 [00:05<00:39, 8911.12it/s] 12%|█▏        | 47290/400000 [00:05<00:39, 8959.70it/s] 12%|█▏        | 48316/400000 [00:05<00:37, 9313.77it/s] 12%|█▏        | 49378/400000 [00:05<00:36, 9669.50it/s] 13%|█▎        | 50368/400000 [00:05<00:35, 9736.91it/s] 13%|█▎        | 51417/400000 [00:05<00:35, 9950.61it/s] 13%|█▎        | 52417/400000 [00:05<00:35, 9769.12it/s] 13%|█▎        | 53398/400000 [00:06<00:37, 9335.79it/s] 14%|█▎        | 54362/400000 [00:06<00:36, 9422.00it/s] 14%|█▍        | 55310/400000 [00:06<00:36, 9410.91it/s] 14%|█▍        | 56338/400000 [00:06<00:35, 9653.68it/s] 14%|█▍        | 57371/400000 [00:06<00:34, 9845.96it/s] 15%|█▍        | 58373/400000 [00:06<00:34, 9896.15it/s] 15%|█▍        | 59370/400000 [00:06<00:34, 9916.36it/s] 15%|█▌        | 60364/400000 [00:06<00:35, 9589.70it/s] 15%|█▌        | 61327/400000 [00:06<00:35, 9567.28it/s] 16%|█▌        | 62297/400000 [00:06<00:35, 9606.45it/s] 16%|█▌        | 63275/400000 [00:07<00:34, 9655.43it/s] 16%|█▌        | 64295/400000 [00:07<00:34, 9810.24it/s] 16%|█▋        | 65278/400000 [00:07<00:36, 9193.19it/s] 17%|█▋        | 66207/400000 [00:07<00:37, 8998.36it/s] 17%|█▋        | 67137/400000 [00:07<00:36, 9084.74it/s] 17%|█▋        | 68051/400000 [00:07<00:36, 8975.40it/s] 17%|█▋        | 68953/400000 [00:07<00:36, 8950.05it/s] 17%|█▋        | 69920/400000 [00:07<00:36, 9154.07it/s] 18%|█▊        | 70839/400000 [00:07<00:36, 9047.95it/s] 18%|█▊        | 71747/400000 [00:07<00:36, 8927.54it/s] 18%|█▊        | 72706/400000 [00:08<00:35, 9114.94it/s] 18%|█▊        | 73705/400000 [00:08<00:34, 9358.12it/s] 19%|█▊        | 74645/400000 [00:08<00:34, 9348.76it/s] 19%|█▉        | 75583/400000 [00:08<00:34, 9309.02it/s] 19%|█▉        | 76516/400000 [00:08<00:35, 9205.48it/s] 19%|█▉        | 77526/400000 [00:08<00:34, 9454.56it/s] 20%|█▉        | 78475/400000 [00:08<00:34, 9206.56it/s] 20%|█▉        | 79483/400000 [00:08<00:33, 9450.82it/s] 20%|██        | 80488/400000 [00:08<00:33, 9622.49it/s] 20%|██        | 81481/400000 [00:08<00:32, 9710.18it/s] 21%|██        | 82455/400000 [00:09<00:33, 9340.05it/s] 21%|██        | 83394/400000 [00:09<00:34, 9173.32it/s] 21%|██        | 84427/400000 [00:09<00:33, 9491.19it/s] 21%|██▏       | 85409/400000 [00:09<00:32, 9587.34it/s] 22%|██▏       | 86372/400000 [00:09<00:33, 9286.26it/s] 22%|██▏       | 87306/400000 [00:09<00:34, 9181.51it/s] 22%|██▏       | 88228/400000 [00:09<00:35, 8891.13it/s] 22%|██▏       | 89122/400000 [00:09<00:35, 8764.12it/s] 23%|██▎       | 90047/400000 [00:09<00:34, 8903.61it/s] 23%|██▎       | 90941/400000 [00:10<00:35, 8650.57it/s] 23%|██▎       | 91810/400000 [00:10<00:36, 8550.82it/s] 23%|██▎       | 92668/400000 [00:10<00:35, 8537.41it/s] 23%|██▎       | 93543/400000 [00:10<00:35, 8597.71it/s] 24%|██▎       | 94410/400000 [00:10<00:35, 8616.96it/s] 24%|██▍       | 95273/400000 [00:10<00:36, 8449.01it/s] 24%|██▍       | 96179/400000 [00:10<00:35, 8622.35it/s] 24%|██▍       | 97070/400000 [00:10<00:34, 8705.01it/s] 25%|██▍       | 98007/400000 [00:10<00:33, 8894.38it/s] 25%|██▍       | 98937/400000 [00:10<00:33, 9010.96it/s] 25%|██▍       | 99840/400000 [00:11<00:33, 8955.69it/s] 25%|██▌       | 100839/400000 [00:11<00:32, 9241.97it/s] 25%|██▌       | 101870/400000 [00:11<00:31, 9527.48it/s] 26%|██▌       | 102828/400000 [00:11<00:31, 9418.87it/s] 26%|██▌       | 103774/400000 [00:11<00:31, 9286.42it/s] 26%|██▌       | 104706/400000 [00:11<00:32, 9134.61it/s] 26%|██▋       | 105622/400000 [00:11<00:32, 9009.25it/s] 27%|██▋       | 106526/400000 [00:11<00:32, 8944.08it/s] 27%|██▋       | 107423/400000 [00:11<00:32, 8942.90it/s] 27%|██▋       | 108319/400000 [00:11<00:32, 8944.82it/s] 27%|██▋       | 109215/400000 [00:12<00:32, 8857.82it/s] 28%|██▊       | 110117/400000 [00:12<00:32, 8903.38it/s] 28%|██▊       | 111008/400000 [00:12<00:32, 8883.08it/s] 28%|██▊       | 111897/400000 [00:12<00:32, 8856.97it/s] 28%|██▊       | 112844/400000 [00:12<00:31, 9031.08it/s] 28%|██▊       | 113749/400000 [00:12<00:31, 9026.87it/s] 29%|██▊       | 114653/400000 [00:12<00:31, 8964.51it/s] 29%|██▉       | 115552/400000 [00:12<00:31, 8969.31it/s] 29%|██▉       | 116450/400000 [00:12<00:32, 8818.50it/s] 29%|██▉       | 117358/400000 [00:13<00:31, 8893.91it/s] 30%|██▉       | 118249/400000 [00:13<00:31, 8820.17it/s] 30%|██▉       | 119199/400000 [00:13<00:31, 9011.57it/s] 30%|███       | 120102/400000 [00:13<00:31, 8851.90it/s] 30%|███       | 120989/400000 [00:13<00:31, 8814.36it/s] 30%|███       | 121905/400000 [00:13<00:31, 8915.04it/s] 31%|███       | 122798/400000 [00:13<00:31, 8858.78it/s] 31%|███       | 123747/400000 [00:13<00:30, 9037.93it/s] 31%|███       | 124676/400000 [00:13<00:30, 9111.96it/s] 31%|███▏      | 125657/400000 [00:13<00:29, 9310.63it/s] 32%|███▏      | 126607/400000 [00:14<00:29, 9366.08it/s] 32%|███▏      | 127546/400000 [00:14<00:29, 9226.33it/s] 32%|███▏      | 128471/400000 [00:14<00:29, 9145.99it/s] 32%|███▏      | 129478/400000 [00:14<00:28, 9404.19it/s] 33%|███▎      | 130422/400000 [00:14<00:28, 9338.22it/s] 33%|███▎      | 131358/400000 [00:14<00:29, 9237.60it/s] 33%|███▎      | 132314/400000 [00:14<00:28, 9329.43it/s] 33%|███▎      | 133249/400000 [00:14<00:29, 9184.27it/s] 34%|███▎      | 134169/400000 [00:14<00:29, 9081.90it/s] 34%|███▍      | 135146/400000 [00:14<00:28, 9277.66it/s] 34%|███▍      | 136076/400000 [00:15<00:29, 8930.36it/s] 34%|███▍      | 136974/400000 [00:15<00:29, 8921.23it/s] 34%|███▍      | 137870/400000 [00:15<00:29, 8891.36it/s] 35%|███▍      | 138810/400000 [00:15<00:28, 9037.95it/s] 35%|███▍      | 139716/400000 [00:15<00:28, 8999.54it/s] 35%|███▌      | 140668/400000 [00:15<00:28, 9147.80it/s] 35%|███▌      | 141726/400000 [00:15<00:27, 9533.08it/s] 36%|███▌      | 142738/400000 [00:15<00:26, 9701.83it/s] 36%|███▌      | 143819/400000 [00:15<00:25, 10008.55it/s] 36%|███▌      | 144826/400000 [00:15<00:26, 9679.56it/s]  36%|███▋      | 145801/400000 [00:16<00:26, 9665.64it/s] 37%|███▋      | 146820/400000 [00:16<00:25, 9815.36it/s] 37%|███▋      | 147845/400000 [00:16<00:25, 9940.68it/s] 37%|███▋      | 148849/400000 [00:16<00:25, 9968.02it/s] 37%|███▋      | 149893/400000 [00:16<00:24, 10103.54it/s] 38%|███▊      | 150906/400000 [00:16<00:24, 10038.57it/s] 38%|███▊      | 151912/400000 [00:16<00:25, 9822.49it/s]  38%|███▊      | 152897/400000 [00:16<00:25, 9766.13it/s] 38%|███▊      | 153876/400000 [00:16<00:25, 9650.32it/s] 39%|███▊      | 154903/400000 [00:16<00:24, 9827.32it/s] 39%|███▉      | 155888/400000 [00:17<00:24, 9808.37it/s] 39%|███▉      | 156871/400000 [00:17<00:24, 9814.16it/s] 39%|███▉      | 157854/400000 [00:17<00:24, 9767.50it/s] 40%|███▉      | 158978/400000 [00:17<00:23, 10166.47it/s] 40%|████      | 160000/400000 [00:17<00:23, 10125.69it/s] 40%|████      | 161016/400000 [00:17<00:23, 10045.75it/s] 41%|████      | 162049/400000 [00:17<00:23, 10127.12it/s] 41%|████      | 163064/400000 [00:17<00:24, 9815.32it/s]  41%|████      | 164049/400000 [00:17<00:24, 9483.08it/s] 41%|████▏     | 165062/400000 [00:18<00:24, 9665.59it/s] 42%|████▏     | 166115/400000 [00:18<00:23, 9906.90it/s] 42%|████▏     | 167208/400000 [00:18<00:22, 10192.56it/s] 42%|████▏     | 168252/400000 [00:18<00:22, 10264.13it/s] 42%|████▏     | 169283/400000 [00:18<00:23, 9935.13it/s]  43%|████▎     | 170294/400000 [00:18<00:23, 9985.21it/s] 43%|████▎     | 171297/400000 [00:18<00:23, 9660.23it/s] 43%|████▎     | 172268/400000 [00:18<00:24, 9486.87it/s] 43%|████▎     | 173221/400000 [00:18<00:24, 9430.83it/s] 44%|████▎     | 174179/400000 [00:18<00:23, 9471.12it/s] 44%|████▍     | 175129/400000 [00:19<00:24, 9187.92it/s] 44%|████▍     | 176057/400000 [00:19<00:24, 9214.47it/s] 44%|████▍     | 177029/400000 [00:19<00:23, 9358.44it/s] 45%|████▍     | 178018/400000 [00:19<00:23, 9511.44it/s] 45%|████▍     | 178980/400000 [00:19<00:23, 9541.96it/s] 45%|████▍     | 179936/400000 [00:19<00:23, 9506.00it/s] 45%|████▌     | 180907/400000 [00:19<00:22, 9565.80it/s] 45%|████▌     | 181905/400000 [00:19<00:22, 9684.64it/s] 46%|████▌     | 182885/400000 [00:19<00:22, 9718.03it/s] 46%|████▌     | 183858/400000 [00:19<00:22, 9565.44it/s] 46%|████▌     | 184816/400000 [00:20<00:23, 9275.69it/s] 46%|████▋     | 185747/400000 [00:20<00:23, 9084.55it/s] 47%|████▋     | 186659/400000 [00:20<00:23, 9057.50it/s] 47%|████▋     | 187635/400000 [00:20<00:22, 9256.13it/s] 47%|████▋     | 188639/400000 [00:20<00:22, 9476.15it/s] 47%|████▋     | 189590/400000 [00:20<00:22, 9395.16it/s] 48%|████▊     | 190532/400000 [00:20<00:22, 9386.65it/s] 48%|████▊     | 191473/400000 [00:20<00:22, 9264.74it/s] 48%|████▊     | 192415/400000 [00:20<00:22, 9309.28it/s] 48%|████▊     | 193352/400000 [00:20<00:22, 9325.99it/s] 49%|████▊     | 194286/400000 [00:21<00:22, 9252.81it/s] 49%|████▉     | 195253/400000 [00:21<00:21, 9371.03it/s] 49%|████▉     | 196191/400000 [00:21<00:21, 9280.71it/s] 49%|████▉     | 197188/400000 [00:21<00:21, 9475.60it/s] 50%|████▉     | 198204/400000 [00:21<00:20, 9668.76it/s] 50%|████▉     | 199234/400000 [00:21<00:20, 9847.37it/s] 50%|█████     | 200221/400000 [00:21<00:20, 9568.80it/s] 50%|█████     | 201182/400000 [00:21<00:21, 9224.18it/s] 51%|█████     | 202186/400000 [00:21<00:20, 9454.01it/s] 51%|█████     | 203191/400000 [00:22<00:20, 9624.25it/s] 51%|█████     | 204158/400000 [00:22<00:20, 9575.44it/s] 51%|█████▏    | 205174/400000 [00:22<00:19, 9743.33it/s] 52%|█████▏    | 206152/400000 [00:22<00:20, 9623.99it/s] 52%|█████▏    | 207166/400000 [00:22<00:19, 9771.06it/s] 52%|█████▏    | 208231/400000 [00:22<00:19, 10017.64it/s] 52%|█████▏    | 209292/400000 [00:22<00:18, 10187.01it/s] 53%|█████▎    | 210326/400000 [00:22<00:18, 10230.00it/s] 53%|█████▎    | 211352/400000 [00:22<00:19, 9753.23it/s]  53%|█████▎    | 212334/400000 [00:22<00:19, 9755.15it/s] 53%|█████▎    | 213314/400000 [00:23<00:19, 9674.28it/s] 54%|█████▎    | 214285/400000 [00:23<00:19, 9421.65it/s] 54%|█████▍    | 215231/400000 [00:23<00:19, 9282.40it/s] 54%|█████▍    | 216173/400000 [00:23<00:19, 9320.27it/s] 54%|█████▍    | 217108/400000 [00:23<00:19, 9202.68it/s] 55%|█████▍    | 218134/400000 [00:23<00:19, 9494.55it/s] 55%|█████▍    | 219088/400000 [00:23<00:19, 9411.82it/s] 55%|█████▌    | 220065/400000 [00:23<00:18, 9515.23it/s] 55%|█████▌    | 221050/400000 [00:23<00:18, 9611.77it/s] 56%|█████▌    | 222046/400000 [00:23<00:18, 9711.72it/s] 56%|█████▌    | 223019/400000 [00:24<00:18, 9501.11it/s] 56%|█████▌    | 223972/400000 [00:24<00:18, 9267.79it/s] 56%|█████▌    | 224973/400000 [00:24<00:18, 9477.55it/s] 57%|█████▋    | 226002/400000 [00:24<00:17, 9705.55it/s] 57%|█████▋    | 226977/400000 [00:24<00:18, 9604.10it/s] 57%|█████▋    | 227983/400000 [00:24<00:17, 9736.36it/s] 57%|█████▋    | 228959/400000 [00:24<00:18, 9485.37it/s] 57%|█████▋    | 229911/400000 [00:24<00:18, 9277.96it/s] 58%|█████▊    | 230867/400000 [00:24<00:18, 9359.61it/s] 58%|█████▊    | 231806/400000 [00:25<00:18, 9233.63it/s] 58%|█████▊    | 232732/400000 [00:25<00:18, 9041.66it/s] 58%|█████▊    | 233639/400000 [00:25<00:18, 8979.22it/s] 59%|█████▊    | 234578/400000 [00:25<00:18, 9098.15it/s] 59%|█████▉    | 235530/400000 [00:25<00:17, 9220.59it/s] 59%|█████▉    | 236455/400000 [00:25<00:17, 9228.29it/s] 59%|█████▉    | 237379/400000 [00:25<00:17, 9121.89it/s] 60%|█████▉    | 238327/400000 [00:25<00:17, 9225.96it/s] 60%|█████▉    | 239311/400000 [00:25<00:17, 9401.03it/s] 60%|██████    | 240253/400000 [00:25<00:17, 9233.73it/s] 60%|██████    | 241179/400000 [00:26<00:17, 9154.98it/s] 61%|██████    | 242119/400000 [00:26<00:17, 9225.78it/s] 61%|██████    | 243046/400000 [00:26<00:16, 9237.50it/s] 61%|██████    | 243981/400000 [00:26<00:16, 9268.81it/s] 61%|██████▏   | 245023/400000 [00:26<00:16, 9584.52it/s] 62%|██████▏   | 246008/400000 [00:26<00:15, 9661.61it/s] 62%|██████▏   | 246988/400000 [00:26<00:15, 9702.10it/s] 62%|██████▏   | 247960/400000 [00:26<00:16, 9308.01it/s] 62%|██████▏   | 248992/400000 [00:26<00:15, 9586.23it/s] 63%|██████▎   | 250059/400000 [00:26<00:15, 9887.34it/s] 63%|██████▎   | 251071/400000 [00:27<00:14, 9954.18it/s] 63%|██████▎   | 252102/400000 [00:27<00:14, 10056.49it/s] 63%|██████▎   | 253111/400000 [00:27<00:14, 9894.02it/s]  64%|██████▎   | 254104/400000 [00:27<00:15, 9652.17it/s] 64%|██████▍   | 255073/400000 [00:27<00:15, 9514.93it/s] 64%|██████▍   | 256150/400000 [00:27<00:14, 9856.62it/s] 64%|██████▍   | 257168/400000 [00:27<00:14, 9950.23it/s] 65%|██████▍   | 258167/400000 [00:27<00:14, 9792.82it/s] 65%|██████▍   | 259150/400000 [00:27<00:14, 9538.77it/s] 65%|██████▌   | 260149/400000 [00:27<00:14, 9667.31it/s] 65%|██████▌   | 261119/400000 [00:28<00:14, 9522.09it/s] 66%|██████▌   | 262074/400000 [00:28<00:14, 9390.42it/s] 66%|██████▌   | 263016/400000 [00:28<00:14, 9376.47it/s] 66%|██████▌   | 263956/400000 [00:28<00:14, 9138.24it/s] 66%|██████▌   | 264873/400000 [00:28<00:14, 9028.23it/s] 66%|██████▋   | 265940/400000 [00:28<00:14, 9462.95it/s] 67%|██████▋   | 266996/400000 [00:28<00:13, 9765.67it/s] 67%|██████▋   | 267985/400000 [00:28<00:13, 9800.48it/s] 67%|██████▋   | 268971/400000 [00:28<00:13, 9715.78it/s] 67%|██████▋   | 269947/400000 [00:29<00:13, 9619.42it/s] 68%|██████▊   | 270912/400000 [00:29<00:13, 9406.36it/s] 68%|██████▊   | 271856/400000 [00:29<00:13, 9403.93it/s] 68%|██████▊   | 272863/400000 [00:29<00:13, 9593.85it/s] 68%|██████▊   | 273825/400000 [00:29<00:13, 9523.91it/s] 69%|██████▊   | 274780/400000 [00:29<00:13, 9496.77it/s] 69%|██████▉   | 275731/400000 [00:29<00:13, 9437.31it/s] 69%|██████▉   | 276676/400000 [00:29<00:13, 9260.79it/s] 69%|██████▉   | 277648/400000 [00:29<00:13, 9393.52it/s] 70%|██████▉   | 278612/400000 [00:29<00:12, 9464.50it/s] 70%|██████▉   | 279560/400000 [00:30<00:13, 9136.36it/s] 70%|███████   | 280477/400000 [00:30<00:13, 9046.73it/s] 70%|███████   | 281394/400000 [00:30<00:13, 9081.40it/s] 71%|███████   | 282329/400000 [00:30<00:12, 9159.84it/s] 71%|███████   | 283312/400000 [00:30<00:12, 9348.41it/s] 71%|███████   | 284249/400000 [00:30<00:12, 9271.25it/s] 71%|███████▏  | 285214/400000 [00:30<00:12, 9380.26it/s] 72%|███████▏  | 286154/400000 [00:30<00:12, 9180.14it/s] 72%|███████▏  | 287074/400000 [00:30<00:12, 9095.82it/s] 72%|███████▏  | 287986/400000 [00:30<00:12, 9089.13it/s] 72%|███████▏  | 288896/400000 [00:31<00:12, 9016.61it/s] 72%|███████▏  | 289833/400000 [00:31<00:12, 9119.17it/s] 73%|███████▎  | 290760/400000 [00:31<00:11, 9161.95it/s] 73%|███████▎  | 291727/400000 [00:31<00:11, 9306.11it/s] 73%|███████▎  | 292682/400000 [00:31<00:11, 9376.07it/s] 73%|███████▎  | 293689/400000 [00:31<00:11, 9572.09it/s] 74%|███████▎  | 294648/400000 [00:31<00:11, 9490.79it/s] 74%|███████▍  | 295599/400000 [00:31<00:11, 9372.25it/s] 74%|███████▍  | 296538/400000 [00:31<00:11, 9312.18it/s] 74%|███████▍  | 297471/400000 [00:31<00:11, 9213.49it/s] 75%|███████▍  | 298394/400000 [00:32<00:11, 9184.28it/s] 75%|███████▍  | 299314/400000 [00:32<00:11, 9130.55it/s] 75%|███████▌  | 300228/400000 [00:32<00:11, 8986.65it/s] 75%|███████▌  | 301164/400000 [00:32<00:10, 9092.54it/s] 76%|███████▌  | 302110/400000 [00:32<00:10, 9198.40it/s] 76%|███████▌  | 303127/400000 [00:32<00:10, 9467.72it/s] 76%|███████▌  | 304113/400000 [00:32<00:10, 9581.03it/s] 76%|███████▋  | 305074/400000 [00:32<00:10, 9296.33it/s] 77%|███████▋  | 306007/400000 [00:32<00:10, 9182.65it/s] 77%|███████▋  | 307041/400000 [00:33<00:09, 9499.89it/s] 77%|███████▋  | 308029/400000 [00:33<00:09, 9608.76it/s] 77%|███████▋  | 309035/400000 [00:33<00:09, 9736.45it/s] 78%|███████▊  | 310012/400000 [00:33<00:09, 9567.66it/s] 78%|███████▊  | 310972/400000 [00:33<00:09, 9408.81it/s] 78%|███████▊  | 311955/400000 [00:33<00:09, 9527.80it/s] 78%|███████▊  | 313006/400000 [00:33<00:08, 9802.50it/s] 79%|███████▊  | 314022/400000 [00:33<00:08, 9906.73it/s] 79%|███████▉  | 315062/400000 [00:33<00:08, 10049.07it/s] 79%|███████▉  | 316131/400000 [00:33<00:08, 10232.26it/s] 79%|███████▉  | 317169/400000 [00:34<00:08, 10273.41it/s] 80%|███████▉  | 318199/400000 [00:34<00:08, 9737.13it/s]  80%|███████▉  | 319180/400000 [00:34<00:08, 9475.99it/s] 80%|████████  | 320135/400000 [00:34<00:08, 9350.73it/s] 80%|████████  | 321075/400000 [00:34<00:08, 9253.80it/s] 81%|████████  | 322020/400000 [00:34<00:08, 9310.66it/s] 81%|████████  | 322976/400000 [00:34<00:08, 9381.54it/s] 81%|████████  | 323917/400000 [00:34<00:08, 9254.87it/s] 81%|████████  | 324899/400000 [00:34<00:07, 9417.17it/s] 81%|████████▏ | 325843/400000 [00:34<00:08, 9207.78it/s] 82%|████████▏ | 326767/400000 [00:35<00:08, 9067.59it/s] 82%|████████▏ | 327676/400000 [00:35<00:08, 9019.96it/s] 82%|████████▏ | 328580/400000 [00:35<00:07, 8954.78it/s] 82%|████████▏ | 329506/400000 [00:35<00:07, 9041.50it/s] 83%|████████▎ | 330412/400000 [00:35<00:07, 9008.96it/s] 83%|████████▎ | 331314/400000 [00:35<00:07, 8946.32it/s] 83%|████████▎ | 332260/400000 [00:35<00:07, 9093.54it/s] 83%|████████▎ | 333234/400000 [00:35<00:07, 9275.60it/s] 84%|████████▎ | 334164/400000 [00:35<00:07, 9141.53it/s] 84%|████████▍ | 335080/400000 [00:36<00:07, 8936.60it/s] 84%|████████▍ | 335976/400000 [00:36<00:07, 8887.93it/s] 84%|████████▍ | 336867/400000 [00:36<00:07, 8884.89it/s] 84%|████████▍ | 337770/400000 [00:36<00:06, 8926.30it/s] 85%|████████▍ | 338783/400000 [00:36<00:06, 9254.78it/s] 85%|████████▍ | 339744/400000 [00:36<00:06, 9356.37it/s] 85%|████████▌ | 340683/400000 [00:36<00:06, 9247.94it/s] 85%|████████▌ | 341611/400000 [00:36<00:06, 9156.37it/s] 86%|████████▌ | 342571/400000 [00:36<00:06, 9160.09it/s] 86%|████████▌ | 343525/400000 [00:36<00:06, 9269.62it/s] 86%|████████▌ | 344454/400000 [00:37<00:06, 9195.07it/s] 86%|████████▋ | 345375/400000 [00:37<00:05, 9156.03it/s] 87%|████████▋ | 346292/400000 [00:37<00:06, 8851.97it/s] 87%|████████▋ | 347180/400000 [00:37<00:06, 8712.82it/s] 87%|████████▋ | 348121/400000 [00:37<00:05, 8909.72it/s] 87%|████████▋ | 349180/400000 [00:37<00:05, 9353.54it/s] 88%|████████▊ | 350247/400000 [00:37<00:05, 9712.32it/s] 88%|████████▊ | 351228/400000 [00:37<00:05, 9337.81it/s] 88%|████████▊ | 352172/400000 [00:37<00:05, 9119.93it/s] 88%|████████▊ | 353145/400000 [00:37<00:05, 9294.01it/s] 89%|████████▊ | 354195/400000 [00:38<00:04, 9625.26it/s] 89%|████████▉ | 355198/400000 [00:38<00:04, 9742.91it/s] 89%|████████▉ | 356211/400000 [00:38<00:04, 9853.80it/s] 89%|████████▉ | 357201/400000 [00:38<00:04, 9857.08it/s] 90%|████████▉ | 358190/400000 [00:38<00:04, 9765.19it/s] 90%|████████▉ | 359169/400000 [00:38<00:04, 9765.46it/s] 90%|█████████ | 360148/400000 [00:38<00:04, 9528.32it/s] 90%|█████████ | 361155/400000 [00:38<00:04, 9682.13it/s] 91%|█████████ | 362223/400000 [00:38<00:03, 9959.89it/s] 91%|█████████ | 363274/400000 [00:38<00:03, 10117.14it/s] 91%|█████████ | 364289/400000 [00:39<00:03, 9689.44it/s]  91%|█████████▏| 365283/400000 [00:39<00:03, 9762.39it/s] 92%|█████████▏| 366318/400000 [00:39<00:03, 9929.51it/s] 92%|█████████▏| 367352/400000 [00:39<00:03, 10048.59it/s] 92%|█████████▏| 368360/400000 [00:39<00:03, 9719.46it/s]  92%|█████████▏| 369368/400000 [00:39<00:03, 9824.06it/s] 93%|█████████▎| 370354/400000 [00:39<00:03, 9426.80it/s] 93%|█████████▎| 371303/400000 [00:39<00:03, 9350.62it/s] 93%|█████████▎| 372243/400000 [00:39<00:02, 9333.37it/s] 93%|█████████▎| 373180/400000 [00:40<00:02, 9102.97it/s] 94%|█████████▎| 374094/400000 [00:40<00:02, 9066.70it/s] 94%|█████████▍| 375004/400000 [00:40<00:02, 8978.15it/s] 94%|█████████▍| 375971/400000 [00:40<00:02, 9175.00it/s] 94%|█████████▍| 376891/400000 [00:40<00:02, 9022.85it/s] 94%|█████████▍| 377796/400000 [00:40<00:02, 8996.61it/s] 95%|█████████▍| 378804/400000 [00:40<00:02, 9295.30it/s] 95%|█████████▍| 379814/400000 [00:40<00:02, 9521.78it/s] 95%|█████████▌| 380870/400000 [00:40<00:01, 9808.12it/s] 95%|█████████▌| 381856/400000 [00:40<00:01, 9446.01it/s] 96%|█████████▌| 382807/400000 [00:41<00:01, 9248.91it/s] 96%|█████████▌| 383738/400000 [00:41<00:01, 9244.70it/s] 96%|█████████▌| 384667/400000 [00:41<00:01, 9237.79it/s] 96%|█████████▋| 385594/400000 [00:41<00:01, 9081.70it/s] 97%|█████████▋| 386508/400000 [00:41<00:01, 9096.66it/s] 97%|█████████▋| 387423/400000 [00:41<00:01, 9111.54it/s] 97%|█████████▋| 388362/400000 [00:41<00:01, 9193.19it/s] 97%|█████████▋| 389283/400000 [00:41<00:01, 9130.60it/s] 98%|█████████▊| 390197/400000 [00:41<00:01, 8983.84it/s] 98%|█████████▊| 391131/400000 [00:41<00:00, 9085.35it/s] 98%|█████████▊| 392155/400000 [00:42<00:00, 9401.78it/s] 98%|█████████▊| 393099/400000 [00:42<00:00, 9194.78it/s] 99%|█████████▊| 394034/400000 [00:42<00:00, 9239.14it/s] 99%|█████████▉| 395015/400000 [00:42<00:00, 9401.42it/s] 99%|█████████▉| 395958/400000 [00:42<00:00, 9299.63it/s] 99%|█████████▉| 396922/400000 [00:42<00:00, 9397.16it/s] 99%|█████████▉| 397945/400000 [00:42<00:00, 9630.25it/s]100%|█████████▉| 398911/400000 [00:42<00:00, 9413.04it/s]100%|█████████▉| 399856/400000 [00:42<00:00, 9252.31it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9317.63it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7d6ac200f0>, <torchtext.data.dataset.TabularDataset object at 0x7f7d6ac20240>, <torchtext.vocab.Vocab object at 0x7f7d6ac20160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 27.58 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.61 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.61 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.33 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.23 file/s]2020-06-09 18:18:08.163976: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-09 18:18:08.167874: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-09 18:18:08.168071: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561e474c7e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 18:18:08.168089: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147470.49it/s] 79%|███████▉  | 7823360/9912422 [00:00<00:09, 210500.47it/s]9920512it [00:00, 43178580.05it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 255501.21it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156865.85it/s]1654784it [00:00, 10960459.52it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 77300.79it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
