
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f87d6580b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f87d6580b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f88411f0510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f88411f0510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f886041dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f886041dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87ee598a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87ee598a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87ee598a60> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3178496/9912422 [00:00<00:00, 30829591.25it/s]9920512it [00:00, 37061031.58it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1071849.34it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155174.46it/s]1654784it [00:00, 10372828.72it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193520.39it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87ccfba6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87ccfc8940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f87ee5986a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f87ee5986a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f87ee5986a8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 51.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 74.73 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 74.73 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.87 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.39s/ url]
0 examples [00:00, ? examples/s]2020-08-02 00:10:26.942796: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-02 00:10:26.948180: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-02 00:10:26.948583: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562f147b10c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-02 00:10:26.948607: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
68 examples [00:00, 675.95 examples/s]195 examples [00:00, 785.57 examples/s]318 examples [00:00, 880.36 examples/s]447 examples [00:00, 971.48 examples/s]559 examples [00:00, 1010.38 examples/s]674 examples [00:00, 1047.52 examples/s]798 examples [00:00, 1097.53 examples/s]921 examples [00:00, 1132.72 examples/s]1039 examples [00:00, 1144.11 examples/s]1155 examples [00:01, 1148.09 examples/s]1282 examples [00:01, 1180.33 examples/s]1405 examples [00:01, 1192.51 examples/s]1531 examples [00:01, 1209.88 examples/s]1655 examples [00:01, 1217.04 examples/s]1778 examples [00:01, 1219.38 examples/s]1900 examples [00:01, 1210.19 examples/s]2022 examples [00:01, 1200.93 examples/s]2145 examples [00:01, 1207.83 examples/s]2266 examples [00:01, 1200.14 examples/s]2387 examples [00:02, 1189.01 examples/s]2506 examples [00:02, 1181.90 examples/s]2633 examples [00:02, 1206.74 examples/s]2759 examples [00:02, 1220.76 examples/s]2887 examples [00:02, 1237.31 examples/s]3011 examples [00:02, 1220.36 examples/s]3134 examples [00:02, 1219.98 examples/s]3259 examples [00:02, 1228.44 examples/s]3385 examples [00:02, 1235.74 examples/s]3509 examples [00:02, 1236.35 examples/s]3633 examples [00:03, 1194.83 examples/s]3756 examples [00:03, 1204.52 examples/s]3877 examples [00:03, 1195.41 examples/s]4003 examples [00:03, 1212.55 examples/s]4125 examples [00:03, 1208.81 examples/s]4247 examples [00:03, 1197.20 examples/s]4367 examples [00:03, 1159.20 examples/s]4491 examples [00:03, 1180.80 examples/s]4610 examples [00:03, 1167.56 examples/s]4729 examples [00:03, 1171.84 examples/s]4847 examples [00:04, 1156.57 examples/s]4963 examples [00:04, 1129.09 examples/s]5081 examples [00:04, 1142.71 examples/s]5200 examples [00:04, 1155.12 examples/s]5323 examples [00:04, 1174.15 examples/s]5442 examples [00:04, 1175.92 examples/s]5566 examples [00:04, 1193.77 examples/s]5694 examples [00:04, 1216.75 examples/s]5816 examples [00:04, 1216.24 examples/s]5945 examples [00:04, 1235.93 examples/s]6069 examples [00:05, 1201.16 examples/s]6190 examples [00:05, 1202.27 examples/s]6311 examples [00:05, 1158.16 examples/s]6429 examples [00:05, 1163.66 examples/s]6548 examples [00:05, 1157.44 examples/s]6666 examples [00:05, 1162.64 examples/s]6791 examples [00:05, 1186.59 examples/s]6910 examples [00:05, 1185.13 examples/s]7029 examples [00:05, 1184.50 examples/s]7153 examples [00:06, 1199.37 examples/s]7274 examples [00:06, 1188.15 examples/s]7393 examples [00:06, 1172.48 examples/s]7511 examples [00:06, 1168.91 examples/s]7628 examples [00:06, 1166.41 examples/s]7745 examples [00:06, 1161.06 examples/s]7862 examples [00:06, 1156.67 examples/s]7978 examples [00:06, 1146.32 examples/s]8093 examples [00:06, 1125.13 examples/s]8209 examples [00:06, 1132.04 examples/s]8328 examples [00:07, 1146.34 examples/s]8445 examples [00:07, 1150.78 examples/s]8562 examples [00:07, 1154.57 examples/s]8678 examples [00:07, 1151.64 examples/s]8794 examples [00:07, 1146.35 examples/s]8914 examples [00:07, 1159.63 examples/s]9032 examples [00:07, 1163.48 examples/s]9153 examples [00:07, 1176.86 examples/s]9271 examples [00:07, 1164.74 examples/s]9388 examples [00:07, 1141.91 examples/s]9503 examples [00:08, 1099.91 examples/s]9614 examples [00:08, 1086.57 examples/s]9729 examples [00:08, 1103.17 examples/s]9842 examples [00:08, 1110.66 examples/s]9956 examples [00:08, 1117.09 examples/s]10068 examples [00:08, 1045.43 examples/s]10179 examples [00:08, 1062.77 examples/s]10292 examples [00:08, 1081.66 examples/s]10403 examples [00:08, 1088.98 examples/s]10518 examples [00:09, 1091.94 examples/s]10633 examples [00:09, 1106.09 examples/s]10744 examples [00:09, 1102.44 examples/s]10857 examples [00:09, 1107.99 examples/s]10972 examples [00:09, 1118.71 examples/s]11090 examples [00:09, 1134.09 examples/s]11205 examples [00:09, 1138.22 examples/s]11320 examples [00:09, 1139.10 examples/s]11437 examples [00:09, 1145.70 examples/s]11552 examples [00:09, 1125.92 examples/s]11666 examples [00:10, 1126.81 examples/s]11779 examples [00:10, 1122.33 examples/s]11892 examples [00:10, 1109.08 examples/s]12003 examples [00:10, 1106.84 examples/s]12121 examples [00:10, 1127.44 examples/s]12247 examples [00:10, 1162.00 examples/s]12368 examples [00:10, 1174.12 examples/s]12491 examples [00:10, 1187.84 examples/s]12612 examples [00:10, 1190.67 examples/s]12734 examples [00:10, 1197.68 examples/s]12856 examples [00:11, 1201.62 examples/s]12977 examples [00:11, 1195.64 examples/s]13097 examples [00:11, 1188.97 examples/s]13216 examples [00:11, 1187.97 examples/s]13335 examples [00:11, 1181.33 examples/s]13454 examples [00:11, 1178.59 examples/s]13572 examples [00:11, 1173.73 examples/s]13690 examples [00:11, 1114.09 examples/s]13803 examples [00:11, 1109.36 examples/s]13920 examples [00:11, 1124.74 examples/s]14033 examples [00:12, 1123.08 examples/s]14146 examples [00:12, 1109.10 examples/s]14266 examples [00:12, 1132.94 examples/s]14382 examples [00:12, 1139.14 examples/s]14501 examples [00:12, 1153.76 examples/s]14617 examples [00:12, 1150.90 examples/s]14733 examples [00:12, 1127.21 examples/s]14846 examples [00:12, 1120.42 examples/s]14959 examples [00:12, 1119.25 examples/s]15072 examples [00:12, 1106.44 examples/s]15183 examples [00:13, 1095.86 examples/s]15300 examples [00:13, 1114.53 examples/s]15422 examples [00:13, 1143.17 examples/s]15537 examples [00:13, 1131.91 examples/s]15653 examples [00:13, 1138.26 examples/s]15775 examples [00:13, 1160.47 examples/s]15892 examples [00:13, 1158.90 examples/s]16012 examples [00:13, 1170.50 examples/s]16136 examples [00:13, 1189.51 examples/s]16260 examples [00:14, 1202.96 examples/s]16382 examples [00:14, 1205.81 examples/s]16503 examples [00:14, 1177.08 examples/s]16621 examples [00:14, 1161.80 examples/s]16738 examples [00:14, 1058.40 examples/s]16846 examples [00:14, 1036.36 examples/s]16962 examples [00:14, 1069.66 examples/s]17071 examples [00:14, 1066.99 examples/s]17179 examples [00:14, 1047.74 examples/s]17294 examples [00:14, 1073.81 examples/s]17412 examples [00:15, 1101.18 examples/s]17528 examples [00:15, 1116.21 examples/s]17641 examples [00:15, 1092.47 examples/s]17762 examples [00:15, 1123.23 examples/s]17881 examples [00:15, 1140.52 examples/s]18007 examples [00:15, 1173.79 examples/s]18125 examples [00:15, 1143.38 examples/s]18240 examples [00:15, 1133.51 examples/s]18364 examples [00:15, 1160.02 examples/s]18481 examples [00:15, 1144.95 examples/s]18604 examples [00:16, 1167.61 examples/s]18723 examples [00:16, 1172.87 examples/s]18841 examples [00:16, 1166.52 examples/s]18963 examples [00:16, 1180.51 examples/s]19086 examples [00:16, 1193.13 examples/s]19206 examples [00:16, 1193.05 examples/s]19329 examples [00:16, 1203.05 examples/s]19450 examples [00:16, 1193.14 examples/s]19575 examples [00:16, 1207.32 examples/s]19696 examples [00:17, 1192.94 examples/s]19816 examples [00:17, 1155.12 examples/s]19932 examples [00:17, 1136.57 examples/s]20046 examples [00:17, 1087.81 examples/s]20164 examples [00:17, 1113.08 examples/s]20284 examples [00:17, 1136.96 examples/s]20403 examples [00:17, 1151.25 examples/s]20523 examples [00:17, 1164.36 examples/s]20640 examples [00:17, 1160.58 examples/s]20759 examples [00:17, 1167.32 examples/s]20878 examples [00:18, 1172.56 examples/s]20996 examples [00:18, 1167.36 examples/s]21122 examples [00:18, 1190.97 examples/s]21242 examples [00:18, 1190.58 examples/s]21366 examples [00:18, 1203.98 examples/s]21489 examples [00:18, 1210.60 examples/s]21614 examples [00:18, 1219.28 examples/s]21737 examples [00:18, 1192.63 examples/s]21858 examples [00:18, 1197.12 examples/s]21982 examples [00:18, 1207.72 examples/s]22107 examples [00:19, 1218.65 examples/s]22231 examples [00:19, 1222.26 examples/s]22355 examples [00:19, 1225.80 examples/s]22478 examples [00:19, 1220.97 examples/s]22606 examples [00:19, 1237.80 examples/s]22733 examples [00:19, 1245.82 examples/s]22858 examples [00:19, 1233.63 examples/s]22982 examples [00:19, 1225.58 examples/s]23105 examples [00:19, 1169.57 examples/s]23223 examples [00:19, 1168.89 examples/s]23349 examples [00:20, 1193.00 examples/s]23469 examples [00:20, 1172.92 examples/s]23587 examples [00:20, 1170.62 examples/s]23707 examples [00:20, 1177.96 examples/s]23828 examples [00:20, 1187.06 examples/s]23954 examples [00:20, 1206.76 examples/s]24076 examples [00:20, 1209.64 examples/s]24198 examples [00:20, 1178.83 examples/s]24318 examples [00:20, 1184.54 examples/s]24437 examples [00:21, 1185.85 examples/s]24557 examples [00:21, 1188.20 examples/s]24682 examples [00:21, 1205.92 examples/s]24805 examples [00:21, 1211.98 examples/s]24927 examples [00:21, 1210.36 examples/s]25049 examples [00:21, 1205.38 examples/s]25170 examples [00:21, 1188.90 examples/s]25291 examples [00:21, 1194.63 examples/s]25411 examples [00:21, 1187.04 examples/s]25533 examples [00:21, 1193.98 examples/s]25653 examples [00:22, 1165.42 examples/s]25770 examples [00:22, 1160.36 examples/s]25887 examples [00:22, 1153.68 examples/s]26003 examples [00:22, 1130.76 examples/s]26120 examples [00:22, 1142.13 examples/s]26243 examples [00:22, 1165.82 examples/s]26366 examples [00:22, 1183.60 examples/s]26485 examples [00:22, 1184.39 examples/s]26609 examples [00:22, 1197.95 examples/s]26729 examples [00:22, 1194.05 examples/s]26858 examples [00:23, 1220.85 examples/s]26988 examples [00:23, 1241.77 examples/s]27114 examples [00:23, 1244.24 examples/s]27239 examples [00:23, 1208.70 examples/s]27361 examples [00:23, 1191.98 examples/s]27481 examples [00:23, 1182.54 examples/s]27601 examples [00:23, 1185.29 examples/s]27724 examples [00:23, 1196.01 examples/s]27844 examples [00:23, 1157.63 examples/s]27968 examples [00:23, 1179.35 examples/s]28089 examples [00:24, 1188.00 examples/s]28209 examples [00:24, 1136.74 examples/s]28326 examples [00:24, 1144.70 examples/s]28441 examples [00:24, 1136.25 examples/s]28561 examples [00:24, 1152.26 examples/s]28679 examples [00:24, 1157.79 examples/s]28795 examples [00:24, 1153.80 examples/s]28917 examples [00:24, 1170.40 examples/s]29035 examples [00:24, 1149.50 examples/s]29155 examples [00:25, 1162.64 examples/s]29275 examples [00:25, 1173.42 examples/s]29393 examples [00:25, 1173.45 examples/s]29511 examples [00:25, 1167.52 examples/s]29628 examples [00:25, 1140.23 examples/s]29751 examples [00:25, 1163.99 examples/s]29868 examples [00:25, 1165.71 examples/s]29993 examples [00:25, 1188.99 examples/s]30113 examples [00:25, 1136.34 examples/s]30232 examples [00:25, 1149.09 examples/s]30348 examples [00:26, 1143.19 examples/s]30470 examples [00:26, 1162.22 examples/s]30592 examples [00:26, 1177.37 examples/s]30719 examples [00:26, 1202.62 examples/s]30841 examples [00:26, 1205.77 examples/s]30970 examples [00:26, 1229.73 examples/s]31094 examples [00:26, 1230.85 examples/s]31218 examples [00:26, 1223.70 examples/s]31341 examples [00:26, 1199.74 examples/s]31462 examples [00:26, 1183.03 examples/s]31585 examples [00:27, 1195.25 examples/s]31707 examples [00:27, 1199.96 examples/s]31832 examples [00:27, 1213.60 examples/s]31954 examples [00:27, 1207.65 examples/s]32078 examples [00:27, 1216.27 examples/s]32200 examples [00:27, 1215.95 examples/s]32322 examples [00:27, 1208.38 examples/s]32443 examples [00:27, 1208.66 examples/s]32564 examples [00:27, 1196.06 examples/s]32684 examples [00:27, 1181.89 examples/s]32803 examples [00:28, 1174.00 examples/s]32921 examples [00:28, 1172.21 examples/s]33039 examples [00:28, 1153.54 examples/s]33155 examples [00:28, 1152.89 examples/s]33279 examples [00:28, 1175.58 examples/s]33403 examples [00:28, 1193.97 examples/s]33528 examples [00:28, 1209.90 examples/s]33651 examples [00:28, 1214.88 examples/s]33773 examples [00:28, 1202.51 examples/s]33894 examples [00:28, 1196.56 examples/s]34015 examples [00:29, 1199.44 examples/s]34141 examples [00:29, 1214.39 examples/s]34270 examples [00:29, 1233.87 examples/s]34394 examples [00:29, 1233.20 examples/s]34518 examples [00:29, 1233.62 examples/s]34642 examples [00:29, 1221.56 examples/s]34765 examples [00:29, 1222.22 examples/s]34891 examples [00:29, 1232.50 examples/s]35015 examples [00:29, 1219.16 examples/s]35137 examples [00:30, 1211.82 examples/s]35259 examples [00:30, 1201.70 examples/s]35380 examples [00:30, 1180.62 examples/s]35501 examples [00:30, 1189.23 examples/s]35621 examples [00:30, 1184.67 examples/s]35740 examples [00:30, 1150.79 examples/s]35865 examples [00:30, 1177.56 examples/s]35991 examples [00:30, 1199.84 examples/s]36118 examples [00:30, 1217.55 examples/s]36241 examples [00:30, 1197.00 examples/s]36364 examples [00:31, 1203.83 examples/s]36488 examples [00:31, 1213.63 examples/s]36614 examples [00:31, 1224.80 examples/s]36740 examples [00:31, 1234.47 examples/s]36864 examples [00:31, 1121.73 examples/s]36982 examples [00:31, 1137.42 examples/s]37104 examples [00:31, 1160.29 examples/s]37230 examples [00:31, 1186.52 examples/s]37350 examples [00:31, 1168.04 examples/s]37468 examples [00:31, 1144.90 examples/s]37595 examples [00:32, 1179.20 examples/s]37714 examples [00:32, 1180.84 examples/s]37834 examples [00:32, 1185.80 examples/s]37960 examples [00:32, 1205.85 examples/s]38081 examples [00:32, 1206.94 examples/s]38207 examples [00:32, 1219.96 examples/s]38333 examples [00:32, 1229.92 examples/s]38457 examples [00:32, 1198.74 examples/s]38584 examples [00:32, 1217.69 examples/s]38707 examples [00:33, 1202.65 examples/s]38830 examples [00:33, 1209.29 examples/s]38952 examples [00:33, 1190.41 examples/s]39072 examples [00:33, 1186.66 examples/s]39193 examples [00:33, 1193.56 examples/s]39313 examples [00:33, 1185.40 examples/s]39433 examples [00:33, 1188.01 examples/s]39555 examples [00:33, 1194.91 examples/s]39675 examples [00:33, 1145.15 examples/s]39793 examples [00:33, 1154.37 examples/s]39909 examples [00:34, 1139.46 examples/s]40024 examples [00:34, 1072.31 examples/s]40139 examples [00:34, 1094.10 examples/s]40257 examples [00:34, 1116.38 examples/s]40376 examples [00:34, 1135.97 examples/s]40495 examples [00:34, 1151.38 examples/s]40616 examples [00:34, 1165.98 examples/s]40739 examples [00:34, 1183.26 examples/s]40859 examples [00:34, 1185.80 examples/s]40983 examples [00:34, 1199.11 examples/s]41105 examples [00:35, 1203.87 examples/s]41228 examples [00:35, 1210.07 examples/s]41350 examples [00:35, 1190.63 examples/s]41470 examples [00:35, 1192.17 examples/s]41591 examples [00:35, 1197.36 examples/s]41717 examples [00:35, 1213.46 examples/s]41839 examples [00:35, 1205.54 examples/s]41964 examples [00:35, 1216.69 examples/s]42086 examples [00:35, 1206.73 examples/s]42207 examples [00:35, 1191.18 examples/s]42328 examples [00:36, 1196.26 examples/s]42448 examples [00:36, 1184.77 examples/s]42568 examples [00:36, 1187.00 examples/s]42691 examples [00:36, 1198.49 examples/s]42811 examples [00:36, 1191.11 examples/s]42933 examples [00:36, 1198.88 examples/s]43053 examples [00:36, 1198.87 examples/s]43181 examples [00:36, 1220.30 examples/s]43306 examples [00:36, 1227.84 examples/s]43429 examples [00:37, 1182.35 examples/s]43548 examples [00:37, 1161.37 examples/s]43670 examples [00:37, 1176.82 examples/s]43793 examples [00:37, 1190.07 examples/s]43918 examples [00:37, 1205.30 examples/s]44039 examples [00:37, 1184.90 examples/s]44159 examples [00:37, 1187.67 examples/s]44281 examples [00:37, 1194.85 examples/s]44406 examples [00:37, 1208.83 examples/s]44528 examples [00:37, 1180.40 examples/s]44647 examples [00:38, 1166.98 examples/s]44766 examples [00:38, 1173.57 examples/s]44884 examples [00:38, 1160.97 examples/s]45002 examples [00:38, 1163.90 examples/s]45121 examples [00:38, 1168.83 examples/s]45238 examples [00:38, 1152.10 examples/s]45354 examples [00:38, 1124.17 examples/s]45473 examples [00:38, 1142.79 examples/s]45588 examples [00:38, 1137.68 examples/s]45708 examples [00:38, 1155.63 examples/s]45827 examples [00:39, 1163.70 examples/s]45948 examples [00:39, 1174.11 examples/s]46070 examples [00:39, 1187.22 examples/s]46196 examples [00:39, 1208.10 examples/s]46324 examples [00:39, 1227.72 examples/s]46447 examples [00:39, 1223.86 examples/s]46570 examples [00:39, 1224.47 examples/s]46697 examples [00:39, 1235.49 examples/s]46821 examples [00:39, 1232.12 examples/s]46946 examples [00:39, 1236.86 examples/s]47070 examples [00:40, 1193.27 examples/s]47190 examples [00:40, 1194.26 examples/s]47310 examples [00:40, 1144.74 examples/s]47428 examples [00:40, 1153.73 examples/s]47549 examples [00:40, 1169.29 examples/s]47669 examples [00:40, 1176.63 examples/s]47790 examples [00:40, 1184.42 examples/s]47914 examples [00:40, 1199.88 examples/s]48037 examples [00:40, 1208.57 examples/s]48165 examples [00:40, 1227.81 examples/s]48288 examples [00:41, 1223.15 examples/s]48411 examples [00:41, 1173.55 examples/s]48534 examples [00:41, 1187.98 examples/s]48660 examples [00:41, 1206.60 examples/s]48782 examples [00:41, 1165.11 examples/s]48900 examples [00:41, 1153.57 examples/s]49016 examples [00:41, 1148.40 examples/s]49132 examples [00:41, 1095.31 examples/s]49244 examples [00:41, 1100.43 examples/s]49355 examples [00:42, 1098.64 examples/s]49466 examples [00:42, 1092.62 examples/s]49578 examples [00:42, 1100.21 examples/s]49693 examples [00:42, 1112.80 examples/s]49811 examples [00:42, 1131.76 examples/s]49925 examples [00:42, 1131.71 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7818/50000 [00:00<00:00, 78178.21 examples/s] 43%|████▎     | 21361/50000 [00:00<00:00, 89532.79 examples/s] 71%|███████   | 35372/50000 [00:00<00:00, 100401.48 examples/s] 99%|█████████▉| 49667/50000 [00:00<00:00, 110245.11 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 896.65 examples/s]211 examples [00:00, 971.77 examples/s]334 examples [00:00, 1036.20 examples/s]461 examples [00:00, 1096.69 examples/s]580 examples [00:00, 1122.37 examples/s]688 examples [00:00, 1108.57 examples/s]807 examples [00:00, 1130.63 examples/s]927 examples [00:00, 1148.74 examples/s]1051 examples [00:00, 1174.09 examples/s]1170 examples [00:01, 1177.50 examples/s]1286 examples [00:01, 1144.85 examples/s]1402 examples [00:01, 1149.04 examples/s]1522 examples [00:01, 1163.09 examples/s]1645 examples [00:01, 1180.20 examples/s]1766 examples [00:01, 1186.93 examples/s]1885 examples [00:01, 1164.81 examples/s]2002 examples [00:01, 1164.45 examples/s]2119 examples [00:01, 1157.02 examples/s]2239 examples [00:01, 1166.60 examples/s]2356 examples [00:02, 1165.11 examples/s]2473 examples [00:02, 1152.55 examples/s]2593 examples [00:02, 1164.38 examples/s]2711 examples [00:02, 1168.90 examples/s]2830 examples [00:02, 1174.40 examples/s]2948 examples [00:02, 1173.66 examples/s]3066 examples [00:02, 1112.76 examples/s]3186 examples [00:02, 1137.24 examples/s]3304 examples [00:02, 1148.24 examples/s]3421 examples [00:02, 1153.20 examples/s]3537 examples [00:03, 1146.80 examples/s]3652 examples [00:03, 1139.77 examples/s]3767 examples [00:03, 1141.06 examples/s]3889 examples [00:03, 1161.05 examples/s]4011 examples [00:03, 1177.45 examples/s]4129 examples [00:03, 1175.45 examples/s]4247 examples [00:03, 1156.60 examples/s]4363 examples [00:03, 1156.60 examples/s]4479 examples [00:03, 1153.30 examples/s]4605 examples [00:03, 1182.34 examples/s]4731 examples [00:04, 1203.00 examples/s]4852 examples [00:04, 1192.51 examples/s]4975 examples [00:04, 1201.16 examples/s]5097 examples [00:04, 1205.11 examples/s]5218 examples [00:04, 1203.29 examples/s]5342 examples [00:04, 1211.74 examples/s]5464 examples [00:04, 1198.80 examples/s]5587 examples [00:04, 1206.26 examples/s]5708 examples [00:04, 1196.71 examples/s]5828 examples [00:04, 1196.97 examples/s]5951 examples [00:05, 1204.05 examples/s]6072 examples [00:05, 1194.72 examples/s]6192 examples [00:05, 1183.81 examples/s]6314 examples [00:05, 1192.06 examples/s]6436 examples [00:05, 1200.22 examples/s]6557 examples [00:05, 1202.98 examples/s]6678 examples [00:05, 1187.92 examples/s]6798 examples [00:05, 1189.02 examples/s]6921 examples [00:05, 1198.90 examples/s]7043 examples [00:05, 1204.36 examples/s]7164 examples [00:06, 1200.96 examples/s]7285 examples [00:06, 1193.08 examples/s]7405 examples [00:06, 1192.50 examples/s]7525 examples [00:06, 1175.57 examples/s]7646 examples [00:06, 1185.33 examples/s]7765 examples [00:06, 1183.19 examples/s]7884 examples [00:06, 1175.89 examples/s]8006 examples [00:06, 1186.34 examples/s]8125 examples [00:06, 1184.72 examples/s]8244 examples [00:07, 1148.41 examples/s]8364 examples [00:07, 1162.34 examples/s]8481 examples [00:07, 1155.87 examples/s]8598 examples [00:07, 1158.08 examples/s]8716 examples [00:07, 1163.94 examples/s]8838 examples [00:07, 1178.12 examples/s]8956 examples [00:07, 1170.93 examples/s]9074 examples [00:07, 1154.78 examples/s]9190 examples [00:07, 1134.53 examples/s]9310 examples [00:07, 1151.21 examples/s]9429 examples [00:08, 1162.35 examples/s]9546 examples [00:08, 1159.69 examples/s]9663 examples [00:08, 1143.88 examples/s]9784 examples [00:08, 1160.55 examples/s]9902 examples [00:08, 1164.28 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX33H2X/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteX33H2X/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87ee598a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87ee598a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87ee598a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8839f050b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f877880f400>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87ee598a60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87ee598a60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87ee598a60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87ccfc8278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87ccfc8080>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.05 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.05 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.05 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.59 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.59 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.74 file/s]2020-08-02 00:11:25.561943: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-02 00:11:25.566131: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-08-02 00:11:25.566419: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e5c7443410 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-02 00:11:25.566492: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 150398.16it/s] 69%|██████▉   | 6856704/9912422 [00:00<00:14, 214652.22it/s]9920512it [00:00, 43229765.07it/s]                           
0it [00:00, ?it/s]32768it [00:00, 493850.02it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:25, 65036.89it/s]1654784it [00:00, 5430605.76it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168687.26it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
