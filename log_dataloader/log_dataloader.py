
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6fe8a3b73a576c75c565441d8ffa2353b386e363', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fd1823caae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fd1823caae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fd1ed040488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fd1ed040488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fd20c26cea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fd20c26cea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd19a3e79d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd19a3e79d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd19a3e79d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 306513.81it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 393862.62it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 544810.54it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 769143.28it/s] 75%|███████▍  | 7397376/9912422 [00:00<00:02, 1086842.84it/s]9920512it [00:00, 10322253.33it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 138470.14it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158754.99it/s]  6%|▌         | 98304/1648877 [00:00<00:07, 204400.57it/s] 26%|██▋       | 434176/1648877 [00:00<00:04, 282334.05it/s]1654784it [00:00, 2608226.15it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50398.21it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd178e0b6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd178e1b940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd19a3e7620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd19a3e7620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fd19a3e7620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:21,  1.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:21,  1.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:20,  1.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:19,  1.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<01:38,  1.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:38,  1.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<01:38,  1.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<01:37,  1.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<01:36,  1.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<01:36,  1.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<01:07,  2.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<01:07,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<01:07,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<01:07,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<01:06,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:06,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:05,  2.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:46,  3.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:46,  3.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:46,  3.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:45,  3.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:45,  3.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:45,  3.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:32,  4.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:32,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:32,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:31,  4.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:22,  6.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:22,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:22,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:22,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:22,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:15,  8.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 10.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 14.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 22.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 22.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 22.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 22.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:04, 22.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:04, 22.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 26.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 31.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 35.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:02, 35.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 38.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 41.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 41.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 43.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 44.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 45.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 45.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 47.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 47.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 47.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 47.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 47.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 47.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 47.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:01, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:01, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 47.83 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 48.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 48.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 48.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 48.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 48.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 48.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 48.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 48.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 49.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 49.07 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 49.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 49.31 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 48.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 48.89 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 49.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 49.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 49.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 49.52 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 48.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 48.81 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 48.19 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 48.19 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.71 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.30s/ url]
0 examples [00:00, ? examples/s]2020-07-25 18:08:54.270203: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-25 18:08:54.286115: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-25 18:08:54.286408: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561e05f9a960 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-25 18:08:54.286430: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
14 examples [00:00, 139.46 examples/s]114 examples [00:00, 187.97 examples/s]216 examples [00:00, 248.79 examples/s]312 examples [00:00, 319.86 examples/s]405 examples [00:00, 398.16 examples/s]503 examples [00:00, 484.19 examples/s]600 examples [00:00, 569.16 examples/s]700 examples [00:00, 652.82 examples/s]797 examples [00:00, 722.85 examples/s]898 examples [00:01, 789.77 examples/s]997 examples [00:01, 840.52 examples/s]1100 examples [00:01, 889.30 examples/s]1198 examples [00:01, 912.59 examples/s]1296 examples [00:01, 920.93 examples/s]1394 examples [00:01, 935.88 examples/s]1491 examples [00:01, 921.50 examples/s]1586 examples [00:01, 918.20 examples/s]1684 examples [00:01, 935.29 examples/s]1781 examples [00:01, 944.64 examples/s]1879 examples [00:02, 954.92 examples/s]1983 examples [00:02, 978.77 examples/s]2083 examples [00:02, 983.92 examples/s]2182 examples [00:02, 974.39 examples/s]2281 examples [00:02, 978.70 examples/s]2380 examples [00:02, 971.65 examples/s]2479 examples [00:02, 974.53 examples/s]2577 examples [00:02, 960.21 examples/s]2674 examples [00:02, 958.64 examples/s]2770 examples [00:02, 947.69 examples/s]2868 examples [00:03, 957.13 examples/s]2964 examples [00:03, 948.19 examples/s]3064 examples [00:03, 962.62 examples/s]3162 examples [00:03, 967.56 examples/s]3259 examples [00:03, 963.45 examples/s]3363 examples [00:03, 982.40 examples/s]3464 examples [00:03, 987.97 examples/s]3565 examples [00:03, 993.53 examples/s]3671 examples [00:03, 1011.43 examples/s]3773 examples [00:03, 1003.86 examples/s]3876 examples [00:04, 1010.45 examples/s]3978 examples [00:04, 1006.19 examples/s]4079 examples [00:04, 1000.56 examples/s]4180 examples [00:04, 995.93 examples/s] 4280 examples [00:04, 981.97 examples/s]4379 examples [00:04, 983.08 examples/s]4478 examples [00:04, 983.57 examples/s]4580 examples [00:04, 991.89 examples/s]4681 examples [00:04, 996.99 examples/s]4781 examples [00:04, 975.83 examples/s]4882 examples [00:05, 985.37 examples/s]4983 examples [00:05, 990.37 examples/s]5084 examples [00:05, 993.24 examples/s]5184 examples [00:05, 981.86 examples/s]5283 examples [00:05, 977.84 examples/s]5386 examples [00:05, 992.07 examples/s]5489 examples [00:05, 1002.49 examples/s]5596 examples [00:05, 1021.75 examples/s]5699 examples [00:05, 1018.72 examples/s]5801 examples [00:05, 994.09 examples/s] 5901 examples [00:06, 989.93 examples/s]6001 examples [00:06, 963.42 examples/s]6098 examples [00:06, 885.30 examples/s]6188 examples [00:06, 883.63 examples/s]6278 examples [00:06, 877.20 examples/s]6367 examples [00:06, 873.60 examples/s]6455 examples [00:06, 853.95 examples/s]6546 examples [00:06, 870.01 examples/s]6634 examples [00:06, 867.73 examples/s]6722 examples [00:07, 840.17 examples/s]6818 examples [00:07, 871.38 examples/s]6906 examples [00:07, 862.34 examples/s]6993 examples [00:07, 861.37 examples/s]7086 examples [00:07, 878.51 examples/s]7187 examples [00:07, 913.41 examples/s]7289 examples [00:07, 942.03 examples/s]7388 examples [00:07, 954.61 examples/s]7484 examples [00:07, 947.02 examples/s]7582 examples [00:07, 955.79 examples/s]7681 examples [00:08, 965.06 examples/s]7783 examples [00:08, 980.65 examples/s]7887 examples [00:08, 997.52 examples/s]7989 examples [00:08, 1003.37 examples/s]8090 examples [00:08, 998.67 examples/s] 8192 examples [00:08, 1003.85 examples/s]8300 examples [00:08, 1021.74 examples/s]8403 examples [00:08, 1023.72 examples/s]8510 examples [00:08, 1036.47 examples/s]8614 examples [00:08, 1023.48 examples/s]8717 examples [00:09, 1017.44 examples/s]8819 examples [00:09, 997.34 examples/s] 8923 examples [00:09, 1007.27 examples/s]9025 examples [00:09, 1010.95 examples/s]9127 examples [00:09, 999.11 examples/s] 9228 examples [00:09, 1000.13 examples/s]9330 examples [00:09, 1005.74 examples/s]9435 examples [00:09, 1017.96 examples/s]9537 examples [00:09, 1009.49 examples/s]9639 examples [00:10, 999.10 examples/s] 9742 examples [00:10, 1006.34 examples/s]9846 examples [00:10, 1014.78 examples/s]9948 examples [00:10, 1009.78 examples/s]10050 examples [00:10, 942.76 examples/s]10149 examples [00:10, 956.39 examples/s]10253 examples [00:10, 977.84 examples/s]10355 examples [00:10, 987.97 examples/s]10456 examples [00:10, 994.45 examples/s]10559 examples [00:10, 1003.88 examples/s]10660 examples [00:11, 984.84 examples/s] 10759 examples [00:11, 968.86 examples/s]10857 examples [00:11, 971.82 examples/s]10961 examples [00:11, 991.26 examples/s]11061 examples [00:11, 989.64 examples/s]11161 examples [00:11, 965.70 examples/s]11258 examples [00:11, 966.02 examples/s]11355 examples [00:11, 957.67 examples/s]11454 examples [00:11, 965.30 examples/s]11551 examples [00:11, 932.60 examples/s]11645 examples [00:12, 917.45 examples/s]11740 examples [00:12, 925.39 examples/s]11838 examples [00:12, 938.85 examples/s]11934 examples [00:12, 942.86 examples/s]12029 examples [00:12, 941.35 examples/s]12124 examples [00:12, 936.49 examples/s]12218 examples [00:12, 933.82 examples/s]12316 examples [00:12, 945.43 examples/s]12411 examples [00:12, 934.40 examples/s]12505 examples [00:13, 933.73 examples/s]12599 examples [00:13, 934.00 examples/s]12697 examples [00:13, 946.68 examples/s]12795 examples [00:13, 954.72 examples/s]12893 examples [00:13, 961.38 examples/s]12992 examples [00:13, 968.06 examples/s]13089 examples [00:13, 951.35 examples/s]13189 examples [00:13, 962.83 examples/s]13286 examples [00:13, 912.11 examples/s]13378 examples [00:13, 898.41 examples/s]13473 examples [00:14, 911.14 examples/s]13565 examples [00:14, 911.91 examples/s]13661 examples [00:14, 923.90 examples/s]13754 examples [00:14, 914.19 examples/s]13856 examples [00:14, 942.96 examples/s]13951 examples [00:14, 937.40 examples/s]14045 examples [00:14, 927.36 examples/s]14140 examples [00:14, 932.04 examples/s]14237 examples [00:14, 942.06 examples/s]14332 examples [00:14, 939.05 examples/s]14426 examples [00:15, 931.88 examples/s]14520 examples [00:15, 907.57 examples/s]14621 examples [00:15, 934.96 examples/s]14718 examples [00:15, 944.40 examples/s]14814 examples [00:15, 947.50 examples/s]14912 examples [00:15, 956.73 examples/s]15008 examples [00:15, 936.99 examples/s]15107 examples [00:15, 949.95 examples/s]15203 examples [00:15, 941.65 examples/s]15298 examples [00:15, 919.13 examples/s]15394 examples [00:16, 930.67 examples/s]15488 examples [00:16, 917.50 examples/s]15580 examples [00:16, 876.56 examples/s]15671 examples [00:16, 884.25 examples/s]15760 examples [00:16, 878.37 examples/s]15854 examples [00:16, 894.58 examples/s]15944 examples [00:16, 895.78 examples/s]16034 examples [00:16, 896.09 examples/s]16130 examples [00:16, 911.72 examples/s]16222 examples [00:17, 901.22 examples/s]16315 examples [00:17, 909.18 examples/s]16411 examples [00:17, 923.43 examples/s]16509 examples [00:17, 936.90 examples/s]16603 examples [00:17, 933.80 examples/s]16701 examples [00:17, 945.12 examples/s]16796 examples [00:17, 930.61 examples/s]16890 examples [00:17, 911.87 examples/s]16982 examples [00:17, 895.91 examples/s]17073 examples [00:17, 899.81 examples/s]17167 examples [00:18, 909.27 examples/s]17259 examples [00:18, 904.43 examples/s]17351 examples [00:18, 906.61 examples/s]17444 examples [00:18, 911.22 examples/s]17536 examples [00:18, 901.66 examples/s]17633 examples [00:18, 918.38 examples/s]17729 examples [00:18, 926.83 examples/s]17823 examples [00:18, 929.28 examples/s]17916 examples [00:18, 906.67 examples/s]18008 examples [00:18, 910.41 examples/s]18100 examples [00:19, 907.72 examples/s]18199 examples [00:19, 929.16 examples/s]18300 examples [00:19, 951.62 examples/s]18405 examples [00:19, 977.47 examples/s]18507 examples [00:19, 987.20 examples/s]18606 examples [00:19, 987.20 examples/s]18708 examples [00:19, 994.12 examples/s]18808 examples [00:19, 994.40 examples/s]18908 examples [00:19, 994.55 examples/s]19009 examples [00:19, 997.26 examples/s]19112 examples [00:20, 1005.44 examples/s]19213 examples [00:20, 988.00 examples/s] 19312 examples [00:20, 980.92 examples/s]19412 examples [00:20, 985.95 examples/s]19513 examples [00:20, 990.66 examples/s]19613 examples [00:20, 989.52 examples/s]19713 examples [00:20, 991.44 examples/s]19815 examples [00:20, 999.08 examples/s]19915 examples [00:20, 989.30 examples/s]20014 examples [00:21, 931.23 examples/s]20108 examples [00:21, 920.43 examples/s]20205 examples [00:21, 932.62 examples/s]20308 examples [00:21, 958.96 examples/s]20405 examples [00:21, 915.40 examples/s]20498 examples [00:21, 874.36 examples/s]20587 examples [00:21, 877.26 examples/s]20676 examples [00:21, 865.44 examples/s]20764 examples [00:21, 865.45 examples/s]20857 examples [00:21, 882.08 examples/s]20952 examples [00:22, 900.93 examples/s]21046 examples [00:22, 910.92 examples/s]21138 examples [00:22, 902.81 examples/s]21236 examples [00:22, 922.37 examples/s]21330 examples [00:22, 926.44 examples/s]21426 examples [00:22, 934.87 examples/s]21527 examples [00:22, 954.74 examples/s]21623 examples [00:22, 945.67 examples/s]21718 examples [00:22, 889.54 examples/s]21808 examples [00:23, 875.52 examples/s]21901 examples [00:23, 889.86 examples/s]22000 examples [00:23, 915.59 examples/s]22093 examples [00:23, 910.17 examples/s]22187 examples [00:23, 916.21 examples/s]22282 examples [00:23, 924.85 examples/s]22375 examples [00:23, 925.55 examples/s]22468 examples [00:23, 921.34 examples/s]22561 examples [00:23, 895.92 examples/s]22651 examples [00:23, 891.50 examples/s]22750 examples [00:24, 918.67 examples/s]22843 examples [00:24, 894.54 examples/s]22934 examples [00:24, 898.46 examples/s]23025 examples [00:24, 887.32 examples/s]23118 examples [00:24, 898.01 examples/s]23214 examples [00:24, 915.72 examples/s]23315 examples [00:24, 941.76 examples/s]23412 examples [00:24, 948.38 examples/s]23512 examples [00:24, 962.04 examples/s]23609 examples [00:24, 952.10 examples/s]23706 examples [00:25, 955.93 examples/s]23802 examples [00:25, 939.21 examples/s]23897 examples [00:25, 927.18 examples/s]23994 examples [00:25, 937.74 examples/s]24088 examples [00:25, 901.85 examples/s]24181 examples [00:25, 908.67 examples/s]24279 examples [00:25, 928.09 examples/s]24378 examples [00:25, 943.02 examples/s]24473 examples [00:25, 931.76 examples/s]24567 examples [00:25, 924.99 examples/s]24660 examples [00:26, 885.08 examples/s]24749 examples [00:26, 847.01 examples/s]24835 examples [00:26, 830.33 examples/s]24920 examples [00:26, 832.99 examples/s]25008 examples [00:26, 844.19 examples/s]25097 examples [00:26, 854.95 examples/s]25187 examples [00:26, 865.85 examples/s]25281 examples [00:26, 884.48 examples/s]25373 examples [00:26, 892.94 examples/s]25463 examples [00:27, 892.47 examples/s]25553 examples [00:27, 879.90 examples/s]25648 examples [00:27, 899.14 examples/s]25739 examples [00:27, 897.27 examples/s]25831 examples [00:27, 903.87 examples/s]25922 examples [00:27, 867.33 examples/s]26014 examples [00:27, 882.45 examples/s]26109 examples [00:27, 901.47 examples/s]26200 examples [00:27, 883.13 examples/s]26289 examples [00:27, 845.03 examples/s]26380 examples [00:28, 858.05 examples/s]26468 examples [00:28, 863.27 examples/s]26561 examples [00:28, 878.21 examples/s]26650 examples [00:28, 849.13 examples/s]26748 examples [00:28, 882.69 examples/s]26840 examples [00:28, 891.47 examples/s]26931 examples [00:28, 894.84 examples/s]27023 examples [00:28, 900.25 examples/s]27114 examples [00:28, 892.82 examples/s]27206 examples [00:29, 897.89 examples/s]27298 examples [00:29, 901.90 examples/s]27389 examples [00:29, 903.47 examples/s]27484 examples [00:29, 915.09 examples/s]27585 examples [00:29, 941.05 examples/s]27680 examples [00:29, 942.09 examples/s]27777 examples [00:29, 947.73 examples/s]27873 examples [00:29, 949.43 examples/s]27969 examples [00:29, 948.70 examples/s]28064 examples [00:29, 939.82 examples/s]28160 examples [00:30, 945.15 examples/s]28257 examples [00:30, 951.69 examples/s]28353 examples [00:30, 932.88 examples/s]28449 examples [00:30, 939.07 examples/s]28543 examples [00:30, 834.11 examples/s]28635 examples [00:30, 857.78 examples/s]28723 examples [00:30, 828.81 examples/s]28814 examples [00:30, 851.15 examples/s]28901 examples [00:30, 843.73 examples/s]28992 examples [00:30, 861.36 examples/s]29083 examples [00:31, 873.94 examples/s]29173 examples [00:31, 880.26 examples/s]29262 examples [00:31, 880.41 examples/s]29351 examples [00:31, 852.86 examples/s]29437 examples [00:31, 850.02 examples/s]29532 examples [00:31, 877.45 examples/s]29631 examples [00:31, 905.89 examples/s]29727 examples [00:31, 920.20 examples/s]29820 examples [00:31, 903.23 examples/s]29912 examples [00:32, 907.33 examples/s]30003 examples [00:32, 819.03 examples/s]30098 examples [00:32, 853.54 examples/s]30192 examples [00:32, 875.59 examples/s]30281 examples [00:32, 866.37 examples/s]30376 examples [00:32, 889.27 examples/s]30467 examples [00:32, 894.78 examples/s]30563 examples [00:32, 911.27 examples/s]30656 examples [00:32, 916.12 examples/s]30748 examples [00:32, 887.61 examples/s]30846 examples [00:33, 911.07 examples/s]30942 examples [00:33, 924.78 examples/s]31037 examples [00:33, 932.08 examples/s]31131 examples [00:33, 915.02 examples/s]31223 examples [00:33, 890.20 examples/s]31316 examples [00:33, 901.65 examples/s]31410 examples [00:33, 912.16 examples/s]31510 examples [00:33, 934.46 examples/s]31604 examples [00:33, 931.10 examples/s]31698 examples [00:33, 901.85 examples/s]31790 examples [00:34, 904.50 examples/s]31888 examples [00:34, 925.48 examples/s]31984 examples [00:34, 935.49 examples/s]32078 examples [00:34, 924.17 examples/s]32174 examples [00:34, 931.94 examples/s]32271 examples [00:34, 942.53 examples/s]32366 examples [00:34, 939.44 examples/s]32461 examples [00:34, 914.04 examples/s]32553 examples [00:34, 905.55 examples/s]32644 examples [00:35, 891.49 examples/s]32734 examples [00:35, 870.87 examples/s]32825 examples [00:35, 881.54 examples/s]32920 examples [00:35, 900.51 examples/s]33011 examples [00:35, 880.24 examples/s]33100 examples [00:35, 883.11 examples/s]33190 examples [00:35, 887.71 examples/s]33287 examples [00:35, 910.89 examples/s]33379 examples [00:35, 912.55 examples/s]33478 examples [00:35, 931.98 examples/s]33572 examples [00:36, 922.79 examples/s]33665 examples [00:36, 892.95 examples/s]33755 examples [00:36, 863.28 examples/s]33845 examples [00:36, 871.67 examples/s]33934 examples [00:36, 876.66 examples/s]34022 examples [00:36, 850.56 examples/s]34110 examples [00:36, 858.57 examples/s]34197 examples [00:36, 857.10 examples/s]34283 examples [00:36, 850.30 examples/s]34372 examples [00:36, 861.65 examples/s]34469 examples [00:37, 890.90 examples/s]34566 examples [00:37, 911.42 examples/s]34659 examples [00:37, 916.70 examples/s]34756 examples [00:37, 929.93 examples/s]34850 examples [00:37, 926.33 examples/s]34943 examples [00:37, 920.00 examples/s]35036 examples [00:37, 914.93 examples/s]35134 examples [00:37, 930.85 examples/s]35228 examples [00:37, 904.58 examples/s]35319 examples [00:38, 885.43 examples/s]35408 examples [00:38, 886.40 examples/s]35499 examples [00:38, 891.27 examples/s]35598 examples [00:38, 916.71 examples/s]35699 examples [00:38, 942.32 examples/s]35794 examples [00:38, 937.99 examples/s]35889 examples [00:38, 905.51 examples/s]35989 examples [00:38, 929.02 examples/s]36087 examples [00:38, 942.25 examples/s]36186 examples [00:38, 956.08 examples/s]36282 examples [00:39, 915.56 examples/s]36377 examples [00:39, 924.32 examples/s]36476 examples [00:39, 942.72 examples/s]36571 examples [00:39, 941.94 examples/s]36666 examples [00:39, 929.92 examples/s]36760 examples [00:39, 914.62 examples/s]36852 examples [00:39, 911.03 examples/s]36948 examples [00:39, 924.75 examples/s]37048 examples [00:39, 943.88 examples/s]37143 examples [00:39, 940.34 examples/s]37238 examples [00:40, 922.04 examples/s]37331 examples [00:40, 910.24 examples/s]37428 examples [00:40, 925.51 examples/s]37523 examples [00:40, 931.06 examples/s]37620 examples [00:40, 939.39 examples/s]37715 examples [00:40, 923.65 examples/s]37808 examples [00:40, 925.18 examples/s]37902 examples [00:40, 928.78 examples/s]37995 examples [00:40, 893.06 examples/s]38086 examples [00:41, 897.35 examples/s]38184 examples [00:41, 919.76 examples/s]38280 examples [00:41, 929.41 examples/s]38375 examples [00:41, 933.90 examples/s]38472 examples [00:41, 944.05 examples/s]38567 examples [00:41, 928.91 examples/s]38666 examples [00:41, 945.22 examples/s]38769 examples [00:41, 967.29 examples/s]38870 examples [00:41, 978.01 examples/s]38973 examples [00:41, 991.78 examples/s]39073 examples [00:42, 981.39 examples/s]39173 examples [00:42, 986.28 examples/s]39272 examples [00:42, 977.00 examples/s]39372 examples [00:42, 982.24 examples/s]39471 examples [00:42, 980.18 examples/s]39570 examples [00:42, 978.94 examples/s]39668 examples [00:42, 957.90 examples/s]39764 examples [00:42, 941.89 examples/s]39859 examples [00:42, 928.11 examples/s]39952 examples [00:42, 904.97 examples/s]40043 examples [00:43, 834.76 examples/s]40132 examples [00:43, 848.86 examples/s]40224 examples [00:43, 865.45 examples/s]40312 examples [00:43, 859.23 examples/s]40402 examples [00:43, 870.96 examples/s]40490 examples [00:43, 854.98 examples/s]40578 examples [00:43, 860.97 examples/s]40672 examples [00:43, 881.47 examples/s]40761 examples [00:43, 882.63 examples/s]40856 examples [00:44, 899.67 examples/s]40951 examples [00:44, 913.94 examples/s]41045 examples [00:44, 919.72 examples/s]41140 examples [00:44, 927.02 examples/s]41235 examples [00:44, 933.49 examples/s]41330 examples [00:44, 938.06 examples/s]41424 examples [00:44, 933.99 examples/s]41522 examples [00:44, 944.66 examples/s]41617 examples [00:44, 936.54 examples/s]41716 examples [00:44, 949.80 examples/s]41812 examples [00:45, 943.50 examples/s]41907 examples [00:45, 943.75 examples/s]42002 examples [00:45, 931.56 examples/s]42096 examples [00:45, 918.97 examples/s]42196 examples [00:45, 941.05 examples/s]42295 examples [00:45, 952.75 examples/s]42391 examples [00:45, 940.89 examples/s]42486 examples [00:45, 914.31 examples/s]42581 examples [00:45, 924.69 examples/s]42677 examples [00:45, 934.06 examples/s]42773 examples [00:46, 941.10 examples/s]42868 examples [00:46, 937.10 examples/s]42962 examples [00:46, 920.85 examples/s]43055 examples [00:46, 884.79 examples/s]43144 examples [00:46, 879.03 examples/s]43242 examples [00:46, 904.79 examples/s]43334 examples [00:46, 906.30 examples/s]43429 examples [00:46, 918.51 examples/s]43522 examples [00:46, 916.61 examples/s]43619 examples [00:46, 929.48 examples/s]43722 examples [00:47, 955.68 examples/s]43818 examples [00:47, 927.29 examples/s]43912 examples [00:47, 924.88 examples/s]44009 examples [00:47, 937.71 examples/s]44104 examples [00:47, 933.29 examples/s]44198 examples [00:47, 930.74 examples/s]44292 examples [00:47, 918.83 examples/s]44385 examples [00:47, 909.16 examples/s]44477 examples [00:47, 888.84 examples/s]44567 examples [00:48, 891.95 examples/s]44657 examples [00:48, 865.48 examples/s]44744 examples [00:48, 813.02 examples/s]44836 examples [00:48, 841.94 examples/s]44922 examples [00:48, 841.66 examples/s]45007 examples [00:48, 834.91 examples/s]45091 examples [00:48, 811.31 examples/s]45184 examples [00:48, 843.23 examples/s]45270 examples [00:48, 847.73 examples/s]45358 examples [00:48, 855.05 examples/s]45450 examples [00:49, 872.76 examples/s]45543 examples [00:49, 888.62 examples/s]45641 examples [00:49, 911.67 examples/s]45740 examples [00:49, 931.90 examples/s]45837 examples [00:49, 941.68 examples/s]45932 examples [00:49, 927.67 examples/s]46026 examples [00:49, 929.10 examples/s]46123 examples [00:49, 939.39 examples/s]46219 examples [00:49, 944.13 examples/s]46314 examples [00:49, 919.55 examples/s]46407 examples [00:50, 864.78 examples/s]46496 examples [00:50, 871.20 examples/s]46591 examples [00:50, 892.86 examples/s]46692 examples [00:50, 924.89 examples/s]46788 examples [00:50, 934.13 examples/s]46889 examples [00:50, 952.97 examples/s]46986 examples [00:50, 955.66 examples/s]47084 examples [00:50, 955.26 examples/s]47181 examples [00:50, 958.82 examples/s]47278 examples [00:51, 936.03 examples/s]47372 examples [00:51, 913.52 examples/s]47464 examples [00:51, 856.68 examples/s]47560 examples [00:51, 883.30 examples/s]47656 examples [00:51, 903.03 examples/s]47751 examples [00:51, 915.07 examples/s]47844 examples [00:51, 911.33 examples/s]47939 examples [00:51, 920.70 examples/s]48033 examples [00:51, 926.18 examples/s]48127 examples [00:51, 928.10 examples/s]48220 examples [00:52, 927.29 examples/s]48313 examples [00:52, 927.21 examples/s]48412 examples [00:52, 944.40 examples/s]48509 examples [00:52, 951.02 examples/s]48608 examples [00:52, 961.28 examples/s]48705 examples [00:52, 959.45 examples/s]48802 examples [00:52, 938.99 examples/s]48902 examples [00:52, 955.20 examples/s]48998 examples [00:52, 944.10 examples/s]49093 examples [00:52, 920.41 examples/s]49186 examples [00:53, 918.05 examples/s]49278 examples [00:53, 917.02 examples/s]49373 examples [00:53, 924.97 examples/s]49466 examples [00:53, 923.15 examples/s]49559 examples [00:53, 916.87 examples/s]49656 examples [00:53, 930.28 examples/s]49750 examples [00:53, 932.76 examples/s]49844 examples [00:53, 926.26 examples/s]49937 examples [00:53, 924.27 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6664/50000 [00:00<00:00, 66639.11 examples/s] 35%|███▌      | 17699/50000 [00:00<00:00, 75625.65 examples/s] 58%|█████▊    | 28778/50000 [00:00<00:00, 83583.14 examples/s] 82%|████████▏ | 41000/50000 [00:00<00:00, 92339.72 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 732.82 examples/s]171 examples [00:00, 789.52 examples/s]265 examples [00:00, 827.10 examples/s]363 examples [00:00, 866.56 examples/s]459 examples [00:00, 892.02 examples/s]555 examples [00:00, 911.18 examples/s]653 examples [00:00, 929.22 examples/s]744 examples [00:00, 920.89 examples/s]837 examples [00:00, 923.27 examples/s]935 examples [00:01, 937.85 examples/s]1033 examples [00:01, 947.67 examples/s]1130 examples [00:01, 952.70 examples/s]1225 examples [00:01, 940.66 examples/s]1319 examples [00:01, 940.19 examples/s]1413 examples [00:01, 928.81 examples/s]1513 examples [00:01, 947.38 examples/s]1608 examples [00:01, 943.68 examples/s]1703 examples [00:01, 927.99 examples/s]1800 examples [00:01, 938.39 examples/s]1896 examples [00:02, 943.58 examples/s]1998 examples [00:02, 962.98 examples/s]2095 examples [00:02, 964.54 examples/s]2192 examples [00:02, 956.39 examples/s]2288 examples [00:02, 953.25 examples/s]2385 examples [00:02, 956.15 examples/s]2481 examples [00:02, 935.02 examples/s]2575 examples [00:02, 919.00 examples/s]2668 examples [00:02, 911.69 examples/s]2767 examples [00:02, 932.56 examples/s]2861 examples [00:03, 931.61 examples/s]2955 examples [00:03, 898.83 examples/s]3049 examples [00:03, 908.03 examples/s]3141 examples [00:03, 896.89 examples/s]3239 examples [00:03, 918.39 examples/s]3332 examples [00:03, 921.18 examples/s]3430 examples [00:03, 937.96 examples/s]3528 examples [00:03, 946.83 examples/s]3626 examples [00:03, 953.78 examples/s]3726 examples [00:03, 966.17 examples/s]3826 examples [00:04, 973.43 examples/s]3924 examples [00:04, 965.81 examples/s]4022 examples [00:04, 968.74 examples/s]4119 examples [00:04, 963.00 examples/s]4216 examples [00:04, 957.35 examples/s]4312 examples [00:04, 947.31 examples/s]4407 examples [00:04, 945.83 examples/s]4502 examples [00:04, 939.22 examples/s]4596 examples [00:04, 925.07 examples/s]4699 examples [00:04, 952.27 examples/s]4799 examples [00:05, 965.57 examples/s]4897 examples [00:05, 969.60 examples/s]4996 examples [00:05, 972.93 examples/s]5094 examples [00:05, 942.91 examples/s]5189 examples [00:05, 921.06 examples/s]5285 examples [00:05, 931.18 examples/s]5380 examples [00:05, 935.58 examples/s]5474 examples [00:05, 895.63 examples/s]5570 examples [00:05, 912.85 examples/s]5666 examples [00:06, 924.07 examples/s]5761 examples [00:06, 929.35 examples/s]5856 examples [00:06, 934.44 examples/s]5951 examples [00:06, 938.76 examples/s]6046 examples [00:06, 939.98 examples/s]6143 examples [00:06, 948.15 examples/s]6238 examples [00:06, 945.38 examples/s]6334 examples [00:06, 949.37 examples/s]6429 examples [00:06, 936.83 examples/s]6524 examples [00:06, 940.60 examples/s]6620 examples [00:07, 945.00 examples/s]6715 examples [00:07, 924.89 examples/s]6809 examples [00:07, 928.21 examples/s]6902 examples [00:07, 924.60 examples/s]6999 examples [00:07, 937.28 examples/s]7098 examples [00:07, 951.43 examples/s]7197 examples [00:07, 961.24 examples/s]7297 examples [00:07, 971.57 examples/s]7395 examples [00:07, 955.09 examples/s]7494 examples [00:07, 962.68 examples/s]7592 examples [00:08, 966.70 examples/s]7689 examples [00:08, 938.26 examples/s]7784 examples [00:08, 940.92 examples/s]7879 examples [00:08, 938.58 examples/s]7976 examples [00:08, 947.36 examples/s]8071 examples [00:08, 927.60 examples/s]8169 examples [00:08, 940.92 examples/s]8267 examples [00:08, 949.14 examples/s]8363 examples [00:08, 939.84 examples/s]8463 examples [00:08, 954.36 examples/s]8559 examples [00:09, 943.33 examples/s]8658 examples [00:09, 956.53 examples/s]8762 examples [00:09, 977.84 examples/s]8860 examples [00:09, 971.36 examples/s]8960 examples [00:09, 977.48 examples/s]9062 examples [00:09, 987.18 examples/s]9161 examples [00:09, 986.63 examples/s]9260 examples [00:09, 985.78 examples/s]9359 examples [00:09, 953.35 examples/s]9455 examples [00:10, 954.18 examples/s]9551 examples [00:10, 952.30 examples/s]9652 examples [00:10, 967.79 examples/s]9750 examples [00:10, 969.13 examples/s]9848 examples [00:10, 946.40 examples/s]9943 examples [00:10, 939.71 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSXK3XH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSXK3XH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd19a3e79d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd19a3e79d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd19a3e79d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd1205d36a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd1206a82b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd19a3e79d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd19a3e79d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd19a3e79d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd1206a8d68>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd178e1b6d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fd11ceac620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fd11ceac620> 

  function with postional parmater data_info <function split_train_valid at 0x7fd11ceac620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fd11ceac730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fd11ceac730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fd11ceac730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=0703593bcbdf19cb1aac81677d5d319e45cbc5702fa3c6649d58ecb83fb49058
  Stored in directory: /tmp/pip-ephem-wheel-cache-fe3mijlw/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<10:21:05, 23.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<7:16:41, 32.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.78M/862M [00:00<5:04:32, 47.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.1M/862M [00:00<3:30:39, 67.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.9M/862M [00:00<2:25:58, 95.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:00<1:41:02, 137kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:00<1:09:58, 195kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:01<48:30, 279kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<34:05, 394kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<13:23:09, 16.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:03<9:21:54, 23.9kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<6:34:11, 33.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<4:36:54, 48.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:06<3:14:56, 68.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<2:17:04, 97.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:08<1:37:39, 136kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:09<1:09:00, 192kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:10<50:12, 262kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<35:51, 367kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:12<27:04, 484kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:12<19:33, 669kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:14<15:43, 829kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:14<11:42, 1.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:16<10:14, 1.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:16<07:52, 1.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:18<07:32, 1.71MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:18<05:58, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<06:13, 2.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:20<05:02, 2.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<05:33, 2.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:22<04:35, 2.78MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:13, 2.43MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<04:21, 2.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<05:02, 2.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:13, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:57, 2.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:09, 3.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<04:53, 2.55MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<04:06, 3.04MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<04:50, 2.56MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:04, 3.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<03:19, 3.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<7:34:40, 27.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:34<5:17:02, 38.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<3:48:24, 53.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<2:40:48, 76.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<1:53:50, 107kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<1:20:33, 152kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<57:55, 210kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<41:11, 295kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<30:37, 395kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<22:32, 536kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<17:28, 688kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<12:52, 932kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<10:53, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<08:44, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<07:50, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<06:11, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<06:08, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<04:56, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<05:18, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<04:22, 2.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<04:54, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<04:18, 2.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<04:44, 2.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<03:59, 2.90MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<04:34, 2.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<03:53, 2.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:28, 2.56MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<03:48, 3.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<04:24, 2.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<03:46, 3.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<04:22, 2.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<03:40, 3.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<03:00, 3.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<6:55:51, 27.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<4:49:53, 38.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<3:28:25, 53.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<2:26:43, 76.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<1:43:48, 107kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<1:13:12, 152kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<52:44, 210kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<37:36, 294kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<27:51, 395kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<20:04, 547kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<15:44, 695kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<11:38, 939kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<09:49, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<07:25, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<06:52, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<05:23, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<05:28, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<04:25, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<04:46, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<03:53, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<04:23, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<03:38, 2.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<04:13, 2.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:26<03:25, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<04:04, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<03:25, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<04:02, 2.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<03:24, 3.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:00, 2.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<03:22, 3.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:58, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<03:20, 3.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<02:43, 3.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<6:14:04, 27.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:36<4:20:37, 38.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<3:07:32, 54.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<2:12:01, 76.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<1:33:21, 108kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<1:05:49, 153kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<47:24, 211kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<33:41, 296kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<25:02, 396kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<18:03, 549kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<14:08, 697kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<10:25, 944kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<08:48, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<06:37, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<06:10, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<04:53, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<04:54, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<03:57, 2.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<04:16, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<03:30, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:57, 2.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:33, 2.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:53, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<03:29, 2.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:50, 2.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:32, 2.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [01:59<02:33, 3.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<10:30, 885kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<08:07, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<07:01, 1.31MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<05:40, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<04:17, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<5:30:27, 27.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<3:50:53, 39.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<2:42:46, 56.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<1:54:48, 79.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:07<1:19:58, 113kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<1:03:08, 143kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<44:54, 201kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<32:34, 276kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<23:32, 381kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<17:41, 503kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<13:07, 678kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<10:27, 846kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<07:58, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:15<05:38, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<10:32, 832kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<08:02, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:17<05:40, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<20:02, 434kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<14:43, 591kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<11:31, 749kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<08:46, 984kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<07:22, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<05:53, 1.45MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<05:22, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<04:23, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:24<03:08, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<08:59, 936kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<07:01, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<06:07, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:58, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<04:41, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:59, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<03:07, 2.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<4:46:07, 28.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<3:19:53, 41.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<2:20:50, 58.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<1:39:11, 82.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<1:10:07, 116kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<49:42, 163kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<35:44, 225kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<25:39, 313kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<19:00, 419kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<13:57, 570kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<10:52, 726kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<08:22, 942kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<05:53, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<10:47, 726kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<08:10, 956kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<06:50, 1.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<05:25, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<04:55, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:47<04:04, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<03:58, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<03:25, 2.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<03:30, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<03:04, 2.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<03:15, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<02:53, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<03:07, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<02:47, 2.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<03:02, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:44, 2.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:12, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<4:12:02, 28.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<2:56:07, 41.3kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<2:03:51, 58.3kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<1:27:14, 82.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:01<1:00:41, 118kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<48:21, 148kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<34:33, 207kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:03<24:04, 295kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<24:36, 288kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<17:47, 398kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:05<12:25, 566kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<16:39, 422kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<12:13, 574kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<09:31, 730kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:08<07:15, 958kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<06:03, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<04:44, 1.45MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:11<03:21, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<07:02, 968kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<05:29, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<04:49, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<03:56, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<03:44, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<03:09, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<03:11, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:45, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:53, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:33, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:20<02:37, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:48, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<02:31, 2.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:42, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:51, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<02:12, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<3:49:56, 27.6kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<2:40:31, 39.4kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<1:52:49, 55.7kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<1:19:25, 79.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<55:58, 111kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<39:38, 157kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<28:24, 216kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<20:23, 301kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:32<14:11, 428kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<20:31, 296kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<14:52, 408kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:34<10:21, 580kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<1:02:47, 95.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<44:23, 135kB/s]   .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<31:39, 188kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<22:38, 262kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<16:33, 354kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<12:03, 486kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<09:13, 629kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:41<06:55, 837kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<05:39, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<04:25, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<03:54, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<03:09, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:45<02:14, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<05:46, 968kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<04:30, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<03:57, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<03:13, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<02:27, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<3:12:25, 28.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<2:14:08, 40.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<1:34:22, 57.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<1:06:21, 81.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:53<45:58, 116kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<1:00:57, 87.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<43:03, 124kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<30:34, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<21:49, 241kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<15:52, 328kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<11:31, 451kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<08:44, 587kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<06:31, 785kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<05:16, 960kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<04:06, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<03:35, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:04<03:07, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:05<02:12, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<04:24, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<03:29, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<03:08, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<02:35, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<02:30, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<02:09, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<02:12, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:57, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:12<01:23, 3.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<07:58, 583kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<05:58, 777kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<04:48, 953kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<03:44, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<02:45, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<2:43:31, 27.7kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<1:53:52, 39.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<1:19:57, 55.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<56:16, 79.2kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:20<38:57, 113kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<31:16, 141kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<22:14, 197kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<15:59, 271kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<11:32, 374kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<08:36, 495kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<06:21, 668kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<05:01, 835kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<03:51, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<03:16, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<02:52, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:30<02:01, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<06:59, 580kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:31<05:09, 783kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:31<03:37, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<04:51, 819kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<03:50, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:33<02:40, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<33:25, 117kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<23:40, 165kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<16:53, 228kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<12:07, 317kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<08:54, 424kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:39<06:32, 577kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<05:03, 733kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<03:50, 963kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<02:47, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<2:09:54, 28.2kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<1:30:15, 40.2kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<1:03:21, 56.7kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<44:34, 80.4kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<31:11, 113kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<22:04, 159kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<15:42, 220kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<11:15, 306kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<08:14, 410kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:50<06:01, 561kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:51<04:09, 796kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<19:25, 171kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<13:50, 239kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<09:59, 325kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:54<07:14, 447kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<05:27, 583kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<04:04, 779kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<03:15, 953kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<02:31, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<02:11, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:47, 1.70MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:40, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:25, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:24, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:13, 2.38MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:15, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:06, 2.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<00:52, 3.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:36:57, 28.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<1:07:10, 41.0kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<47:01, 57.8kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<33:07, 81.9kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:10<22:49, 117kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<17:13, 154kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<12:25, 213kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:12<08:34, 303kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<06:59, 368kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<05:05, 504kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<03:51, 650kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<03:01, 831kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:16<02:05, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<04:33, 536kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<03:21, 725kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:19<02:39, 894kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:19<02:04, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:45, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:26, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:21<00:59, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<24:49, 90.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:23<17:30, 127kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<12:14, 177kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<08:43, 248kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<06:14, 336kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<04:31, 462kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<03:22, 600kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<02:30, 808kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:59, 981kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<01:31, 1.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<01:07, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<1:07:11, 28.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:33<46:13, 40.6kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<32:14, 57.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<22:35, 81.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<15:34, 114kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<10:59, 161kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<07:41, 222kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<05:26, 311kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<03:56, 416kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<02:52, 567kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<02:10, 722kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:42<01:38, 949kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<01:19, 1.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:44<01:03, 1.41MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<00:55, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:46<00:45, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:42, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:36, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:36, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:29, 2.59MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:52<00:31, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<00:27, 2.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:28, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:25, 2.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:26, 2.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [05:56<00:23, 2.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<00:18, 3.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:57<35:55, 28.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:58<24:18, 41.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<16:42, 58.0kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<11:41, 82.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<07:47, 116kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<05:26, 163kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<03:41, 225kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<02:37, 315kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<01:48, 421kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<01:18, 574kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:57, 729kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:41, 985kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:32, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:25, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:21, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:16, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:14, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:12, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:11, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:15<00:09, 2.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:08, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<00:07, 2.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:06, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:05, 2.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:21<00:05, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:04, 2.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:02, 3.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<05:35, 29.4kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:23<02:58, 42.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<01:37, 58.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<01:03, 83.5kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:13, 117kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:06, 166kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 572/400000 [00:00<01:09, 5718.11it/s]  0%|          | 1423/400000 [00:00<01:02, 6341.00it/s]  1%|          | 2274/400000 [00:00<00:57, 6865.10it/s]  1%|          | 3146/400000 [00:00<00:54, 7332.66it/s]  1%|          | 3979/400000 [00:00<00:52, 7605.22it/s]  1%|          | 4822/400000 [00:00<00:50, 7834.45it/s]  1%|▏         | 5637/400000 [00:00<00:49, 7926.17it/s]  2%|▏         | 6434/400000 [00:00<00:49, 7938.83it/s]  2%|▏         | 7206/400000 [00:00<00:49, 7870.74it/s]  2%|▏         | 8053/400000 [00:01<00:48, 8041.03it/s]  2%|▏         | 8912/400000 [00:01<00:47, 8197.67it/s]  2%|▏         | 9731/400000 [00:01<00:47, 8193.82it/s]  3%|▎         | 10546/400000 [00:01<00:47, 8174.79it/s]  3%|▎         | 11394/400000 [00:01<00:47, 8262.03it/s]  3%|▎         | 12237/400000 [00:01<00:46, 8309.60it/s]  3%|▎         | 13103/400000 [00:01<00:46, 8409.43it/s]  3%|▎         | 13944/400000 [00:01<00:46, 8323.40it/s]  4%|▎         | 14777/400000 [00:01<00:47, 8087.78it/s]  4%|▍         | 15588/400000 [00:01<00:47, 8018.03it/s]  4%|▍         | 16412/400000 [00:02<00:47, 8080.15it/s]  4%|▍         | 17244/400000 [00:02<00:46, 8149.41it/s]  5%|▍         | 18081/400000 [00:02<00:46, 8213.77it/s]  5%|▍         | 18904/400000 [00:02<00:47, 8095.86it/s]  5%|▍         | 19715/400000 [00:02<00:47, 8006.23it/s]  5%|▌         | 20548/400000 [00:02<00:46, 8099.54it/s]  5%|▌         | 21379/400000 [00:02<00:46, 8161.38it/s]  6%|▌         | 22242/400000 [00:02<00:45, 8294.63it/s]  6%|▌         | 23073/400000 [00:02<00:45, 8237.76it/s]  6%|▌         | 23909/400000 [00:02<00:45, 8273.54it/s]  6%|▌         | 24794/400000 [00:03<00:44, 8437.11it/s]  6%|▋         | 25661/400000 [00:03<00:44, 8504.29it/s]  7%|▋         | 26566/400000 [00:03<00:43, 8660.22it/s]  7%|▋         | 27434/400000 [00:03<00:43, 8502.68it/s]  7%|▋         | 28286/400000 [00:03<00:44, 8371.30it/s]  7%|▋         | 29154/400000 [00:03<00:43, 8458.23it/s]  8%|▊         | 30025/400000 [00:03<00:43, 8531.62it/s]  8%|▊         | 30880/400000 [00:03<00:43, 8486.81it/s]  8%|▊         | 31730/400000 [00:03<00:44, 8366.60it/s]  8%|▊         | 32585/400000 [00:03<00:43, 8419.14it/s]  8%|▊         | 33430/400000 [00:04<00:43, 8426.74it/s]  9%|▊         | 34329/400000 [00:04<00:42, 8587.51it/s]  9%|▉         | 35275/400000 [00:04<00:41, 8830.89it/s]  9%|▉         | 36161/400000 [00:04<00:41, 8678.27it/s]  9%|▉         | 37087/400000 [00:04<00:41, 8844.21it/s]  9%|▉         | 37984/400000 [00:04<00:40, 8879.24it/s] 10%|▉         | 38935/400000 [00:04<00:39, 9059.21it/s] 10%|▉         | 39844/400000 [00:04<00:40, 8829.91it/s] 10%|█         | 40730/400000 [00:04<00:42, 8362.49it/s] 10%|█         | 41574/400000 [00:04<00:43, 8168.44it/s] 11%|█         | 42397/400000 [00:05<00:44, 7991.12it/s] 11%|█         | 43236/400000 [00:05<00:44, 8101.42it/s] 11%|█         | 44051/400000 [00:05<00:44, 8085.05it/s] 11%|█         | 44863/400000 [00:05<00:44, 7994.91it/s] 11%|█▏        | 45684/400000 [00:05<00:43, 8056.57it/s] 12%|█▏        | 46531/400000 [00:05<00:43, 8174.86it/s] 12%|█▏        | 47403/400000 [00:05<00:42, 8330.28it/s] 12%|█▏        | 48238/400000 [00:05<00:42, 8319.98it/s] 12%|█▏        | 49072/400000 [00:05<00:42, 8273.56it/s] 12%|█▏        | 49997/400000 [00:06<00:40, 8543.39it/s] 13%|█▎        | 50865/400000 [00:06<00:40, 8582.67it/s] 13%|█▎        | 51749/400000 [00:06<00:40, 8655.66it/s] 13%|█▎        | 52652/400000 [00:06<00:39, 8764.53it/s] 13%|█▎        | 53530/400000 [00:06<00:40, 8643.03it/s] 14%|█▎        | 54413/400000 [00:06<00:39, 8697.92it/s] 14%|█▍        | 55301/400000 [00:06<00:39, 8750.85it/s] 14%|█▍        | 56177/400000 [00:06<00:40, 8559.56it/s] 14%|█▍        | 57071/400000 [00:06<00:39, 8669.02it/s] 14%|█▍        | 57940/400000 [00:06<00:40, 8459.64it/s] 15%|█▍        | 58789/400000 [00:07<00:40, 8384.37it/s] 15%|█▍        | 59645/400000 [00:07<00:40, 8433.75it/s] 15%|█▌        | 60490/400000 [00:07<00:40, 8336.18it/s] 15%|█▌        | 61325/400000 [00:07<00:41, 8141.87it/s] 16%|█▌        | 62141/400000 [00:07<00:41, 8102.04it/s] 16%|█▌        | 63007/400000 [00:07<00:40, 8260.96it/s] 16%|█▌        | 63857/400000 [00:07<00:40, 8330.06it/s] 16%|█▌        | 64717/400000 [00:07<00:39, 8408.75it/s] 16%|█▋        | 65559/400000 [00:07<00:39, 8389.81it/s] 17%|█▋        | 66399/400000 [00:07<00:40, 8283.40it/s] 17%|█▋        | 67260/400000 [00:08<00:39, 8376.00it/s] 17%|█▋        | 68135/400000 [00:08<00:39, 8483.97it/s] 17%|█▋        | 69037/400000 [00:08<00:38, 8637.59it/s] 17%|█▋        | 69903/400000 [00:08<00:38, 8519.21it/s] 18%|█▊        | 70757/400000 [00:08<00:38, 8446.43it/s] 18%|█▊        | 71603/400000 [00:08<00:39, 8338.05it/s] 18%|█▊        | 72456/400000 [00:08<00:39, 8394.20it/s] 18%|█▊        | 73337/400000 [00:08<00:38, 8512.15it/s] 19%|█▊        | 74206/400000 [00:08<00:38, 8562.06it/s] 19%|█▉        | 75063/400000 [00:08<00:38, 8459.00it/s] 19%|█▉        | 75910/400000 [00:09<00:38, 8461.99it/s] 19%|█▉        | 76788/400000 [00:09<00:37, 8554.04it/s] 19%|█▉        | 77673/400000 [00:09<00:37, 8640.65it/s] 20%|█▉        | 78563/400000 [00:09<00:36, 8712.26it/s] 20%|█▉        | 79435/400000 [00:09<00:37, 8543.22it/s] 20%|██        | 80291/400000 [00:09<00:37, 8414.60it/s] 20%|██        | 81152/400000 [00:09<00:37, 8471.59it/s] 21%|██        | 82072/400000 [00:09<00:36, 8676.24it/s] 21%|██        | 82942/400000 [00:09<00:37, 8506.12it/s] 21%|██        | 83795/400000 [00:09<00:37, 8395.59it/s] 21%|██        | 84637/400000 [00:10<00:37, 8368.47it/s] 21%|██▏       | 85476/400000 [00:10<00:37, 8342.17it/s] 22%|██▏       | 86348/400000 [00:10<00:37, 8451.36it/s] 22%|██▏       | 87209/400000 [00:10<00:36, 8498.16it/s] 22%|██▏       | 88060/400000 [00:10<00:38, 8199.78it/s] 22%|██▏       | 88883/400000 [00:10<00:38, 8103.09it/s] 22%|██▏       | 89708/400000 [00:10<00:38, 8145.35it/s] 23%|██▎       | 90592/400000 [00:10<00:37, 8341.84it/s] 23%|██▎       | 91454/400000 [00:10<00:36, 8421.57it/s] 23%|██▎       | 92298/400000 [00:11<00:37, 8311.01it/s] 23%|██▎       | 93143/400000 [00:11<00:36, 8349.97it/s] 24%|██▎       | 94003/400000 [00:11<00:36, 8421.93it/s] 24%|██▎       | 94906/400000 [00:11<00:35, 8595.21it/s] 24%|██▍       | 95779/400000 [00:11<00:35, 8634.93it/s] 24%|██▍       | 96644/400000 [00:11<00:36, 8348.76it/s] 24%|██▍       | 97482/400000 [00:11<00:36, 8288.90it/s] 25%|██▍       | 98315/400000 [00:11<00:36, 8299.05it/s] 25%|██▍       | 99147/400000 [00:11<00:38, 7870.81it/s] 25%|██▍       | 99940/400000 [00:11<00:39, 7513.80it/s] 25%|██▌       | 100774/400000 [00:12<00:38, 7742.67it/s] 25%|██▌       | 101640/400000 [00:12<00:37, 7996.68it/s] 26%|██▌       | 102500/400000 [00:12<00:36, 8166.67it/s] 26%|██▌       | 103323/400000 [00:12<00:36, 8056.01it/s] 26%|██▌       | 104175/400000 [00:12<00:36, 8187.23it/s] 26%|██▋       | 105001/400000 [00:12<00:35, 8205.85it/s] 26%|██▋       | 105853/400000 [00:12<00:35, 8297.53it/s] 27%|██▋       | 106774/400000 [00:12<00:34, 8550.42it/s] 27%|██▋       | 107633/400000 [00:12<00:34, 8493.84it/s] 27%|██▋       | 108485/400000 [00:12<00:34, 8395.09it/s] 27%|██▋       | 109327/400000 [00:13<00:34, 8345.95it/s] 28%|██▊       | 110164/400000 [00:13<00:34, 8314.64it/s] 28%|██▊       | 110997/400000 [00:13<00:36, 7976.26it/s] 28%|██▊       | 111890/400000 [00:13<00:34, 8240.12it/s] 28%|██▊       | 112743/400000 [00:13<00:34, 8323.47it/s] 28%|██▊       | 113579/400000 [00:13<00:34, 8301.47it/s] 29%|██▊       | 114422/400000 [00:13<00:34, 8338.01it/s] 29%|██▉       | 115284/400000 [00:13<00:33, 8420.08it/s] 29%|██▉       | 116128/400000 [00:13<00:34, 8347.35it/s] 29%|██▉       | 116964/400000 [00:14<00:34, 8223.53it/s] 29%|██▉       | 117809/400000 [00:14<00:34, 8288.08it/s] 30%|██▉       | 118693/400000 [00:14<00:33, 8443.13it/s] 30%|██▉       | 119576/400000 [00:14<00:32, 8554.68it/s] 30%|███       | 120498/400000 [00:14<00:31, 8742.18it/s] 30%|███       | 121377/400000 [00:14<00:31, 8753.59it/s] 31%|███       | 122254/400000 [00:14<00:32, 8652.11it/s] 31%|███       | 123121/400000 [00:14<00:33, 8385.07it/s] 31%|███       | 123963/400000 [00:14<00:33, 8194.59it/s] 31%|███       | 124812/400000 [00:14<00:33, 8279.74it/s] 31%|███▏      | 125647/400000 [00:15<00:33, 8297.95it/s] 32%|███▏      | 126535/400000 [00:15<00:32, 8464.30it/s] 32%|███▏      | 127418/400000 [00:15<00:31, 8568.41it/s] 32%|███▏      | 128308/400000 [00:15<00:31, 8665.06it/s] 32%|███▏      | 129188/400000 [00:15<00:31, 8702.18it/s] 33%|███▎      | 130060/400000 [00:15<00:31, 8531.98it/s] 33%|███▎      | 130934/400000 [00:15<00:31, 8591.36it/s] 33%|███▎      | 131813/400000 [00:15<00:31, 8647.68it/s] 33%|███▎      | 132679/400000 [00:15<00:30, 8643.51it/s] 33%|███▎      | 133544/400000 [00:15<00:30, 8622.84it/s] 34%|███▎      | 134407/400000 [00:16<00:30, 8622.00it/s] 34%|███▍      | 135270/400000 [00:16<00:30, 8574.78it/s] 34%|███▍      | 136162/400000 [00:16<00:30, 8673.67it/s] 34%|███▍      | 137094/400000 [00:16<00:29, 8856.39it/s] 34%|███▍      | 137981/400000 [00:16<00:30, 8557.12it/s] 35%|███▍      | 138840/400000 [00:16<00:31, 8358.01it/s] 35%|███▍      | 139734/400000 [00:16<00:30, 8524.09it/s] 35%|███▌      | 140631/400000 [00:16<00:29, 8651.92it/s] 35%|███▌      | 141500/400000 [00:16<00:29, 8662.65it/s] 36%|███▌      | 142369/400000 [00:16<00:29, 8626.64it/s] 36%|███▌      | 143233/400000 [00:17<00:30, 8540.48it/s] 36%|███▌      | 144118/400000 [00:17<00:29, 8629.76it/s] 36%|███▋      | 145032/400000 [00:17<00:29, 8776.12it/s] 36%|███▋      | 145911/400000 [00:17<00:29, 8631.68it/s] 37%|███▋      | 146815/400000 [00:17<00:28, 8748.43it/s] 37%|███▋      | 147692/400000 [00:17<00:29, 8633.42it/s] 37%|███▋      | 148557/400000 [00:17<00:29, 8452.84it/s] 37%|███▋      | 149422/400000 [00:17<00:29, 8508.10it/s] 38%|███▊      | 150275/400000 [00:17<00:29, 8371.08it/s] 38%|███▊      | 151191/400000 [00:17<00:28, 8592.05it/s] 38%|███▊      | 152053/400000 [00:18<00:29, 8460.99it/s] 38%|███▊      | 152915/400000 [00:18<00:29, 8507.39it/s] 38%|███▊      | 153795/400000 [00:18<00:28, 8591.99it/s] 39%|███▊      | 154673/400000 [00:18<00:28, 8646.40it/s] 39%|███▉      | 155539/400000 [00:18<00:28, 8590.18it/s] 39%|███▉      | 156399/400000 [00:18<00:28, 8547.58it/s] 39%|███▉      | 157260/400000 [00:18<00:28, 8564.01it/s] 40%|███▉      | 158150/400000 [00:18<00:27, 8662.06it/s] 40%|███▉      | 159022/400000 [00:18<00:27, 8677.40it/s] 40%|███▉      | 159891/400000 [00:19<00:27, 8629.69it/s] 40%|████      | 160761/400000 [00:19<00:27, 8647.82it/s] 40%|████      | 161637/400000 [00:19<00:27, 8679.78it/s] 41%|████      | 162557/400000 [00:19<00:26, 8828.99it/s] 41%|████      | 163452/400000 [00:19<00:26, 8863.23it/s] 41%|████      | 164343/400000 [00:19<00:26, 8876.39it/s] 41%|████▏     | 165232/400000 [00:19<00:27, 8561.83it/s] 42%|████▏     | 166091/400000 [00:19<00:27, 8398.68it/s] 42%|████▏     | 166967/400000 [00:19<00:27, 8501.86it/s] 42%|████▏     | 167866/400000 [00:19<00:26, 8640.60it/s] 42%|████▏     | 168733/400000 [00:20<00:26, 8583.94it/s] 42%|████▏     | 169593/400000 [00:20<00:26, 8567.48it/s] 43%|████▎     | 170464/400000 [00:20<00:26, 8608.96it/s] 43%|████▎     | 171361/400000 [00:20<00:26, 8712.28it/s] 43%|████▎     | 172234/400000 [00:20<00:26, 8710.01it/s] 43%|████▎     | 173106/400000 [00:20<00:26, 8653.33it/s] 43%|████▎     | 173972/400000 [00:20<00:26, 8608.88it/s] 44%|████▎     | 174834/400000 [00:20<00:26, 8576.57it/s] 44%|████▍     | 175708/400000 [00:20<00:26, 8622.60it/s] 44%|████▍     | 176599/400000 [00:20<00:25, 8704.20it/s] 44%|████▍     | 177502/400000 [00:21<00:25, 8797.49it/s] 45%|████▍     | 178383/400000 [00:21<00:25, 8791.01it/s] 45%|████▍     | 179263/400000 [00:21<00:25, 8652.74it/s] 45%|████▌     | 180130/400000 [00:21<00:25, 8610.61it/s] 45%|████▌     | 181033/400000 [00:21<00:25, 8729.46it/s] 45%|████▌     | 181938/400000 [00:21<00:24, 8822.07it/s] 46%|████▌     | 182821/400000 [00:21<00:24, 8755.86it/s] 46%|████▌     | 183698/400000 [00:21<00:24, 8678.83it/s] 46%|████▌     | 184567/400000 [00:21<00:25, 8446.85it/s] 46%|████▋     | 185414/400000 [00:21<00:26, 8225.61it/s] 47%|████▋     | 186286/400000 [00:22<00:25, 8367.45it/s] 47%|████▋     | 187126/400000 [00:22<00:25, 8374.79it/s] 47%|████▋     | 187969/400000 [00:22<00:25, 8389.82it/s] 47%|████▋     | 188817/400000 [00:22<00:25, 8416.20it/s] 47%|████▋     | 189672/400000 [00:22<00:24, 8455.39it/s] 48%|████▊     | 190560/400000 [00:22<00:24, 8577.24it/s] 48%|████▊     | 191444/400000 [00:22<00:24, 8652.11it/s] 48%|████▊     | 192310/400000 [00:22<00:24, 8596.23it/s] 48%|████▊     | 193171/400000 [00:22<00:24, 8406.17it/s] 49%|████▊     | 194013/400000 [00:22<00:25, 8229.03it/s] 49%|████▊     | 194842/400000 [00:23<00:24, 8246.95it/s] 49%|████▉     | 195668/400000 [00:23<00:25, 8093.60it/s] 49%|████▉     | 196544/400000 [00:23<00:24, 8281.80it/s] 49%|████▉     | 197386/400000 [00:23<00:24, 8322.05it/s] 50%|████▉     | 198275/400000 [00:23<00:23, 8484.10it/s] 50%|████▉     | 199152/400000 [00:23<00:23, 8567.29it/s] 50%|█████     | 200017/400000 [00:23<00:23, 8591.62it/s] 50%|█████     | 200878/400000 [00:23<00:23, 8428.90it/s] 50%|█████     | 201740/400000 [00:23<00:23, 8483.75it/s] 51%|█████     | 202590/400000 [00:23<00:23, 8486.00it/s] 51%|█████     | 203442/400000 [00:24<00:23, 8491.59it/s] 51%|█████     | 204292/400000 [00:24<00:23, 8209.89it/s] 51%|█████▏    | 205160/400000 [00:24<00:23, 8343.28it/s] 52%|█████▏    | 206067/400000 [00:24<00:22, 8546.64it/s] 52%|█████▏    | 206925/400000 [00:24<00:22, 8483.81it/s] 52%|█████▏    | 207777/400000 [00:24<00:22, 8491.54it/s] 52%|█████▏    | 208648/400000 [00:24<00:22, 8553.97it/s] 52%|█████▏    | 209505/400000 [00:24<00:22, 8317.79it/s] 53%|█████▎    | 210428/400000 [00:24<00:22, 8571.42it/s] 53%|█████▎    | 211358/400000 [00:25<00:21, 8777.29it/s] 53%|█████▎    | 212280/400000 [00:25<00:21, 8905.42it/s] 53%|█████▎    | 213174/400000 [00:25<00:21, 8834.56it/s] 54%|█████▎    | 214069/400000 [00:25<00:20, 8867.41it/s] 54%|█████▎    | 214958/400000 [00:25<00:20, 8870.54it/s] 54%|█████▍    | 215847/400000 [00:25<00:20, 8816.28it/s] 54%|█████▍    | 216730/400000 [00:25<00:20, 8777.41it/s] 54%|█████▍    | 217609/400000 [00:25<00:21, 8595.30it/s] 55%|█████▍    | 218470/400000 [00:25<00:21, 8580.35it/s] 55%|█████▍    | 219383/400000 [00:25<00:20, 8737.69it/s] 55%|█████▌    | 220259/400000 [00:26<00:20, 8656.11it/s] 55%|█████▌    | 221126/400000 [00:26<00:21, 8503.08it/s] 55%|█████▌    | 221978/400000 [00:26<00:21, 8384.84it/s] 56%|█████▌    | 222890/400000 [00:26<00:20, 8591.58it/s] 56%|█████▌    | 223769/400000 [00:26<00:20, 8649.56it/s] 56%|█████▌    | 224640/400000 [00:26<00:20, 8665.77it/s] 56%|█████▋    | 225531/400000 [00:26<00:19, 8734.12it/s] 57%|█████▋    | 226406/400000 [00:26<00:20, 8656.97it/s] 57%|█████▋    | 227273/400000 [00:26<00:20, 8559.97it/s] 57%|█████▋    | 228140/400000 [00:26<00:20, 8592.34it/s] 57%|█████▋    | 229024/400000 [00:27<00:19, 8662.36it/s] 57%|█████▋    | 229912/400000 [00:27<00:19, 8726.33it/s] 58%|█████▊    | 230786/400000 [00:27<00:19, 8649.55it/s] 58%|█████▊    | 231652/400000 [00:27<00:19, 8498.04it/s] 58%|█████▊    | 232536/400000 [00:27<00:19, 8595.68it/s] 58%|█████▊    | 233397/400000 [00:27<00:19, 8556.40it/s] 59%|█████▊    | 234255/400000 [00:27<00:19, 8563.26it/s] 59%|█████▉    | 235112/400000 [00:27<00:19, 8536.58it/s] 59%|█████▉    | 235967/400000 [00:27<00:19, 8445.60it/s] 59%|█████▉    | 236888/400000 [00:27<00:18, 8658.68it/s] 59%|█████▉    | 237791/400000 [00:28<00:18, 8765.38it/s] 60%|█████▉    | 238675/400000 [00:28<00:18, 8785.62it/s] 60%|█████▉    | 239555/400000 [00:28<00:18, 8626.99it/s] 60%|██████    | 240420/400000 [00:28<00:18, 8453.46it/s] 60%|██████    | 241268/400000 [00:28<00:18, 8406.36it/s] 61%|██████    | 242133/400000 [00:28<00:18, 8475.03it/s] 61%|██████    | 243012/400000 [00:28<00:18, 8566.25it/s] 61%|██████    | 243873/400000 [00:28<00:18, 8579.12it/s] 61%|██████    | 244732/400000 [00:28<00:18, 8484.30it/s] 61%|██████▏   | 245582/400000 [00:28<00:18, 8417.34it/s] 62%|██████▏   | 246445/400000 [00:29<00:18, 8478.41it/s] 62%|██████▏   | 247306/400000 [00:29<00:17, 8513.96it/s] 62%|██████▏   | 248158/400000 [00:29<00:18, 8320.29it/s] 62%|██████▏   | 248992/400000 [00:29<00:18, 8264.76it/s] 62%|██████▏   | 249913/400000 [00:29<00:17, 8524.99it/s] 63%|██████▎   | 250769/400000 [00:29<00:17, 8466.80it/s] 63%|██████▎   | 251660/400000 [00:29<00:17, 8594.84it/s] 63%|██████▎   | 252592/400000 [00:29<00:16, 8798.24it/s] 63%|██████▎   | 253475/400000 [00:29<00:16, 8767.92it/s] 64%|██████▎   | 254403/400000 [00:30<00:16, 8913.53it/s] 64%|██████▍   | 255298/400000 [00:30<00:16, 8924.04it/s] 64%|██████▍   | 256228/400000 [00:30<00:15, 9031.49it/s] 64%|██████▍   | 257163/400000 [00:30<00:15, 9122.65it/s] 65%|██████▍   | 258077/400000 [00:30<00:16, 8839.14it/s] 65%|██████▍   | 258964/400000 [00:30<00:15, 8837.20it/s] 65%|██████▍   | 259850/400000 [00:30<00:15, 8769.56it/s] 65%|██████▌   | 260729/400000 [00:30<00:15, 8745.57it/s] 65%|██████▌   | 261605/400000 [00:30<00:15, 8659.14it/s] 66%|██████▌   | 262472/400000 [00:30<00:16, 8543.26it/s] 66%|██████▌   | 263364/400000 [00:31<00:15, 8649.68it/s] 66%|██████▌   | 264275/400000 [00:31<00:15, 8781.75it/s] 66%|██████▋   | 265165/400000 [00:31<00:15, 8815.40it/s] 67%|██████▋   | 266060/400000 [00:31<00:15, 8854.07it/s] 67%|██████▋   | 266947/400000 [00:31<00:15, 8665.33it/s] 67%|██████▋   | 267836/400000 [00:31<00:15, 8730.83it/s] 67%|██████▋   | 268745/400000 [00:31<00:14, 8835.17it/s] 67%|██████▋   | 269630/400000 [00:31<00:14, 8804.04it/s] 68%|██████▊   | 270512/400000 [00:31<00:15, 8561.69it/s] 68%|██████▊   | 271373/400000 [00:31<00:14, 8575.50it/s] 68%|██████▊   | 272232/400000 [00:32<00:14, 8518.13it/s] 68%|██████▊   | 273100/400000 [00:32<00:14, 8561.70it/s] 68%|██████▊   | 273998/400000 [00:32<00:14, 8682.26it/s] 69%|██████▊   | 274868/400000 [00:32<00:14, 8604.89it/s] 69%|██████▉   | 275730/400000 [00:32<00:15, 8195.82it/s] 69%|██████▉   | 276606/400000 [00:32<00:14, 8357.18it/s] 69%|██████▉   | 277446/400000 [00:32<00:14, 8304.32it/s] 70%|██████▉   | 278338/400000 [00:32<00:14, 8477.46it/s] 70%|██████▉   | 279189/400000 [00:32<00:14, 8324.91it/s] 70%|███████   | 280025/400000 [00:32<00:14, 8315.90it/s] 70%|███████   | 280922/400000 [00:33<00:14, 8499.31it/s] 70%|███████   | 281788/400000 [00:33<00:13, 8543.46it/s] 71%|███████   | 282675/400000 [00:33<00:13, 8638.67it/s] 71%|███████   | 283541/400000 [00:33<00:13, 8643.46it/s] 71%|███████   | 284407/400000 [00:33<00:13, 8513.10it/s] 71%|███████▏  | 285260/400000 [00:33<00:13, 8483.97it/s] 72%|███████▏  | 286117/400000 [00:33<00:13, 8507.64it/s] 72%|███████▏  | 286980/400000 [00:33<00:13, 8542.09it/s] 72%|███████▏  | 287835/400000 [00:33<00:13, 8458.66it/s] 72%|███████▏  | 288682/400000 [00:33<00:13, 8433.82it/s] 72%|███████▏  | 289526/400000 [00:34<00:13, 8420.98it/s] 73%|███████▎  | 290407/400000 [00:34<00:12, 8531.78it/s] 73%|███████▎  | 291271/400000 [00:34<00:12, 8563.10it/s] 73%|███████▎  | 292128/400000 [00:34<00:12, 8499.56it/s] 73%|███████▎  | 292979/400000 [00:34<00:12, 8327.32it/s] 73%|███████▎  | 293813/400000 [00:34<00:12, 8319.44it/s] 74%|███████▎  | 294670/400000 [00:34<00:12, 8391.70it/s] 74%|███████▍  | 295510/400000 [00:34<00:12, 8181.70it/s] 74%|███████▍  | 296349/400000 [00:34<00:12, 8242.83it/s] 74%|███████▍  | 297175/400000 [00:35<00:12, 8165.34it/s] 74%|███████▍  | 297997/400000 [00:35<00:12, 8181.52it/s] 75%|███████▍  | 298881/400000 [00:35<00:12, 8368.19it/s] 75%|███████▍  | 299728/400000 [00:35<00:11, 8398.01it/s] 75%|███████▌  | 300606/400000 [00:35<00:11, 8508.62it/s] 75%|███████▌  | 301458/400000 [00:35<00:11, 8367.49it/s] 76%|███████▌  | 302297/400000 [00:35<00:11, 8336.76it/s] 76%|███████▌  | 303200/400000 [00:35<00:11, 8529.75it/s] 76%|███████▌  | 304055/400000 [00:35<00:11, 8473.04it/s] 76%|███████▌  | 304904/400000 [00:35<00:11, 8437.44it/s] 76%|███████▋  | 305779/400000 [00:36<00:11, 8528.24it/s] 77%|███████▋  | 306652/400000 [00:36<00:10, 8585.45it/s] 77%|███████▋  | 307525/400000 [00:36<00:10, 8625.68it/s] 77%|███████▋  | 308412/400000 [00:36<00:10, 8696.43it/s] 77%|███████▋  | 309283/400000 [00:36<00:10, 8689.88it/s] 78%|███████▊  | 310153/400000 [00:36<00:10, 8534.48it/s] 78%|███████▊  | 311051/400000 [00:36<00:10, 8663.31it/s] 78%|███████▊  | 311965/400000 [00:36<00:10, 8800.04it/s] 78%|███████▊  | 312850/400000 [00:36<00:09, 8811.73it/s] 78%|███████▊  | 313733/400000 [00:36<00:09, 8677.17it/s] 79%|███████▊  | 314602/400000 [00:37<00:10, 8485.52it/s] 79%|███████▉  | 315460/400000 [00:37<00:09, 8512.92it/s] 79%|███████▉  | 316313/400000 [00:37<00:09, 8417.34it/s] 79%|███████▉  | 317185/400000 [00:37<00:09, 8503.86it/s] 80%|███████▉  | 318037/400000 [00:37<00:09, 8210.19it/s] 80%|███████▉  | 318899/400000 [00:37<00:09, 8326.29it/s] 80%|███████▉  | 319773/400000 [00:37<00:09, 8445.56it/s] 80%|████████  | 320635/400000 [00:37<00:09, 8496.23it/s] 80%|████████  | 321487/400000 [00:37<00:09, 8438.37it/s] 81%|████████  | 322333/400000 [00:37<00:09, 8372.16it/s] 81%|████████  | 323172/400000 [00:38<00:09, 8293.05it/s] 81%|████████  | 324056/400000 [00:38<00:08, 8448.06it/s] 81%|████████  | 324948/400000 [00:38<00:08, 8583.03it/s] 81%|████████▏ | 325830/400000 [00:38<00:08, 8651.37it/s] 82%|████████▏ | 326697/400000 [00:38<00:08, 8563.64it/s] 82%|████████▏ | 327555/400000 [00:38<00:08, 8480.05it/s] 82%|████████▏ | 328444/400000 [00:38<00:08, 8597.71it/s] 82%|████████▏ | 329305/400000 [00:38<00:08, 8309.34it/s] 83%|████████▎ | 330139/400000 [00:38<00:08, 8185.27it/s] 83%|████████▎ | 330960/400000 [00:39<00:08, 8108.21it/s] 83%|████████▎ | 331816/400000 [00:39<00:08, 8238.54it/s] 83%|████████▎ | 332702/400000 [00:39<00:07, 8415.03it/s] 83%|████████▎ | 333592/400000 [00:39<00:07, 8554.05it/s] 84%|████████▎ | 334489/400000 [00:39<00:07, 8673.33it/s] 84%|████████▍ | 335366/400000 [00:39<00:07, 8701.47it/s] 84%|████████▍ | 336238/400000 [00:39<00:07, 8513.10it/s] 84%|████████▍ | 337092/400000 [00:39<00:07, 8411.39it/s] 84%|████████▍ | 337942/400000 [00:39<00:07, 8435.87it/s] 85%|████████▍ | 338787/400000 [00:39<00:07, 8317.92it/s] 85%|████████▍ | 339620/400000 [00:40<00:07, 8197.43it/s] 85%|████████▌ | 340495/400000 [00:40<00:07, 8354.07it/s] 85%|████████▌ | 341388/400000 [00:40<00:06, 8518.23it/s] 86%|████████▌ | 342265/400000 [00:40<00:06, 8589.47it/s] 86%|████████▌ | 343126/400000 [00:40<00:06, 8443.66it/s] 86%|████████▌ | 343972/400000 [00:40<00:06, 8439.79it/s] 86%|████████▌ | 344853/400000 [00:40<00:06, 8547.15it/s] 86%|████████▋ | 345709/400000 [00:40<00:06, 8485.10it/s] 87%|████████▋ | 346584/400000 [00:40<00:06, 8561.09it/s] 87%|████████▋ | 347461/400000 [00:40<00:06, 8618.67it/s] 87%|████████▋ | 348324/400000 [00:41<00:06, 8275.43it/s] 87%|████████▋ | 349155/400000 [00:41<00:06, 8210.74it/s] 88%|████████▊ | 350027/400000 [00:41<00:05, 8355.77it/s] 88%|████████▊ | 350936/400000 [00:41<00:05, 8560.85it/s] 88%|████████▊ | 351795/400000 [00:41<00:05, 8551.10it/s] 88%|████████▊ | 352653/400000 [00:41<00:05, 8377.76it/s] 88%|████████▊ | 353493/400000 [00:41<00:05, 8327.80it/s] 89%|████████▊ | 354365/400000 [00:41<00:05, 8439.61it/s] 89%|████████▉ | 355211/400000 [00:41<00:05, 8230.83it/s] 89%|████████▉ | 356037/400000 [00:41<00:05, 8172.19it/s] 89%|████████▉ | 356856/400000 [00:42<00:05, 8102.70it/s] 89%|████████▉ | 357717/400000 [00:42<00:05, 8246.40it/s] 90%|████████▉ | 358603/400000 [00:42<00:04, 8419.42it/s] 90%|████████▉ | 359501/400000 [00:42<00:04, 8577.80it/s] 90%|█████████ | 360397/400000 [00:42<00:04, 8687.42it/s] 90%|█████████ | 361268/400000 [00:42<00:04, 8594.69it/s] 91%|█████████ | 362167/400000 [00:42<00:04, 8701.20it/s] 91%|█████████ | 363039/400000 [00:42<00:04, 8668.98it/s] 91%|█████████ | 363913/400000 [00:42<00:04, 8687.40it/s] 91%|█████████ | 364814/400000 [00:42<00:04, 8778.98it/s] 91%|█████████▏| 365693/400000 [00:43<00:03, 8694.88it/s] 92%|█████████▏| 366585/400000 [00:43<00:03, 8759.51it/s] 92%|█████████▏| 367501/400000 [00:43<00:03, 8873.19it/s] 92%|█████████▏| 368394/400000 [00:43<00:03, 8888.09it/s] 92%|█████████▏| 369284/400000 [00:43<00:03, 8881.84it/s] 93%|█████████▎| 370173/400000 [00:43<00:03, 8650.38it/s] 93%|█████████▎| 371040/400000 [00:43<00:03, 8510.45it/s] 93%|█████████▎| 371893/400000 [00:43<00:03, 8418.92it/s] 93%|█████████▎| 372737/400000 [00:43<00:03, 8324.74it/s] 93%|█████████▎| 373571/400000 [00:44<00:03, 8100.73it/s] 94%|█████████▎| 374384/400000 [00:44<00:03, 7873.85it/s] 94%|█████████▍| 375265/400000 [00:44<00:03, 8132.83it/s] 94%|█████████▍| 376157/400000 [00:44<00:02, 8353.55it/s] 94%|█████████▍| 377030/400000 [00:44<00:02, 8462.56it/s] 94%|█████████▍| 377956/400000 [00:44<00:02, 8686.03it/s] 95%|█████████▍| 378829/400000 [00:44<00:02, 8534.33it/s] 95%|█████████▍| 379686/400000 [00:44<00:02, 8445.98it/s] 95%|█████████▌| 380543/400000 [00:44<00:02, 8482.57it/s] 95%|█████████▌| 381418/400000 [00:44<00:02, 8559.39it/s] 96%|█████████▌| 382322/400000 [00:45<00:02, 8696.14it/s] 96%|█████████▌| 383200/400000 [00:45<00:01, 8721.11it/s] 96%|█████████▌| 384085/400000 [00:45<00:01, 8758.34it/s] 96%|█████████▌| 384977/400000 [00:45<00:01, 8806.10it/s] 96%|█████████▋| 385859/400000 [00:45<00:01, 8783.48it/s] 97%|█████████▋| 386747/400000 [00:45<00:01, 8811.26it/s] 97%|█████████▋| 387629/400000 [00:45<00:01, 8471.88it/s] 97%|█████████▋| 388492/400000 [00:45<00:01, 8517.16it/s] 97%|█████████▋| 389346/400000 [00:45<00:01, 8425.23it/s] 98%|█████████▊| 390191/400000 [00:45<00:01, 8043.90it/s] 98%|█████████▊| 391001/400000 [00:46<00:01, 8025.04it/s] 98%|█████████▊| 391815/400000 [00:46<00:01, 8057.29it/s] 98%|█████████▊| 392666/400000 [00:46<00:00, 8186.63it/s] 98%|█████████▊| 393565/400000 [00:46<00:00, 8408.69it/s] 99%|█████████▊| 394420/400000 [00:46<00:00, 8444.12it/s] 99%|█████████▉| 395298/400000 [00:46<00:00, 8541.55it/s] 99%|█████████▉| 396154/400000 [00:46<00:00, 8543.15it/s] 99%|█████████▉| 397081/400000 [00:46<00:00, 8747.86it/s] 99%|█████████▉| 397958/400000 [00:46<00:00, 8651.38it/s]100%|█████████▉| 398882/400000 [00:46<00:00, 8818.87it/s]100%|█████████▉| 399766/400000 [00:47<00:00, 8803.16it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8489.80it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fd11cece208>, <torchtext.data.dataset.TabularDataset object at 0x7fd11cece358>, <torchtext.vocab.Vocab object at 0x7fd11cece278>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 21.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 41.91 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 21.23 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 21.23 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.70 file/s]2020-07-25 18:18:06.991347: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-25 18:18:06.995385: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-25 18:18:06.996116: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559823de89d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-25 18:18:06.996133: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:33, 297605.44it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 384568.74it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 519385.38it/s] 57%|█████▋    | 5636096/9912422 [00:00<00:05, 738524.77it/s] 96%|█████████▌| 9527296/9912422 [00:00<00:00, 1044253.97it/s]9920512it [00:00, 10281871.27it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 136290.84it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 295257.36it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 382966.78it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 529572.96it/s]1654784it [00:00, 2508639.21it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 48742.82it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
