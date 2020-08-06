
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2076d24ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2076d24ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f20e1994510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f20e1994510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2100bc1ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f2100bc1ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f208ed3ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f208ed3ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f208ed3ca60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161690.60it/s] 20%|█▉        | 1966080/9912422 [00:00<00:34, 230139.80it/s] 56%|█████▋    | 5586944/9912422 [00:00<00:13, 327877.90it/s]9920512it [00:00, 24718707.58it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1430701.96it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464688.14it/s]1654784it [00:00, 12195822.81it/s]                         
0it [00:00, ?it/s]8192it [00:00, 263763.95it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f208cda0278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2075f749b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f208ed3c6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f208ed3c6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f208ed3c6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:56,  1.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:56,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:56,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:55,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:54,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:53,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:53,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:52,  1.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:18,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:18,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:18,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:17,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:17,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:16,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:16,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:15,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:15,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:14,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:14,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:52,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:52,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:51,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:51,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:50,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:50,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:50,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:49,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:49,  2.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:49,  2.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:34,  3.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:34,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:34,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:34,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:33,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:33,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:33,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:33,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:32,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:32,  3.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:23,  5.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:23,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:22,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:22,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:22,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:22,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:22,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:21,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:21,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:21,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:21,  5.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:14,  7.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:09, 10.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:06, 14.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:04, 19.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 25.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 25.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 31.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 31.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 39.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 47.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 47.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 55.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 55.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.86 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.86 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.50 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]
0 examples [00:00, ? examples/s]2020-08-06 06:08:43.867668: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-06 06:08:43.888381: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-06 06:08:43.888565: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55883b1b8440 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-06 06:08:43.888585: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  8.79 examples/s]93 examples [00:00, 12.51 examples/s]191 examples [00:00, 17.77 examples/s]281 examples [00:00, 25.18 examples/s]376 examples [00:00, 35.57 examples/s]471 examples [00:00, 50.00 examples/s]558 examples [00:00, 69.71 examples/s]654 examples [00:00, 96.57 examples/s]761 examples [00:00, 132.79 examples/s]856 examples [00:01, 178.90 examples/s]966 examples [00:01, 238.83 examples/s]1064 examples [00:01, 306.12 examples/s]1167 examples [00:01, 387.62 examples/s]1266 examples [00:01, 473.99 examples/s]1364 examples [00:01, 554.36 examples/s]1460 examples [00:01, 631.94 examples/s]1558 examples [00:01, 706.47 examples/s]1657 examples [00:01, 770.82 examples/s]1759 examples [00:01, 830.92 examples/s]1858 examples [00:02, 864.59 examples/s]1960 examples [00:02, 905.65 examples/s]2060 examples [00:02, 930.30 examples/s]2162 examples [00:02, 954.21 examples/s]2267 examples [00:02, 980.27 examples/s]2369 examples [00:02, 983.34 examples/s]2470 examples [00:02, 975.38 examples/s]2570 examples [00:02, 964.72 examples/s]2668 examples [00:02, 936.53 examples/s]2768 examples [00:02, 952.26 examples/s]2864 examples [00:03, 940.28 examples/s]2961 examples [00:03, 948.15 examples/s]3058 examples [00:03, 953.75 examples/s]3154 examples [00:03, 953.30 examples/s]3255 examples [00:03, 969.56 examples/s]3353 examples [00:03, 961.13 examples/s]3456 examples [00:03, 978.36 examples/s]3556 examples [00:03, 984.70 examples/s]3658 examples [00:03, 994.29 examples/s]3758 examples [00:03, 968.81 examples/s]3856 examples [00:04, 964.59 examples/s]3954 examples [00:04, 967.22 examples/s]4051 examples [00:04, 955.49 examples/s]4147 examples [00:04, 948.36 examples/s]4242 examples [00:04, 941.86 examples/s]4337 examples [00:04, 941.73 examples/s]4436 examples [00:04, 953.44 examples/s]4534 examples [00:04, 958.79 examples/s]4640 examples [00:04, 986.68 examples/s]4741 examples [00:05, 993.26 examples/s]4841 examples [00:05, 974.34 examples/s]4940 examples [00:05, 978.68 examples/s]5039 examples [00:05, 973.42 examples/s]5144 examples [00:05, 993.35 examples/s]5250 examples [00:05, 1011.00 examples/s]5352 examples [00:05, 998.50 examples/s] 5453 examples [00:05, 999.37 examples/s]5554 examples [00:05, 989.40 examples/s]5659 examples [00:05, 1004.83 examples/s]5761 examples [00:06, 1007.90 examples/s]5862 examples [00:06, 978.37 examples/s] 5961 examples [00:06, 979.97 examples/s]6060 examples [00:06, 976.23 examples/s]6158 examples [00:06, 953.68 examples/s]6256 examples [00:06, 958.80 examples/s]6353 examples [00:06, 946.69 examples/s]6448 examples [00:06, 940.49 examples/s]6543 examples [00:06, 930.44 examples/s]6637 examples [00:06, 926.22 examples/s]6730 examples [00:07, 919.11 examples/s]6822 examples [00:07, 909.04 examples/s]6913 examples [00:07, 862.72 examples/s]7005 examples [00:07, 878.48 examples/s]7098 examples [00:07, 892.24 examples/s]7191 examples [00:07, 902.97 examples/s]7283 examples [00:07, 907.48 examples/s]7379 examples [00:07, 920.86 examples/s]7479 examples [00:07, 941.50 examples/s]7574 examples [00:08, 923.94 examples/s]7667 examples [00:08, 920.72 examples/s]7760 examples [00:08, 920.29 examples/s]7859 examples [00:08, 938.17 examples/s]7959 examples [00:08, 955.68 examples/s]8062 examples [00:08, 976.30 examples/s]8165 examples [00:08, 989.63 examples/s]8265 examples [00:08, 971.65 examples/s]8365 examples [00:08, 977.64 examples/s]8468 examples [00:08, 990.76 examples/s]8568 examples [00:09, 982.08 examples/s]8668 examples [00:09, 986.61 examples/s]8767 examples [00:09, 970.33 examples/s]8867 examples [00:09, 978.37 examples/s]8971 examples [00:09, 994.19 examples/s]9078 examples [00:09, 1012.88 examples/s]9180 examples [00:09, 1005.08 examples/s]9281 examples [00:09, 1000.40 examples/s]9385 examples [00:09, 1009.00 examples/s]9486 examples [00:09, 993.83 examples/s] 9587 examples [00:10, 996.91 examples/s]9689 examples [00:10, 1001.87 examples/s]9790 examples [00:10, 985.57 examples/s] 9891 examples [00:10, 989.84 examples/s]9996 examples [00:10, 1005.79 examples/s]10097 examples [00:10, 934.54 examples/s]10193 examples [00:10, 939.41 examples/s]10299 examples [00:10, 970.09 examples/s]10398 examples [00:10, 975.56 examples/s]10498 examples [00:10, 980.72 examples/s]10597 examples [00:11, 954.50 examples/s]10700 examples [00:11, 975.44 examples/s]10801 examples [00:11, 982.98 examples/s]10900 examples [00:11, 977.59 examples/s]10998 examples [00:11, 953.02 examples/s]11094 examples [00:11, 941.08 examples/s]11189 examples [00:11, 936.49 examples/s]11286 examples [00:11, 943.98 examples/s]11382 examples [00:11, 947.51 examples/s]11484 examples [00:12, 966.33 examples/s]11591 examples [00:12, 994.53 examples/s]11691 examples [00:12, 973.48 examples/s]11800 examples [00:12, 1004.28 examples/s]11905 examples [00:12, 1015.97 examples/s]12007 examples [00:12, 1008.82 examples/s]12109 examples [00:12, 999.80 examples/s] 12210 examples [00:12, 970.16 examples/s]12313 examples [00:12, 985.65 examples/s]12412 examples [00:12, 972.22 examples/s]12511 examples [00:13, 972.22 examples/s]12609 examples [00:13, 937.26 examples/s]12704 examples [00:13, 910.47 examples/s]12804 examples [00:13, 934.50 examples/s]12898 examples [00:13, 906.27 examples/s]13001 examples [00:13, 939.20 examples/s]13106 examples [00:13, 969.21 examples/s]13204 examples [00:13, 953.01 examples/s]13301 examples [00:13, 957.12 examples/s]13408 examples [00:13, 988.10 examples/s]13508 examples [00:14, 972.71 examples/s]13612 examples [00:14, 989.38 examples/s]13712 examples [00:14, 984.89 examples/s]13813 examples [00:14, 991.25 examples/s]13914 examples [00:14, 994.70 examples/s]14016 examples [00:14, 999.25 examples/s]14117 examples [00:14, 995.35 examples/s]14218 examples [00:14, 999.57 examples/s]14324 examples [00:14, 1016.47 examples/s]14428 examples [00:15, 1023.33 examples/s]14535 examples [00:15, 1034.17 examples/s]14639 examples [00:15, 1032.18 examples/s]14743 examples [00:15, 1026.61 examples/s]14846 examples [00:15, 997.74 examples/s] 14951 examples [00:15, 1012.39 examples/s]15053 examples [00:15, 1010.71 examples/s]15155 examples [00:15, 984.12 examples/s] 15254 examples [00:15, 979.66 examples/s]15359 examples [00:15, 998.17 examples/s]15460 examples [00:16, 995.50 examples/s]15562 examples [00:16, 1002.16 examples/s]15663 examples [00:16, 997.47 examples/s] 15763 examples [00:16, 981.37 examples/s]15866 examples [00:16, 993.65 examples/s]15966 examples [00:16, 968.79 examples/s]16064 examples [00:16, 964.25 examples/s]16161 examples [00:16, 965.20 examples/s]16258 examples [00:16, 889.25 examples/s]16349 examples [00:16, 875.35 examples/s]16438 examples [00:17, 846.74 examples/s]16530 examples [00:17, 866.40 examples/s]16618 examples [00:17, 847.22 examples/s]16708 examples [00:17, 859.02 examples/s]16800 examples [00:17, 874.13 examples/s]16888 examples [00:17, 869.71 examples/s]16980 examples [00:17, 882.94 examples/s]17074 examples [00:17, 899.22 examples/s]17172 examples [00:17, 920.54 examples/s]17271 examples [00:18, 940.16 examples/s]17368 examples [00:18, 948.26 examples/s]17467 examples [00:18, 959.70 examples/s]17564 examples [00:18, 948.43 examples/s]17660 examples [00:18, 949.99 examples/s]17757 examples [00:18, 954.40 examples/s]17855 examples [00:18, 961.33 examples/s]17960 examples [00:18, 985.18 examples/s]18061 examples [00:18, 989.75 examples/s]18161 examples [00:18, 991.05 examples/s]18262 examples [00:19, 996.06 examples/s]18365 examples [00:19, 1004.20 examples/s]18466 examples [00:19, 987.55 examples/s] 18565 examples [00:19, 987.85 examples/s]18665 examples [00:19, 990.82 examples/s]18770 examples [00:19, 1006.74 examples/s]18871 examples [00:19, 997.45 examples/s] 18976 examples [00:19, 1009.82 examples/s]19081 examples [00:19, 1020.70 examples/s]19185 examples [00:19, 1024.40 examples/s]19289 examples [00:20, 1028.19 examples/s]19392 examples [00:20, 1016.55 examples/s]19495 examples [00:20, 1018.28 examples/s]19597 examples [00:20, 997.42 examples/s] 19703 examples [00:20, 1014.48 examples/s]19806 examples [00:20, 1018.61 examples/s]19911 examples [00:20, 1025.00 examples/s]20014 examples [00:20, 949.76 examples/s] 20111 examples [00:20, 946.27 examples/s]20207 examples [00:20, 947.42 examples/s]20313 examples [00:21, 976.77 examples/s]20415 examples [00:21, 986.20 examples/s]20515 examples [00:21, 968.10 examples/s]20613 examples [00:21, 963.06 examples/s]20710 examples [00:21, 959.40 examples/s]20813 examples [00:21, 977.07 examples/s]20911 examples [00:21, 976.56 examples/s]21010 examples [00:21, 979.01 examples/s]21109 examples [00:21, 963.23 examples/s]21213 examples [00:22, 982.52 examples/s]21313 examples [00:22, 985.29 examples/s]21420 examples [00:22, 1008.50 examples/s]21523 examples [00:22, 1013.22 examples/s]21625 examples [00:22, 1008.32 examples/s]21726 examples [00:22, 1003.83 examples/s]21827 examples [00:22, 965.95 examples/s] 21924 examples [00:22, 963.13 examples/s]22028 examples [00:22, 983.93 examples/s]22129 examples [00:22, 989.54 examples/s]22234 examples [00:23, 1005.81 examples/s]22335 examples [00:23, 994.21 examples/s] 22435 examples [00:23, 981.60 examples/s]22534 examples [00:23, 965.48 examples/s]22631 examples [00:23, 944.36 examples/s]22727 examples [00:23, 948.77 examples/s]22837 examples [00:23, 988.38 examples/s]22941 examples [00:23, 994.38 examples/s]23047 examples [00:23, 1012.81 examples/s]23149 examples [00:23, 989.46 examples/s] 23253 examples [00:24, 1001.01 examples/s]23357 examples [00:24, 1009.65 examples/s]23459 examples [00:24, 987.26 examples/s] 23561 examples [00:24, 996.24 examples/s]23661 examples [00:24, 979.30 examples/s]23760 examples [00:24, 978.16 examples/s]23863 examples [00:24, 992.14 examples/s]23964 examples [00:24, 994.90 examples/s]24064 examples [00:24, 987.89 examples/s]24163 examples [00:25, 960.95 examples/s]24268 examples [00:25, 985.85 examples/s]24376 examples [00:25, 1011.32 examples/s]24481 examples [00:25, 1020.35 examples/s]24590 examples [00:25, 1037.81 examples/s]24696 examples [00:25, 1042.83 examples/s]24801 examples [00:25, 1001.61 examples/s]24902 examples [00:25, 994.65 examples/s] 25005 examples [00:25, 1004.09 examples/s]25108 examples [00:25, 1009.59 examples/s]25210 examples [00:26, 987.44 examples/s] 25313 examples [00:26, 998.59 examples/s]25414 examples [00:26, 987.37 examples/s]25516 examples [00:26, 996.46 examples/s]25616 examples [00:26, 996.56 examples/s]25716 examples [00:26, 955.18 examples/s]25816 examples [00:26, 967.75 examples/s]25921 examples [00:26, 989.86 examples/s]26021 examples [00:26, 986.98 examples/s]26120 examples [00:26, 982.68 examples/s]26220 examples [00:27, 987.22 examples/s]26319 examples [00:27, 940.38 examples/s]26420 examples [00:27, 959.98 examples/s]26517 examples [00:27, 957.46 examples/s]26614 examples [00:27, 960.64 examples/s]26711 examples [00:27, 944.13 examples/s]26806 examples [00:27, 930.65 examples/s]26908 examples [00:27, 952.21 examples/s]27012 examples [00:27, 975.93 examples/s]27114 examples [00:27, 986.89 examples/s]27213 examples [00:28, 970.80 examples/s]27311 examples [00:28, 967.23 examples/s]27409 examples [00:28, 970.88 examples/s]27509 examples [00:28, 979.17 examples/s]27612 examples [00:28, 992.51 examples/s]27712 examples [00:28, 970.34 examples/s]27810 examples [00:28, 967.97 examples/s]27907 examples [00:28, 956.43 examples/s]28009 examples [00:28, 971.74 examples/s]28107 examples [00:29, 964.82 examples/s]28207 examples [00:29, 972.62 examples/s]28305 examples [00:29, 969.82 examples/s]28403 examples [00:29, 972.26 examples/s]28507 examples [00:29, 990.45 examples/s]28607 examples [00:29, 992.33 examples/s]28712 examples [00:29, 1006.87 examples/s]28813 examples [00:29, 1006.68 examples/s]28914 examples [00:29, 995.06 examples/s] 29020 examples [00:29, 1011.94 examples/s]29122 examples [00:30, 1005.90 examples/s]29223 examples [00:30, 1003.36 examples/s]29324 examples [00:30, 998.38 examples/s] 29424 examples [00:30, 990.54 examples/s]29524 examples [00:30, 992.97 examples/s]29624 examples [00:30, 986.09 examples/s]29723 examples [00:30, 981.95 examples/s]29822 examples [00:30, 981.33 examples/s]29925 examples [00:30, 994.28 examples/s]30025 examples [00:30, 939.14 examples/s]30120 examples [00:31, 937.05 examples/s]30221 examples [00:31, 957.25 examples/s]30328 examples [00:31, 987.69 examples/s]30428 examples [00:31, 978.39 examples/s]30530 examples [00:31, 988.72 examples/s]30630 examples [00:31, 972.22 examples/s]30728 examples [00:31, 957.54 examples/s]30826 examples [00:31, 962.21 examples/s]30926 examples [00:31, 972.75 examples/s]31032 examples [00:31, 996.95 examples/s]31132 examples [00:32, 992.14 examples/s]31232 examples [00:32, 988.77 examples/s]31332 examples [00:32, 982.80 examples/s]31434 examples [00:32, 992.50 examples/s]31538 examples [00:32, 1004.50 examples/s]31639 examples [00:32, 1002.07 examples/s]31742 examples [00:32, 1009.86 examples/s]31844 examples [00:32, 1004.68 examples/s]31945 examples [00:32, 1002.27 examples/s]32046 examples [00:33, 958.59 examples/s] 32143 examples [00:33, 937.60 examples/s]32241 examples [00:33, 949.18 examples/s]32344 examples [00:33, 971.13 examples/s]32444 examples [00:33, 978.97 examples/s]32543 examples [00:33, 973.37 examples/s]32641 examples [00:33, 959.60 examples/s]32738 examples [00:33, 961.21 examples/s]32835 examples [00:33, 952.60 examples/s]32931 examples [00:33, 938.64 examples/s]33025 examples [00:34, 934.00 examples/s]33120 examples [00:34, 937.09 examples/s]33222 examples [00:34, 958.38 examples/s]33324 examples [00:34, 974.50 examples/s]33423 examples [00:34, 975.44 examples/s]33521 examples [00:34, 970.05 examples/s]33619 examples [00:34, 968.39 examples/s]33719 examples [00:34, 975.54 examples/s]33822 examples [00:34, 990.02 examples/s]33932 examples [00:34, 1019.58 examples/s]34035 examples [00:35, 1013.57 examples/s]34137 examples [00:35, 994.90 examples/s] 34238 examples [00:35, 999.15 examples/s]34339 examples [00:35, 1001.28 examples/s]34445 examples [00:35, 1016.81 examples/s]34552 examples [00:35, 1029.69 examples/s]34656 examples [00:35, 1012.33 examples/s]34758 examples [00:35, 991.55 examples/s] 34858 examples [00:35, 972.83 examples/s]34967 examples [00:35, 1004.86 examples/s]35068 examples [00:36, 986.85 examples/s] 35168 examples [00:36, 974.68 examples/s]35269 examples [00:36, 982.84 examples/s]35368 examples [00:36, 980.92 examples/s]35470 examples [00:36, 991.97 examples/s]35571 examples [00:36, 994.26 examples/s]35671 examples [00:36, 984.58 examples/s]35773 examples [00:36, 992.45 examples/s]35873 examples [00:36, 993.29 examples/s]35977 examples [00:37, 1002.89 examples/s]36081 examples [00:37, 1013.02 examples/s]36183 examples [00:37, 1004.17 examples/s]36284 examples [00:37, 997.06 examples/s] 36384 examples [00:37, 996.58 examples/s]36485 examples [00:37, 998.47 examples/s]36585 examples [00:37, 988.00 examples/s]36684 examples [00:37, 958.09 examples/s]36786 examples [00:37, 974.48 examples/s]36885 examples [00:37, 978.05 examples/s]36987 examples [00:38, 989.60 examples/s]37087 examples [00:38, 990.41 examples/s]37187 examples [00:38, 965.96 examples/s]37285 examples [00:38, 968.48 examples/s]37384 examples [00:38, 974.61 examples/s]37483 examples [00:38, 978.07 examples/s]37586 examples [00:38, 991.00 examples/s]37686 examples [00:38, 982.82 examples/s]37785 examples [00:38, 978.26 examples/s]37886 examples [00:38, 984.87 examples/s]37985 examples [00:39, 962.84 examples/s]38082 examples [00:39, 962.45 examples/s]38179 examples [00:39, 931.21 examples/s]38278 examples [00:39, 945.84 examples/s]38381 examples [00:39, 967.42 examples/s]38479 examples [00:39, 964.14 examples/s]38577 examples [00:39, 966.16 examples/s]38674 examples [00:39, 953.81 examples/s]38770 examples [00:39, 953.51 examples/s]38871 examples [00:39, 968.37 examples/s]38969 examples [00:40, 969.67 examples/s]39068 examples [00:40, 975.39 examples/s]39166 examples [00:40, 956.16 examples/s]39264 examples [00:40, 962.63 examples/s]39366 examples [00:40, 976.38 examples/s]39465 examples [00:40, 980.04 examples/s]39564 examples [00:40, 981.17 examples/s]39663 examples [00:40, 963.08 examples/s]39760 examples [00:40, 950.85 examples/s]39862 examples [00:41, 969.15 examples/s]39960 examples [00:41, 929.61 examples/s]40054 examples [00:41, 883.90 examples/s]40159 examples [00:41, 927.82 examples/s]40264 examples [00:41, 959.84 examples/s]40365 examples [00:41, 972.85 examples/s]40469 examples [00:41, 991.72 examples/s]40569 examples [00:41, 967.35 examples/s]40669 examples [00:41, 975.24 examples/s]40772 examples [00:41, 988.56 examples/s]40877 examples [00:42, 1005.74 examples/s]40978 examples [00:42, 1003.09 examples/s]41079 examples [00:42, 1002.28 examples/s]41181 examples [00:42, 1005.80 examples/s]41282 examples [00:42, 1000.10 examples/s]41383 examples [00:42, 987.58 examples/s] 41482 examples [00:42, 981.88 examples/s]41582 examples [00:42, 984.67 examples/s]41684 examples [00:42, 994.39 examples/s]41785 examples [00:42, 998.00 examples/s]41885 examples [00:43, 991.02 examples/s]41987 examples [00:43, 997.00 examples/s]42087 examples [00:43, 993.09 examples/s]42187 examples [00:43, 978.06 examples/s]42287 examples [00:43, 982.22 examples/s]42387 examples [00:43, 987.00 examples/s]42492 examples [00:43, 1001.92 examples/s]42593 examples [00:43, 984.65 examples/s] 42694 examples [00:43, 990.47 examples/s]42795 examples [00:43, 995.73 examples/s]42895 examples [00:44, 980.64 examples/s]42994 examples [00:44, 981.28 examples/s]43093 examples [00:44, 971.82 examples/s]43194 examples [00:44, 981.48 examples/s]43297 examples [00:44, 995.27 examples/s]43399 examples [00:44, 1000.70 examples/s]43500 examples [00:44, 996.61 examples/s] 43600 examples [00:44, 976.80 examples/s]43698 examples [00:44, 965.75 examples/s]43803 examples [00:45, 988.93 examples/s]43908 examples [00:45, 1005.01 examples/s]44013 examples [00:45, 1014.54 examples/s]44115 examples [00:45, 1008.93 examples/s]44217 examples [00:45, 996.08 examples/s] 44322 examples [00:45, 1009.59 examples/s]44424 examples [00:45, 978.10 examples/s] 44523 examples [00:45, 967.27 examples/s]44620 examples [00:45, 953.83 examples/s]44721 examples [00:45, 969.53 examples/s]44823 examples [00:46, 983.39 examples/s]44926 examples [00:46, 994.89 examples/s]45026 examples [00:46, 995.69 examples/s]45126 examples [00:46, 982.78 examples/s]45228 examples [00:46, 991.55 examples/s]45328 examples [00:46, 991.69 examples/s]45428 examples [00:46, 988.38 examples/s]45529 examples [00:46, 988.54 examples/s]45628 examples [00:46, 972.43 examples/s]45730 examples [00:46, 985.13 examples/s]45835 examples [00:47, 1003.22 examples/s]45939 examples [00:47, 1013.94 examples/s]46041 examples [00:47, 1005.59 examples/s]46142 examples [00:47, 969.45 examples/s] 46240 examples [00:47, 968.77 examples/s]46338 examples [00:47, 960.73 examples/s]46437 examples [00:47, 966.96 examples/s]46535 examples [00:47, 968.19 examples/s]46632 examples [00:47, 939.26 examples/s]46727 examples [00:48, 917.50 examples/s]46820 examples [00:48, 900.58 examples/s]46913 examples [00:48, 908.37 examples/s]47007 examples [00:48, 915.01 examples/s]47099 examples [00:48, 907.65 examples/s]47194 examples [00:48, 918.30 examples/s]47297 examples [00:48, 947.61 examples/s]47397 examples [00:48, 962.19 examples/s]47499 examples [00:48, 976.20 examples/s]47597 examples [00:48, 965.37 examples/s]47696 examples [00:49, 970.61 examples/s]47794 examples [00:49, 972.31 examples/s]47892 examples [00:49, 967.68 examples/s]47991 examples [00:49, 973.47 examples/s]48089 examples [00:49, 965.93 examples/s]48190 examples [00:49, 976.72 examples/s]48288 examples [00:49, 974.29 examples/s]48390 examples [00:49, 987.28 examples/s]48496 examples [00:49, 1005.82 examples/s]48597 examples [00:49, 980.13 examples/s] 48697 examples [00:50, 985.69 examples/s]48797 examples [00:50, 988.50 examples/s]48897 examples [00:50, 990.41 examples/s]48999 examples [00:50, 998.23 examples/s]49099 examples [00:50, 982.67 examples/s]49211 examples [00:50, 1018.46 examples/s]49315 examples [00:50, 1022.53 examples/s]49420 examples [00:50, 1029.10 examples/s]49524 examples [00:50, 1018.67 examples/s]49627 examples [00:50, 1003.18 examples/s]49728 examples [00:51, 996.30 examples/s] 49829 examples [00:51, 998.25 examples/s]49933 examples [00:51, 1010.29 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7132/50000 [00:00<00:00, 71317.18 examples/s] 39%|███▉      | 19510/50000 [00:00<00:00, 81705.64 examples/s] 65%|██████▍   | 32286/50000 [00:00<00:00, 91612.07 examples/s] 91%|█████████ | 45277/50000 [00:00<00:00, 100500.24 examples/s]                                                                0 examples [00:00, ? examples/s]78 examples [00:00, 773.22 examples/s]177 examples [00:00, 827.12 examples/s]279 examples [00:00, 875.07 examples/s]389 examples [00:00, 931.89 examples/s]494 examples [00:00, 962.43 examples/s]597 examples [00:00, 980.15 examples/s]698 examples [00:00, 988.20 examples/s]799 examples [00:00, 993.82 examples/s]903 examples [00:00, 1006.10 examples/s]1004 examples [00:01, 1007.23 examples/s]1106 examples [00:01, 1009.30 examples/s]1206 examples [00:01, 991.80 examples/s] 1305 examples [00:01, 961.58 examples/s]1401 examples [00:01, 939.15 examples/s]1497 examples [00:01, 942.88 examples/s]1592 examples [00:01, 917.10 examples/s]1684 examples [00:01, 897.39 examples/s]1777 examples [00:01, 904.67 examples/s]1872 examples [00:01, 917.76 examples/s]1967 examples [00:02, 924.96 examples/s]2063 examples [00:02, 932.83 examples/s]2164 examples [00:02, 952.40 examples/s]2266 examples [00:02, 969.59 examples/s]2367 examples [00:02, 979.41 examples/s]2468 examples [00:02, 987.27 examples/s]2568 examples [00:02, 989.27 examples/s]2668 examples [00:02, 957.62 examples/s]2765 examples [00:02, 951.60 examples/s]2861 examples [00:02, 942.01 examples/s]2970 examples [00:03, 981.05 examples/s]3072 examples [00:03, 991.91 examples/s]3172 examples [00:03, 985.67 examples/s]3271 examples [00:03, 981.56 examples/s]3370 examples [00:03, 982.04 examples/s]3472 examples [00:03, 992.02 examples/s]3576 examples [00:03, 1005.88 examples/s]3678 examples [00:03, 1009.97 examples/s]3783 examples [00:03, 1019.85 examples/s]3886 examples [00:03, 1003.50 examples/s]3987 examples [00:04, 985.95 examples/s] 4086 examples [00:04, 973.17 examples/s]4189 examples [00:04, 984.83 examples/s]4295 examples [00:04, 1003.95 examples/s]4401 examples [00:04, 1018.80 examples/s]4512 examples [00:04, 1042.98 examples/s]4617 examples [00:04, 1033.05 examples/s]4721 examples [00:04, 1024.65 examples/s]4824 examples [00:04, 1018.68 examples/s]4926 examples [00:05, 978.61 examples/s] 5032 examples [00:05, 999.60 examples/s]5138 examples [00:05, 1015.85 examples/s]5240 examples [00:05, 996.39 examples/s] 5340 examples [00:05, 994.37 examples/s]5445 examples [00:05, 1010.40 examples/s]5552 examples [00:05, 1025.30 examples/s]5655 examples [00:05, 1020.48 examples/s]5758 examples [00:05, 991.45 examples/s] 5858 examples [00:05, 989.36 examples/s]5958 examples [00:06, 984.82 examples/s]6057 examples [00:06, 978.57 examples/s]6155 examples [00:06, 973.97 examples/s]6258 examples [00:06, 989.04 examples/s]6360 examples [00:06, 995.38 examples/s]6460 examples [00:06, 983.33 examples/s]6561 examples [00:06, 990.15 examples/s]6662 examples [00:06, 991.46 examples/s]6762 examples [00:06, 985.86 examples/s]6861 examples [00:06, 984.86 examples/s]6960 examples [00:07, 984.26 examples/s]7059 examples [00:07, 974.44 examples/s]7157 examples [00:07, 965.29 examples/s]7255 examples [00:07, 968.59 examples/s]7359 examples [00:07, 988.68 examples/s]7463 examples [00:07, 1003.40 examples/s]7571 examples [00:07, 1022.99 examples/s]7679 examples [00:07, 1036.91 examples/s]7783 examples [00:07, 999.21 examples/s] 7887 examples [00:07, 1009.74 examples/s]7993 examples [00:08, 1022.90 examples/s]8096 examples [00:08, 989.26 examples/s] 8199 examples [00:08, 998.69 examples/s]8300 examples [00:08, 982.97 examples/s]8404 examples [00:08, 998.64 examples/s]8506 examples [00:08, 1003.55 examples/s]8616 examples [00:08, 1030.42 examples/s]8720 examples [00:08, 1030.13 examples/s]8824 examples [00:08, 1012.37 examples/s]8931 examples [00:09, 1027.34 examples/s]9046 examples [00:09, 1061.00 examples/s]9153 examples [00:09, 1012.14 examples/s]9255 examples [00:09, 1009.73 examples/s]9357 examples [00:09, 1006.51 examples/s]9464 examples [00:09, 1023.60 examples/s]9567 examples [00:09, 1021.29 examples/s]9670 examples [00:09, 1006.23 examples/s]9771 examples [00:09, 981.75 examples/s] 9870 examples [00:09, 983.53 examples/s]9969 examples [00:10, 955.70 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEKIF0M/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteEKIF0M/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f208ed3ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f208ed3ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f208ed3ca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f20da6a90f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2019029320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f208ed3ca60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f208ed3ca60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f208ed3ca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2075f74588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2075f74470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.06 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.59 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 15.59 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.70 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.04 file/s]2020-08-06 06:09:53.399713: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-06 06:09:53.403293: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-06 06:09:53.403446: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a3f914ef90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-06 06:09:53.403461: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 21%|██        | 2105344/9912422 [00:00<00:00, 20825311.62it/s] 98%|█████████▊| 9674752/9912422 [00:00<00:00, 26610907.57it/s]9920512it [00:00, 18839401.07it/s]                             
0it [00:00, ?it/s]32768it [00:00, 653012.81it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158728.58it/s]1654784it [00:00, 12223939.27it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 48724.43it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
