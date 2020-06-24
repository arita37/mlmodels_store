
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f513aacfea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f513aacfea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f51a5d8f1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f51a5d8f1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f51c40d6e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f51c40d6e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f51530b6488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f51530b6488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f51530b6488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:12, 135805.49it/s] 59%|█████▉    | 5881856/9912422 [00:00<00:20, 193802.82it/s]9920512it [00:00, 38531086.42it/s]                           
0it [00:00, ?it/s]32768it [00:00, 592222.94it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1025520.19it/s]1654784it [00:00, 12501516.88it/s]                           
0it [00:00, ?it/s]8192it [00:00, 211400.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f513a1e0390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f513a1c8630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f51530b60d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f51530b60d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f51530b60d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:08,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:44,  3.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:29,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:29,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:29,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:19,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:18,  6.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:12,  8.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 12.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:08, 12.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 22.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 28.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 35.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 44.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 52.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 62.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 71.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 79.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.06s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ url]
0 examples [00:00, ? examples/s]2020-06-24 18:11:05.427434: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-24 18:11:05.445921: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-24 18:11:05.446191: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e639edd730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-24 18:11:05.446211: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
14 examples [00:00, 139.25 examples/s]107 examples [00:00, 186.81 examples/s]209 examples [00:00, 247.30 examples/s]315 examples [00:00, 321.05 examples/s]415 examples [00:00, 402.95 examples/s]510 examples [00:00, 486.61 examples/s]600 examples [00:00, 564.01 examples/s]699 examples [00:00, 646.64 examples/s]802 examples [00:00, 726.51 examples/s]895 examples [00:01, 776.27 examples/s]993 examples [00:01, 827.65 examples/s]1096 examples [00:01, 878.13 examples/s]1194 examples [00:01, 904.77 examples/s]1295 examples [00:01, 932.44 examples/s]1394 examples [00:01, 947.45 examples/s]1496 examples [00:01, 966.73 examples/s]1597 examples [00:01, 977.54 examples/s]1698 examples [00:01, 984.37 examples/s]1800 examples [00:01, 993.65 examples/s]1901 examples [00:02, 996.28 examples/s]2002 examples [00:02, 999.88 examples/s]2103 examples [00:02, 959.78 examples/s]2200 examples [00:02, 946.26 examples/s]2303 examples [00:02, 969.50 examples/s]2404 examples [00:02, 979.78 examples/s]2506 examples [00:02, 991.30 examples/s]2608 examples [00:02, 999.53 examples/s]2710 examples [00:02, 1004.03 examples/s]2811 examples [00:02, 1000.05 examples/s]2913 examples [00:03, 1003.58 examples/s]3014 examples [00:03, 1002.11 examples/s]3115 examples [00:03, 972.44 examples/s] 3217 examples [00:03, 985.60 examples/s]3319 examples [00:03, 992.72 examples/s]3419 examples [00:03, 979.43 examples/s]3518 examples [00:03, 981.56 examples/s]3620 examples [00:03, 991.40 examples/s]3720 examples [00:03, 987.31 examples/s]3819 examples [00:03, 982.75 examples/s]3918 examples [00:04, 981.16 examples/s]4023 examples [00:04, 1000.05 examples/s]4124 examples [00:04, 996.76 examples/s] 4226 examples [00:04, 1003.48 examples/s]4328 examples [00:04, 1005.95 examples/s]4429 examples [00:04, 994.34 examples/s] 4531 examples [00:04, 999.92 examples/s]4632 examples [00:04, 1001.45 examples/s]4733 examples [00:04, 969.68 examples/s] 4836 examples [00:04, 985.40 examples/s]4935 examples [00:05, 977.33 examples/s]5037 examples [00:05, 988.60 examples/s]5137 examples [00:05, 951.65 examples/s]5236 examples [00:05, 961.38 examples/s]5335 examples [00:05, 969.02 examples/s]5433 examples [00:05, 963.13 examples/s]5530 examples [00:05, 937.16 examples/s]5625 examples [00:05, 940.36 examples/s]5725 examples [00:05, 956.55 examples/s]5823 examples [00:06, 961.89 examples/s]5921 examples [00:06, 966.90 examples/s]6018 examples [00:06, 955.69 examples/s]6115 examples [00:06, 957.49 examples/s]6211 examples [00:06, 952.13 examples/s]6307 examples [00:06, 950.88 examples/s]6403 examples [00:06, 935.45 examples/s]6500 examples [00:06, 944.54 examples/s]6602 examples [00:06, 965.32 examples/s]6699 examples [00:06, 947.05 examples/s]6800 examples [00:07, 963.42 examples/s]6900 examples [00:07, 973.40 examples/s]7004 examples [00:07, 992.34 examples/s]7104 examples [00:07, 987.71 examples/s]7203 examples [00:07, 979.39 examples/s]7304 examples [00:07, 988.34 examples/s]7406 examples [00:07, 996.44 examples/s]7506 examples [00:07, 987.54 examples/s]7605 examples [00:07, 976.44 examples/s]7703 examples [00:07, 964.32 examples/s]7807 examples [00:08, 984.21 examples/s]7908 examples [00:08, 990.62 examples/s]8008 examples [00:08, 971.22 examples/s]8106 examples [00:08, 972.77 examples/s]8207 examples [00:08, 981.25 examples/s]8307 examples [00:08, 986.18 examples/s]8406 examples [00:08, 985.33 examples/s]8505 examples [00:08, 985.71 examples/s]8606 examples [00:08, 991.68 examples/s]8708 examples [00:08, 998.66 examples/s]8811 examples [00:09, 1005.55 examples/s]8912 examples [00:09, 1004.82 examples/s]9017 examples [00:09, 1017.08 examples/s]9119 examples [00:09, 972.99 examples/s] 9224 examples [00:09, 993.06 examples/s]9324 examples [00:09, 988.06 examples/s]9426 examples [00:09, 995.58 examples/s]9533 examples [00:09, 1014.07 examples/s]9638 examples [00:09, 1024.52 examples/s]9745 examples [00:09, 1037.45 examples/s]9852 examples [00:10, 1044.54 examples/s]9957 examples [00:10, 1020.72 examples/s]10060 examples [00:10, 934.90 examples/s]10162 examples [00:10, 957.59 examples/s]10265 examples [00:10, 976.07 examples/s]10364 examples [00:10, 971.33 examples/s]10463 examples [00:10, 974.71 examples/s]10561 examples [00:10, 947.45 examples/s]10660 examples [00:10, 958.01 examples/s]10757 examples [00:11, 957.76 examples/s]10854 examples [00:11, 957.47 examples/s]10955 examples [00:11, 971.09 examples/s]11053 examples [00:11, 951.28 examples/s]11158 examples [00:11, 978.88 examples/s]11257 examples [00:11, 944.77 examples/s]11357 examples [00:11, 958.29 examples/s]11460 examples [00:11, 976.56 examples/s]11561 examples [00:11, 984.06 examples/s]11660 examples [00:11, 981.87 examples/s]11760 examples [00:12, 984.55 examples/s]11859 examples [00:12, 979.07 examples/s]11958 examples [00:12, 974.03 examples/s]12061 examples [00:12, 989.29 examples/s]12168 examples [00:12, 1009.89 examples/s]12270 examples [00:12, 1000.64 examples/s]12371 examples [00:12, 975.20 examples/s] 12473 examples [00:12, 986.85 examples/s]12578 examples [00:12, 1002.39 examples/s]12685 examples [00:12, 1021.60 examples/s]12788 examples [00:13, 998.43 examples/s] 12889 examples [00:13, 999.63 examples/s]12990 examples [00:13, 995.44 examples/s]13091 examples [00:13, 999.48 examples/s]13193 examples [00:13, 1003.56 examples/s]13296 examples [00:13, 1010.25 examples/s]13398 examples [00:13, 1006.88 examples/s]13500 examples [00:13, 1008.01 examples/s]13603 examples [00:13, 1012.26 examples/s]13706 examples [00:14, 1015.41 examples/s]13808 examples [00:14, 1016.51 examples/s]13910 examples [00:14, 975.14 examples/s] 14008 examples [00:14, 950.12 examples/s]14107 examples [00:14, 961.13 examples/s]14208 examples [00:14, 969.55 examples/s]14306 examples [00:14, 954.27 examples/s]14402 examples [00:14, 940.22 examples/s]14508 examples [00:14, 973.13 examples/s]14613 examples [00:14, 993.01 examples/s]14720 examples [00:15, 1014.31 examples/s]14826 examples [00:15, 1024.98 examples/s]14930 examples [00:15, 1029.22 examples/s]15034 examples [00:15, 1015.35 examples/s]15136 examples [00:15, 999.79 examples/s] 15237 examples [00:15, 1001.14 examples/s]15338 examples [00:15, 998.90 examples/s] 15439 examples [00:15, 1000.13 examples/s]15540 examples [00:15, 996.34 examples/s] 15640 examples [00:15, 991.91 examples/s]15740 examples [00:16, 966.58 examples/s]15845 examples [00:16, 989.92 examples/s]15946 examples [00:16, 995.30 examples/s]16050 examples [00:16, 1006.24 examples/s]16154 examples [00:16, 1013.64 examples/s]16259 examples [00:16, 1022.09 examples/s]16365 examples [00:16, 1031.57 examples/s]16469 examples [00:16, 1020.77 examples/s]16572 examples [00:16, 1008.64 examples/s]16674 examples [00:16, 1009.17 examples/s]16776 examples [00:17, 1011.06 examples/s]16878 examples [00:17, 1006.72 examples/s]16979 examples [00:17, 1001.69 examples/s]17080 examples [00:17, 1002.03 examples/s]17182 examples [00:17, 1005.13 examples/s]17283 examples [00:17, 967.81 examples/s] 17386 examples [00:17, 985.46 examples/s]17485 examples [00:17, 975.70 examples/s]17584 examples [00:17, 978.67 examples/s]17685 examples [00:18, 986.09 examples/s]17786 examples [00:18, 990.83 examples/s]17886 examples [00:18, 993.48 examples/s]17986 examples [00:18, 974.24 examples/s]18087 examples [00:18, 982.98 examples/s]18186 examples [00:18, 971.95 examples/s]18286 examples [00:18, 979.37 examples/s]18389 examples [00:18, 993.16 examples/s]18489 examples [00:18, 988.12 examples/s]18588 examples [00:18, 987.63 examples/s]18689 examples [00:19, 993.47 examples/s]18790 examples [00:19, 997.93 examples/s]18892 examples [00:19, 1003.56 examples/s]18993 examples [00:19, 999.38 examples/s] 19093 examples [00:19, 977.62 examples/s]19191 examples [00:19, 959.16 examples/s]19288 examples [00:19, 946.87 examples/s]19386 examples [00:19, 954.80 examples/s]19485 examples [00:19, 964.30 examples/s]19586 examples [00:19, 974.44 examples/s]19684 examples [00:20, 954.13 examples/s]19785 examples [00:20, 969.49 examples/s]19885 examples [00:20, 976.04 examples/s]19984 examples [00:20, 978.28 examples/s]20082 examples [00:20, 930.62 examples/s]20184 examples [00:20, 954.55 examples/s]20280 examples [00:20, 943.52 examples/s]20382 examples [00:20, 964.70 examples/s]20485 examples [00:20, 980.99 examples/s]20591 examples [00:20, 1001.78 examples/s]20697 examples [00:21, 1015.88 examples/s]20800 examples [00:21, 1018.93 examples/s]20903 examples [00:21, 989.84 examples/s] 21003 examples [00:21, 988.08 examples/s]21103 examples [00:21, 989.39 examples/s]21205 examples [00:21, 997.88 examples/s]21306 examples [00:21, 1000.00 examples/s]21407 examples [00:21, 998.79 examples/s] 21508 examples [00:21, 1001.28 examples/s]21609 examples [00:22, 1001.55 examples/s]21710 examples [00:22, 1003.59 examples/s]21811 examples [00:22, 1000.75 examples/s]21912 examples [00:22, 977.06 examples/s] 22010 examples [00:22, 974.68 examples/s]22114 examples [00:22, 991.92 examples/s]22214 examples [00:22, 992.94 examples/s]22316 examples [00:22, 1000.22 examples/s]22417 examples [00:22, 986.42 examples/s] 22516 examples [00:22, 986.70 examples/s]22622 examples [00:23, 1005.92 examples/s]22723 examples [00:23, 1003.74 examples/s]22824 examples [00:23, 997.63 examples/s] 22924 examples [00:23, 988.27 examples/s]23024 examples [00:23, 990.76 examples/s]23126 examples [00:23, 997.57 examples/s]23229 examples [00:23, 1006.51 examples/s]23330 examples [00:23, 980.41 examples/s] 23434 examples [00:23, 995.56 examples/s]23534 examples [00:23, 996.31 examples/s]23636 examples [00:24, 1001.53 examples/s]23737 examples [00:24, 995.73 examples/s] 23840 examples [00:24, 1005.20 examples/s]23941 examples [00:24, 999.67 examples/s] 24042 examples [00:24, 974.82 examples/s]24140 examples [00:24, 962.06 examples/s]24237 examples [00:24, 945.65 examples/s]24332 examples [00:24, 926.32 examples/s]24434 examples [00:24, 951.87 examples/s]24539 examples [00:24, 978.03 examples/s]24646 examples [00:25, 1002.64 examples/s]24752 examples [00:25, 1017.74 examples/s]24855 examples [00:25, 1021.30 examples/s]24958 examples [00:25, 1019.88 examples/s]25063 examples [00:25, 1027.39 examples/s]25166 examples [00:25, 1019.27 examples/s]25269 examples [00:25, 1015.19 examples/s]25371 examples [00:25, 1012.72 examples/s]25473 examples [00:25, 993.02 examples/s] 25573 examples [00:26, 986.86 examples/s]25677 examples [00:26, 1000.47 examples/s]25778 examples [00:26, 988.01 examples/s] 25877 examples [00:26, 975.71 examples/s]25975 examples [00:26, 976.04 examples/s]26076 examples [00:26, 983.76 examples/s]26176 examples [00:26, 985.99 examples/s]26280 examples [00:26, 1000.78 examples/s]26381 examples [00:26, 977.11 examples/s] 26484 examples [00:26, 990.04 examples/s]26584 examples [00:27, 992.02 examples/s]26686 examples [00:27, 999.24 examples/s]26790 examples [00:27, 1009.63 examples/s]26892 examples [00:27, 1011.79 examples/s]26994 examples [00:27, 1012.80 examples/s]27096 examples [00:27, 1009.16 examples/s]27197 examples [00:27, 1002.48 examples/s]27298 examples [00:27, 1002.46 examples/s]27399 examples [00:27, 987.16 examples/s] 27501 examples [00:27, 996.69 examples/s]27606 examples [00:28, 1011.40 examples/s]27708 examples [00:28, 1008.31 examples/s]27809 examples [00:28, 992.12 examples/s] 27909 examples [00:28, 953.80 examples/s]28008 examples [00:28, 962.82 examples/s]28110 examples [00:28, 977.42 examples/s]28209 examples [00:28, 977.14 examples/s]28307 examples [00:28, 977.78 examples/s]28405 examples [00:28, 976.98 examples/s]28503 examples [00:28, 970.55 examples/s]28602 examples [00:29, 976.04 examples/s]28701 examples [00:29, 979.96 examples/s]28800 examples [00:29, 982.02 examples/s]28899 examples [00:29, 969.46 examples/s]28997 examples [00:29, 971.72 examples/s]29095 examples [00:29, 969.01 examples/s]29196 examples [00:29, 978.43 examples/s]29303 examples [00:29, 1002.00 examples/s]29404 examples [00:29, 974.84 examples/s] 29506 examples [00:29, 986.17 examples/s]29605 examples [00:30, 987.25 examples/s]29707 examples [00:30, 994.86 examples/s]29807 examples [00:30, 991.71 examples/s]29907 examples [00:30, 989.02 examples/s]30006 examples [00:30, 927.85 examples/s]30109 examples [00:30, 956.21 examples/s]30208 examples [00:30, 965.61 examples/s]30311 examples [00:30, 983.70 examples/s]30412 examples [00:30, 989.44 examples/s]30512 examples [00:31, 982.97 examples/s]30613 examples [00:31, 989.25 examples/s]30713 examples [00:31, 980.84 examples/s]30817 examples [00:31, 996.23 examples/s]30920 examples [00:31, 1003.19 examples/s]31021 examples [00:31, 991.45 examples/s] 31121 examples [00:31, 991.08 examples/s]31221 examples [00:31, 983.50 examples/s]31321 examples [00:31, 986.53 examples/s]31420 examples [00:31, 985.46 examples/s]31521 examples [00:32, 990.37 examples/s]31623 examples [00:32, 997.95 examples/s]31723 examples [00:32, 998.46 examples/s]31823 examples [00:32, 978.74 examples/s]31926 examples [00:32, 992.79 examples/s]32028 examples [00:32, 998.29 examples/s]32131 examples [00:32, 1006.77 examples/s]32232 examples [00:32, 984.60 examples/s] 32333 examples [00:32, 991.46 examples/s]32436 examples [00:32, 1000.65 examples/s]32537 examples [00:33, 978.95 examples/s] 32641 examples [00:33, 994.91 examples/s]32741 examples [00:33, 982.85 examples/s]32840 examples [00:33, 961.97 examples/s]32940 examples [00:33, 971.83 examples/s]33038 examples [00:33, 964.08 examples/s]33139 examples [00:33, 975.01 examples/s]33239 examples [00:33, 979.77 examples/s]33341 examples [00:33, 990.93 examples/s]33441 examples [00:33, 989.74 examples/s]33542 examples [00:34, 992.87 examples/s]33644 examples [00:34, 999.48 examples/s]33744 examples [00:34, 995.79 examples/s]33846 examples [00:34, 1002.06 examples/s]33947 examples [00:34, 962.07 examples/s] 34044 examples [00:34, 944.98 examples/s]34151 examples [00:34, 977.48 examples/s]34254 examples [00:34, 990.66 examples/s]34354 examples [00:34, 954.99 examples/s]34453 examples [00:35, 962.30 examples/s]34550 examples [00:35, 961.46 examples/s]34652 examples [00:35, 976.11 examples/s]34750 examples [00:35, 972.78 examples/s]34848 examples [00:35, 972.19 examples/s]34946 examples [00:35, 956.59 examples/s]35045 examples [00:35, 964.11 examples/s]35142 examples [00:35, 958.90 examples/s]35238 examples [00:35, 951.83 examples/s]35344 examples [00:35, 979.55 examples/s]35443 examples [00:36, 941.50 examples/s]35541 examples [00:36, 950.08 examples/s]35638 examples [00:36, 954.06 examples/s]35734 examples [00:36, 952.50 examples/s]35832 examples [00:36, 959.40 examples/s]35931 examples [00:36, 967.34 examples/s]36029 examples [00:36, 968.64 examples/s]36133 examples [00:36, 988.67 examples/s]36237 examples [00:36, 1002.31 examples/s]36339 examples [00:36, 1006.90 examples/s]36443 examples [00:37, 1016.17 examples/s]36546 examples [00:37, 1019.44 examples/s]36649 examples [00:37, 1020.54 examples/s]36753 examples [00:37, 1025.83 examples/s]36856 examples [00:37, 1026.99 examples/s]36963 examples [00:37, 1039.41 examples/s]37067 examples [00:37, 1035.12 examples/s]37172 examples [00:37, 1038.77 examples/s]37276 examples [00:37, 1033.31 examples/s]37380 examples [00:37, 1023.59 examples/s]37485 examples [00:38, 1028.99 examples/s]37588 examples [00:38, 1024.79 examples/s]37694 examples [00:38, 1034.95 examples/s]37801 examples [00:38, 1042.97 examples/s]37908 examples [00:38, 1049.80 examples/s]38015 examples [00:38, 1054.55 examples/s]38121 examples [00:38, 1046.74 examples/s]38228 examples [00:38, 1053.26 examples/s]38334 examples [00:38, 1030.30 examples/s]38438 examples [00:39, 995.58 examples/s] 38538 examples [00:39, 942.35 examples/s]38638 examples [00:39, 957.81 examples/s]38737 examples [00:39, 965.69 examples/s]38838 examples [00:39, 977.84 examples/s]38938 examples [00:39, 983.79 examples/s]39037 examples [00:39, 970.39 examples/s]39137 examples [00:39, 977.59 examples/s]39242 examples [00:39, 997.06 examples/s]39345 examples [00:39, 1004.62 examples/s]39449 examples [00:40, 1012.23 examples/s]39554 examples [00:40, 1022.97 examples/s]39658 examples [00:40, 1025.47 examples/s]39765 examples [00:40, 1035.63 examples/s]39872 examples [00:40, 1042.86 examples/s]39977 examples [00:40, 1042.85 examples/s]40082 examples [00:40, 986.17 examples/s] 40183 examples [00:40, 992.68 examples/s]40287 examples [00:40, 1004.88 examples/s]40389 examples [00:40, 1008.95 examples/s]40491 examples [00:41, 991.75 examples/s] 40596 examples [00:41, 1006.33 examples/s]40701 examples [00:41, 1016.86 examples/s]40803 examples [00:41, 1014.97 examples/s]40905 examples [00:41, 963.75 examples/s] 41010 examples [00:41, 986.87 examples/s]41111 examples [00:41, 991.03 examples/s]41211 examples [00:41, 983.83 examples/s]41310 examples [00:41, 966.64 examples/s]41407 examples [00:42, 922.73 examples/s]41506 examples [00:42, 940.22 examples/s]41601 examples [00:42, 939.59 examples/s]41698 examples [00:42, 947.75 examples/s]41795 examples [00:42, 953.20 examples/s]41897 examples [00:42, 971.32 examples/s]41998 examples [00:42, 981.07 examples/s]42097 examples [00:42, 975.50 examples/s]42197 examples [00:42, 981.88 examples/s]42296 examples [00:42, 969.03 examples/s]42394 examples [00:43, 967.14 examples/s]42495 examples [00:43, 977.85 examples/s]42593 examples [00:43, 966.52 examples/s]42695 examples [00:43, 980.23 examples/s]42794 examples [00:43, 980.73 examples/s]42893 examples [00:43, 975.23 examples/s]42995 examples [00:43, 987.09 examples/s]43094 examples [00:43, 924.75 examples/s]43188 examples [00:43, 915.84 examples/s]43291 examples [00:43, 946.15 examples/s]43397 examples [00:44, 976.76 examples/s]43500 examples [00:44, 990.06 examples/s]43603 examples [00:44, 999.33 examples/s]43704 examples [00:44, 992.68 examples/s]43804 examples [00:44, 991.12 examples/s]43906 examples [00:44, 998.93 examples/s]44007 examples [00:44, 1001.97 examples/s]44112 examples [00:44, 1013.32 examples/s]44215 examples [00:44, 1015.80 examples/s]44321 examples [00:44, 1027.78 examples/s]44424 examples [00:45, 1026.64 examples/s]44531 examples [00:45, 1039.26 examples/s]44636 examples [00:45, 1013.87 examples/s]44742 examples [00:45, 1025.07 examples/s]44845 examples [00:45, 1024.01 examples/s]44953 examples [00:45, 1037.53 examples/s]45059 examples [00:45, 1042.28 examples/s]45165 examples [00:45, 1047.41 examples/s]45272 examples [00:45, 1053.86 examples/s]45378 examples [00:45, 1055.40 examples/s]45484 examples [00:46, 1054.80 examples/s]45591 examples [00:46, 1058.88 examples/s]45697 examples [00:46, 1056.32 examples/s]45803 examples [00:46, 1039.53 examples/s]45908 examples [00:46, 1035.30 examples/s]46012 examples [00:46, 1031.64 examples/s]46117 examples [00:46, 1036.49 examples/s]46221 examples [00:46, 1025.60 examples/s]46324 examples [00:46, 1008.88 examples/s]46426 examples [00:47, 1010.37 examples/s]46528 examples [00:47, 977.45 examples/s] 46627 examples [00:47, 949.87 examples/s]46723 examples [00:47, 927.12 examples/s]46821 examples [00:47, 941.84 examples/s]46927 examples [00:47, 972.44 examples/s]47033 examples [00:47, 996.41 examples/s]47138 examples [00:47, 1010.95 examples/s]47244 examples [00:47, 1001.63 examples/s]47345 examples [00:47, 977.85 examples/s] 47445 examples [00:48, 983.25 examples/s]47544 examples [00:48, 966.85 examples/s]47641 examples [00:48, 943.30 examples/s]47740 examples [00:48, 956.39 examples/s]47842 examples [00:48, 972.11 examples/s]47948 examples [00:48, 994.79 examples/s]48052 examples [00:48, 1006.78 examples/s]48155 examples [00:48, 1011.17 examples/s]48261 examples [00:48, 1024.63 examples/s]48364 examples [00:48, 1023.13 examples/s]48467 examples [00:49, 997.56 examples/s] 48570 examples [00:49, 1006.35 examples/s]48676 examples [00:49, 1021.58 examples/s]48780 examples [00:49, 1025.55 examples/s]48883 examples [00:49, 1022.87 examples/s]48988 examples [00:49, 1028.98 examples/s]49091 examples [00:49, 1022.51 examples/s]49194 examples [00:49, 1004.60 examples/s]49299 examples [00:49, 1017.21 examples/s]49401 examples [00:49, 1014.44 examples/s]49504 examples [00:50, 1019.00 examples/s]49610 examples [00:50, 1030.52 examples/s]49716 examples [00:50, 1038.28 examples/s]49820 examples [00:50, 1022.81 examples/s]49924 examples [00:50, 1026.38 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5993/50000 [00:00<00:00, 59925.06 examples/s] 37%|███▋      | 18747/50000 [00:00<00:00, 71257.45 examples/s] 63%|██████▎   | 31471/50000 [00:00<00:00, 82056.93 examples/s] 89%|████████▉ | 44395/50000 [00:00<00:00, 92149.19 examples/s]                                                               0 examples [00:00, ? examples/s]85 examples [00:00, 844.77 examples/s]186 examples [00:00, 887.14 examples/s]278 examples [00:00, 895.18 examples/s]385 examples [00:00, 940.27 examples/s]491 examples [00:00, 972.44 examples/s]597 examples [00:00, 994.53 examples/s]699 examples [00:00, 1001.82 examples/s]798 examples [00:00, 996.44 examples/s] 900 examples [00:00, 1002.46 examples/s]1003 examples [00:01, 1008.51 examples/s]1107 examples [00:01, 1015.45 examples/s]1215 examples [00:01, 1033.38 examples/s]1321 examples [00:01, 1038.95 examples/s]1425 examples [00:01, 1022.18 examples/s]1527 examples [00:01, 998.87 examples/s] 1628 examples [00:01, 1001.55 examples/s]1729 examples [00:01, 990.59 examples/s] 1830 examples [00:01, 993.71 examples/s]1933 examples [00:01, 1003.29 examples/s]2034 examples [00:02, 1005.15 examples/s]2135 examples [00:02, 1001.45 examples/s]2236 examples [00:02, 995.23 examples/s] 2336 examples [00:02, 986.85 examples/s]2438 examples [00:02, 993.71 examples/s]2540 examples [00:02, 999.67 examples/s]2641 examples [00:02, 1001.82 examples/s]2743 examples [00:02, 1007.12 examples/s]2844 examples [00:02, 1006.75 examples/s]2945 examples [00:02, 994.51 examples/s] 3046 examples [00:03, 997.24 examples/s]3147 examples [00:03, 1000.25 examples/s]3250 examples [00:03, 1008.60 examples/s]3351 examples [00:03, 984.25 examples/s] 3452 examples [00:03, 990.49 examples/s]3558 examples [00:03, 1008.72 examples/s]3660 examples [00:03, 1003.79 examples/s]3765 examples [00:03, 1014.99 examples/s]3868 examples [00:03, 1018.07 examples/s]3970 examples [00:03, 1011.84 examples/s]4072 examples [00:04, 1010.74 examples/s]4174 examples [00:04, 1002.11 examples/s]4277 examples [00:04, 1010.31 examples/s]4379 examples [00:04, 1013.10 examples/s]4487 examples [00:04, 1030.41 examples/s]4593 examples [00:04, 1038.69 examples/s]4698 examples [00:04, 1039.92 examples/s]4803 examples [00:04, 1042.35 examples/s]4909 examples [00:04, 1044.87 examples/s]5015 examples [00:04, 1046.76 examples/s]5120 examples [00:05, 1033.03 examples/s]5226 examples [00:05, 1040.74 examples/s]5331 examples [00:05, 1037.90 examples/s]5437 examples [00:05, 1043.29 examples/s]5544 examples [00:05, 1048.81 examples/s]5650 examples [00:05, 1050.87 examples/s]5756 examples [00:05, 1051.80 examples/s]5862 examples [00:05, 1044.67 examples/s]5967 examples [00:05, 1033.17 examples/s]6071 examples [00:05, 1007.38 examples/s]6174 examples [00:06, 1012.64 examples/s]6280 examples [00:06, 1024.37 examples/s]6384 examples [00:06, 1028.70 examples/s]6487 examples [00:06, 984.98 examples/s] 6594 examples [00:06, 1006.80 examples/s]6699 examples [00:06, 1016.92 examples/s]6802 examples [00:06, 1013.99 examples/s]6908 examples [00:06, 1025.20 examples/s]7013 examples [00:06, 1032.33 examples/s]7119 examples [00:07, 1039.93 examples/s]7224 examples [00:07, 1040.96 examples/s]7330 examples [00:07, 1046.12 examples/s]7438 examples [00:07, 1053.26 examples/s]7544 examples [00:07, 1040.61 examples/s]7649 examples [00:07, 1040.34 examples/s]7754 examples [00:07, 1027.60 examples/s]7857 examples [00:07, 1004.55 examples/s]7962 examples [00:07, 1016.82 examples/s]8064 examples [00:07, 1012.94 examples/s]8167 examples [00:08, 1017.48 examples/s]8273 examples [00:08, 1029.73 examples/s]8381 examples [00:08, 1042.26 examples/s]8487 examples [00:08, 1046.59 examples/s]8594 examples [00:08, 1051.52 examples/s]8701 examples [00:08, 1055.58 examples/s]8807 examples [00:08, 1042.61 examples/s]8912 examples [00:08, 1004.37 examples/s]9013 examples [00:08, 1002.13 examples/s]9114 examples [00:08, 998.96 examples/s] 9215 examples [00:09, 995.48 examples/s]9318 examples [00:09, 1003.83 examples/s]9424 examples [00:09, 1017.33 examples/s]9526 examples [00:09, 1015.09 examples/s]9628 examples [00:09, 958.28 examples/s] 9730 examples [00:09, 975.86 examples/s]9834 examples [00:09, 992.78 examples/s]9939 examples [00:09, 1008.67 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNP18TZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNP18TZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f51530b6488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f51530b6488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f51530b6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f51502e04e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f50d8536940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f51530b6488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f51530b6488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f51530b6488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f513a1c8400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f513a1c8128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f50d4d851e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f50d4d851e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f50d4d851e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f50d4d852f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f50d4d852f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f50d4d852f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=50433eb2bc82726b9102bff0ca270d179e4394b51a872f5489863cb952eeac15
  Stored in directory: /tmp/pip-ephem-wheel-cache-oaaxq9wk/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:51:52, 11.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:32:17, 15.4kB/s].vector_cache/glove.6B.zip:   0%|          | 180k/862M [00:01<10:56:55, 21.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 729k/862M [00:01<7:40:35, 31.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.92M/862M [00:01<5:21:50, 44.5kB/s].vector_cache/glove.6B.zip:   1%|          | 6.99M/862M [00:01<3:44:19, 63.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.2M/862M [00:01<2:36:21, 90.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<1:49:02, 129kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.2M/862M [00:01<1:15:58, 185kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:01<53:04, 263kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:01<37:01, 375kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<25:55, 534kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<18:07, 759kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:02<12:44, 1.07MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<08:56, 1.52MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<06:21, 2.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<05:11, 2.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:03<03:43, 3.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<1:57:03, 115kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<1:24:00, 160kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<59:14, 226kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<43:44, 305kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<32:18, 413kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<23:00, 579kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<18:41, 711kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<14:27, 919kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<10:25, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<10:24, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<10:02, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<07:41, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:11<05:29, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<55:02, 239kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<39:53, 330kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<28:12, 465kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<22:45, 575kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<18:35, 703kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<13:40, 955kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:15<09:42, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<2:35:34, 83.7kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<1:50:11, 118kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<1:17:19, 168kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<57:00, 227kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<42:35, 304kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.2M/862M [00:18<30:27, 425kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<21:23, 602kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<1:46:53, 120kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<1:16:09, 169kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<53:31, 240kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<40:22, 317kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<30:51, 415kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<22:14, 575kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<17:34, 725kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<14:53, 855kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<10:59, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<07:51, 1.61MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:01, 1.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:59, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:36, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:34, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:50, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:02, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:23, 2.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:05, 1.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:39, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:40, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:53, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:06, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:34, 2.71MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:06, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:46, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:18, 2.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<03:50, 3.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<14:14, 866kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:13, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<08:09, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:34, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:14, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:18, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:52, 3.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:06:57, 16.8kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<8:31:11, 23.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<5:58:00, 34.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<4:09:48, 48.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<3:09:05, 64.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:13:31, 90.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<1:33:34, 129kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<1:08:08, 177kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<50:09, 240kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<35:36, 338kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<25:00, 480kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<24:10, 496kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:07, 661kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<12:55, 925kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<11:49, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:45, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:06, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:34, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:31, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:49, 2.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:07, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:47, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:21, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:52, 3.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<44:59, 260kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<32:41, 358kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<23:07, 505kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:52, 617kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<15:35, 747kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<11:24, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<08:11, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:15, 1.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:40, 1.51MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:39, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:37, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:58, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:27, 2.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:40, 2.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:17, 1.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<04:59, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:37, 3.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:35:19, 119kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<1:07:50, 168kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<47:40, 238kB/s]  .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<35:54, 315kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<27:26, 412kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<19:42, 573kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<13:51, 811kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<26:18, 427kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<19:34, 574kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<13:57, 803kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<12:21, 903kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:46, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<07:06, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:36, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:26, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:45, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:56, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:24, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:58, 2.22MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<03:38, 3.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<07:05, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:03, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:31, 2.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:42, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:05, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:48, 2.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:12, 2.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:44, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:35, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:03, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:37, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:27, 3.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:39, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:24, 2.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:16, 3.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:48, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:47, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:27, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<20:03, 526kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<16:05, 655kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:46, 895kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<08:19, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<22:21, 469kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<17:36, 595kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<12:49, 816kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<09:02, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<32:46, 318kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<25:00, 416kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<18:04, 575kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<12:50, 808kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:29, 828kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<15:04, 686kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<12:05, 855kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:49, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<06:26, 1.60MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:11, 1.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<08:30, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:41, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:56, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:42, 2.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<14:35, 699kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<15:44, 649kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<12:24, 822kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<09:02, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<06:31, 1.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<08:43, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<11:06, 913kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<08:52, 1.14MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:31, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:56, 2.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:38, 2.76MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<25:57, 388kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<23:04, 436kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<17:28, 576kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<12:30, 803kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<08:58, 1.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<10:50, 922kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<12:28, 802kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<10:02, 995kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:22, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<05:22, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<08:18, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<11:10, 888kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:05, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:42, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:54, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:59, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<10:27, 943kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:32, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:16, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<04:33, 2.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:29, 2.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<8:20:04, 19.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<5:54:46, 27.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<4:09:24, 39.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<2:54:33, 55.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<2:02:04, 79.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<1:30:20, 108kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<1:07:57, 143kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<48:43, 199kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<34:19, 283kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<24:09, 400kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<17:12, 561kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<1:49:05, 88.5kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<1:21:04, 119kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<57:55, 167kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<40:44, 236kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<28:38, 335kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<23:48, 402kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<21:23, 448kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<15:56, 601kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<11:28, 833kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<08:12, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<06:03, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<4:03:13, 39.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<2:54:53, 54.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<2:03:30, 77.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<1:26:33, 110kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<1:00:35, 156kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<42:39, 221kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<3:10:15, 49.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<2:17:48, 68.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<1:37:34, 96.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:08:29, 138kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<48:01, 196kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<38:12, 245kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<31:22, 299kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<23:05, 406kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<16:24, 570kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<11:38, 801kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<11:59, 777kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<13:27, 692kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<10:32, 882kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:38, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<05:32, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<08:27, 1.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:25, 886kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<08:26, 1.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:09, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:29, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:14, 1.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<09:34, 958kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<07:49, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:46, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:14, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:50, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<10:28, 869kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<08:26, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:09, 1.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:29, 2.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:40, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:06, 991kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<07:25, 1.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:28, 1.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<04:00, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:43, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<10:04, 890kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<08:05, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:57, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:21, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:15, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<09:42, 915kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:46, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:40, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:07, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:32, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:12, 958kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:25, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<05:28, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:59, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<10:36, 825kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<11:00, 795kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:29, 1.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<06:11, 1.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:31, 1.92MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<13:35, 639kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<13:06, 662kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<10:03, 863kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<07:13, 1.20MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:12, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<09:23, 917kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<09:49, 877kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<07:35, 1.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:30, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<06:28, 1.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<07:32, 1.13MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:03, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:23, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:12, 2.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<28:06, 302kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<22:36, 375kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<16:32, 512kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<11:44, 719kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<10:08, 829kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<10:04, 835kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<07:37, 1.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<05:30, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:03, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:57, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:16, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:21, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:37, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:24, 2.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:02, 915kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<09:03, 913kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<07:00, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<05:03, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:48, 1.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:24, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:05, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<03:41, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:18, 2.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<5:36:40, 24.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<3:56:38, 34.3kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<2:45:31, 49.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:56:47, 69.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:26:05, 93.7kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<1:01:06, 132kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<42:56, 187kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<30:01, 266kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<34:39, 231kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<26:32, 301kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<19:02, 419kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<13:25, 592kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<12:03, 657kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:18, 769kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:41, 1.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<05:27, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:52, 998kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:26, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:42, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<04:06, 1.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:03, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:13, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:47, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<03:28, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:35, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:53, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:32, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:18, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:33, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:39, 1.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:20, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:08, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<06:02, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:42, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:20, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:07, 2.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<16:03, 468kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<12:35, 596kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<09:08, 819kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<06:26, 1.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<43:42, 170kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<32:23, 230kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<23:05, 322kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<16:10, 457kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<16:07, 457kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<12:54, 571kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<09:25, 780kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<06:39, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<11:03, 660kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<09:26, 773kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:58, 1.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<04:56, 1.47MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<07:22, 982kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:50, 1.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:07, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<03:40, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:37, 1.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:35, 1.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:19, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<03:06, 2.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:36, 932kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:24, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:44, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:33, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:40, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:35, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:35, 2.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:57, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:38, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:15, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:05, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:28, 1.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:35, 1.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:34, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:34, 2.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<50:30, 135kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<36:17, 188kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<25:31, 266kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<19:02, 354kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<15:52, 425kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<11:38, 579kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<08:14, 815kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<07:23, 904kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<06:22, 1.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:45, 1.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:25, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:24, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:17, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:08, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:11, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:23, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:40, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:55, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:12, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:32, 2.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:49, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:07, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:28, 2.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:46, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:20, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:30, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:38, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:59, 2.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:14, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:32, 2.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:47, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:07, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<01:49, 3.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:16, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:33, 2.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:47, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:02, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:24, 2.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:39, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:27, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:48, 2.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:03, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:10, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:57, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:01, 1.93MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:05, 1.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:11, 1.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:26, 1.68MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:30, 2.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:14, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:15, 1.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:29, 2.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<01:47, 3.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<13:13, 427kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<11:13, 503kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<08:21, 675kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<05:55, 947kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:36, 996kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:44, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:47, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:23, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:19, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:31, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<01:49, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:45, 806kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:40, 958kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<04:09, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<03:02, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:34, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:24, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:36, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:43, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:49, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:12, 2.39MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:27, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:05, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:29, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:48, 2.86MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:37, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:11, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:22, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:39, 1.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:39, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:00, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:11, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<02:54, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:40, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:00, 2.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:27, 3.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<18:27, 269kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<14:42, 337kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<10:55, 453kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<07:48, 633kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<05:30, 891kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:52, 832kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:55, 991kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:37, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:21, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:02, 1.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:14, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:21, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:56, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:20, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:25, 1.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<01:53, 2.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:06, 2.19MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:39, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:06, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:31, 2.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:53, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:46, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:07, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:14, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:08, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:31, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:50, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:21, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:23, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:49, 2.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:21, 3.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:29, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:29, 1.74MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:55, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:22, 3.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<1:24:14, 50.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<59:25, 71.8kB/s]  .vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<41:31, 102kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<29:36, 142kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<22:10, 189kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<15:48, 265kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<11:03, 376kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<08:46, 471kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:49, 604kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:56, 833kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:07, 987kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<04:17, 947kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:17, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:25, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:25, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:23, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:50, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:57, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:35, 2.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:45, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:34, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:07, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:33, 2.45MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:07, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:07, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:38, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:46, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:31, 1.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:02, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:30, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:02, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:02, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:34, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:42, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:26, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:00, 1.78MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:28, 2.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:59, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:58, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:31, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:09, 2.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:51, 1.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:10, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:43, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:15, 2.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:55, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:08, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:39, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:13, 2.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:41, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:56, 1.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:31, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:07, 2.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:57, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:31, 2.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:07, 2.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:38, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:52, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:29, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:04, 2.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:07, 992kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:52, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:08, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:31, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:42, 1.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:36, 1.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:58, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:24, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:15, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:16, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:43, 1.70MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:14, 2.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:03, 1.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:01, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:33, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:07, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:31, 622kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:49, 736kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:48, 998kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:00, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:04, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:04, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:36, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:09, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:43, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:49, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:24, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:00, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:44, 950kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:30, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:52, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:19, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:56, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:54, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:27, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:02, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:57, 2.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<1:51:00, 22.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<1:17:43, 31.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<53:51, 45.2kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<37:45, 63.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<26:57, 89.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<18:55, 126kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<13:01, 180kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<10:50, 215kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<08:07, 287kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<05:46, 402kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<04:00, 569kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:39, 619kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:04, 734kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:15, 996kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:34, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:49, 777kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:27, 889kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:49, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:16, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:48, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:39, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:15, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<00:52, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<07:13, 284kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<05:31, 371kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<03:57, 516kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<02:45, 729kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:32, 782kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:14, 887kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:39, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<01:09, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<02:01, 945kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:50, 1.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:58, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:10, 850kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:51, 989kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:22, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:14, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:14, 1.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:56, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:41, 2.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:53, 1.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:56, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:44, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:31, 3.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<06:16, 262kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<04:41, 350kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<03:18, 490kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<02:17, 693kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:25, 647kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:04, 754kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:32, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:04, 1.41MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:47, 838kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:37, 925kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:12, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:50, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:36, 894kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:27, 983kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:05, 1.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:45, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:03, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:58, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:43, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:30, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:28, 881kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:18, 983kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:58, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:40, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:58, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:54, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:40, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:28, 2.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<02:27, 470kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:59, 577kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:26, 788kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:59, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:24, 775kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:10, 923kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:51, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:44, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:42, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:31, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:22, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:48, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:45, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:34, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:23, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:16, 693kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:05, 805kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:47, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:32, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:56, 858kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:50, 964kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:25, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:33, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:33, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:25, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:17, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:47, 858kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:40, 1.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:29, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:19, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<03:40, 164kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<02:42, 222kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<01:53, 312kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<01:16, 442kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:00, 528kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:49, 641kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:35, 877kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:24, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:23, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:20, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:09, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<02:02, 194kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<01:31, 259kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<01:03, 363kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:39, 515kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:38, 514kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 622kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:22, 851kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:14, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:14, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:12, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:09, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:05, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.89MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.95MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<26:42:56,  4.16it/s]  0%|          | 831/400000 [00:00<18:39:58,  5.94it/s]  0%|          | 1629/400000 [00:00<13:02:39,  8.48it/s]  1%|          | 2441/400000 [00:00<9:06:59, 12.11it/s]   1%|          | 3240/400000 [00:00<6:22:22, 17.29it/s]  1%|          | 4024/400000 [00:00<4:27:23, 24.68it/s]  1%|          | 4815/400000 [00:00<3:07:02, 35.21it/s]  1%|▏         | 5563/400000 [00:00<2:10:56, 50.20it/s]  2%|▏         | 6365/400000 [00:01<1:31:43, 71.53it/s]  2%|▏         | 7163/400000 [00:01<1:04:19, 101.79it/s]  2%|▏         | 7922/400000 [00:01<45:11, 144.58it/s]    2%|▏         | 8710/400000 [00:01<31:49, 204.93it/s]  2%|▏         | 9509/400000 [00:01<22:28, 289.57it/s]  3%|▎         | 10294/400000 [00:01<15:56, 407.23it/s]  3%|▎         | 11135/400000 [00:01<11:22, 569.92it/s]  3%|▎         | 11965/400000 [00:01<08:10, 790.90it/s]  3%|▎         | 12794/400000 [00:01<05:56, 1085.43it/s]  3%|▎         | 13608/400000 [00:01<04:23, 1464.45it/s]  4%|▎         | 14415/400000 [00:02<03:18, 1939.02it/s]  4%|▍         | 15219/400000 [00:02<02:34, 2494.10it/s]  4%|▍         | 16007/400000 [00:02<02:02, 3126.10it/s]  4%|▍         | 16807/400000 [00:02<01:40, 3824.87it/s]  4%|▍         | 17635/400000 [00:02<01:23, 4560.57it/s]  5%|▍         | 18484/400000 [00:02<01:12, 5294.95it/s]  5%|▍         | 19298/400000 [00:02<01:04, 5914.12it/s]  5%|▌         | 20164/400000 [00:02<00:58, 6535.73it/s]  5%|▌         | 21007/400000 [00:02<00:54, 7008.14it/s]  5%|▌         | 21865/400000 [00:02<00:50, 7414.53it/s]  6%|▌         | 22706/400000 [00:03<00:49, 7609.53it/s]  6%|▌         | 23538/400000 [00:03<00:49, 7662.21it/s]  6%|▌         | 24354/400000 [00:03<00:48, 7788.55it/s]  6%|▋         | 25202/400000 [00:03<00:46, 7982.24it/s]  7%|▋         | 26056/400000 [00:03<00:45, 8139.12it/s]  7%|▋         | 26910/400000 [00:03<00:45, 8255.31it/s]  7%|▋         | 27766/400000 [00:03<00:44, 8343.55it/s]  7%|▋         | 28627/400000 [00:03<00:44, 8420.22it/s]  7%|▋         | 29477/400000 [00:03<00:44, 8399.32it/s]  8%|▊         | 30339/400000 [00:03<00:43, 8463.25it/s]  8%|▊         | 31189/400000 [00:04<00:43, 8464.69it/s]  8%|▊         | 32038/400000 [00:04<00:44, 8253.62it/s]  8%|▊         | 32867/400000 [00:04<00:44, 8224.33it/s]  8%|▊         | 33720/400000 [00:04<00:44, 8310.96it/s]  9%|▊         | 34585/400000 [00:04<00:43, 8409.05it/s]  9%|▉         | 35443/400000 [00:04<00:43, 8459.18it/s]  9%|▉         | 36290/400000 [00:04<00:45, 8045.29it/s]  9%|▉         | 37123/400000 [00:04<00:44, 8127.42it/s]  9%|▉         | 37961/400000 [00:04<00:44, 8201.39it/s] 10%|▉         | 38823/400000 [00:04<00:43, 8322.56it/s] 10%|▉         | 39658/400000 [00:05<00:43, 8225.45it/s] 10%|█         | 40483/400000 [00:05<00:43, 8171.61it/s] 10%|█         | 41303/400000 [00:05<00:43, 8177.70it/s] 11%|█         | 42122/400000 [00:05<00:44, 8072.01it/s] 11%|█         | 42931/400000 [00:05<00:44, 7998.99it/s] 11%|█         | 43765/400000 [00:05<00:44, 8096.08it/s] 11%|█         | 44576/400000 [00:05<00:44, 8044.12it/s] 11%|█▏        | 45382/400000 [00:05<00:44, 7891.78it/s] 12%|█▏        | 46173/400000 [00:05<00:44, 7891.31it/s] 12%|█▏        | 46967/400000 [00:06<00:44, 7903.64it/s] 12%|█▏        | 47758/400000 [00:06<00:44, 7898.60it/s] 12%|█▏        | 48557/400000 [00:06<00:44, 7925.54it/s] 12%|█▏        | 49350/400000 [00:06<00:44, 7879.01it/s] 13%|█▎        | 50142/400000 [00:06<00:44, 7890.31it/s] 13%|█▎        | 50932/400000 [00:06<00:44, 7857.98it/s] 13%|█▎        | 51725/400000 [00:06<00:44, 7878.11it/s] 13%|█▎        | 52527/400000 [00:06<00:43, 7918.19it/s] 13%|█▎        | 53319/400000 [00:06<00:43, 7891.04it/s] 14%|█▎        | 54113/400000 [00:06<00:43, 7903.67it/s] 14%|█▎        | 54906/400000 [00:07<00:43, 7910.31it/s] 14%|█▍        | 55698/400000 [00:07<00:43, 7832.47it/s] 14%|█▍        | 56494/400000 [00:07<00:43, 7867.99it/s] 14%|█▍        | 57288/400000 [00:07<00:43, 7888.58it/s] 15%|█▍        | 58078/400000 [00:07<00:43, 7825.84it/s] 15%|█▍        | 58869/400000 [00:07<00:43, 7848.80it/s] 15%|█▍        | 59655/400000 [00:07<00:43, 7823.34it/s] 15%|█▌        | 60438/400000 [00:07<00:44, 7651.11it/s] 15%|█▌        | 61233/400000 [00:07<00:43, 7736.31it/s] 16%|█▌        | 62023/400000 [00:07<00:43, 7783.26it/s] 16%|█▌        | 62803/400000 [00:08<00:43, 7759.48it/s] 16%|█▌        | 63600/400000 [00:08<00:43, 7821.19it/s] 16%|█▌        | 64450/400000 [00:08<00:41, 8010.50it/s] 16%|█▋        | 65255/400000 [00:08<00:41, 8022.04it/s] 17%|█▋        | 66059/400000 [00:08<00:41, 7967.13it/s] 17%|█▋        | 66857/400000 [00:08<00:42, 7875.71it/s] 17%|█▋        | 67646/400000 [00:08<00:42, 7865.44it/s] 17%|█▋        | 68438/400000 [00:08<00:42, 7880.44it/s] 17%|█▋        | 69230/400000 [00:08<00:41, 7891.69it/s] 18%|█▊        | 70020/400000 [00:08<00:42, 7839.87it/s] 18%|█▊        | 70832/400000 [00:09<00:41, 7918.33it/s] 18%|█▊        | 71669/400000 [00:09<00:40, 8048.32it/s] 18%|█▊        | 72525/400000 [00:09<00:39, 8193.86it/s] 18%|█▊        | 73375/400000 [00:09<00:39, 8281.54it/s] 19%|█▊        | 74205/400000 [00:09<00:39, 8254.79it/s] 19%|█▉        | 75032/400000 [00:09<00:39, 8124.98it/s] 19%|█▉        | 75846/400000 [00:09<00:40, 8065.12it/s] 19%|█▉        | 76696/400000 [00:09<00:39, 8190.45it/s] 19%|█▉        | 77551/400000 [00:09<00:38, 8292.79it/s] 20%|█▉        | 78388/400000 [00:09<00:38, 8311.28it/s] 20%|█▉        | 79220/400000 [00:10<00:38, 8292.85it/s] 20%|██        | 80062/400000 [00:10<00:38, 8329.17it/s] 20%|██        | 80907/400000 [00:10<00:38, 8362.32it/s] 20%|██        | 81764/400000 [00:10<00:37, 8417.00it/s] 21%|██        | 82607/400000 [00:10<00:38, 8250.15it/s] 21%|██        | 83433/400000 [00:10<00:39, 8094.62it/s] 21%|██        | 84261/400000 [00:10<00:38, 8146.54it/s] 21%|██▏       | 85124/400000 [00:10<00:38, 8285.28it/s] 21%|██▏       | 85971/400000 [00:10<00:37, 8339.64it/s] 22%|██▏       | 86840/400000 [00:10<00:37, 8440.89it/s] 22%|██▏       | 87695/400000 [00:11<00:36, 8471.38it/s] 22%|██▏       | 88543/400000 [00:11<00:36, 8434.22it/s] 22%|██▏       | 89387/400000 [00:11<00:36, 8428.95it/s] 23%|██▎       | 90231/400000 [00:11<00:37, 8370.76it/s] 23%|██▎       | 91088/400000 [00:11<00:36, 8427.99it/s] 23%|██▎       | 91932/400000 [00:11<00:36, 8425.50it/s] 23%|██▎       | 92775/400000 [00:11<00:36, 8357.70it/s] 23%|██▎       | 93612/400000 [00:11<00:36, 8360.78it/s] 24%|██▎       | 94466/400000 [00:11<00:36, 8412.34it/s] 24%|██▍       | 95308/400000 [00:11<00:36, 8267.01it/s] 24%|██▍       | 96136/400000 [00:12<00:37, 8091.81it/s] 24%|██▍       | 96947/400000 [00:12<00:38, 7963.12it/s] 24%|██▍       | 97765/400000 [00:12<00:37, 8025.68it/s] 25%|██▍       | 98577/400000 [00:12<00:37, 8053.70it/s] 25%|██▍       | 99417/400000 [00:12<00:36, 8152.27it/s] 25%|██▌       | 100248/400000 [00:12<00:36, 8198.22it/s] 25%|██▌       | 101069/400000 [00:12<00:36, 8186.08it/s] 25%|██▌       | 101889/400000 [00:12<00:36, 8158.49it/s] 26%|██▌       | 102725/400000 [00:12<00:36, 8215.67it/s] 26%|██▌       | 103547/400000 [00:13<00:36, 8114.22it/s] 26%|██▌       | 104359/400000 [00:13<00:37, 7973.30it/s] 26%|██▋       | 105158/400000 [00:13<00:37, 7944.14it/s] 26%|██▋       | 105954/400000 [00:13<00:37, 7853.73it/s] 27%|██▋       | 106741/400000 [00:13<00:37, 7819.24it/s] 27%|██▋       | 107531/400000 [00:13<00:37, 7841.43it/s] 27%|██▋       | 108389/400000 [00:13<00:36, 8046.81it/s] 27%|██▋       | 109255/400000 [00:13<00:35, 8218.94it/s] 28%|██▊       | 110103/400000 [00:13<00:34, 8294.71it/s] 28%|██▊       | 110959/400000 [00:13<00:34, 8370.78it/s] 28%|██▊       | 111821/400000 [00:14<00:34, 8441.26it/s] 28%|██▊       | 112679/400000 [00:14<00:33, 8479.83it/s] 28%|██▊       | 113528/400000 [00:14<00:33, 8474.70it/s] 29%|██▊       | 114377/400000 [00:14<00:33, 8441.29it/s] 29%|██▉       | 115222/400000 [00:14<00:34, 8291.82it/s] 29%|██▉       | 116053/400000 [00:14<00:34, 8181.10it/s] 29%|██▉       | 116873/400000 [00:14<00:35, 7991.55it/s] 29%|██▉       | 117699/400000 [00:14<00:34, 8067.71it/s] 30%|██▉       | 118527/400000 [00:14<00:34, 8129.69it/s] 30%|██▉       | 119378/400000 [00:14<00:34, 8238.56it/s] 30%|███       | 120203/400000 [00:15<00:34, 8202.86it/s] 30%|███       | 121032/400000 [00:15<00:33, 8227.20it/s] 30%|███       | 121869/400000 [00:15<00:33, 8266.80it/s] 31%|███       | 122697/400000 [00:15<00:33, 8222.43it/s] 31%|███       | 123520/400000 [00:15<00:33, 8195.89it/s] 31%|███       | 124340/400000 [00:15<00:33, 8109.74it/s] 31%|███▏      | 125152/400000 [00:15<00:33, 8100.29it/s] 31%|███▏      | 125968/400000 [00:15<00:33, 8116.30it/s] 32%|███▏      | 126780/400000 [00:15<00:34, 8031.99it/s] 32%|███▏      | 127584/400000 [00:15<00:34, 7916.76it/s] 32%|███▏      | 128377/400000 [00:16<00:34, 7866.70it/s] 32%|███▏      | 129165/400000 [00:16<00:34, 7822.45it/s] 32%|███▏      | 129995/400000 [00:16<00:33, 7957.46it/s] 33%|███▎      | 130824/400000 [00:16<00:33, 8051.88it/s] 33%|███▎      | 131653/400000 [00:16<00:33, 8121.30it/s] 33%|███▎      | 132502/400000 [00:16<00:32, 8226.59it/s] 33%|███▎      | 133338/400000 [00:16<00:32, 8265.00it/s] 34%|███▎      | 134166/400000 [00:16<00:32, 8123.01it/s] 34%|███▎      | 134980/400000 [00:16<00:33, 7973.69it/s] 34%|███▍      | 135779/400000 [00:16<00:33, 7930.24it/s] 34%|███▍      | 136576/400000 [00:17<00:33, 7939.28it/s] 34%|███▍      | 137375/400000 [00:17<00:33, 7953.24it/s] 35%|███▍      | 138171/400000 [00:17<00:32, 7952.34it/s] 35%|███▍      | 138994/400000 [00:17<00:32, 8031.52it/s] 35%|███▍      | 139840/400000 [00:17<00:31, 8154.11it/s] 35%|███▌      | 140677/400000 [00:17<00:31, 8216.16it/s] 35%|███▌      | 141501/400000 [00:17<00:31, 8222.86it/s] 36%|███▌      | 142329/400000 [00:17<00:31, 8237.14it/s] 36%|███▌      | 143154/400000 [00:17<00:31, 8213.60it/s] 36%|███▌      | 144017/400000 [00:17<00:30, 8332.70it/s] 36%|███▌      | 144864/400000 [00:18<00:30, 8372.25it/s] 36%|███▋      | 145747/400000 [00:18<00:29, 8502.42it/s] 37%|███▋      | 146599/400000 [00:18<00:30, 8389.87it/s] 37%|███▋      | 147439/400000 [00:18<00:31, 8133.77it/s] 37%|███▋      | 148255/400000 [00:18<00:31, 8051.36it/s] 37%|███▋      | 149062/400000 [00:18<00:31, 7967.11it/s] 37%|███▋      | 149872/400000 [00:18<00:31, 8004.64it/s] 38%|███▊      | 150684/400000 [00:18<00:31, 8037.76it/s] 38%|███▊      | 151539/400000 [00:18<00:30, 8182.61it/s] 38%|███▊      | 152383/400000 [00:19<00:29, 8257.64it/s] 38%|███▊      | 153235/400000 [00:19<00:29, 8332.52it/s] 39%|███▊      | 154098/400000 [00:19<00:29, 8416.94it/s] 39%|███▊      | 154970/400000 [00:19<00:28, 8503.90it/s] 39%|███▉      | 155822/400000 [00:19<00:28, 8504.31it/s] 39%|███▉      | 156680/400000 [00:19<00:28, 8525.00it/s] 39%|███▉      | 157533/400000 [00:19<00:28, 8515.82it/s] 40%|███▉      | 158385/400000 [00:19<00:29, 8249.57it/s] 40%|███▉      | 159212/400000 [00:19<00:29, 8228.61it/s] 40%|████      | 160066/400000 [00:19<00:28, 8317.06it/s] 40%|████      | 160908/400000 [00:20<00:28, 8346.99it/s] 40%|████      | 161744/400000 [00:20<00:29, 8162.12it/s] 41%|████      | 162562/400000 [00:20<00:29, 8139.35it/s] 41%|████      | 163378/400000 [00:20<00:29, 8081.53it/s] 41%|████      | 164188/400000 [00:20<00:29, 8080.28it/s] 41%|████▏     | 165024/400000 [00:20<00:28, 8162.14it/s] 41%|████▏     | 165900/400000 [00:20<00:28, 8332.26it/s] 42%|████▏     | 166761/400000 [00:20<00:27, 8413.13it/s] 42%|████▏     | 167618/400000 [00:20<00:27, 8458.82it/s] 42%|████▏     | 168473/400000 [00:20<00:27, 8483.45it/s] 42%|████▏     | 169334/400000 [00:21<00:27, 8520.08it/s] 43%|████▎     | 170199/400000 [00:21<00:26, 8556.48it/s] 43%|████▎     | 171073/400000 [00:21<00:26, 8608.04it/s] 43%|████▎     | 171943/400000 [00:21<00:26, 8634.03it/s] 43%|████▎     | 172807/400000 [00:21<00:26, 8564.89it/s] 43%|████▎     | 173670/400000 [00:21<00:26, 8583.30it/s] 44%|████▎     | 174529/400000 [00:21<00:26, 8546.69it/s] 44%|████▍     | 175384/400000 [00:21<00:26, 8462.41it/s] 44%|████▍     | 176231/400000 [00:21<00:26, 8339.22it/s] 44%|████▍     | 177066/400000 [00:21<00:27, 8234.50it/s] 44%|████▍     | 177891/400000 [00:22<00:27, 8203.01it/s] 45%|████▍     | 178713/400000 [00:22<00:26, 8207.27it/s] 45%|████▍     | 179551/400000 [00:22<00:26, 8257.10it/s] 45%|████▌     | 180396/400000 [00:22<00:26, 8312.38it/s] 45%|████▌     | 181255/400000 [00:22<00:26, 8392.68it/s] 46%|████▌     | 182107/400000 [00:22<00:25, 8428.02it/s] 46%|████▌     | 182951/400000 [00:22<00:26, 8326.48it/s] 46%|████▌     | 183785/400000 [00:22<00:26, 8243.55it/s] 46%|████▌     | 184610/400000 [00:22<00:26, 8198.66it/s] 46%|████▋     | 185431/400000 [00:22<00:26, 8145.32it/s] 47%|████▋     | 186246/400000 [00:23<00:26, 8051.64it/s] 47%|████▋     | 187052/400000 [00:23<00:27, 7878.04it/s] 47%|████▋     | 187843/400000 [00:23<00:26, 7881.01it/s] 47%|████▋     | 188632/400000 [00:23<00:26, 7864.82it/s] 47%|████▋     | 189420/400000 [00:23<00:26, 7828.01it/s] 48%|████▊     | 190229/400000 [00:23<00:26, 7902.33it/s] 48%|████▊     | 191042/400000 [00:23<00:26, 7966.16it/s] 48%|████▊     | 191845/400000 [00:23<00:26, 7982.58it/s] 48%|████▊     | 192648/400000 [00:23<00:25, 7994.32it/s] 48%|████▊     | 193458/400000 [00:23<00:25, 8023.14it/s] 49%|████▊     | 194261/400000 [00:24<00:25, 7999.76it/s] 49%|████▉     | 195065/400000 [00:24<00:25, 8010.43it/s] 49%|████▉     | 195890/400000 [00:24<00:25, 8078.65it/s] 49%|████▉     | 196727/400000 [00:24<00:24, 8162.15it/s] 49%|████▉     | 197570/400000 [00:24<00:24, 8239.17it/s] 50%|████▉     | 198437/400000 [00:24<00:24, 8362.44it/s] 50%|████▉     | 199274/400000 [00:24<00:24, 8151.17it/s] 50%|█████     | 200114/400000 [00:24<00:24, 8223.36it/s] 50%|█████     | 200947/400000 [00:24<00:24, 8253.07it/s] 50%|█████     | 201790/400000 [00:24<00:23, 8303.60it/s] 51%|█████     | 202622/400000 [00:25<00:24, 8212.15it/s] 51%|█████     | 203444/400000 [00:25<00:24, 8182.09it/s] 51%|█████     | 204263/400000 [00:25<00:24, 8129.50it/s] 51%|█████▏    | 205077/400000 [00:25<00:24, 8099.40it/s] 51%|█████▏    | 205924/400000 [00:25<00:23, 8205.46it/s] 52%|█████▏    | 206793/400000 [00:25<00:23, 8342.76it/s] 52%|█████▏    | 207662/400000 [00:25<00:22, 8442.45it/s] 52%|█████▏    | 208508/400000 [00:25<00:22, 8446.45it/s] 52%|█████▏    | 209354/400000 [00:25<00:22, 8392.30it/s] 53%|█████▎    | 210194/400000 [00:26<00:23, 8219.61it/s] 53%|█████▎    | 211065/400000 [00:26<00:22, 8358.71it/s] 53%|█████▎    | 211903/400000 [00:26<00:22, 8356.51it/s] 53%|█████▎    | 212763/400000 [00:26<00:22, 8425.83it/s] 53%|█████▎    | 213607/400000 [00:26<00:22, 8363.84it/s] 54%|█████▎    | 214452/400000 [00:26<00:22, 8388.40it/s] 54%|█████▍    | 215316/400000 [00:26<00:21, 8460.08it/s] 54%|█████▍    | 216163/400000 [00:26<00:22, 8305.22it/s] 54%|█████▍    | 216995/400000 [00:26<00:22, 8213.92it/s] 54%|█████▍    | 217830/400000 [00:26<00:22, 8253.22it/s] 55%|█████▍    | 218672/400000 [00:27<00:21, 8300.06it/s] 55%|█████▍    | 219541/400000 [00:27<00:21, 8412.25it/s] 55%|█████▌    | 220390/400000 [00:27<00:21, 8432.88it/s] 55%|█████▌    | 221244/400000 [00:27<00:21, 8462.71it/s] 56%|█████▌    | 222091/400000 [00:27<00:21, 8388.84it/s] 56%|█████▌    | 222931/400000 [00:27<00:21, 8368.15it/s] 56%|█████▌    | 223769/400000 [00:27<00:21, 8135.30it/s] 56%|█████▌    | 224585/400000 [00:27<00:21, 8034.09it/s] 56%|█████▋    | 225390/400000 [00:27<00:21, 8011.46it/s] 57%|█████▋    | 226193/400000 [00:27<00:21, 7996.79it/s] 57%|█████▋    | 226994/400000 [00:28<00:21, 7968.20it/s] 57%|█████▋    | 227817/400000 [00:28<00:21, 8044.75it/s] 57%|█████▋    | 228636/400000 [00:28<00:21, 8085.48it/s] 57%|█████▋    | 229445/400000 [00:28<00:21, 8003.81it/s] 58%|█████▊    | 230246/400000 [00:28<00:21, 7928.50it/s] 58%|█████▊    | 231040/400000 [00:28<00:21, 7922.81it/s] 58%|█████▊    | 231888/400000 [00:28<00:20, 8079.83it/s] 58%|█████▊    | 232768/400000 [00:28<00:20, 8281.58it/s] 58%|█████▊    | 233622/400000 [00:28<00:19, 8355.87it/s] 59%|█████▊    | 234461/400000 [00:28<00:19, 8365.19it/s] 59%|█████▉    | 235305/400000 [00:29<00:19, 8387.41it/s] 59%|█████▉    | 236166/400000 [00:29<00:19, 8451.62it/s] 59%|█████▉    | 237027/400000 [00:29<00:19, 8495.67it/s] 59%|█████▉    | 237885/400000 [00:29<00:19, 8520.41it/s] 60%|█████▉    | 238738/400000 [00:29<00:19, 8459.27it/s] 60%|█████▉    | 239585/400000 [00:29<00:19, 8331.47it/s] 60%|██████    | 240419/400000 [00:29<00:19, 8249.83it/s] 60%|██████    | 241245/400000 [00:29<00:19, 8163.17it/s] 61%|██████    | 242062/400000 [00:29<00:19, 8118.31it/s] 61%|██████    | 242875/400000 [00:29<00:19, 8030.44it/s] 61%|██████    | 243679/400000 [00:30<00:19, 7876.62it/s] 61%|██████    | 244470/400000 [00:30<00:19, 7885.07it/s] 61%|██████▏   | 245273/400000 [00:30<00:19, 7925.84it/s] 62%|██████▏   | 246067/400000 [00:30<00:19, 7914.69it/s] 62%|██████▏   | 246894/400000 [00:30<00:19, 8015.30it/s] 62%|██████▏   | 247704/400000 [00:30<00:18, 8039.34it/s] 62%|██████▏   | 248564/400000 [00:30<00:18, 8199.00it/s] 62%|██████▏   | 249400/400000 [00:30<00:18, 8245.45it/s] 63%|██████▎   | 250226/400000 [00:30<00:18, 8190.08it/s] 63%|██████▎   | 251046/400000 [00:30<00:18, 8166.85it/s] 63%|██████▎   | 251881/400000 [00:31<00:18, 8219.22it/s] 63%|██████▎   | 252749/400000 [00:31<00:17, 8351.87it/s] 63%|██████▎   | 253623/400000 [00:31<00:17, 8462.51it/s] 64%|██████▎   | 254472/400000 [00:31<00:17, 8457.53it/s] 64%|██████▍   | 255323/400000 [00:31<00:17, 8472.85it/s] 64%|██████▍   | 256171/400000 [00:31<00:17, 8408.29it/s] 64%|██████▍   | 257040/400000 [00:31<00:16, 8489.67it/s] 64%|██████▍   | 257890/400000 [00:31<00:16, 8477.19it/s] 65%|██████▍   | 258739/400000 [00:31<00:16, 8400.85it/s] 65%|██████▍   | 259587/400000 [00:31<00:16, 8422.83it/s] 65%|██████▌   | 260430/400000 [00:32<00:16, 8342.72it/s] 65%|██████▌   | 261281/400000 [00:32<00:16, 8389.50it/s] 66%|██████▌   | 262121/400000 [00:32<00:16, 8364.56it/s] 66%|██████▌   | 262958/400000 [00:32<00:16, 8246.51it/s] 66%|██████▌   | 263784/400000 [00:32<00:16, 8215.21it/s] 66%|██████▌   | 264606/400000 [00:32<00:16, 8148.60it/s] 66%|██████▋   | 265430/400000 [00:32<00:16, 8175.73it/s] 67%|██████▋   | 266248/400000 [00:32<00:16, 8127.64it/s] 67%|██████▋   | 267062/400000 [00:32<00:16, 8074.48it/s] 67%|██████▋   | 267870/400000 [00:33<00:16, 8010.63it/s] 67%|██████▋   | 268672/400000 [00:33<00:16, 7949.00it/s] 67%|██████▋   | 269512/400000 [00:33<00:16, 8077.95it/s] 68%|██████▊   | 270353/400000 [00:33<00:15, 8173.40it/s] 68%|██████▊   | 271192/400000 [00:33<00:15, 8236.70it/s] 68%|██████▊   | 272017/400000 [00:33<00:15, 8184.78it/s] 68%|██████▊   | 272837/400000 [00:33<00:15, 7988.97it/s] 68%|██████▊   | 273638/400000 [00:33<00:15, 7918.28it/s] 69%|██████▊   | 274431/400000 [00:33<00:16, 7696.61it/s] 69%|██████▉   | 275236/400000 [00:33<00:15, 7799.10it/s] 69%|██████▉   | 276079/400000 [00:34<00:15, 7976.02it/s] 69%|██████▉   | 276890/400000 [00:34<00:15, 8013.59it/s] 69%|██████▉   | 277709/400000 [00:34<00:15, 8063.59it/s] 70%|██████▉   | 278517/400000 [00:34<00:15, 7975.79it/s] 70%|██████▉   | 279316/400000 [00:34<00:15, 7980.00it/s] 70%|███████   | 280115/400000 [00:34<00:15, 7941.01it/s] 70%|███████   | 280910/400000 [00:34<00:15, 7904.33it/s] 70%|███████   | 281732/400000 [00:34<00:14, 7995.74it/s] 71%|███████   | 282582/400000 [00:34<00:14, 8140.60it/s] 71%|███████   | 283436/400000 [00:34<00:14, 8255.92it/s] 71%|███████   | 284282/400000 [00:35<00:13, 8314.99it/s] 71%|███████▏  | 285115/400000 [00:35<00:14, 8174.26it/s] 71%|███████▏  | 285934/400000 [00:35<00:14, 8111.48it/s] 72%|███████▏  | 286747/400000 [00:35<00:14, 8087.83it/s] 72%|███████▏  | 287557/400000 [00:35<00:13, 8073.40it/s] 72%|███████▏  | 288365/400000 [00:35<00:13, 8011.30it/s] 72%|███████▏  | 289193/400000 [00:35<00:13, 8089.83it/s] 73%|███████▎  | 290038/400000 [00:35<00:13, 8192.27it/s] 73%|███████▎  | 290902/400000 [00:35<00:13, 8320.82it/s] 73%|███████▎  | 291768/400000 [00:35<00:12, 8418.89it/s] 73%|███████▎  | 292622/400000 [00:36<00:12, 8454.54it/s] 73%|███████▎  | 293469/400000 [00:36<00:12, 8458.81it/s] 74%|███████▎  | 294316/400000 [00:36<00:12, 8460.45it/s] 74%|███████▍  | 295176/400000 [00:36<00:12, 8499.04it/s] 74%|███████▍  | 296027/400000 [00:36<00:12, 8395.75it/s] 74%|███████▍  | 296868/400000 [00:36<00:12, 8035.73it/s] 74%|███████▍  | 297709/400000 [00:36<00:12, 8143.99it/s] 75%|███████▍  | 298527/400000 [00:36<00:12, 8130.49it/s] 75%|███████▍  | 299343/400000 [00:36<00:12, 8031.41it/s] 75%|███████▌  | 300148/400000 [00:36<00:12, 8017.11it/s] 75%|███████▌  | 300998/400000 [00:37<00:12, 8154.57it/s] 75%|███████▌  | 301851/400000 [00:37<00:11, 8262.77it/s] 76%|███████▌  | 302697/400000 [00:37<00:11, 8319.90it/s] 76%|███████▌  | 303550/400000 [00:37<00:11, 8379.84it/s] 76%|███████▌  | 304423/400000 [00:37<00:11, 8479.26it/s] 76%|███████▋  | 305275/400000 [00:37<00:11, 8490.84it/s] 77%|███████▋  | 306125/400000 [00:37<00:11, 8165.76it/s] 77%|███████▋  | 306945/400000 [00:37<00:11, 8065.18it/s] 77%|███████▋  | 307757/400000 [00:37<00:11, 8080.60it/s] 77%|███████▋  | 308567/400000 [00:38<00:11, 8035.37it/s] 77%|███████▋  | 309372/400000 [00:38<00:11, 7962.60it/s] 78%|███████▊  | 310190/400000 [00:38<00:11, 8026.53it/s] 78%|███████▊  | 311034/400000 [00:38<00:10, 8145.12it/s] 78%|███████▊  | 311857/400000 [00:38<00:10, 8167.63it/s] 78%|███████▊  | 312675/400000 [00:38<00:10, 8021.29it/s] 78%|███████▊  | 313479/400000 [00:38<00:10, 7930.55it/s] 79%|███████▊  | 314285/400000 [00:38<00:10, 7966.28it/s] 79%|███████▉  | 315144/400000 [00:38<00:10, 8142.08it/s] 79%|███████▉  | 315978/400000 [00:38<00:10, 8197.91it/s] 79%|███████▉  | 316837/400000 [00:39<00:10, 8309.86it/s] 79%|███████▉  | 317690/400000 [00:39<00:09, 8372.86it/s] 80%|███████▉  | 318533/400000 [00:39<00:09, 8388.13it/s] 80%|███████▉  | 319373/400000 [00:39<00:09, 8353.97it/s] 80%|████████  | 320209/400000 [00:39<00:09, 8198.61it/s] 80%|████████  | 321030/400000 [00:39<00:09, 8136.30it/s] 80%|████████  | 321845/400000 [00:39<00:09, 8059.36it/s] 81%|████████  | 322652/400000 [00:39<00:09, 8000.52it/s] 81%|████████  | 323453/400000 [00:39<00:09, 7927.68it/s] 81%|████████  | 324247/400000 [00:39<00:09, 7792.31it/s] 81%|████████▏ | 325050/400000 [00:40<00:09, 7861.28it/s] 81%|████████▏ | 325837/400000 [00:40<00:09, 7859.68it/s] 82%|████████▏ | 326634/400000 [00:40<00:09, 7891.33it/s] 82%|████████▏ | 327432/400000 [00:40<00:09, 7915.81it/s] 82%|████████▏ | 328224/400000 [00:40<00:09, 7820.58it/s] 82%|████████▏ | 329014/400000 [00:40<00:09, 7843.05it/s] 82%|████████▏ | 329879/400000 [00:40<00:08, 8067.16it/s] 83%|████████▎ | 330727/400000 [00:40<00:08, 8186.33it/s] 83%|████████▎ | 331596/400000 [00:40<00:08, 8330.67it/s] 83%|████████▎ | 332432/400000 [00:40<00:08, 8336.85it/s] 83%|████████▎ | 333281/400000 [00:41<00:07, 8379.96it/s] 84%|████████▎ | 334133/400000 [00:41<00:07, 8420.75it/s] 84%|████████▎ | 334983/400000 [00:41<00:07, 8441.62it/s] 84%|████████▍ | 335848/400000 [00:41<00:07, 8501.80it/s] 84%|████████▍ | 336699/400000 [00:41<00:07, 8454.16it/s] 84%|████████▍ | 337559/400000 [00:41<00:07, 8495.09it/s] 85%|████████▍ | 338409/400000 [00:41<00:07, 8354.49it/s] 85%|████████▍ | 339246/400000 [00:41<00:07, 8189.28it/s] 85%|████████▌ | 340067/400000 [00:41<00:07, 8112.18it/s] 85%|████████▌ | 340880/400000 [00:41<00:07, 8055.32it/s] 85%|████████▌ | 341695/400000 [00:42<00:07, 8075.21it/s] 86%|████████▌ | 342504/400000 [00:42<00:07, 8033.64it/s] 86%|████████▌ | 343308/400000 [00:42<00:07, 7913.89it/s] 86%|████████▌ | 344101/400000 [00:42<00:07, 7831.30it/s] 86%|████████▌ | 344885/400000 [00:42<00:07, 7797.57it/s] 86%|████████▋ | 345718/400000 [00:42<00:06, 7949.31it/s] 87%|████████▋ | 346537/400000 [00:42<00:06, 8017.75it/s] 87%|████████▋ | 347340/400000 [00:42<00:06, 7977.91it/s] 87%|████████▋ | 348139/400000 [00:42<00:06, 7914.32it/s] 87%|████████▋ | 348932/400000 [00:42<00:06, 7806.02it/s] 87%|████████▋ | 349714/400000 [00:43<00:06, 7738.79it/s] 88%|████████▊ | 350492/400000 [00:43<00:06, 7748.97it/s] 88%|████████▊ | 351278/400000 [00:43<00:06, 7780.48it/s] 88%|████████▊ | 352057/400000 [00:43<00:06, 7764.93it/s] 88%|████████▊ | 352834/400000 [00:43<00:06, 7733.58it/s] 88%|████████▊ | 353608/400000 [00:43<00:06, 7692.54it/s] 89%|████████▊ | 354378/400000 [00:43<00:05, 7652.26it/s] 89%|████████▉ | 355155/400000 [00:43<00:05, 7685.59it/s] 89%|████████▉ | 355932/400000 [00:43<00:05, 7709.76it/s] 89%|████████▉ | 356749/400000 [00:44<00:05, 7840.68it/s] 89%|████████▉ | 357594/400000 [00:44<00:05, 8012.03it/s] 90%|████████▉ | 358406/400000 [00:44<00:05, 8042.69it/s] 90%|████████▉ | 359224/400000 [00:44<00:05, 8081.32it/s] 90%|█████████ | 360033/400000 [00:44<00:04, 8024.66it/s] 90%|█████████ | 360867/400000 [00:44<00:04, 8116.40it/s] 90%|█████████ | 361722/400000 [00:44<00:04, 8241.20it/s] 91%|█████████ | 362548/400000 [00:44<00:04, 8184.87it/s] 91%|█████████ | 363368/400000 [00:44<00:04, 8103.42it/s] 91%|█████████ | 364180/400000 [00:44<00:04, 8069.38it/s] 91%|█████████ | 364988/400000 [00:45<00:04, 8022.15it/s] 91%|█████████▏| 365791/400000 [00:45<00:04, 7988.13it/s] 92%|█████████▏| 366591/400000 [00:45<00:04, 7935.71it/s] 92%|█████████▏| 367385/400000 [00:45<00:04, 7932.11it/s] 92%|█████████▏| 368179/400000 [00:45<00:04, 7873.42it/s] 92%|█████████▏| 368967/400000 [00:45<00:03, 7821.50it/s] 92%|█████████▏| 369782/400000 [00:45<00:03, 7916.58it/s] 93%|█████████▎| 370640/400000 [00:45<00:03, 8102.19it/s] 93%|█████████▎| 371487/400000 [00:45<00:03, 8207.76it/s] 93%|█████████▎| 372333/400000 [00:45<00:03, 8280.90it/s] 93%|█████████▎| 373188/400000 [00:46<00:03, 8354.39it/s] 94%|█████████▎| 374028/400000 [00:46<00:03, 8366.82it/s] 94%|█████████▎| 374887/400000 [00:46<00:02, 8430.08it/s] 94%|█████████▍| 375731/400000 [00:46<00:02, 8395.00it/s] 94%|█████████▍| 376573/400000 [00:46<00:02, 8402.22it/s] 94%|█████████▍| 377421/400000 [00:46<00:02, 8425.05it/s] 95%|█████████▍| 378264/400000 [00:46<00:02, 8246.39it/s] 95%|█████████▍| 379090/400000 [00:46<00:02, 8149.58it/s] 95%|█████████▍| 379906/400000 [00:46<00:02, 7950.92it/s] 95%|█████████▌| 380708/400000 [00:46<00:02, 7962.33it/s] 95%|█████████▌| 381506/400000 [00:47<00:02, 7924.08it/s] 96%|█████████▌| 382305/400000 [00:47<00:02, 7943.57it/s] 96%|█████████▌| 383100/400000 [00:47<00:02, 7934.62it/s] 96%|█████████▌| 383894/400000 [00:47<00:02, 7898.99it/s] 96%|█████████▌| 384685/400000 [00:47<00:01, 7845.49it/s] 96%|█████████▋| 385490/400000 [00:47<00:01, 7904.64it/s] 97%|█████████▋| 386339/400000 [00:47<00:01, 8069.91it/s] 97%|█████████▋| 387191/400000 [00:47<00:01, 8197.82it/s] 97%|█████████▋| 388013/400000 [00:47<00:01, 8184.97it/s] 97%|█████████▋| 388846/400000 [00:47<00:01, 8227.85it/s] 97%|█████████▋| 389670/400000 [00:48<00:01, 8174.50it/s] 98%|█████████▊| 390488/400000 [00:48<00:01, 8047.00it/s] 98%|█████████▊| 391294/400000 [00:48<00:01, 8033.26it/s] 98%|█████████▊| 392098/400000 [00:48<00:01, 7888.97it/s] 98%|█████████▊| 392900/400000 [00:48<00:00, 7926.11it/s] 98%|█████████▊| 393750/400000 [00:48<00:00, 8088.34it/s] 99%|█████████▊| 394603/400000 [00:48<00:00, 8213.66it/s] 99%|█████████▉| 395456/400000 [00:48<00:00, 8305.70it/s] 99%|█████████▉| 396295/400000 [00:48<00:00, 8327.98it/s] 99%|█████████▉| 397139/400000 [00:48<00:00, 8360.85it/s] 99%|█████████▉| 397981/400000 [00:49<00:00, 8378.39it/s]100%|█████████▉| 398820/400000 [00:49<00:00, 8355.84it/s]100%|█████████▉| 399659/400000 [00:49<00:00, 8365.65it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8111.70it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f50d4d9a6d8>, <torchtext.data.dataset.TabularDataset object at 0x7f50d4d9a828>, <torchtext.vocab.Vocab object at 0x7f50d4d9a748>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.08 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 25.02 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 15.21 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 15.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.96 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.64 file/s]2020-06-24 18:20:22.923552: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-24 18:20:22.928277: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-24 18:20:22.928449: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fb4a8db230 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-24 18:20:22.928465: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 24%|██▍       | 2400256/9912422 [00:00<00:00, 23453582.41it/s]9920512it [00:00, 30132787.57it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1123445.51it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1041121.68it/s]1654784it [00:00, 12694848.38it/s]                           
0it [00:00, ?it/s]8192it [00:00, 144310.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
