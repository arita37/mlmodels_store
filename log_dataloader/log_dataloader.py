
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f174d6dc0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f174d6dc0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f17b8a261e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f17b8a261e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f17d6d6de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f17d6d6de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1765d53620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1765d53620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1765d53620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 25%|██▌       | 2482176/9912422 [00:00<00:00, 24804633.39it/s] 88%|████████▊ | 8749056/9912422 [00:00<00:00, 30295612.58it/s]9920512it [00:00, 29752520.09it/s]                             
0it [00:00, ?it/s]32768it [00:00, 676775.80it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 145891.31it/s]1654784it [00:00, 10087885.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 233319.11it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f174cd92940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f174cd86be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1765d53268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1765d53268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1765d53268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.89 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:19,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13, 10.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:13, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:12, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12, 10.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 13.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 13.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 17.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 22.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 33.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 39.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 45.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 49.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 55.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 61.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 61.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 66.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.09s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.99 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.99 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.65s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.09s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 66.99 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.65s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.65s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 28.68 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.65s/ url]
0 examples [00:00, ? examples/s]2020-07-10 18:12:01.044975: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-10 18:12:01.059814: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-10 18:12:01.060038: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c51318820 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-10 18:12:01.060059: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
5 examples [00:00, 47.57 examples/s]89 examples [00:00, 66.33 examples/s]177 examples [00:00, 91.80 examples/s]267 examples [00:00, 125.59 examples/s]357 examples [00:00, 169.20 examples/s]443 examples [00:00, 222.85 examples/s]525 examples [00:00, 285.05 examples/s]616 examples [00:00, 358.76 examples/s]705 examples [00:00, 436.88 examples/s]788 examples [00:01, 507.99 examples/s]874 examples [00:01, 579.04 examples/s]963 examples [00:01, 645.58 examples/s]1053 examples [00:01, 688.14 examples/s]1137 examples [00:01, 720.78 examples/s]1224 examples [00:01, 759.49 examples/s]1309 examples [00:01, 772.89 examples/s]1393 examples [00:01, 781.28 examples/s]1476 examples [00:01, 792.70 examples/s]1559 examples [00:01, 802.26 examples/s]1644 examples [00:02, 814.92 examples/s]1728 examples [00:02, 821.93 examples/s]1812 examples [00:02, 825.58 examples/s]1899 examples [00:02, 837.57 examples/s]1988 examples [00:02, 851.60 examples/s]2074 examples [00:02, 844.80 examples/s]2163 examples [00:02, 856.64 examples/s]2251 examples [00:02, 861.44 examples/s]2338 examples [00:02, 832.53 examples/s]2424 examples [00:02, 838.98 examples/s]2509 examples [00:03, 841.79 examples/s]2595 examples [00:03, 844.62 examples/s]2680 examples [00:03, 836.62 examples/s]2764 examples [00:03, 831.88 examples/s]2848 examples [00:03, 828.24 examples/s]2931 examples [00:03, 824.29 examples/s]3020 examples [00:03, 841.10 examples/s]3105 examples [00:03, 825.25 examples/s]3194 examples [00:03, 841.40 examples/s]3284 examples [00:03, 857.51 examples/s]3373 examples [00:04, 865.01 examples/s]3460 examples [00:04, 865.43 examples/s]3547 examples [00:04, 857.69 examples/s]3633 examples [00:04, 848.87 examples/s]3721 examples [00:04, 856.86 examples/s]3808 examples [00:04, 858.23 examples/s]3895 examples [00:04, 860.66 examples/s]3982 examples [00:04, 852.81 examples/s]4068 examples [00:04, 854.44 examples/s]4158 examples [00:04, 862.60 examples/s]4245 examples [00:05, 847.82 examples/s]4330 examples [00:05, 837.98 examples/s]4416 examples [00:05, 843.54 examples/s]4505 examples [00:05, 854.19 examples/s]4591 examples [00:05, 822.20 examples/s]4679 examples [00:05, 836.82 examples/s]4766 examples [00:05, 846.05 examples/s]4851 examples [00:05, 847.08 examples/s]4936 examples [00:05, 840.10 examples/s]5029 examples [00:06, 864.16 examples/s]5120 examples [00:06, 875.10 examples/s]5208 examples [00:06, 867.63 examples/s]5295 examples [00:06, 844.59 examples/s]5380 examples [00:06, 843.50 examples/s]5468 examples [00:06, 852.13 examples/s]5557 examples [00:06, 861.41 examples/s]5644 examples [00:06, 814.17 examples/s]5727 examples [00:06, 780.69 examples/s]5812 examples [00:06, 799.73 examples/s]5897 examples [00:07, 811.44 examples/s]5983 examples [00:07, 824.87 examples/s]6070 examples [00:07, 837.12 examples/s]6155 examples [00:07, 838.67 examples/s]6243 examples [00:07, 848.52 examples/s]6329 examples [00:07, 848.32 examples/s]6415 examples [00:07, 850.53 examples/s]6501 examples [00:07, 845.55 examples/s]6588 examples [00:07, 850.58 examples/s]6675 examples [00:07, 855.12 examples/s]6763 examples [00:08, 862.12 examples/s]6850 examples [00:08, 858.36 examples/s]6936 examples [00:08, 856.92 examples/s]7024 examples [00:08, 862.53 examples/s]7111 examples [00:08, 864.27 examples/s]7198 examples [00:08, 848.44 examples/s]7285 examples [00:08, 853.59 examples/s]7371 examples [00:08, 829.60 examples/s]7458 examples [00:08, 841.08 examples/s]7545 examples [00:09, 848.99 examples/s]7633 examples [00:09, 857.70 examples/s]7719 examples [00:09, 855.90 examples/s]7806 examples [00:09, 859.94 examples/s]7893 examples [00:09, 859.66 examples/s]7980 examples [00:09, 860.14 examples/s]8067 examples [00:09, 851.83 examples/s]8153 examples [00:09, 845.08 examples/s]8238 examples [00:09, 829.91 examples/s]8322 examples [00:09, 825.44 examples/s]8405 examples [00:10, 826.51 examples/s]8488 examples [00:10, 824.12 examples/s]8571 examples [00:10, 808.57 examples/s]8652 examples [00:10, 784.08 examples/s]8737 examples [00:10, 801.01 examples/s]8825 examples [00:10, 821.08 examples/s]8913 examples [00:10, 836.41 examples/s]9000 examples [00:10, 845.98 examples/s]9086 examples [00:10, 848.55 examples/s]9175 examples [00:10, 858.94 examples/s]9263 examples [00:11, 864.80 examples/s]9354 examples [00:11, 875.28 examples/s]9443 examples [00:11, 877.19 examples/s]9531 examples [00:11, 868.00 examples/s]9618 examples [00:11, 862.58 examples/s]9705 examples [00:11, 839.67 examples/s]9790 examples [00:11, 836.91 examples/s]9875 examples [00:11, 839.06 examples/s]9959 examples [00:11, 831.82 examples/s]10043 examples [00:12, 762.01 examples/s]10121 examples [00:12, 758.86 examples/s]10202 examples [00:12, 771.54 examples/s]10289 examples [00:12, 797.45 examples/s]10378 examples [00:12, 823.05 examples/s]10464 examples [00:12, 830.86 examples/s]10548 examples [00:12, 831.38 examples/s]10632 examples [00:12, 827.82 examples/s]10716 examples [00:12, 793.76 examples/s]10802 examples [00:12, 812.25 examples/s]10889 examples [00:13, 827.94 examples/s]10974 examples [00:13, 832.94 examples/s]11062 examples [00:13, 845.41 examples/s]11147 examples [00:13, 841.89 examples/s]11232 examples [00:13, 837.99 examples/s]11316 examples [00:13, 838.37 examples/s]11400 examples [00:13, 822.71 examples/s]11483 examples [00:13, 807.91 examples/s]11566 examples [00:13, 812.42 examples/s]11650 examples [00:13, 819.78 examples/s]11739 examples [00:14, 839.05 examples/s]11829 examples [00:14, 854.82 examples/s]11915 examples [00:14, 833.27 examples/s]11999 examples [00:14, 806.48 examples/s]12081 examples [00:14, 807.26 examples/s]12169 examples [00:14, 827.38 examples/s]12258 examples [00:14, 843.82 examples/s]12344 examples [00:14, 847.66 examples/s]12429 examples [00:14, 840.79 examples/s]12514 examples [00:14, 831.81 examples/s]12598 examples [00:15, 832.57 examples/s]12683 examples [00:15, 836.89 examples/s]12767 examples [00:15, 834.31 examples/s]12851 examples [00:15, 823.41 examples/s]12934 examples [00:15, 823.34 examples/s]13019 examples [00:15, 830.56 examples/s]13103 examples [00:15, 822.39 examples/s]13186 examples [00:15, 807.63 examples/s]13268 examples [00:15, 810.38 examples/s]13357 examples [00:16, 831.05 examples/s]13441 examples [00:16, 832.87 examples/s]13527 examples [00:16, 839.17 examples/s]13613 examples [00:16, 845.05 examples/s]13699 examples [00:16, 848.72 examples/s]13785 examples [00:16, 849.36 examples/s]13870 examples [00:16, 826.50 examples/s]13958 examples [00:16, 839.90 examples/s]14045 examples [00:16, 846.95 examples/s]14130 examples [00:16, 827.62 examples/s]14216 examples [00:17, 835.78 examples/s]14300 examples [00:17, 826.03 examples/s]14387 examples [00:17, 836.82 examples/s]14472 examples [00:17, 838.28 examples/s]14559 examples [00:17, 846.39 examples/s]14644 examples [00:17, 831.60 examples/s]14728 examples [00:17, 825.00 examples/s]14815 examples [00:17, 837.25 examples/s]14903 examples [00:17, 849.10 examples/s]14990 examples [00:17, 854.63 examples/s]15078 examples [00:18, 860.51 examples/s]15165 examples [00:18, 839.52 examples/s]15253 examples [00:18, 849.52 examples/s]15341 examples [00:18, 856.08 examples/s]15427 examples [00:18, 857.02 examples/s]15514 examples [00:18, 858.94 examples/s]15601 examples [00:18, 860.28 examples/s]15688 examples [00:18, 846.75 examples/s]15775 examples [00:18, 851.75 examples/s]15861 examples [00:18, 847.62 examples/s]15946 examples [00:19, 838.97 examples/s]16034 examples [00:19, 848.26 examples/s]16120 examples [00:19, 849.03 examples/s]16205 examples [00:19, 835.99 examples/s]16290 examples [00:19, 839.04 examples/s]16380 examples [00:19, 854.22 examples/s]16469 examples [00:19, 863.33 examples/s]16558 examples [00:19, 870.79 examples/s]16649 examples [00:19, 880.45 examples/s]16738 examples [00:19, 875.50 examples/s]16828 examples [00:20, 879.88 examples/s]16917 examples [00:20, 880.44 examples/s]17006 examples [00:20, 883.23 examples/s]17095 examples [00:20, 873.19 examples/s]17183 examples [00:20, 855.43 examples/s]17269 examples [00:20, 854.25 examples/s]17359 examples [00:20, 865.25 examples/s]17448 examples [00:20, 871.79 examples/s]17536 examples [00:20, 869.78 examples/s]17624 examples [00:21, 868.88 examples/s]17711 examples [00:21, 865.27 examples/s]17798 examples [00:21, 844.98 examples/s]17883 examples [00:21, 837.11 examples/s]17967 examples [00:21, 797.57 examples/s]18055 examples [00:21, 819.09 examples/s]18140 examples [00:21, 826.40 examples/s]18224 examples [00:21, 829.31 examples/s]18310 examples [00:21, 836.88 examples/s]18394 examples [00:21, 829.30 examples/s]18478 examples [00:22, 812.25 examples/s]18564 examples [00:22, 823.70 examples/s]18652 examples [00:22, 837.37 examples/s]18737 examples [00:22, 839.31 examples/s]18822 examples [00:22, 842.01 examples/s]18909 examples [00:22, 849.73 examples/s]18995 examples [00:22, 848.93 examples/s]19080 examples [00:22, 836.76 examples/s]19167 examples [00:22, 844.15 examples/s]19253 examples [00:22, 848.76 examples/s]19340 examples [00:23, 852.39 examples/s]19426 examples [00:23, 850.89 examples/s]19514 examples [00:23, 858.08 examples/s]19602 examples [00:23, 863.13 examples/s]19689 examples [00:23, 861.39 examples/s]19777 examples [00:23, 866.09 examples/s]19864 examples [00:23, 794.54 examples/s]19948 examples [00:23, 805.87 examples/s]20030 examples [00:23, 741.60 examples/s]20118 examples [00:24, 776.77 examples/s]20203 examples [00:24, 794.92 examples/s]20293 examples [00:24, 822.56 examples/s]20378 examples [00:24, 828.94 examples/s]20462 examples [00:24, 814.44 examples/s]20545 examples [00:24, 815.24 examples/s]20627 examples [00:24, 812.28 examples/s]20710 examples [00:24, 815.51 examples/s]20794 examples [00:24, 822.64 examples/s]20877 examples [00:24, 818.75 examples/s]20960 examples [00:25, 819.54 examples/s]21043 examples [00:25, 813.47 examples/s]21128 examples [00:25, 823.06 examples/s]21214 examples [00:25, 832.06 examples/s]21298 examples [00:25, 823.30 examples/s]21381 examples [00:25, 819.01 examples/s]21463 examples [00:25, 805.86 examples/s]21547 examples [00:25, 815.34 examples/s]21631 examples [00:25, 821.66 examples/s]21715 examples [00:25, 824.23 examples/s]21802 examples [00:26, 835.04 examples/s]21886 examples [00:26, 832.90 examples/s]21970 examples [00:26, 818.29 examples/s]22052 examples [00:26, 804.02 examples/s]22134 examples [00:26, 808.65 examples/s]22220 examples [00:26, 822.00 examples/s]22303 examples [00:26, 815.64 examples/s]22389 examples [00:26, 827.52 examples/s]22475 examples [00:26, 836.34 examples/s]22560 examples [00:26, 838.91 examples/s]22645 examples [00:27, 840.12 examples/s]22730 examples [00:27, 823.12 examples/s]22813 examples [00:27, 815.45 examples/s]22899 examples [00:27, 826.19 examples/s]22982 examples [00:27, 824.96 examples/s]23065 examples [00:27, 820.45 examples/s]23149 examples [00:27, 823.54 examples/s]23233 examples [00:27, 827.55 examples/s]23324 examples [00:27, 850.02 examples/s]23413 examples [00:28, 859.65 examples/s]23500 examples [00:28, 861.39 examples/s]23587 examples [00:28, 832.31 examples/s]23674 examples [00:28, 840.68 examples/s]23759 examples [00:28, 837.54 examples/s]23844 examples [00:28, 838.45 examples/s]23928 examples [00:28, 806.85 examples/s]24011 examples [00:28, 811.31 examples/s]24099 examples [00:28, 828.36 examples/s]24183 examples [00:28, 829.62 examples/s]24267 examples [00:29, 765.71 examples/s]24352 examples [00:29, 787.20 examples/s]24432 examples [00:29, 784.53 examples/s]24517 examples [00:29, 801.67 examples/s]24598 examples [00:29, 799.53 examples/s]24685 examples [00:29, 818.39 examples/s]24768 examples [00:29, 805.16 examples/s]24858 examples [00:29, 829.07 examples/s]24942 examples [00:29, 828.35 examples/s]25026 examples [00:30, 806.75 examples/s]25107 examples [00:30, 790.10 examples/s]25187 examples [00:30, 779.75 examples/s]25266 examples [00:30, 776.57 examples/s]25350 examples [00:30, 794.24 examples/s]25430 examples [00:30, 793.44 examples/s]25512 examples [00:30, 800.05 examples/s]25593 examples [00:30, 760.13 examples/s]25676 examples [00:30, 777.44 examples/s]25762 examples [00:30, 799.61 examples/s]25848 examples [00:31, 814.63 examples/s]25935 examples [00:31, 830.19 examples/s]26019 examples [00:31, 825.05 examples/s]26102 examples [00:31, 823.02 examples/s]26185 examples [00:31, 776.30 examples/s]26268 examples [00:31, 789.58 examples/s]26348 examples [00:31, 791.03 examples/s]26429 examples [00:31, 795.49 examples/s]26517 examples [00:31, 816.64 examples/s]26599 examples [00:31, 783.73 examples/s]26678 examples [00:32, 762.69 examples/s]26757 examples [00:32, 768.99 examples/s]26836 examples [00:32, 773.62 examples/s]26916 examples [00:32, 780.91 examples/s]26998 examples [00:32, 789.98 examples/s]27081 examples [00:32, 801.06 examples/s]27166 examples [00:32, 814.18 examples/s]27250 examples [00:32, 819.20 examples/s]27335 examples [00:32, 825.81 examples/s]27424 examples [00:33, 843.26 examples/s]27512 examples [00:33, 852.88 examples/s]27598 examples [00:33, 839.94 examples/s]27687 examples [00:33, 853.82 examples/s]27775 examples [00:33, 859.10 examples/s]27862 examples [00:33, 853.88 examples/s]27948 examples [00:33, 855.08 examples/s]28034 examples [00:33, 855.22 examples/s]28120 examples [00:33, 850.43 examples/s]28206 examples [00:33, 853.12 examples/s]28292 examples [00:34, 841.24 examples/s]28377 examples [00:34, 834.48 examples/s]28461 examples [00:34, 833.83 examples/s]28547 examples [00:34, 838.78 examples/s]28631 examples [00:34, 839.12 examples/s]28720 examples [00:34, 850.89 examples/s]28809 examples [00:34, 861.03 examples/s]28898 examples [00:34, 868.02 examples/s]28985 examples [00:34, 853.47 examples/s]29071 examples [00:34, 843.35 examples/s]29156 examples [00:35, 838.61 examples/s]29240 examples [00:35, 831.92 examples/s]29324 examples [00:35, 789.40 examples/s]29408 examples [00:35, 801.71 examples/s]29494 examples [00:35, 818.10 examples/s]29582 examples [00:35, 833.30 examples/s]29668 examples [00:35, 841.11 examples/s]29753 examples [00:35, 821.64 examples/s]29840 examples [00:35, 834.86 examples/s]29924 examples [00:35, 833.49 examples/s]30008 examples [00:36, 780.50 examples/s]30093 examples [00:36, 799.19 examples/s]30179 examples [00:36, 814.45 examples/s]30267 examples [00:36, 832.27 examples/s]30354 examples [00:36, 840.92 examples/s]30443 examples [00:36, 852.88 examples/s]30529 examples [00:36, 848.87 examples/s]30615 examples [00:36, 848.19 examples/s]30701 examples [00:36, 851.56 examples/s]30787 examples [00:37, 844.88 examples/s]30872 examples [00:37, 833.43 examples/s]30957 examples [00:37, 838.29 examples/s]31041 examples [00:37, 824.37 examples/s]31124 examples [00:37, 812.25 examples/s]31206 examples [00:37, 803.78 examples/s]31294 examples [00:37, 823.06 examples/s]31381 examples [00:37, 834.92 examples/s]31465 examples [00:37, 809.15 examples/s]31550 examples [00:37, 818.96 examples/s]31633 examples [00:38, 819.39 examples/s]31716 examples [00:38, 816.79 examples/s]31798 examples [00:38, 792.31 examples/s]31878 examples [00:38, 791.14 examples/s]31958 examples [00:38, 786.31 examples/s]32040 examples [00:38, 795.82 examples/s]32123 examples [00:38, 803.34 examples/s]32204 examples [00:38, 786.58 examples/s]32284 examples [00:38, 790.51 examples/s]32367 examples [00:38, 798.63 examples/s]32447 examples [00:39, 785.33 examples/s]32529 examples [00:39, 792.44 examples/s]32609 examples [00:39, 781.70 examples/s]32688 examples [00:39, 779.88 examples/s]32771 examples [00:39, 792.38 examples/s]32857 examples [00:39, 808.79 examples/s]32939 examples [00:39, 796.52 examples/s]33019 examples [00:39, 791.79 examples/s]33099 examples [00:39, 782.96 examples/s]33178 examples [00:40, 784.75 examples/s]33258 examples [00:40, 786.72 examples/s]33345 examples [00:40, 808.88 examples/s]33428 examples [00:40, 814.42 examples/s]33517 examples [00:40, 834.45 examples/s]33604 examples [00:40, 844.74 examples/s]33693 examples [00:40, 856.34 examples/s]33780 examples [00:40, 858.11 examples/s]33866 examples [00:40, 840.87 examples/s]33951 examples [00:40, 837.57 examples/s]34038 examples [00:41, 845.81 examples/s]34123 examples [00:41, 845.49 examples/s]34213 examples [00:41, 860.43 examples/s]34300 examples [00:41, 854.57 examples/s]34387 examples [00:41, 859.05 examples/s]34473 examples [00:41, 836.37 examples/s]34557 examples [00:41, 826.60 examples/s]34643 examples [00:41, 833.32 examples/s]34727 examples [00:41, 811.32 examples/s]34812 examples [00:41, 822.02 examples/s]34895 examples [00:42, 793.17 examples/s]34980 examples [00:42, 807.96 examples/s]35065 examples [00:42, 818.43 examples/s]35149 examples [00:42, 823.08 examples/s]35232 examples [00:42, 797.53 examples/s]35315 examples [00:42, 805.42 examples/s]35401 examples [00:42, 821.03 examples/s]35484 examples [00:42, 796.91 examples/s]35568 examples [00:42, 809.11 examples/s]35655 examples [00:42, 824.98 examples/s]35740 examples [00:43, 831.31 examples/s]35824 examples [00:43, 831.29 examples/s]35911 examples [00:43, 839.70 examples/s]35997 examples [00:43, 843.00 examples/s]36088 examples [00:43, 861.05 examples/s]36175 examples [00:43, 839.78 examples/s]36264 examples [00:43, 853.66 examples/s]36350 examples [00:43, 852.73 examples/s]36436 examples [00:43, 853.39 examples/s]36524 examples [00:44, 859.84 examples/s]36611 examples [00:44, 813.15 examples/s]36693 examples [00:44, 803.94 examples/s]36782 examples [00:44, 826.60 examples/s]36867 examples [00:44, 832.45 examples/s]36954 examples [00:44, 842.78 examples/s]37039 examples [00:44, 841.74 examples/s]37124 examples [00:44, 831.84 examples/s]37208 examples [00:44, 832.07 examples/s]37292 examples [00:44, 827.69 examples/s]37375 examples [00:45, 808.69 examples/s]37461 examples [00:45, 821.28 examples/s]37547 examples [00:45, 832.12 examples/s]37633 examples [00:45, 839.42 examples/s]37718 examples [00:45, 838.23 examples/s]37804 examples [00:45, 843.72 examples/s]37892 examples [00:45, 853.12 examples/s]37980 examples [00:45, 860.44 examples/s]38067 examples [00:45, 856.91 examples/s]38153 examples [00:45, 844.89 examples/s]38240 examples [00:46, 851.67 examples/s]38332 examples [00:46, 869.59 examples/s]38421 examples [00:46, 872.95 examples/s]38509 examples [00:46, 866.74 examples/s]38596 examples [00:46, 860.05 examples/s]38684 examples [00:46, 865.78 examples/s]38771 examples [00:46, 867.02 examples/s]38858 examples [00:46, 865.41 examples/s]38945 examples [00:46, 861.80 examples/s]39032 examples [00:46, 862.40 examples/s]39119 examples [00:47, 841.54 examples/s]39204 examples [00:47, 822.92 examples/s]39289 examples [00:47, 830.26 examples/s]39375 examples [00:47, 836.86 examples/s]39459 examples [00:47, 835.60 examples/s]39543 examples [00:47, 833.03 examples/s]39631 examples [00:47, 846.10 examples/s]39716 examples [00:47, 837.67 examples/s]39800 examples [00:47, 812.73 examples/s]39885 examples [00:48, 822.59 examples/s]39968 examples [00:48, 791.62 examples/s]40048 examples [00:48, 745.45 examples/s]40133 examples [00:48, 773.31 examples/s]40212 examples [00:48, 764.58 examples/s]40294 examples [00:48, 778.34 examples/s]40381 examples [00:48, 802.43 examples/s]40469 examples [00:48, 824.05 examples/s]40552 examples [00:48, 820.48 examples/s]40639 examples [00:48, 832.93 examples/s]40724 examples [00:49, 835.50 examples/s]40808 examples [00:49, 804.85 examples/s]40889 examples [00:49, 805.38 examples/s]40970 examples [00:49, 745.83 examples/s]41054 examples [00:49, 770.70 examples/s]41138 examples [00:49, 790.10 examples/s]41218 examples [00:49, 773.45 examples/s]41303 examples [00:49, 794.85 examples/s]41390 examples [00:49, 813.22 examples/s]41472 examples [00:50, 802.82 examples/s]41559 examples [00:50, 820.37 examples/s]41644 examples [00:50, 826.65 examples/s]41731 examples [00:50, 839.13 examples/s]41819 examples [00:50, 850.71 examples/s]41905 examples [00:50, 843.03 examples/s]41990 examples [00:50, 839.69 examples/s]42075 examples [00:50, 837.83 examples/s]42162 examples [00:50, 846.32 examples/s]42247 examples [00:50, 846.73 examples/s]42332 examples [00:51, 843.69 examples/s]42417 examples [00:51, 832.76 examples/s]42501 examples [00:51, 827.28 examples/s]42590 examples [00:51, 843.18 examples/s]42675 examples [00:51, 836.04 examples/s]42759 examples [00:51, 792.01 examples/s]42839 examples [00:51, 781.56 examples/s]42920 examples [00:51, 789.78 examples/s]43002 examples [00:51, 798.54 examples/s]43093 examples [00:51, 828.19 examples/s]43177 examples [00:52, 797.94 examples/s]43262 examples [00:52, 809.67 examples/s]43346 examples [00:52, 816.49 examples/s]43434 examples [00:52, 832.44 examples/s]43518 examples [00:52, 816.08 examples/s]43602 examples [00:52, 822.60 examples/s]43690 examples [00:52, 837.02 examples/s]43774 examples [00:52, 807.09 examples/s]43858 examples [00:52, 815.06 examples/s]43940 examples [00:53, 808.04 examples/s]44022 examples [00:53, 808.21 examples/s]44108 examples [00:53, 821.83 examples/s]44194 examples [00:53, 831.63 examples/s]44282 examples [00:53, 843.81 examples/s]44367 examples [00:53, 840.54 examples/s]44452 examples [00:53, 840.25 examples/s]44538 examples [00:53, 844.49 examples/s]44623 examples [00:53, 843.33 examples/s]44708 examples [00:53, 831.56 examples/s]44793 examples [00:54, 836.57 examples/s]44877 examples [00:54, 833.77 examples/s]44962 examples [00:54, 836.77 examples/s]45047 examples [00:54, 840.24 examples/s]45132 examples [00:54, 833.50 examples/s]45220 examples [00:54, 846.91 examples/s]45306 examples [00:54, 847.46 examples/s]45394 examples [00:54, 855.45 examples/s]45483 examples [00:54, 864.95 examples/s]45574 examples [00:54, 875.17 examples/s]45663 examples [00:55, 879.39 examples/s]45751 examples [00:55, 874.03 examples/s]45839 examples [00:55, 850.85 examples/s]45925 examples [00:55, 836.82 examples/s]46009 examples [00:55, 837.15 examples/s]46097 examples [00:55, 848.02 examples/s]46182 examples [00:55, 840.98 examples/s]46267 examples [00:55, 839.31 examples/s]46353 examples [00:55, 842.38 examples/s]46438 examples [00:55, 836.12 examples/s]46522 examples [00:56, 799.60 examples/s]46603 examples [00:56, 801.50 examples/s]46686 examples [00:56, 808.55 examples/s]46768 examples [00:56, 805.80 examples/s]46853 examples [00:56, 816.87 examples/s]46937 examples [00:56, 821.22 examples/s]47020 examples [00:56, 798.57 examples/s]47106 examples [00:56, 814.76 examples/s]47191 examples [00:56, 822.79 examples/s]47274 examples [00:57, 800.48 examples/s]47355 examples [00:57, 801.70 examples/s]47436 examples [00:57, 799.74 examples/s]47521 examples [00:57, 814.10 examples/s]47607 examples [00:57, 827.34 examples/s]47695 examples [00:57, 839.72 examples/s]47780 examples [00:57, 828.29 examples/s]47863 examples [00:57, 820.96 examples/s]47946 examples [00:57, 783.25 examples/s]48029 examples [00:57, 796.58 examples/s]48111 examples [00:58, 802.22 examples/s]48197 examples [00:58, 818.16 examples/s]48280 examples [00:58, 811.07 examples/s]48363 examples [00:58, 815.90 examples/s]48445 examples [00:58, 814.86 examples/s]48527 examples [00:58, 801.93 examples/s]48610 examples [00:58, 809.03 examples/s]48693 examples [00:58, 813.52 examples/s]48775 examples [00:58, 811.45 examples/s]48862 examples [00:58, 826.88 examples/s]48948 examples [00:59, 833.99 examples/s]49035 examples [00:59, 842.34 examples/s]49120 examples [00:59, 837.88 examples/s]49204 examples [00:59, 831.94 examples/s]49288 examples [00:59, 808.97 examples/s]49370 examples [00:59, 810.13 examples/s]49452 examples [00:59, 797.13 examples/s]49532 examples [00:59, 784.67 examples/s]49618 examples [00:59, 804.89 examples/s]49702 examples [00:59, 813.41 examples/s]49785 examples [01:00, 817.79 examples/s]49867 examples [01:00, 801.01 examples/s]49950 examples [01:00, 807.40 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5310/50000 [00:00<00:00, 53098.66 examples/s] 32%|███▏      | 15752/50000 [00:00<00:00, 62281.39 examples/s] 52%|█████▏    | 26182/50000 [00:00<00:00, 70842.33 examples/s] 73%|███████▎  | 36412/50000 [00:00<00:00, 78041.41 examples/s] 94%|█████████▍| 46891/50000 [00:00<00:00, 84512.63 examples/s]                                                               0 examples [00:00, ? examples/s]64 examples [00:00, 629.45 examples/s]148 examples [00:00, 679.18 examples/s]227 examples [00:00, 708.05 examples/s]313 examples [00:00, 745.47 examples/s]401 examples [00:00, 779.75 examples/s]485 examples [00:00, 796.29 examples/s]564 examples [00:00, 792.73 examples/s]639 examples [00:00, 770.60 examples/s]728 examples [00:00, 801.05 examples/s]817 examples [00:01, 823.10 examples/s]903 examples [00:01, 832.74 examples/s]989 examples [00:01, 838.58 examples/s]1075 examples [00:01, 844.22 examples/s]1162 examples [00:01, 849.40 examples/s]1252 examples [00:01, 862.39 examples/s]1341 examples [00:01, 868.56 examples/s]1430 examples [00:01, 873.67 examples/s]1518 examples [00:01, 871.28 examples/s]1606 examples [00:01, 846.14 examples/s]1693 examples [00:02, 850.29 examples/s]1779 examples [00:02, 847.77 examples/s]1868 examples [00:02, 858.58 examples/s]1955 examples [00:02, 861.87 examples/s]2043 examples [00:02, 866.28 examples/s]2133 examples [00:02, 873.89 examples/s]2221 examples [00:02, 871.12 examples/s]2310 examples [00:02, 876.52 examples/s]2398 examples [00:02, 876.85 examples/s]2487 examples [00:02, 880.14 examples/s]2576 examples [00:03, 878.78 examples/s]2664 examples [00:03, 877.42 examples/s]2753 examples [00:03, 879.26 examples/s]2841 examples [00:03, 877.39 examples/s]2929 examples [00:03, 872.67 examples/s]3017 examples [00:03, 865.74 examples/s]3105 examples [00:03, 869.78 examples/s]3192 examples [00:03, 860.54 examples/s]3279 examples [00:03, 855.64 examples/s]3365 examples [00:03, 851.98 examples/s]3451 examples [00:04, 854.12 examples/s]3537 examples [00:04, 836.56 examples/s]3621 examples [00:04, 837.31 examples/s]3707 examples [00:04, 842.42 examples/s]3792 examples [00:04, 825.66 examples/s]3880 examples [00:04, 840.50 examples/s]3965 examples [00:04, 819.90 examples/s]4054 examples [00:04, 839.45 examples/s]4142 examples [00:04, 848.92 examples/s]4229 examples [00:04, 854.90 examples/s]4315 examples [00:05, 850.48 examples/s]4401 examples [00:05, 826.71 examples/s]4484 examples [00:05, 812.36 examples/s]4573 examples [00:05, 834.02 examples/s]4657 examples [00:05, 822.53 examples/s]4740 examples [00:05, 821.95 examples/s]4828 examples [00:05, 837.09 examples/s]4912 examples [00:05, 802.21 examples/s]4994 examples [00:05, 805.72 examples/s]5080 examples [00:06, 820.02 examples/s]5163 examples [00:06, 818.70 examples/s]5246 examples [00:06, 804.60 examples/s]5333 examples [00:06, 822.65 examples/s]5416 examples [00:06, 817.25 examples/s]5499 examples [00:06, 820.40 examples/s]5585 examples [00:06, 829.54 examples/s]5673 examples [00:06, 843.06 examples/s]5758 examples [00:06, 843.64 examples/s]5843 examples [00:06, 824.09 examples/s]5926 examples [00:07, 816.17 examples/s]6008 examples [00:07, 813.89 examples/s]6095 examples [00:07, 829.83 examples/s]6182 examples [00:07, 839.60 examples/s]6269 examples [00:07, 845.95 examples/s]6357 examples [00:07, 854.85 examples/s]6447 examples [00:07, 866.58 examples/s]6534 examples [00:07, 839.97 examples/s]6619 examples [00:07, 830.59 examples/s]6703 examples [00:07, 828.74 examples/s]6787 examples [00:08, 830.63 examples/s]6874 examples [00:08, 840.52 examples/s]6964 examples [00:08, 856.49 examples/s]7054 examples [00:08, 867.04 examples/s]7141 examples [00:08, 854.09 examples/s]7229 examples [00:08, 859.86 examples/s]7316 examples [00:08, 840.87 examples/s]7401 examples [00:08, 837.94 examples/s]7486 examples [00:08, 840.58 examples/s]7571 examples [00:09, 825.14 examples/s]7657 examples [00:09, 834.78 examples/s]7741 examples [00:09, 834.06 examples/s]7825 examples [00:09, 832.91 examples/s]7909 examples [00:09, 823.50 examples/s]7993 examples [00:09, 827.95 examples/s]8081 examples [00:09, 841.57 examples/s]8169 examples [00:09, 850.68 examples/s]8256 examples [00:09, 854.18 examples/s]8345 examples [00:09, 861.88 examples/s]8432 examples [00:10, 853.49 examples/s]8522 examples [00:10, 865.13 examples/s]8610 examples [00:10, 869.51 examples/s]8698 examples [00:10, 864.08 examples/s]8785 examples [00:10, 849.92 examples/s]8871 examples [00:10, 836.70 examples/s]8960 examples [00:10, 851.99 examples/s]9047 examples [00:10, 856.25 examples/s]9135 examples [00:10, 862.55 examples/s]9223 examples [00:10, 866.75 examples/s]9311 examples [00:11, 870.55 examples/s]9399 examples [00:11, 846.30 examples/s]9486 examples [00:11, 851.75 examples/s]9572 examples [00:11, 854.17 examples/s]9662 examples [00:11, 865.05 examples/s]9749 examples [00:11, 863.14 examples/s]9836 examples [00:11, 863.37 examples/s]9923 examples [00:11, 862.64 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 89%|████████▊ | 8860/10000 [00:00<00:00, 88592.06 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWQT98H/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteWQT98H/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1765d53620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1765d53620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1765d53620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16ef1c45c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f16ef201978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1765d53620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1765d53620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1765d53620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f174cd865c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f174cd86358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f16e38b01e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f16e38b01e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f16e38b01e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f16e38b02f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f16e38b02f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f16e38b02f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=58959b1134c1e52c64ed7cc0008241b5f885a8698d1c2d6ed7f3ade512a63242
  Stored in directory: /tmp/pip-ephem-wheel-cache-4qiflq4u/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:23:03, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:06:16, 18.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:13:38, 25.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:28:05, 37.0kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:31:00, 52.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.29M/862M [00:01<3:08:33, 75.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:01<2:11:11, 108kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:31:18, 154kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<1:03:32, 219kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<44:26, 312kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<31:00, 445kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:01<21:42, 632kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.4M/862M [00:02<15:10, 898kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:02<10:43, 1.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<07:48, 1.73MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<07:21, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<05:08, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:04<03:44, 3.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<31:11, 429kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<23:25, 570kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<16:43, 798kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<14:24, 923kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<11:24, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:08<08:18, 1.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<08:57, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<08:57, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<06:56, 1.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<06:58, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:15, 2.10MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:12<04:42, 2.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<05:49, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:14<04:24, 2.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<06:56, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<05:32, 2.35MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<04:01, 3.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<52:30, 247kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<38:04, 340kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:18<26:56, 480kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<21:48, 591kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<17:53, 720kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<13:10, 977kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<11:16, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<09:11, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<06:45, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<07:42, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<08:06, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<06:20, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:33, 2.78MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<49:01, 259kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<35:38, 356kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<25:14, 501kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<20:33, 613kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<17:04, 739kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:36, 999kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<08:57, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:35:32, 131kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:08:08, 184kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<47:52, 261kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<36:20, 343kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<28:29, 438kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<20:42, 602kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<14:36, 850kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<16:35, 748kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<13:09, 942kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:34, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<09:11, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:56, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:52, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:36, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:58, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:28, 2.73MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:57, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:39, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:16, 2.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:26, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:19, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:05, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:11, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<03:57, 3.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:12, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:06, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:53, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:09, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:04, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:54, 3.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:08, 2.31MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:33, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:12, 2.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:51, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:52, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:33, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:12, 2.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:46, 1.73MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:58, 2.93MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<10:12, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:33, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:20, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:46, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:43, 1.50MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:01, 1.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:22, 2.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:15, 1.58MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:17, 1.83MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:45, 2.41MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:38, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:07, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:51, 2.96MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<02:53, 3.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:16, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:52, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:51, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:22, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<07:13, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:45, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:10, 2.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<09:22, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<07:55, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:53, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:22, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:11, 1.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:06, 2.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:16, 3.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:05, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:59, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:46, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:36, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:32, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:23, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:41, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:23, 2.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:12, 3.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:15, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<08:27, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:18, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<04:32, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:10, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<09:00, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<06:46, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:32, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:31, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:58, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<03:37, 2.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:28, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:05, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:26, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:35, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:46, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:25, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:15, 3.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:02, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:46, 1.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:07, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:46, 2.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:25, 1.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:19, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:30, 2.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:40, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:15, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:28, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:57, 2.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:12, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:15, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:13, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:05, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<04:44, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:02, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:18, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:09, 2.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:37, 2.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:00, 2.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:57, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:27, 2.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:52, 2.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:48, 2.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<02:45, 3.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<23:00, 434kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<17:51, 559kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:51, 775kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<09:05, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<15:07, 656kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<12:23, 801kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:05, 1.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:00, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:19, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:32, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:32, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:39, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:18, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:08, 3.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:59, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:37, 1.47MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:02, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:09, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:22, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:10, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:32, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:51, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:49, 2.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:16, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:22, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:24, 2.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:07, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:32, 2.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:34, 2.63MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:05, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:33, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:36, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:05, 2.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:30, 2.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:33, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:03, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:33, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:02, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:25, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:29, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:59, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:23, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:25, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:28, 3.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<21:31, 419kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<16:40, 541kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<12:03, 747kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<09:54, 903kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<08:30, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<06:20, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:55, 1.50MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:46, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:22, 2.02MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:33, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:46, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:43, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:04, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:27, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:26, 2.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<02:32, 3.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:16, 1.65MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:13, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:01, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:16, 2.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:29, 1.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<03:27, 2.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:33, 3.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:22, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:15, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:03, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:15, 1.99MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:28, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:26, 2.45MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:30, 3.36MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<09:11, 914kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<07:57, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:56, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:32, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:47, 1.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:30, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:00, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:58, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:58, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:47, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<02:44, 3.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<08:13, 996kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:11, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:20, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<03:48, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<27:59, 290kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<21:02, 386kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<15:03, 539kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<10:34, 763kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<18:36, 433kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<14:25, 558kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<10:26, 771kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:37, 927kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:28, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:34, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:13, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:02, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<03:52, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<02:46, 2.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<24:56, 315kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<18:49, 417kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<13:26, 583kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<09:26, 825kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<17:52, 436kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<13:41, 569kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:54, 784kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<07:00, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:20, 746kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<08:35, 899kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:17, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<04:28, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<10:34, 724kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<08:46, 871kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:28, 1.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:47, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:22, 1.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:02, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:53, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<10:07, 742kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<08:26, 889kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<06:11, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<05:35, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:12, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:55, 1.90MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<02:52, 2.57MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:25, 1.67MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:23, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:20, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:28, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:08, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:10, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:14, 2.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:29, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:42, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:54, 2.48MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:14, 2.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:34, 2.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:45, 2.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:00, 3.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:42, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:14, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:58, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:57, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:02, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:08, 2.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:21, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<03:34, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:48, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:07, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:24, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:37, 2.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<01:55, 3.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:56, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:20, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<04:02, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:57, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:56, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:00, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<02:10, 3.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<09:18, 719kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<07:43, 866kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<05:38, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:17<03:59, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<11:08, 594kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:57, 739kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:30, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<04:38, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:00, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:23, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:00, 1.63MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<02:52, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:16, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:31, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<04:08, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:58, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:53, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:00, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:10, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:19, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:33, 2.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<01:52, 3.36MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:45, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:25, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:19, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:24, 2.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:53, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:32, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:27, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:26, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:29, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:42, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:54, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:06, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:23, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<01:46, 3.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:28, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:29, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:42, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:53, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:03, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:21, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<01:42, 3.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<05:39, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:01, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:46, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:36, 1.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:32, 1.64MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:43, 2.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:59, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:15, 1.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:29, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:33, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:15, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:17, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:32, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:42, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:54, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:16, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:43, 2.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:08, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:25, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:38, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:02, 2.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:30, 3.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:02, 1.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:45, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:52, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:53, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:57, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:15, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<01:38, 3.20MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:43, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:39, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:32, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<02:36, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:58, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:44, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:04, 2.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:27, 2.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:34, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:01, 2.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:16, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:28, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:56, 2.59MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:12, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:10, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:39, 2.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:12, 4.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<20:22, 240kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<15:55, 308kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<11:32, 424kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<08:07, 598kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<06:57, 693kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:42, 845kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:09, 1.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:57, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:33, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<04:53, 972kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:50, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:11, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:02, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:19, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:25, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:13, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:36, 1.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:55, 2.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:33, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:34, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:59, 2.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:08, 2.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:16, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:46, 2.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:59, 2.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:24, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:01, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:31, 2.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:56, 2.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:07, 2.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:40, 2.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:53, 2.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:03, 2.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:37, 2.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:50, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:54, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:33, 2.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:09, 3.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:57, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:57, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:46, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:18, 3.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:59, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:06, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:21, 2.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:54, 2.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:01, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:34, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:46, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:55, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:30, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:42, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:52, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:28, 2.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:40, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:50, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<01:26, 2.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:38, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:33, 2.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:11, 3.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<00:53, 4.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:24, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:52, 1.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:19, 1.56MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:41, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:09, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:07, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:36, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:09, 3.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:45, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:32, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:55, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:56, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:30, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:04, 3.15MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<20:32, 165kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<14:56, 226kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<10:32, 319kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<07:24, 451kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<06:04, 546kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:48, 689kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:28, 950kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<02:26, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<05:24, 601kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<04:19, 749kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:09, 1.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<11:18, 281kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<08:27, 375kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<06:01, 524kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<04:11, 742kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<06:10, 503kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:52, 637kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:32, 874kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:32, 1.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:48, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<16:16, 187kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<11:53, 255kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<08:23, 360kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<05:54, 508kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<04:55, 603kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:56, 752kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:51, 1.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:02, 1.44MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:24, 1.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:10, 1.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:38, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:09, 2.44MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<03:19, 853kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:48, 1.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:03, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:29, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:49, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:44, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:19, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:57, 2.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<02:36, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:16, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:42, 1.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:37, 1.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:34, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:11, 2.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:52, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:45, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:40, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:16, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<12:45, 195kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<09:21, 266kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<06:36, 373kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<04:57, 488kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<03:52, 624kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:47, 859kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:19, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<02:01, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:30, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:25, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:22, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:03, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:44, 2.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<22:08, 100kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<15:51, 139kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<11:07, 197kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<07:41, 281kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<06:28, 332kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<04:52, 439kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:29, 610kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:44, 758kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:15, 915kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:39, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:28, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:22, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:02, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:02, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:24, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:07, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:49, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:36, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:52, 651kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:19, 804kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:41, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:27, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:19, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:59, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:42, 2.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:14, 1.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:09, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:52, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:53, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:40, 2.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:29, 3.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:16, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:10, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:52, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:51, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:51, 1.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:39, 2.28MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:41, 2.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:41, 2.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:30, 2.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:22, 3.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:05, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:06, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:51, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:35, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:14, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:11, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:53, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:38, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:52, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:40, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:28, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:48, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:50, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:38, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:27, 2.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:08, 966kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:02, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:46, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:32, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:49, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:48, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:36, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:26, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:32, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:19, 2.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:35, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:37, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:20, 2.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:34, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:35, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:27, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:18, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:56, 811kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:49, 927kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:35, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:25, 1.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:28, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:29, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:15, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:46, 811kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:40, 929kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:29, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:19, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:44, 747kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:38, 868kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:27, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:18, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:49, 590kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:41, 698kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:29, 949kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:20, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:19, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:18, 1.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:08, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:11, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:03, 3.29MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:17, 710kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:16, 762kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:12, 997kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:07, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:07, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:25<00:01, 2.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:08, 482kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:07, 569kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 764kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 1.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.04MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 511/400000 [00:00<01:18, 5107.09it/s]  0%|          | 1206/400000 [00:00<01:11, 5547.88it/s]  0%|          | 1886/400000 [00:00<01:07, 5871.41it/s]  1%|          | 2593/400000 [00:00<01:04, 6185.59it/s]  1%|          | 3256/400000 [00:00<01:02, 6310.93it/s]  1%|          | 3945/400000 [00:00<01:01, 6472.05it/s]  1%|          | 4655/400000 [00:00<00:59, 6648.34it/s]  1%|▏         | 5316/400000 [00:00<00:59, 6636.37it/s]  1%|▏         | 5955/400000 [00:00<01:00, 6558.33it/s]  2%|▏         | 6619/400000 [00:01<00:59, 6582.34it/s]  2%|▏         | 7319/400000 [00:01<00:58, 6701.03it/s]  2%|▏         | 8032/400000 [00:01<00:57, 6823.34it/s]  2%|▏         | 8710/400000 [00:01<00:57, 6782.47it/s]  2%|▏         | 9419/400000 [00:01<00:56, 6870.43it/s]  3%|▎         | 10139/400000 [00:01<00:55, 6965.98it/s]  3%|▎         | 10867/400000 [00:01<00:55, 7057.09it/s]  3%|▎         | 11577/400000 [00:01<00:54, 7069.19it/s]  3%|▎         | 12285/400000 [00:01<00:54, 7071.02it/s]  3%|▎         | 13005/400000 [00:01<00:54, 7108.52it/s]  3%|▎         | 13716/400000 [00:02<00:54, 7099.13it/s]  4%|▎         | 14444/400000 [00:02<00:53, 7150.76it/s]  4%|▍         | 15174/400000 [00:02<00:53, 7192.35it/s]  4%|▍         | 15894/400000 [00:02<00:54, 7099.85it/s]  4%|▍         | 16605/400000 [00:02<00:54, 7096.06it/s]  4%|▍         | 17315/400000 [00:02<00:55, 6894.36it/s]  5%|▍         | 18035/400000 [00:02<00:54, 6982.85it/s]  5%|▍         | 18735/400000 [00:02<00:56, 6778.33it/s]  5%|▍         | 19416/400000 [00:02<00:58, 6460.91it/s]  5%|▌         | 20067/400000 [00:02<00:59, 6421.44it/s]  5%|▌         | 20792/400000 [00:03<00:57, 6648.75it/s]  5%|▌         | 21501/400000 [00:03<00:55, 6772.40it/s]  6%|▌         | 22217/400000 [00:03<00:54, 6882.69it/s]  6%|▌         | 22926/400000 [00:03<00:54, 6942.30it/s]  6%|▌         | 23634/400000 [00:03<00:53, 6982.53it/s]  6%|▌         | 24353/400000 [00:03<00:53, 7041.36it/s]  6%|▋         | 25079/400000 [00:03<00:52, 7104.61it/s]  6%|▋         | 25791/400000 [00:03<00:53, 7038.19it/s]  7%|▋         | 26496/400000 [00:03<00:53, 6980.28it/s]  7%|▋         | 27195/400000 [00:03<00:53, 6934.76it/s]  7%|▋         | 27894/400000 [00:04<00:53, 6950.57it/s]  7%|▋         | 28590/400000 [00:04<00:53, 6897.97it/s]  7%|▋         | 29296/400000 [00:04<00:53, 6944.96it/s]  7%|▋         | 29991/400000 [00:04<00:53, 6932.91it/s]  8%|▊         | 30695/400000 [00:04<00:53, 6962.62it/s]  8%|▊         | 31392/400000 [00:04<00:53, 6836.42it/s]  8%|▊         | 32102/400000 [00:04<00:53, 6912.43it/s]  8%|▊         | 32794/400000 [00:04<00:53, 6873.25it/s]  8%|▊         | 33483/400000 [00:04<00:53, 6877.53it/s]  9%|▊         | 34177/400000 [00:04<00:53, 6893.53it/s]  9%|▊         | 34867/400000 [00:05<00:53, 6810.74it/s]  9%|▉         | 35575/400000 [00:05<00:52, 6888.69it/s]  9%|▉         | 36277/400000 [00:05<00:52, 6925.59it/s]  9%|▉         | 36970/400000 [00:05<00:52, 6919.67it/s]  9%|▉         | 37691/400000 [00:05<00:51, 7001.44it/s] 10%|▉         | 38392/400000 [00:05<00:51, 6958.42it/s] 10%|▉         | 39091/400000 [00:05<00:51, 6967.26it/s] 10%|▉         | 39808/400000 [00:05<00:51, 7025.43it/s] 10%|█         | 40511/400000 [00:05<00:51, 6996.38it/s] 10%|█         | 41250/400000 [00:05<00:50, 7109.40it/s] 10%|█         | 41962/400000 [00:06<00:50, 7069.42it/s] 11%|█         | 42670/400000 [00:06<00:50, 7033.77it/s] 11%|█         | 43374/400000 [00:06<00:51, 6992.45it/s] 11%|█         | 44074/400000 [00:06<00:51, 6955.35it/s] 11%|█         | 44770/400000 [00:06<00:51, 6941.78it/s] 11%|█▏        | 45478/400000 [00:06<00:50, 6980.63it/s] 12%|█▏        | 46185/400000 [00:06<00:50, 7007.08it/s] 12%|█▏        | 46894/400000 [00:06<00:50, 7029.10it/s] 12%|█▏        | 47598/400000 [00:06<00:50, 6949.26it/s] 12%|█▏        | 48294/400000 [00:06<00:51, 6787.87it/s] 12%|█▏        | 48995/400000 [00:07<00:51, 6851.17it/s] 12%|█▏        | 49700/400000 [00:07<00:50, 6906.92it/s] 13%|█▎        | 50419/400000 [00:07<00:50, 6988.41it/s] 13%|█▎        | 51122/400000 [00:07<00:49, 6998.69it/s] 13%|█▎        | 51823/400000 [00:07<00:51, 6825.93it/s] 13%|█▎        | 52529/400000 [00:07<00:50, 6893.32it/s] 13%|█▎        | 53235/400000 [00:07<00:49, 6940.90it/s] 13%|█▎        | 53943/400000 [00:07<00:49, 6982.02it/s] 14%|█▎        | 54642/400000 [00:07<00:49, 6945.13it/s] 14%|█▍        | 55362/400000 [00:08<00:49, 7019.16it/s] 14%|█▍        | 56069/400000 [00:08<00:48, 7033.64it/s] 14%|█▍        | 56783/400000 [00:08<00:48, 7064.92it/s] 14%|█▍        | 57503/400000 [00:08<00:48, 7103.97it/s] 15%|█▍        | 58214/400000 [00:08<00:48, 7102.43it/s] 15%|█▍        | 58934/400000 [00:08<00:47, 7129.69it/s] 15%|█▍        | 59663/400000 [00:08<00:47, 7176.74it/s] 15%|█▌        | 60381/400000 [00:08<00:47, 7129.26it/s] 15%|█▌        | 61095/400000 [00:08<00:47, 7131.10it/s] 15%|█▌        | 61815/400000 [00:08<00:47, 7149.63it/s] 16%|█▌        | 62531/400000 [00:09<00:47, 7149.72it/s] 16%|█▌        | 63247/400000 [00:09<00:47, 7116.47it/s] 16%|█▌        | 63959/400000 [00:09<00:47, 7081.61it/s] 16%|█▌        | 64668/400000 [00:09<00:47, 6994.79it/s] 16%|█▋        | 65368/400000 [00:09<00:48, 6925.88it/s] 17%|█▋        | 66065/400000 [00:09<00:48, 6938.92it/s] 17%|█▋        | 66783/400000 [00:09<00:47, 7009.34it/s] 17%|█▋        | 67504/400000 [00:09<00:47, 7066.01it/s] 17%|█▋        | 68232/400000 [00:09<00:46, 7126.51it/s] 17%|█▋        | 68946/400000 [00:09<00:47, 7005.72it/s] 17%|█▋        | 69648/400000 [00:10<00:47, 6961.45it/s] 18%|█▊        | 70345/400000 [00:10<00:47, 6936.78it/s] 18%|█▊        | 71040/400000 [00:10<00:47, 6931.38it/s] 18%|█▊        | 71741/400000 [00:10<00:47, 6951.90it/s] 18%|█▊        | 72449/400000 [00:10<00:46, 6988.17it/s] 18%|█▊        | 73149/400000 [00:10<00:46, 6981.04it/s] 18%|█▊        | 73857/400000 [00:10<00:46, 6988.75it/s] 19%|█▊        | 74556/400000 [00:10<00:47, 6863.95it/s] 19%|█▉        | 75243/400000 [00:10<00:49, 6570.03it/s] 19%|█▉        | 75923/400000 [00:10<00:48, 6635.19it/s] 19%|█▉        | 76590/400000 [00:11<00:48, 6645.10it/s] 19%|█▉        | 77306/400000 [00:11<00:47, 6789.43it/s] 20%|█▉        | 78010/400000 [00:11<00:46, 6861.30it/s] 20%|█▉        | 78702/400000 [00:11<00:47, 6773.57it/s] 20%|█▉        | 79381/400000 [00:11<00:47, 6721.67it/s] 20%|██        | 80087/400000 [00:11<00:46, 6818.08it/s] 20%|██        | 80813/400000 [00:11<00:45, 6943.76it/s] 20%|██        | 81535/400000 [00:11<00:45, 7021.75it/s] 21%|██        | 82239/400000 [00:11<00:45, 6974.09it/s] 21%|██        | 82938/400000 [00:11<00:45, 6975.01it/s] 21%|██        | 83637/400000 [00:12<00:46, 6860.74it/s] 21%|██        | 84338/400000 [00:12<00:45, 6902.66it/s] 21%|██▏       | 85029/400000 [00:12<00:45, 6856.82it/s] 21%|██▏       | 85747/400000 [00:12<00:45, 6949.22it/s] 22%|██▏       | 86443/400000 [00:12<00:46, 6673.09it/s] 22%|██▏       | 87114/400000 [00:12<00:47, 6621.43it/s] 22%|██▏       | 87808/400000 [00:12<00:46, 6713.23it/s] 22%|██▏       | 88512/400000 [00:12<00:45, 6806.83it/s] 22%|██▏       | 89195/400000 [00:12<00:45, 6777.96it/s] 22%|██▏       | 89874/400000 [00:13<00:47, 6524.65it/s] 23%|██▎       | 90581/400000 [00:13<00:46, 6679.21it/s] 23%|██▎       | 91318/400000 [00:13<00:44, 6871.48it/s] 23%|██▎       | 92060/400000 [00:13<00:43, 7026.01it/s] 23%|██▎       | 92802/400000 [00:13<00:43, 7137.26it/s] 23%|██▎       | 93539/400000 [00:13<00:42, 7202.58it/s] 24%|██▎       | 94271/400000 [00:13<00:42, 7234.80it/s] 24%|██▎       | 94999/400000 [00:13<00:42, 7161.03it/s] 24%|██▍       | 95717/400000 [00:13<00:42, 7127.81it/s] 24%|██▍       | 96431/400000 [00:13<00:43, 7052.35it/s] 24%|██▍       | 97138/400000 [00:14<00:43, 7026.49it/s] 24%|██▍       | 97851/400000 [00:14<00:42, 7055.14it/s] 25%|██▍       | 98558/400000 [00:14<00:42, 7057.39it/s] 25%|██▍       | 99265/400000 [00:14<00:42, 7019.41it/s] 25%|██▍       | 99968/400000 [00:14<00:43, 6912.12it/s] 25%|██▌       | 100714/400000 [00:14<00:42, 7065.36it/s] 25%|██▌       | 101442/400000 [00:14<00:41, 7127.43it/s] 26%|██▌       | 102156/400000 [00:14<00:41, 7110.70it/s] 26%|██▌       | 102868/400000 [00:14<00:42, 6939.28it/s] 26%|██▌       | 103564/400000 [00:14<00:43, 6790.81it/s] 26%|██▌       | 104250/400000 [00:15<00:43, 6811.09it/s] 26%|██▌       | 104950/400000 [00:15<00:42, 6865.96it/s] 26%|██▋       | 105661/400000 [00:15<00:42, 6934.49it/s] 27%|██▋       | 106356/400000 [00:15<00:43, 6790.85it/s] 27%|██▋       | 107046/400000 [00:15<00:42, 6821.10it/s] 27%|██▋       | 107737/400000 [00:15<00:42, 6846.03it/s] 27%|██▋       | 108440/400000 [00:15<00:42, 6899.29it/s] 27%|██▋       | 109145/400000 [00:15<00:41, 6943.13it/s] 27%|██▋       | 109884/400000 [00:15<00:41, 7070.23it/s] 28%|██▊       | 110625/400000 [00:15<00:40, 7167.18it/s] 28%|██▊       | 111343/400000 [00:16<00:40, 7137.22it/s] 28%|██▊       | 112059/400000 [00:16<00:40, 7143.01it/s] 28%|██▊       | 112774/400000 [00:16<00:40, 7106.22it/s] 28%|██▊       | 113486/400000 [00:16<00:40, 7090.51it/s] 29%|██▊       | 114196/400000 [00:16<00:40, 7063.68it/s] 29%|██▊       | 114908/400000 [00:16<00:40, 7079.00it/s] 29%|██▉       | 115617/400000 [00:16<00:40, 7054.42it/s] 29%|██▉       | 116323/400000 [00:16<00:40, 6972.36it/s] 29%|██▉       | 117034/400000 [00:16<00:40, 7012.85it/s] 29%|██▉       | 117756/400000 [00:16<00:39, 7071.22it/s] 30%|██▉       | 118464/400000 [00:17<00:39, 7067.28it/s] 30%|██▉       | 119171/400000 [00:17<00:39, 7061.73it/s] 30%|██▉       | 119896/400000 [00:17<00:39, 7115.48it/s] 30%|███       | 120621/400000 [00:17<00:39, 7153.10it/s] 30%|███       | 121337/400000 [00:17<00:39, 7058.94it/s] 31%|███       | 122044/400000 [00:17<00:39, 7045.69it/s] 31%|███       | 122755/400000 [00:17<00:39, 7062.40it/s] 31%|███       | 123478/400000 [00:17<00:38, 7111.32it/s] 31%|███       | 124210/400000 [00:17<00:38, 7171.32it/s] 31%|███       | 124954/400000 [00:17<00:37, 7249.48it/s] 31%|███▏      | 125680/400000 [00:18<00:38, 7119.64it/s] 32%|███▏      | 126393/400000 [00:18<00:38, 7115.07it/s] 32%|███▏      | 127106/400000 [00:18<00:38, 7113.53it/s] 32%|███▏      | 127818/400000 [00:18<00:38, 7063.73it/s] 32%|███▏      | 128527/400000 [00:18<00:38, 7070.76it/s] 32%|███▏      | 129235/400000 [00:18<00:38, 6975.27it/s] 32%|███▏      | 129933/400000 [00:18<00:38, 6966.36it/s] 33%|███▎      | 130678/400000 [00:18<00:37, 7103.91it/s] 33%|███▎      | 131390/400000 [00:18<00:37, 7106.96it/s] 33%|███▎      | 132102/400000 [00:18<00:37, 7103.96it/s] 33%|███▎      | 132813/400000 [00:19<00:37, 7037.00it/s] 33%|███▎      | 133523/400000 [00:19<00:37, 7053.99it/s] 34%|███▎      | 134229/400000 [00:19<00:37, 7004.75it/s] 34%|███▎      | 134954/400000 [00:19<00:37, 7075.10it/s] 34%|███▍      | 135677/400000 [00:19<00:37, 7119.96it/s] 34%|███▍      | 136390/400000 [00:19<00:37, 7088.50it/s] 34%|███▍      | 137108/400000 [00:19<00:36, 7115.10it/s] 34%|███▍      | 137831/400000 [00:19<00:36, 7148.20it/s] 35%|███▍      | 138547/400000 [00:19<00:37, 6969.92it/s] 35%|███▍      | 139262/400000 [00:20<00:37, 7022.46it/s] 35%|███▍      | 139966/400000 [00:20<00:37, 7013.61it/s] 35%|███▌      | 140668/400000 [00:20<00:37, 6981.22it/s] 35%|███▌      | 141375/400000 [00:20<00:36, 7006.12it/s] 36%|███▌      | 142094/400000 [00:20<00:36, 7058.81it/s] 36%|███▌      | 142801/400000 [00:20<00:37, 6771.65it/s] 36%|███▌      | 143481/400000 [00:20<00:38, 6684.05it/s] 36%|███▌      | 144187/400000 [00:20<00:37, 6792.52it/s] 36%|███▌      | 144882/400000 [00:20<00:37, 6837.14it/s] 36%|███▋      | 145605/400000 [00:20<00:36, 6948.01it/s] 37%|███▋      | 146302/400000 [00:21<00:36, 6924.81it/s] 37%|███▋      | 146996/400000 [00:21<00:36, 6885.62it/s] 37%|███▋      | 147716/400000 [00:21<00:36, 6976.57it/s] 37%|███▋      | 148426/400000 [00:21<00:35, 7010.38it/s] 37%|███▋      | 149131/400000 [00:21<00:35, 7022.05it/s] 37%|███▋      | 149842/400000 [00:21<00:35, 7046.95it/s] 38%|███▊      | 150548/400000 [00:21<00:35, 7026.47it/s] 38%|███▊      | 151251/400000 [00:21<00:36, 6907.33it/s] 38%|███▊      | 151968/400000 [00:21<00:35, 6982.80it/s] 38%|███▊      | 152667/400000 [00:21<00:36, 6697.68it/s] 38%|███▊      | 153340/400000 [00:22<00:38, 6464.39it/s] 39%|███▊      | 154040/400000 [00:22<00:37, 6615.68it/s] 39%|███▊      | 154706/400000 [00:22<00:38, 6375.01it/s] 39%|███▉      | 155365/400000 [00:22<00:38, 6437.22it/s] 39%|███▉      | 156088/400000 [00:22<00:36, 6654.04it/s] 39%|███▉      | 156798/400000 [00:22<00:35, 6780.24it/s] 39%|███▉      | 157499/400000 [00:22<00:35, 6845.71it/s] 40%|███▉      | 158212/400000 [00:22<00:34, 6927.82it/s] 40%|███▉      | 158918/400000 [00:22<00:34, 6966.22it/s] 40%|███▉      | 159626/400000 [00:22<00:34, 6998.94it/s] 40%|████      | 160327/400000 [00:23<00:34, 6955.45it/s] 40%|████      | 161048/400000 [00:23<00:34, 7027.99it/s] 40%|████      | 161769/400000 [00:23<00:33, 7078.48it/s] 41%|████      | 162501/400000 [00:23<00:33, 7149.21it/s] 41%|████      | 163225/400000 [00:23<00:33, 7173.74it/s] 41%|████      | 163943/400000 [00:23<00:32, 7155.67it/s] 41%|████      | 164664/400000 [00:23<00:32, 7169.55it/s] 41%|████▏     | 165382/400000 [00:23<00:33, 7106.76it/s] 42%|████▏     | 166093/400000 [00:23<00:33, 7040.71it/s] 42%|████▏     | 166805/400000 [00:23<00:33, 7063.92it/s] 42%|████▏     | 167524/400000 [00:24<00:32, 7099.84it/s] 42%|████▏     | 168235/400000 [00:24<00:32, 7037.09it/s] 42%|████▏     | 168971/400000 [00:24<00:32, 7129.47it/s] 42%|████▏     | 169713/400000 [00:24<00:31, 7211.51it/s] 43%|████▎     | 170444/400000 [00:24<00:31, 7237.80it/s] 43%|████▎     | 171171/400000 [00:24<00:31, 7246.74it/s] 43%|████▎     | 171896/400000 [00:24<00:31, 7231.33it/s] 43%|████▎     | 172620/400000 [00:24<00:33, 6880.99it/s] 43%|████▎     | 173368/400000 [00:24<00:32, 7048.17it/s] 44%|████▎     | 174100/400000 [00:25<00:31, 7126.84it/s] 44%|████▎     | 174822/400000 [00:25<00:31, 7150.71it/s] 44%|████▍     | 175540/400000 [00:25<00:31, 7113.12it/s] 44%|████▍     | 176253/400000 [00:25<00:32, 6984.99it/s] 44%|████▍     | 176953/400000 [00:25<00:31, 6977.01it/s] 44%|████▍     | 177679/400000 [00:25<00:31, 7057.62it/s] 45%|████▍     | 178399/400000 [00:25<00:31, 7098.13it/s] 45%|████▍     | 179110/400000 [00:25<00:31, 7092.49it/s] 45%|████▍     | 179820/400000 [00:25<00:31, 7036.88it/s] 45%|████▌     | 180525/400000 [00:25<00:31, 6967.78it/s] 45%|████▌     | 181223/400000 [00:26<00:31, 6895.26it/s] 45%|████▌     | 181914/400000 [00:26<00:31, 6863.94it/s] 46%|████▌     | 182619/400000 [00:26<00:31, 6916.38it/s] 46%|████▌     | 183342/400000 [00:26<00:30, 7005.22it/s] 46%|████▌     | 184059/400000 [00:26<00:30, 7051.95it/s] 46%|████▌     | 184770/400000 [00:26<00:30, 7067.52it/s] 46%|████▋     | 185478/400000 [00:26<00:30, 7016.66it/s] 47%|████▋     | 186180/400000 [00:26<00:30, 6996.18it/s] 47%|████▋     | 186880/400000 [00:26<00:30, 6996.68it/s] 47%|████▋     | 187606/400000 [00:26<00:30, 7070.65it/s] 47%|████▋     | 188314/400000 [00:27<00:29, 7069.75it/s] 47%|████▋     | 189022/400000 [00:27<00:29, 7072.40it/s] 47%|████▋     | 189730/400000 [00:27<00:30, 6927.75it/s] 48%|████▊     | 190433/400000 [00:27<00:30, 6955.52it/s] 48%|████▊     | 191130/400000 [00:27<00:30, 6953.68it/s] 48%|████▊     | 191851/400000 [00:27<00:29, 7026.94it/s] 48%|████▊     | 192556/400000 [00:27<00:29, 7031.45it/s] 48%|████▊     | 193260/400000 [00:27<00:30, 6832.89it/s] 48%|████▊     | 193945/400000 [00:27<00:30, 6820.57it/s] 49%|████▊     | 194651/400000 [00:27<00:29, 6890.26it/s] 49%|████▉     | 195368/400000 [00:28<00:29, 6970.03it/s] 49%|████▉     | 196089/400000 [00:28<00:28, 7039.54it/s] 49%|████▉     | 196802/400000 [00:28<00:28, 7064.49it/s] 49%|████▉     | 197514/400000 [00:28<00:28, 7079.30it/s] 50%|████▉     | 198223/400000 [00:28<00:28, 7042.07it/s] 50%|████▉     | 198928/400000 [00:28<00:28, 7019.78it/s] 50%|████▉     | 199631/400000 [00:28<00:29, 6822.99it/s] 50%|█████     | 200332/400000 [00:28<00:29, 6877.74it/s] 50%|█████     | 201080/400000 [00:28<00:28, 7045.33it/s] 50%|█████     | 201787/400000 [00:28<00:28, 7016.65it/s] 51%|█████     | 202490/400000 [00:29<00:28, 6870.66it/s] 51%|█████     | 203206/400000 [00:29<00:28, 6954.82it/s] 51%|█████     | 203909/400000 [00:29<00:28, 6974.85it/s] 51%|█████     | 204625/400000 [00:29<00:27, 7027.76it/s] 51%|█████▏    | 205329/400000 [00:29<00:27, 7030.59it/s] 52%|█████▏    | 206067/400000 [00:29<00:27, 7131.41it/s] 52%|█████▏    | 206781/400000 [00:29<00:27, 7084.37it/s] 52%|█████▏    | 207510/400000 [00:29<00:26, 7144.06it/s] 52%|█████▏    | 208225/400000 [00:29<00:27, 6945.27it/s] 52%|█████▏    | 208936/400000 [00:29<00:27, 6992.35it/s] 52%|█████▏    | 209637/400000 [00:30<00:27, 6995.50it/s] 53%|█████▎    | 210351/400000 [00:30<00:26, 7036.76it/s] 53%|█████▎    | 211056/400000 [00:30<00:26, 7008.24it/s] 53%|█████▎    | 211787/400000 [00:30<00:26, 7094.88it/s] 53%|█████▎    | 212499/400000 [00:30<00:26, 7100.83it/s] 53%|█████▎    | 213230/400000 [00:30<00:26, 7161.16it/s] 53%|█████▎    | 213947/400000 [00:30<00:27, 6870.27it/s] 54%|█████▎    | 214637/400000 [00:30<00:27, 6728.20it/s] 54%|█████▍    | 215313/400000 [00:30<00:28, 6569.40it/s] 54%|█████▍    | 216021/400000 [00:31<00:27, 6713.17it/s] 54%|█████▍    | 216723/400000 [00:31<00:26, 6799.80it/s] 54%|█████▍    | 217443/400000 [00:31<00:26, 6913.66it/s] 55%|█████▍    | 218146/400000 [00:31<00:26, 6947.30it/s] 55%|█████▍    | 218843/400000 [00:31<00:26, 6904.08it/s] 55%|█████▍    | 219571/400000 [00:31<00:25, 7012.27it/s] 55%|█████▌    | 220291/400000 [00:31<00:25, 7066.95it/s] 55%|█████▌    | 221017/400000 [00:31<00:25, 7121.37it/s] 55%|█████▌    | 221743/400000 [00:31<00:24, 7160.79it/s] 56%|█████▌    | 222475/400000 [00:31<00:24, 7205.69it/s] 56%|█████▌    | 223208/400000 [00:32<00:24, 7240.41it/s] 56%|█████▌    | 223933/400000 [00:32<00:24, 7178.64it/s] 56%|█████▌    | 224652/400000 [00:32<00:24, 7180.31it/s] 56%|█████▋    | 225371/400000 [00:32<00:25, 6978.83it/s] 57%|█████▋    | 226071/400000 [00:32<00:25, 6922.64it/s] 57%|█████▋    | 226787/400000 [00:32<00:24, 6991.27it/s] 57%|█████▋    | 227503/400000 [00:32<00:24, 7039.04it/s] 57%|█████▋    | 228208/400000 [00:32<00:24, 6992.50it/s] 57%|█████▋    | 228922/400000 [00:32<00:24, 7034.09it/s] 57%|█████▋    | 229638/400000 [00:32<00:24, 7069.70it/s] 58%|█████▊    | 230346/400000 [00:33<00:24, 6982.84it/s] 58%|█████▊    | 231072/400000 [00:33<00:23, 7062.04it/s] 58%|█████▊    | 231808/400000 [00:33<00:23, 7147.43it/s] 58%|█████▊    | 232524/400000 [00:33<00:23, 7114.42it/s] 58%|█████▊    | 233258/400000 [00:33<00:23, 7177.77it/s] 58%|█████▊    | 233990/400000 [00:33<00:22, 7218.45it/s] 59%|█████▊    | 234731/400000 [00:33<00:22, 7273.15it/s] 59%|█████▉    | 235459/400000 [00:33<00:22, 7222.53it/s] 59%|█████▉    | 236182/400000 [00:33<00:22, 7206.75it/s] 59%|█████▉    | 236903/400000 [00:33<00:22, 7176.08it/s] 59%|█████▉    | 237622/400000 [00:34<00:22, 7178.03it/s] 60%|█████▉    | 238345/400000 [00:34<00:22, 7192.28it/s] 60%|█████▉    | 239065/400000 [00:34<00:22, 7032.32it/s] 60%|█████▉    | 239773/400000 [00:34<00:22, 7044.77it/s] 60%|██████    | 240504/400000 [00:34<00:22, 7119.45it/s] 60%|██████    | 241217/400000 [00:34<00:22, 7088.25it/s] 60%|██████    | 241944/400000 [00:34<00:22, 7139.26it/s] 61%|██████    | 242697/400000 [00:34<00:21, 7250.03it/s] 61%|██████    | 243423/400000 [00:34<00:21, 7176.21it/s] 61%|██████    | 244148/400000 [00:34<00:21, 7196.34it/s] 61%|██████    | 244869/400000 [00:35<00:21, 7142.66it/s] 61%|██████▏   | 245587/400000 [00:35<00:21, 7153.77it/s] 62%|██████▏   | 246303/400000 [00:35<00:21, 7059.03it/s] 62%|██████▏   | 247028/400000 [00:35<00:21, 7113.04it/s] 62%|██████▏   | 247754/400000 [00:35<00:21, 7154.01it/s] 62%|██████▏   | 248470/400000 [00:35<00:21, 7113.59it/s] 62%|██████▏   | 249182/400000 [00:35<00:21, 7109.04it/s] 62%|██████▏   | 249894/400000 [00:35<00:22, 6810.99it/s] 63%|██████▎   | 250578/400000 [00:35<00:21, 6803.00it/s] 63%|██████▎   | 251301/400000 [00:35<00:21, 6924.91it/s] 63%|██████▎   | 252031/400000 [00:36<00:21, 7032.12it/s] 63%|██████▎   | 252754/400000 [00:36<00:20, 7089.64it/s] 63%|██████▎   | 253486/400000 [00:36<00:20, 7156.57it/s] 64%|██████▎   | 254203/400000 [00:36<00:20, 7059.48it/s] 64%|██████▎   | 254911/400000 [00:36<00:20, 7039.92it/s] 64%|██████▍   | 255625/400000 [00:36<00:20, 7066.12it/s] 64%|██████▍   | 256345/400000 [00:36<00:20, 7103.15it/s] 64%|██████▍   | 257085/400000 [00:36<00:19, 7186.91it/s] 64%|██████▍   | 257805/400000 [00:36<00:19, 7143.04it/s] 65%|██████▍   | 258520/400000 [00:37<00:20, 7020.51it/s] 65%|██████▍   | 259225/400000 [00:37<00:20, 7028.46it/s] 65%|██████▍   | 259941/400000 [00:37<00:19, 7066.54it/s] 65%|██████▌   | 260650/400000 [00:37<00:19, 7071.33it/s] 65%|██████▌   | 261358/400000 [00:37<00:19, 7065.14it/s] 66%|██████▌   | 262065/400000 [00:37<00:19, 7054.87it/s] 66%|██████▌   | 262771/400000 [00:37<00:19, 6935.98it/s] 66%|██████▌   | 263513/400000 [00:37<00:19, 7071.96it/s] 66%|██████▌   | 264235/400000 [00:37<00:19, 7115.68it/s] 66%|██████▌   | 264948/400000 [00:37<00:19, 7097.23it/s] 66%|██████▋   | 265671/400000 [00:38<00:18, 7136.19it/s] 67%|██████▋   | 266386/400000 [00:38<00:19, 6909.92it/s] 67%|██████▋   | 267079/400000 [00:38<00:19, 6737.04it/s] 67%|██████▋   | 267799/400000 [00:38<00:19, 6867.89it/s] 67%|██████▋   | 268515/400000 [00:38<00:18, 6951.05it/s] 67%|██████▋   | 269238/400000 [00:38<00:18, 7029.98it/s] 67%|██████▋   | 269962/400000 [00:38<00:18, 7089.70it/s] 68%|██████▊   | 270685/400000 [00:38<00:18, 7128.79it/s] 68%|██████▊   | 271399/400000 [00:38<00:18, 7111.17it/s] 68%|██████▊   | 272111/400000 [00:38<00:18, 7059.85it/s] 68%|██████▊   | 272818/400000 [00:39<00:18, 6872.63it/s] 68%|██████▊   | 273508/400000 [00:39<00:18, 6879.51it/s] 69%|██████▊   | 274208/400000 [00:39<00:18, 6913.18it/s] 69%|██████▊   | 274912/400000 [00:39<00:18, 6946.69it/s] 69%|██████▉   | 275608/400000 [00:39<00:18, 6732.58it/s] 69%|██████▉   | 276284/400000 [00:39<00:18, 6726.07it/s] 69%|██████▉   | 276987/400000 [00:39<00:18, 6813.56it/s] 69%|██████▉   | 277708/400000 [00:39<00:17, 6927.62it/s] 70%|██████▉   | 278442/400000 [00:39<00:17, 7045.71it/s] 70%|██████▉   | 279148/400000 [00:39<00:17, 7026.49it/s] 70%|██████▉   | 279852/400000 [00:40<00:17, 7013.63it/s] 70%|███████   | 280578/400000 [00:40<00:16, 7085.15it/s] 70%|███████   | 281288/400000 [00:40<00:17, 6970.07it/s] 70%|███████   | 281998/400000 [00:40<00:16, 7007.86it/s] 71%|███████   | 282720/400000 [00:40<00:16, 7066.72it/s] 71%|███████   | 283431/400000 [00:40<00:16, 7076.06it/s] 71%|███████   | 284153/400000 [00:40<00:16, 7113.67it/s] 71%|███████   | 284865/400000 [00:40<00:17, 6713.62it/s] 71%|███████▏  | 285557/400000 [00:40<00:16, 6773.94it/s] 72%|███████▏  | 286259/400000 [00:40<00:16, 6841.78it/s] 72%|███████▏  | 286946/400000 [00:41<00:16, 6800.69it/s] 72%|███████▏  | 287628/400000 [00:41<00:16, 6752.21it/s] 72%|███████▏  | 288305/400000 [00:41<00:16, 6738.78it/s] 72%|███████▏  | 288991/400000 [00:41<00:16, 6772.95it/s] 72%|███████▏  | 289691/400000 [00:41<00:16, 6837.66it/s] 73%|███████▎  | 290405/400000 [00:41<00:15, 6923.35it/s] 73%|███████▎  | 291103/400000 [00:41<00:15, 6940.11it/s] 73%|███████▎  | 291811/400000 [00:41<00:15, 6978.65it/s] 73%|███████▎  | 292510/400000 [00:41<00:15, 6876.37it/s] 73%|███████▎  | 293205/400000 [00:41<00:15, 6895.43it/s] 73%|███████▎  | 293900/400000 [00:42<00:15, 6911.05it/s] 74%|███████▎  | 294592/400000 [00:42<00:15, 6866.66it/s] 74%|███████▍  | 295279/400000 [00:42<00:15, 6737.34it/s] 74%|███████▍  | 296004/400000 [00:42<00:15, 6882.54it/s] 74%|███████▍  | 296694/400000 [00:42<00:15, 6884.70it/s] 74%|███████▍  | 297384/400000 [00:42<00:15, 6831.30it/s] 75%|███████▍  | 298068/400000 [00:42<00:14, 6832.13it/s] 75%|███████▍  | 298757/400000 [00:42<00:14, 6849.36it/s] 75%|███████▍  | 299443/400000 [00:42<00:14, 6836.12it/s] 75%|███████▌  | 300151/400000 [00:43<00:14, 6906.82it/s] 75%|███████▌  | 300843/400000 [00:43<00:14, 6856.61it/s] 75%|███████▌  | 301556/400000 [00:43<00:14, 6935.73it/s] 76%|███████▌  | 302260/400000 [00:43<00:14, 6964.74it/s] 76%|███████▌  | 302957/400000 [00:43<00:13, 6944.59it/s] 76%|███████▌  | 303667/400000 [00:43<00:13, 6990.22it/s] 76%|███████▌  | 304382/400000 [00:43<00:13, 7035.35it/s] 76%|███████▋  | 305086/400000 [00:43<00:13, 7011.34it/s] 76%|███████▋  | 305788/400000 [00:43<00:13, 7007.97it/s] 77%|███████▋  | 306507/400000 [00:43<00:13, 7061.02it/s] 77%|███████▋  | 307214/400000 [00:44<00:13, 7055.58it/s] 77%|███████▋  | 307920/400000 [00:44<00:13, 7018.78it/s] 77%|███████▋  | 308623/400000 [00:44<00:13, 6993.32it/s] 77%|███████▋  | 309323/400000 [00:44<00:13, 6809.43it/s] 78%|███████▊  | 310031/400000 [00:44<00:13, 6886.69it/s] 78%|███████▊  | 310731/400000 [00:44<00:12, 6919.17it/s] 78%|███████▊  | 311458/400000 [00:44<00:12, 7020.24it/s] 78%|███████▊  | 312170/400000 [00:44<00:12, 7048.08it/s] 78%|███████▊  | 312891/400000 [00:44<00:12, 7094.75it/s] 78%|███████▊  | 313601/400000 [00:44<00:12, 7025.70it/s] 79%|███████▊  | 314306/400000 [00:45<00:12, 7030.84it/s] 79%|███████▉  | 315010/400000 [00:45<00:12, 6998.74it/s] 79%|███████▉  | 315732/400000 [00:45<00:11, 7063.32it/s] 79%|███████▉  | 316450/400000 [00:45<00:11, 7095.67it/s] 79%|███████▉  | 317163/400000 [00:45<00:11, 7104.01it/s] 79%|███████▉  | 317874/400000 [00:45<00:11, 7036.30it/s] 80%|███████▉  | 318593/400000 [00:45<00:11, 7080.72it/s] 80%|███████▉  | 319322/400000 [00:45<00:11, 7139.81it/s] 80%|████████  | 320037/400000 [00:45<00:11, 7100.08it/s] 80%|████████  | 320752/400000 [00:45<00:11, 7114.59it/s] 80%|████████  | 321464/400000 [00:46<00:11, 7047.75it/s] 81%|████████  | 322170/400000 [00:46<00:11, 7030.60it/s] 81%|████████  | 322900/400000 [00:46<00:10, 7108.24it/s] 81%|████████  | 323633/400000 [00:46<00:10, 7171.81it/s] 81%|████████  | 324358/400000 [00:46<00:10, 7193.79it/s] 81%|████████▏ | 325078/400000 [00:46<00:10, 7175.45it/s] 81%|████████▏ | 325796/400000 [00:46<00:10, 7021.07it/s] 82%|████████▏ | 326499/400000 [00:46<00:10, 6899.39it/s] 82%|████████▏ | 327191/400000 [00:46<00:10, 6664.97it/s] 82%|████████▏ | 327906/400000 [00:46<00:10, 6802.52it/s] 82%|████████▏ | 328630/400000 [00:47<00:10, 6925.26it/s] 82%|████████▏ | 329330/400000 [00:47<00:10, 6946.53it/s] 83%|████████▎ | 330079/400000 [00:47<00:09, 7100.19it/s] 83%|████████▎ | 330803/400000 [00:47<00:09, 7139.25it/s] 83%|████████▎ | 331558/400000 [00:47<00:09, 7255.94it/s] 83%|████████▎ | 332286/400000 [00:47<00:09, 7124.34it/s] 83%|████████▎ | 333000/400000 [00:47<00:09, 7075.48it/s] 83%|████████▎ | 333709/400000 [00:47<00:09, 6985.55it/s] 84%|████████▎ | 334411/400000 [00:47<00:09, 6994.22it/s] 84%|████████▍ | 335112/400000 [00:47<00:09, 6990.55it/s] 84%|████████▍ | 335828/400000 [00:48<00:09, 7040.40it/s] 84%|████████▍ | 336561/400000 [00:48<00:08, 7124.26it/s] 84%|████████▍ | 337295/400000 [00:48<00:08, 7185.02it/s] 85%|████████▍ | 338028/400000 [00:48<00:08, 7227.46it/s] 85%|████████▍ | 338752/400000 [00:48<00:08, 7171.45it/s] 85%|████████▍ | 339470/400000 [00:48<00:08, 7057.31it/s] 85%|████████▌ | 340199/400000 [00:48<00:08, 7123.84it/s] 85%|████████▌ | 340917/400000 [00:48<00:08, 7140.05it/s] 85%|████████▌ | 341637/400000 [00:48<00:08, 7157.78it/s] 86%|████████▌ | 342368/400000 [00:48<00:08, 7202.70it/s] 86%|████████▌ | 343089/400000 [00:49<00:07, 7191.25it/s] 86%|████████▌ | 343809/400000 [00:49<00:07, 7160.86it/s] 86%|████████▌ | 344552/400000 [00:49<00:07, 7238.89it/s] 86%|████████▋ | 345295/400000 [00:49<00:07, 7294.37it/s] 87%|████████▋ | 346030/400000 [00:49<00:07, 7308.87it/s] 87%|████████▋ | 346762/400000 [00:49<00:07, 7293.44it/s] 87%|████████▋ | 347492/400000 [00:49<00:07, 6981.22it/s] 87%|████████▋ | 348194/400000 [00:49<00:07, 6945.49it/s] 87%|████████▋ | 348896/400000 [00:49<00:07, 6966.80it/s] 87%|████████▋ | 349616/400000 [00:50<00:07, 7033.29it/s] 88%|████████▊ | 350321/400000 [00:50<00:07, 6947.35it/s] 88%|████████▊ | 351037/400000 [00:50<00:06, 7008.48it/s] 88%|████████▊ | 351739/400000 [00:50<00:06, 6990.91it/s] 88%|████████▊ | 352443/400000 [00:50<00:06, 7005.48it/s] 88%|████████▊ | 353153/400000 [00:50<00:06, 7032.58it/s] 88%|████████▊ | 353873/400000 [00:50<00:06, 7080.46it/s] 89%|████████▊ | 354582/400000 [00:50<00:06, 7020.83it/s] 89%|████████▉ | 355285/400000 [00:50<00:06, 6969.56it/s] 89%|████████▉ | 355983/400000 [00:50<00:06, 6853.22it/s] 89%|████████▉ | 356669/400000 [00:51<00:06, 6845.04it/s] 89%|████████▉ | 357354/400000 [00:51<00:06, 6776.04it/s] 90%|████████▉ | 358078/400000 [00:51<00:06, 6908.03it/s] 90%|████████▉ | 358811/400000 [00:51<00:05, 7028.01it/s] 90%|████████▉ | 359548/400000 [00:51<00:05, 7125.19it/s] 90%|█████████ | 360262/400000 [00:51<00:05, 7060.41it/s] 90%|█████████ | 360999/400000 [00:51<00:05, 7149.23it/s] 90%|█████████ | 361715/400000 [00:51<00:05, 6989.28it/s] 91%|█████████ | 362442/400000 [00:51<00:05, 7070.52it/s] 91%|█████████ | 363157/400000 [00:51<00:05, 7091.65it/s] 91%|█████████ | 363886/400000 [00:52<00:05, 7147.37it/s] 91%|█████████ | 364602/400000 [00:52<00:05, 7024.68it/s] 91%|█████████▏| 365326/400000 [00:52<00:04, 7082.71it/s] 92%|█████████▏| 366070/400000 [00:52<00:04, 7185.42it/s] 92%|█████████▏| 366790/400000 [00:52<00:04, 7153.31it/s] 92%|█████████▏| 367507/400000 [00:52<00:04, 7125.69it/s] 92%|█████████▏| 368241/400000 [00:52<00:04, 7186.62it/s] 92%|█████████▏| 368961/400000 [00:52<00:04, 7077.35it/s] 92%|█████████▏| 369700/400000 [00:52<00:04, 7168.14it/s] 93%|█████████▎| 370425/400000 [00:52<00:04, 7192.25it/s] 93%|█████████▎| 371166/400000 [00:53<00:03, 7254.32it/s] 93%|█████████▎| 371892/400000 [00:53<00:03, 7191.56it/s] 93%|█████████▎| 372612/400000 [00:53<00:03, 6972.60it/s] 93%|█████████▎| 373312/400000 [00:53<00:03, 6863.56it/s] 94%|█████████▎| 374023/400000 [00:53<00:03, 6931.50it/s] 94%|█████████▎| 374750/400000 [00:53<00:03, 7029.25it/s] 94%|█████████▍| 375455/400000 [00:53<00:03, 7018.83it/s] 94%|█████████▍| 376171/400000 [00:53<00:03, 7059.74it/s] 94%|█████████▍| 376893/400000 [00:53<00:03, 7106.63it/s] 94%|█████████▍| 377605/400000 [00:53<00:03, 7059.44it/s] 95%|█████████▍| 378319/400000 [00:54<00:03, 7083.39it/s] 95%|█████████▍| 379029/400000 [00:54<00:02, 7085.41it/s] 95%|█████████▍| 379749/400000 [00:54<00:02, 7118.53it/s] 95%|█████████▌| 380462/400000 [00:54<00:02, 7121.32it/s] 95%|█████████▌| 381175/400000 [00:54<00:02, 6943.58it/s] 95%|█████████▌| 381871/400000 [00:54<00:02, 6869.29it/s] 96%|█████████▌| 382559/400000 [00:54<00:02, 6832.24it/s] 96%|█████████▌| 383251/400000 [00:54<00:02, 6856.21it/s] 96%|█████████▌| 383938/400000 [00:54<00:02, 6857.13it/s] 96%|█████████▌| 384654/400000 [00:54<00:02, 6942.88it/s] 96%|█████████▋| 385358/400000 [00:55<00:02, 6969.99it/s] 97%|█████████▋| 386056/400000 [00:55<00:02, 6906.91it/s] 97%|█████████▋| 386748/400000 [00:55<00:01, 6730.38it/s] 97%|█████████▋| 387486/400000 [00:55<00:01, 6911.11it/s] 97%|█████████▋| 388217/400000 [00:55<00:01, 7024.42it/s] 97%|█████████▋| 388944/400000 [00:55<00:01, 7095.65it/s] 97%|█████████▋| 389666/400000 [00:55<00:01, 7129.97it/s] 98%|█████████▊| 390382/400000 [00:55<00:01, 7137.39it/s] 98%|█████████▊| 391101/400000 [00:55<00:01, 7149.90it/s] 98%|█████████▊| 391841/400000 [00:56<00:01, 7222.83it/s] 98%|█████████▊| 392579/400000 [00:56<00:01, 7266.22it/s] 98%|█████████▊| 393307/400000 [00:56<00:00, 7132.63it/s] 99%|█████████▊| 394022/400000 [00:56<00:00, 7014.88it/s] 99%|█████████▊| 394725/400000 [00:56<00:00, 6991.52it/s] 99%|█████████▉| 395454/400000 [00:56<00:00, 7075.40it/s] 99%|█████████▉| 396171/400000 [00:56<00:00, 7101.56it/s] 99%|█████████▉| 396887/400000 [00:56<00:00, 7118.15it/s] 99%|█████████▉| 397600/400000 [00:56<00:00, 7119.96it/s]100%|█████████▉| 398313/400000 [00:56<00:00, 7055.40it/s]100%|█████████▉| 399019/400000 [00:57<00:00, 6953.09it/s]100%|█████████▉| 399730/400000 [00:57<00:00, 6998.29it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 6996.16it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f16e38cc1d0>, <torchtext.data.dataset.TabularDataset object at 0x7f16e38cc320>, <torchtext.vocab.Vocab object at 0x7f16e38cc240>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 26.50 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.41 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 17.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 17.41 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.80 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.80 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.82 file/s]2020-07-10 18:21:39.962626: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-10 18:21:39.967815: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-10 18:21:39.968017: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e0d8c63530 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-10 18:21:39.968043: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157295.28it/s] 77%|███████▋  | 7643136/9912422 [00:00<00:10, 224509.07it/s]9920512it [00:00, 42742797.47it/s]                           
0it [00:00, ?it/s]32768it [00:00, 618957.77it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478714.57it/s]1654784it [00:00, 12155326.55it/s]                         
0it [00:00, ?it/s]8192it [00:00, 229476.45it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
