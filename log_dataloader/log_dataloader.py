
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1be13feea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1be13feea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1c4c6bf1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1c4c6bf1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1c6aa07e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1c6aa07e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf99e8488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf99e8488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf99e8488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 18%|█▊        | 1736704/9912422 [00:00<00:00, 17367015.16it/s] 54%|█████▍    | 5357568/9912422 [00:00<00:00, 20529055.85it/s] 93%|█████████▎| 9232384/9912422 [00:00<00:00, 23745436.60it/s]9920512it [00:00, 21959055.41it/s]                             
0it [00:00, ?it/s]32768it [00:00, 675395.61it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1010909.89it/s]1654784it [00:00, 11089754.71it/s]                           
0it [00:00, ?it/s]8192it [00:00, 232594.15it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1be0b14390>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1be0afb630>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1bf99e80d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1bf99e80d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1bf99e80d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:06,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:06,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:06,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:05,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:05,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:04,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:45,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:45,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:44,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:44,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:43,  3.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:31,  4.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:31,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:30,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:29,  4.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:21,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:13,  9.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 12.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 12.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 33.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 39.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 66.86 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.51 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.51 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.54 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ url]
0 examples [00:00, ? examples/s]2020-06-27 00:12:22.557482: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-27 00:12:22.705361: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-27 00:12:22.705679: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a727be8390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-27 00:12:22.705702: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.63 examples/s]85 examples [00:00,  2.32 examples/s]173 examples [00:00,  3.31 examples/s]261 examples [00:00,  4.72 examples/s]348 examples [00:01,  6.73 examples/s]435 examples [00:01,  9.58 examples/s]523 examples [00:01, 13.63 examples/s]602 examples [00:01, 19.32 examples/s]691 examples [00:01, 27.35 examples/s]780 examples [00:01, 38.56 examples/s]867 examples [00:01, 54.05 examples/s]952 examples [00:01, 75.16 examples/s]1038 examples [00:01, 103.47 examples/s]1129 examples [00:01, 140.93 examples/s]1219 examples [00:02, 188.56 examples/s]1307 examples [00:02, 246.21 examples/s]1394 examples [00:02, 313.15 examples/s]1481 examples [00:02, 387.08 examples/s]1569 examples [00:02, 465.22 examples/s]1660 examples [00:02, 545.06 examples/s]1749 examples [00:02, 610.88 examples/s]1837 examples [00:02, 656.30 examples/s]1922 examples [00:02, 694.33 examples/s]2009 examples [00:02, 736.93 examples/s]2098 examples [00:03, 776.42 examples/s]2190 examples [00:03, 811.97 examples/s]2281 examples [00:03, 836.74 examples/s]2371 examples [00:03, 854.08 examples/s]2460 examples [00:03, 861.16 examples/s]2552 examples [00:03, 876.90 examples/s]2642 examples [00:03, 881.09 examples/s]2732 examples [00:03, 838.67 examples/s]2818 examples [00:03, 828.05 examples/s]2907 examples [00:03, 845.52 examples/s]2998 examples [00:04, 861.68 examples/s]3088 examples [00:04, 870.97 examples/s]3176 examples [00:04, 867.42 examples/s]3264 examples [00:04, 864.98 examples/s]3351 examples [00:04, 849.55 examples/s]3437 examples [00:04, 821.12 examples/s]3526 examples [00:04, 840.29 examples/s]3614 examples [00:04, 850.43 examples/s]3704 examples [00:04, 863.07 examples/s]3795 examples [00:05, 875.65 examples/s]3883 examples [00:05, 875.27 examples/s]3972 examples [00:05, 876.80 examples/s]4060 examples [00:05, 846.83 examples/s]4145 examples [00:05, 843.38 examples/s]4230 examples [00:05, 842.26 examples/s]4325 examples [00:05, 869.51 examples/s]4413 examples [00:05, 858.65 examples/s]4500 examples [00:05, 850.84 examples/s]4589 examples [00:05, 861.56 examples/s]4677 examples [00:06, 865.97 examples/s]4766 examples [00:06, 872.07 examples/s]4857 examples [00:06, 882.18 examples/s]4946 examples [00:06, 880.87 examples/s]5035 examples [00:06, 872.94 examples/s]5123 examples [00:06, 874.92 examples/s]5212 examples [00:06, 878.29 examples/s]5300 examples [00:06, 872.10 examples/s]5388 examples [00:06, 869.78 examples/s]5476 examples [00:06, 871.76 examples/s]5564 examples [00:07, 845.60 examples/s]5652 examples [00:07, 854.94 examples/s]5738 examples [00:07, 849.62 examples/s]5824 examples [00:07, 845.82 examples/s]5911 examples [00:07, 851.45 examples/s]5997 examples [00:07, 836.97 examples/s]6085 examples [00:07, 848.65 examples/s]6170 examples [00:07, 835.44 examples/s]6255 examples [00:07, 838.53 examples/s]6344 examples [00:07, 852.32 examples/s]6432 examples [00:08, 859.97 examples/s]6519 examples [00:08, 844.39 examples/s]6607 examples [00:08, 854.71 examples/s]6695 examples [00:08, 860.78 examples/s]6782 examples [00:08, 861.02 examples/s]6869 examples [00:08, 862.51 examples/s]6962 examples [00:08, 880.43 examples/s]7051 examples [00:08, 882.01 examples/s]7140 examples [00:08, 859.14 examples/s]7228 examples [00:09, 862.68 examples/s]7324 examples [00:09, 878.65 examples/s]7413 examples [00:09, 866.70 examples/s]7500 examples [00:09, 852.57 examples/s]7586 examples [00:09, 850.00 examples/s]7677 examples [00:09, 864.49 examples/s]7766 examples [00:09, 869.93 examples/s]7854 examples [00:09, 865.17 examples/s]7941 examples [00:09, 850.31 examples/s]8027 examples [00:09, 840.51 examples/s]8112 examples [00:10, 818.13 examples/s]8201 examples [00:10, 836.79 examples/s]8286 examples [00:10, 839.54 examples/s]8371 examples [00:10, 842.23 examples/s]8460 examples [00:10, 854.38 examples/s]8549 examples [00:10, 864.58 examples/s]8636 examples [00:10, 860.60 examples/s]8724 examples [00:10, 865.08 examples/s]8811 examples [00:10, 864.75 examples/s]8898 examples [00:10, 860.37 examples/s]8986 examples [00:11, 864.75 examples/s]9078 examples [00:11, 879.87 examples/s]9167 examples [00:11, 874.19 examples/s]9255 examples [00:11, 837.39 examples/s]9343 examples [00:11, 848.49 examples/s]9431 examples [00:11, 856.45 examples/s]9521 examples [00:11, 868.97 examples/s]9609 examples [00:11, 863.11 examples/s]9699 examples [00:11, 873.84 examples/s]9787 examples [00:11, 869.57 examples/s]9875 examples [00:12, 871.18 examples/s]9963 examples [00:12, 871.60 examples/s]10051 examples [00:12, 811.31 examples/s]10136 examples [00:12, 821.25 examples/s]10224 examples [00:12, 837.19 examples/s]10314 examples [00:12, 854.57 examples/s]10400 examples [00:12, 855.42 examples/s]10487 examples [00:12, 857.92 examples/s]10574 examples [00:12, 842.54 examples/s]10662 examples [00:13, 851.57 examples/s]10749 examples [00:13, 855.47 examples/s]10835 examples [00:13, 832.63 examples/s]10919 examples [00:13, 819.03 examples/s]11002 examples [00:13, 816.83 examples/s]11084 examples [00:13, 798.78 examples/s]11172 examples [00:13, 821.19 examples/s]11261 examples [00:13, 839.56 examples/s]11346 examples [00:13, 840.00 examples/s]11431 examples [00:13, 829.70 examples/s]11519 examples [00:14, 843.26 examples/s]11607 examples [00:14, 851.74 examples/s]11693 examples [00:14, 821.91 examples/s]11780 examples [00:14, 835.76 examples/s]11871 examples [00:14, 855.10 examples/s]11957 examples [00:14, 845.55 examples/s]12042 examples [00:14, 834.96 examples/s]12126 examples [00:14, 825.82 examples/s]12217 examples [00:14, 847.19 examples/s]12304 examples [00:14, 852.98 examples/s]12394 examples [00:15, 863.85 examples/s]12485 examples [00:15, 874.82 examples/s]12575 examples [00:15, 881.39 examples/s]12664 examples [00:15, 882.92 examples/s]12754 examples [00:15, 887.56 examples/s]12843 examples [00:15, 882.57 examples/s]12932 examples [00:15, 846.08 examples/s]13018 examples [00:15, 849.35 examples/s]13105 examples [00:15, 854.14 examples/s]13192 examples [00:16, 857.06 examples/s]13280 examples [00:16, 863.39 examples/s]13368 examples [00:16, 868.11 examples/s]13458 examples [00:16, 875.58 examples/s]13546 examples [00:16, 864.36 examples/s]13635 examples [00:16, 869.63 examples/s]13726 examples [00:16, 880.80 examples/s]13817 examples [00:16, 887.36 examples/s]13908 examples [00:16, 891.85 examples/s]14000 examples [00:16, 897.40 examples/s]14091 examples [00:17, 898.48 examples/s]14182 examples [00:17, 900.11 examples/s]14273 examples [00:17, 852.26 examples/s]14359 examples [00:17, 846.18 examples/s]14445 examples [00:17, 833.78 examples/s]14529 examples [00:17, 825.69 examples/s]14612 examples [00:17, 803.48 examples/s]14693 examples [00:17, 801.65 examples/s]14774 examples [00:17, 797.86 examples/s]14855 examples [00:17, 799.60 examples/s]14938 examples [00:18, 806.45 examples/s]15025 examples [00:18, 822.35 examples/s]15115 examples [00:18, 841.49 examples/s]15202 examples [00:18, 847.36 examples/s]15295 examples [00:18, 869.32 examples/s]15383 examples [00:18, 865.94 examples/s]15475 examples [00:18, 879.80 examples/s]15564 examples [00:18, 880.13 examples/s]15653 examples [00:18, 869.60 examples/s]15741 examples [00:18, 868.30 examples/s]15830 examples [00:19, 872.30 examples/s]15921 examples [00:19, 881.04 examples/s]16013 examples [00:19, 889.96 examples/s]16103 examples [00:19, 887.42 examples/s]16192 examples [00:19, 865.69 examples/s]16281 examples [00:19, 871.10 examples/s]16369 examples [00:19, 852.44 examples/s]16458 examples [00:19, 861.04 examples/s]16548 examples [00:19, 872.08 examples/s]16638 examples [00:20, 878.33 examples/s]16726 examples [00:20, 849.15 examples/s]16814 examples [00:20, 857.87 examples/s]16901 examples [00:20, 835.53 examples/s]16989 examples [00:20, 843.88 examples/s]17080 examples [00:20, 860.87 examples/s]17168 examples [00:20, 864.93 examples/s]17255 examples [00:20, 857.96 examples/s]17344 examples [00:20, 866.31 examples/s]17438 examples [00:20, 885.24 examples/s]17528 examples [00:21, 888.33 examples/s]17617 examples [00:21, 874.73 examples/s]17705 examples [00:21, 852.28 examples/s]17791 examples [00:21, 845.69 examples/s]17878 examples [00:21, 851.87 examples/s]17967 examples [00:21, 861.84 examples/s]18055 examples [00:21, 865.52 examples/s]18142 examples [00:21, 860.35 examples/s]18230 examples [00:21, 866.01 examples/s]18319 examples [00:21, 871.49 examples/s]18407 examples [00:22, 869.02 examples/s]18494 examples [00:22, 831.19 examples/s]18579 examples [00:22, 835.05 examples/s]18666 examples [00:22, 843.60 examples/s]18753 examples [00:22, 850.99 examples/s]18841 examples [00:22, 858.28 examples/s]18930 examples [00:22, 867.24 examples/s]19018 examples [00:22, 869.55 examples/s]19106 examples [00:22, 868.15 examples/s]19193 examples [00:22, 865.08 examples/s]19281 examples [00:23, 868.66 examples/s]19370 examples [00:23, 873.01 examples/s]19458 examples [00:23, 873.07 examples/s]19546 examples [00:23, 863.34 examples/s]19633 examples [00:23, 848.34 examples/s]19718 examples [00:23, 844.72 examples/s]19803 examples [00:23, 845.91 examples/s]19890 examples [00:23, 850.83 examples/s]19976 examples [00:23, 845.50 examples/s]20061 examples [00:24, 806.65 examples/s]20153 examples [00:24, 835.04 examples/s]20241 examples [00:24, 847.56 examples/s]20333 examples [00:24, 867.12 examples/s]20425 examples [00:24, 880.74 examples/s]20514 examples [00:24, 879.15 examples/s]20603 examples [00:24, 869.29 examples/s]20693 examples [00:24, 875.89 examples/s]20781 examples [00:24, 811.41 examples/s]20864 examples [00:24, 816.71 examples/s]20952 examples [00:25, 834.02 examples/s]21039 examples [00:25, 844.28 examples/s]21126 examples [00:25, 850.54 examples/s]21218 examples [00:25, 869.90 examples/s]21309 examples [00:25, 878.75 examples/s]21398 examples [00:25, 876.27 examples/s]21486 examples [00:25, 867.58 examples/s]21574 examples [00:25, 870.92 examples/s]21665 examples [00:25, 879.84 examples/s]21755 examples [00:25, 885.46 examples/s]21845 examples [00:26, 887.11 examples/s]21934 examples [00:26, 883.72 examples/s]22023 examples [00:26, 861.98 examples/s]22113 examples [00:26, 872.25 examples/s]22202 examples [00:26, 877.28 examples/s]22292 examples [00:26, 881.90 examples/s]22384 examples [00:26, 890.37 examples/s]22474 examples [00:26, 884.17 examples/s]22565 examples [00:26, 889.54 examples/s]22656 examples [00:26, 893.27 examples/s]22746 examples [00:27, 868.64 examples/s]22834 examples [00:27, 846.70 examples/s]22920 examples [00:27, 849.37 examples/s]23010 examples [00:27, 862.50 examples/s]23100 examples [00:27, 872.09 examples/s]23189 examples [00:27, 874.54 examples/s]23277 examples [00:27, 863.39 examples/s]23367 examples [00:27, 871.43 examples/s]23455 examples [00:27, 868.31 examples/s]23544 examples [00:28, 874.32 examples/s]23633 examples [00:28, 877.57 examples/s]23722 examples [00:28, 878.85 examples/s]23810 examples [00:28, 875.19 examples/s]23900 examples [00:28, 881.48 examples/s]23990 examples [00:28, 885.74 examples/s]24079 examples [00:28, 863.33 examples/s]24169 examples [00:28, 873.33 examples/s]24257 examples [00:28, 864.26 examples/s]24344 examples [00:28, 841.08 examples/s]24430 examples [00:29, 846.10 examples/s]24520 examples [00:29, 859.64 examples/s]24611 examples [00:29, 871.59 examples/s]24699 examples [00:29, 864.97 examples/s]24791 examples [00:29, 880.06 examples/s]24885 examples [00:29, 896.43 examples/s]24977 examples [00:29, 903.20 examples/s]25068 examples [00:29, 893.45 examples/s]25162 examples [00:29, 904.66 examples/s]25253 examples [00:29, 905.78 examples/s]25344 examples [00:30, 904.03 examples/s]25435 examples [00:30, 894.59 examples/s]25525 examples [00:30, 887.86 examples/s]25615 examples [00:30, 888.87 examples/s]25704 examples [00:30, 837.22 examples/s]25789 examples [00:30, 831.21 examples/s]25877 examples [00:30, 844.55 examples/s]25965 examples [00:30, 852.16 examples/s]26060 examples [00:30, 876.93 examples/s]26149 examples [00:30, 872.10 examples/s]26239 examples [00:31, 878.06 examples/s]26328 examples [00:31, 881.34 examples/s]26417 examples [00:31, 878.05 examples/s]26505 examples [00:31, 875.05 examples/s]26593 examples [00:31, 870.85 examples/s]26682 examples [00:31, 875.28 examples/s]26770 examples [00:31, 850.11 examples/s]26859 examples [00:31, 858.45 examples/s]26951 examples [00:31, 873.46 examples/s]27042 examples [00:32, 882.96 examples/s]27133 examples [00:32, 889.61 examples/s]27223 examples [00:32, 877.84 examples/s]27311 examples [00:32, 857.02 examples/s]27403 examples [00:32, 874.69 examples/s]27493 examples [00:32, 879.39 examples/s]27582 examples [00:32, 875.49 examples/s]27670 examples [00:32, 875.86 examples/s]27759 examples [00:32, 879.28 examples/s]27847 examples [00:32, 874.28 examples/s]27937 examples [00:33, 880.45 examples/s]28027 examples [00:33, 884.42 examples/s]28116 examples [00:33, 884.31 examples/s]28205 examples [00:33, 878.46 examples/s]28293 examples [00:33, 876.01 examples/s]28381 examples [00:33, 852.51 examples/s]28468 examples [00:33, 855.26 examples/s]28559 examples [00:33, 870.47 examples/s]28651 examples [00:33, 883.58 examples/s]28740 examples [00:33, 880.32 examples/s]28832 examples [00:34, 888.98 examples/s]28922 examples [00:34, 890.30 examples/s]29015 examples [00:34, 901.21 examples/s]29106 examples [00:34, 860.98 examples/s]29193 examples [00:34, 862.44 examples/s]29280 examples [00:34, 859.18 examples/s]29367 examples [00:34, 854.46 examples/s]29453 examples [00:34, 845.15 examples/s]29542 examples [00:34, 857.58 examples/s]29630 examples [00:34, 863.21 examples/s]29722 examples [00:35, 877.49 examples/s]29811 examples [00:35, 878.53 examples/s]29901 examples [00:35, 884.65 examples/s]29992 examples [00:35, 889.77 examples/s]30082 examples [00:35, 830.53 examples/s]30169 examples [00:35, 839.62 examples/s]30260 examples [00:35, 858.59 examples/s]30352 examples [00:35, 873.76 examples/s]30444 examples [00:35, 886.42 examples/s]30533 examples [00:36, 885.71 examples/s]30623 examples [00:36, 887.79 examples/s]30712 examples [00:36, 887.71 examples/s]30801 examples [00:36, 883.78 examples/s]30892 examples [00:36, 890.45 examples/s]30982 examples [00:36, 878.00 examples/s]31070 examples [00:36, 875.66 examples/s]31159 examples [00:36, 879.04 examples/s]31249 examples [00:36, 883.27 examples/s]31338 examples [00:36, 880.85 examples/s]31427 examples [00:37, 862.07 examples/s]31514 examples [00:37, 856.58 examples/s]31600 examples [00:37, 856.59 examples/s]31686 examples [00:37, 845.65 examples/s]31776 examples [00:37, 860.62 examples/s]31864 examples [00:37, 864.14 examples/s]31951 examples [00:37, 851.58 examples/s]32039 examples [00:37, 858.28 examples/s]32125 examples [00:37, 841.71 examples/s]32215 examples [00:37, 856.95 examples/s]32301 examples [00:38, 842.39 examples/s]32391 examples [00:38, 857.36 examples/s]32481 examples [00:38, 868.31 examples/s]32570 examples [00:38, 872.24 examples/s]32658 examples [00:38, 873.79 examples/s]32747 examples [00:38, 878.01 examples/s]32835 examples [00:38, 845.14 examples/s]32924 examples [00:38, 856.20 examples/s]33010 examples [00:38, 850.73 examples/s]33096 examples [00:38, 832.48 examples/s]33184 examples [00:39, 845.36 examples/s]33274 examples [00:39, 858.63 examples/s]33363 examples [00:39, 867.27 examples/s]33450 examples [00:39, 861.35 examples/s]33537 examples [00:39, 857.71 examples/s]33627 examples [00:39, 868.87 examples/s]33717 examples [00:39, 875.55 examples/s]33806 examples [00:39, 879.57 examples/s]33898 examples [00:39, 889.39 examples/s]33989 examples [00:40, 893.38 examples/s]34079 examples [00:40, 893.68 examples/s]34169 examples [00:40, 895.17 examples/s]34260 examples [00:40, 897.35 examples/s]34350 examples [00:40, 897.11 examples/s]34440 examples [00:40, 896.16 examples/s]34530 examples [00:40, 852.44 examples/s]34619 examples [00:40, 861.02 examples/s]34706 examples [00:40, 849.27 examples/s]34792 examples [00:40, 852.32 examples/s]34878 examples [00:41, 853.92 examples/s]34967 examples [00:41, 862.08 examples/s]35054 examples [00:41, 862.38 examples/s]35143 examples [00:41, 870.38 examples/s]35231 examples [00:41, 871.63 examples/s]35319 examples [00:41, 870.16 examples/s]35407 examples [00:41, 859.65 examples/s]35495 examples [00:41, 864.44 examples/s]35582 examples [00:41, 846.22 examples/s]35668 examples [00:41, 845.91 examples/s]35758 examples [00:42, 858.03 examples/s]35846 examples [00:42, 862.14 examples/s]35936 examples [00:42, 871.26 examples/s]36026 examples [00:42, 878.91 examples/s]36114 examples [00:42, 844.59 examples/s]36203 examples [00:42, 856.82 examples/s]36293 examples [00:42, 867.41 examples/s]36381 examples [00:42, 869.10 examples/s]36469 examples [00:42, 862.11 examples/s]36558 examples [00:42, 867.75 examples/s]36645 examples [00:43, 829.10 examples/s]36733 examples [00:43, 843.54 examples/s]36824 examples [00:43, 860.72 examples/s]36913 examples [00:43, 866.50 examples/s]37002 examples [00:43, 871.98 examples/s]37090 examples [00:43, 868.81 examples/s]37178 examples [00:43, 857.67 examples/s]37269 examples [00:43, 870.45 examples/s]37357 examples [00:43, 857.01 examples/s]37449 examples [00:44, 874.95 examples/s]37537 examples [00:44, 873.81 examples/s]37627 examples [00:44, 879.41 examples/s]37718 examples [00:44, 886.89 examples/s]37807 examples [00:44, 880.14 examples/s]37896 examples [00:44, 877.50 examples/s]37984 examples [00:44, 873.57 examples/s]38072 examples [00:44, 872.21 examples/s]38161 examples [00:44, 876.44 examples/s]38249 examples [00:44, 867.71 examples/s]38336 examples [00:45, 773.87 examples/s]38416 examples [00:45, 739.47 examples/s]38505 examples [00:45, 777.26 examples/s]38597 examples [00:45, 815.18 examples/s]38685 examples [00:45, 832.62 examples/s]38777 examples [00:45, 854.92 examples/s]38864 examples [00:45, 856.26 examples/s]38956 examples [00:45, 871.45 examples/s]39047 examples [00:45, 880.81 examples/s]39139 examples [00:45, 890.49 examples/s]39230 examples [00:46, 894.37 examples/s]39320 examples [00:46, 887.50 examples/s]39412 examples [00:46, 895.50 examples/s]39502 examples [00:46, 889.83 examples/s]39592 examples [00:46, 886.77 examples/s]39681 examples [00:46, 882.13 examples/s]39773 examples [00:46, 890.92 examples/s]39863 examples [00:46, 892.78 examples/s]39953 examples [00:46, 894.72 examples/s]40043 examples [00:47, 823.24 examples/s]40132 examples [00:47, 841.83 examples/s]40218 examples [00:47, 795.56 examples/s]40299 examples [00:47, 745.95 examples/s]40385 examples [00:47, 776.72 examples/s]40474 examples [00:47, 807.15 examples/s]40556 examples [00:47, 800.23 examples/s]40642 examples [00:47, 816.04 examples/s]40731 examples [00:47, 835.42 examples/s]40820 examples [00:47, 850.49 examples/s]40910 examples [00:48, 863.79 examples/s]40997 examples [00:48, 865.12 examples/s]41086 examples [00:48, 870.31 examples/s]41176 examples [00:48, 876.07 examples/s]41266 examples [00:48, 882.18 examples/s]41360 examples [00:48, 896.46 examples/s]41450 examples [00:48, 896.21 examples/s]41540 examples [00:48, 895.32 examples/s]41630 examples [00:48, 889.25 examples/s]41719 examples [00:48, 888.82 examples/s]41809 examples [00:49, 889.49 examples/s]41898 examples [00:49, 868.55 examples/s]41985 examples [00:49, 862.98 examples/s]42074 examples [00:49, 870.64 examples/s]42164 examples [00:49, 876.64 examples/s]42253 examples [00:49, 880.11 examples/s]42342 examples [00:49, 880.38 examples/s]42432 examples [00:49, 883.10 examples/s]42524 examples [00:49, 892.94 examples/s]42614 examples [00:50, 876.79 examples/s]42702 examples [00:50, 869.77 examples/s]42793 examples [00:50, 879.05 examples/s]42881 examples [00:50, 872.05 examples/s]42969 examples [00:50, 836.46 examples/s]43054 examples [00:50, 828.01 examples/s]43143 examples [00:50, 844.96 examples/s]43228 examples [00:50, 827.21 examples/s]43312 examples [00:50, 829.49 examples/s]43400 examples [00:50, 843.55 examples/s]43485 examples [00:51, 822.27 examples/s]43568 examples [00:51, 808.15 examples/s]43652 examples [00:51, 815.12 examples/s]43742 examples [00:51, 837.76 examples/s]43829 examples [00:51, 844.56 examples/s]43915 examples [00:51, 848.65 examples/s]44006 examples [00:51, 865.92 examples/s]44093 examples [00:51, 864.42 examples/s]44183 examples [00:51, 873.85 examples/s]44274 examples [00:51, 882.15 examples/s]44363 examples [00:52, 883.69 examples/s]44453 examples [00:52, 886.38 examples/s]44542 examples [00:52, 866.85 examples/s]44629 examples [00:52, 822.40 examples/s]44718 examples [00:52, 839.06 examples/s]44808 examples [00:52, 854.33 examples/s]44895 examples [00:52, 857.70 examples/s]44987 examples [00:52, 873.61 examples/s]45081 examples [00:52, 890.05 examples/s]45171 examples [00:53, 877.10 examples/s]45261 examples [00:53, 881.06 examples/s]45350 examples [00:53, 847.26 examples/s]45438 examples [00:53, 855.86 examples/s]45528 examples [00:53, 867.91 examples/s]45616 examples [00:53, 848.39 examples/s]45705 examples [00:53, 858.82 examples/s]45793 examples [00:53, 862.90 examples/s]45880 examples [00:53, 861.23 examples/s]45967 examples [00:53, 860.26 examples/s]46054 examples [00:54, 862.70 examples/s]46142 examples [00:54, 866.52 examples/s]46231 examples [00:54, 871.75 examples/s]46319 examples [00:54, 872.86 examples/s]46407 examples [00:54, 874.33 examples/s]46495 examples [00:54, 864.66 examples/s]46582 examples [00:54, 838.85 examples/s]46667 examples [00:54, 831.41 examples/s]46757 examples [00:54, 849.73 examples/s]46846 examples [00:54, 859.98 examples/s]46938 examples [00:55, 877.06 examples/s]47028 examples [00:55, 882.89 examples/s]47117 examples [00:55, 884.31 examples/s]47206 examples [00:55, 874.30 examples/s]47294 examples [00:55, 875.94 examples/s]47382 examples [00:55, 875.28 examples/s]47470 examples [00:55, 876.44 examples/s]47558 examples [00:55, 853.18 examples/s]47644 examples [00:55, 830.90 examples/s]47734 examples [00:55, 847.82 examples/s]47824 examples [00:56, 843.67 examples/s]47912 examples [00:56, 852.31 examples/s]47998 examples [00:56, 845.32 examples/s]48086 examples [00:56, 853.24 examples/s]48172 examples [00:56, 849.11 examples/s]48263 examples [00:56, 865.45 examples/s]48353 examples [00:56, 873.58 examples/s]48441 examples [00:56, 874.10 examples/s]48529 examples [00:56, 871.70 examples/s]48619 examples [00:57, 877.33 examples/s]48707 examples [00:57, 870.65 examples/s]48795 examples [00:57, 861.19 examples/s]48882 examples [00:57, 862.19 examples/s]48969 examples [00:57, 855.92 examples/s]49055 examples [00:57, 855.89 examples/s]49141 examples [00:57, 855.85 examples/s]49227 examples [00:57, 847.72 examples/s]49315 examples [00:57, 855.03 examples/s]49402 examples [00:57, 857.05 examples/s]49492 examples [00:58, 866.96 examples/s]49585 examples [00:58, 883.25 examples/s]49675 examples [00:58, 887.57 examples/s]49764 examples [00:58, 886.16 examples/s]49853 examples [00:58, 883.44 examples/s]49943 examples [00:58, 886.35 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6004/50000 [00:00<00:00, 60039.77 examples/s] 35%|███▍      | 17383/50000 [00:00<00:00, 69952.25 examples/s] 56%|█████▌    | 27767/50000 [00:00<00:00, 77543.75 examples/s] 77%|███████▋  | 38536/50000 [00:00<00:00, 84652.60 examples/s] 99%|█████████▉| 49701/50000 [00:00<00:00, 91272.17 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 697.16 examples/s]156 examples [00:00, 738.18 examples/s]235 examples [00:00, 750.00 examples/s]326 examples [00:00, 791.51 examples/s]415 examples [00:00, 818.49 examples/s]497 examples [00:00, 818.75 examples/s]590 examples [00:00, 847.41 examples/s]671 examples [00:00, 835.28 examples/s]755 examples [00:00, 834.91 examples/s]849 examples [00:01, 862.35 examples/s]939 examples [00:01, 872.36 examples/s]1027 examples [00:01, 872.28 examples/s]1114 examples [00:01, 848.97 examples/s]1203 examples [00:01, 860.42 examples/s]1292 examples [00:01, 868.37 examples/s]1380 examples [00:01, 860.88 examples/s]1468 examples [00:01, 865.21 examples/s]1557 examples [00:01, 872.39 examples/s]1650 examples [00:01, 888.14 examples/s]1743 examples [00:02, 898.74 examples/s]1836 examples [00:02, 905.16 examples/s]1928 examples [00:02, 906.70 examples/s]2019 examples [00:02, 905.87 examples/s]2111 examples [00:02, 907.48 examples/s]2202 examples [00:02, 908.03 examples/s]2293 examples [00:02, 904.78 examples/s]2384 examples [00:02, 774.75 examples/s]2472 examples [00:02, 801.41 examples/s]2563 examples [00:02, 830.40 examples/s]2654 examples [00:03, 850.46 examples/s]2741 examples [00:03, 853.64 examples/s]2828 examples [00:03, 841.28 examples/s]2913 examples [00:03, 841.75 examples/s]2998 examples [00:03, 830.80 examples/s]3087 examples [00:03, 847.69 examples/s]3173 examples [00:03, 850.98 examples/s]3262 examples [00:03, 862.10 examples/s]3353 examples [00:03, 873.91 examples/s]3442 examples [00:04, 878.30 examples/s]3531 examples [00:04, 881.09 examples/s]3620 examples [00:04, 851.11 examples/s]3710 examples [00:04, 862.48 examples/s]3797 examples [00:04, 850.65 examples/s]3887 examples [00:04, 863.03 examples/s]3974 examples [00:04, 851.84 examples/s]4060 examples [00:04, 833.27 examples/s]4145 examples [00:04, 835.81 examples/s]4234 examples [00:04, 849.65 examples/s]4320 examples [00:05, 847.73 examples/s]4407 examples [00:05, 852.83 examples/s]4500 examples [00:05, 873.17 examples/s]4590 examples [00:05, 880.35 examples/s]4681 examples [00:05, 886.38 examples/s]4770 examples [00:05, 886.62 examples/s]4859 examples [00:05, 862.31 examples/s]4951 examples [00:05, 877.87 examples/s]5041 examples [00:05, 884.04 examples/s]5133 examples [00:05, 894.43 examples/s]5224 examples [00:06, 897.01 examples/s]5315 examples [00:06, 898.26 examples/s]5405 examples [00:06, 863.86 examples/s]5492 examples [00:06, 819.21 examples/s]5582 examples [00:06, 840.74 examples/s]5675 examples [00:06, 863.37 examples/s]5763 examples [00:06, 866.27 examples/s]5851 examples [00:06, 868.64 examples/s]5941 examples [00:06, 875.88 examples/s]6030 examples [00:06, 877.52 examples/s]6120 examples [00:07, 883.74 examples/s]6211 examples [00:07, 890.08 examples/s]6302 examples [00:07, 895.18 examples/s]6392 examples [00:07, 893.45 examples/s]6482 examples [00:07, 888.89 examples/s]6571 examples [00:07, 888.19 examples/s]6660 examples [00:07, 864.87 examples/s]6747 examples [00:07, 861.66 examples/s]6834 examples [00:07, 863.71 examples/s]6921 examples [00:08, 862.20 examples/s]7011 examples [00:08, 871.11 examples/s]7099 examples [00:08, 871.94 examples/s]7188 examples [00:08, 874.87 examples/s]7278 examples [00:08, 879.67 examples/s]7366 examples [00:08, 861.37 examples/s]7453 examples [00:08, 862.17 examples/s]7540 examples [00:08, 858.00 examples/s]7629 examples [00:08, 867.09 examples/s]7716 examples [00:08, 844.55 examples/s]7801 examples [00:09, 840.40 examples/s]7890 examples [00:09, 854.60 examples/s]7982 examples [00:09, 871.62 examples/s]8076 examples [00:09, 889.80 examples/s]8166 examples [00:09, 858.52 examples/s]8260 examples [00:09, 878.45 examples/s]8349 examples [00:09, 875.94 examples/s]8441 examples [00:09, 887.95 examples/s]8531 examples [00:09, 888.22 examples/s]8623 examples [00:09, 896.79 examples/s]8713 examples [00:10, 840.02 examples/s]8800 examples [00:10, 848.67 examples/s]8889 examples [00:10, 858.08 examples/s]8977 examples [00:10, 864.10 examples/s]9066 examples [00:10, 870.28 examples/s]9160 examples [00:10, 887.92 examples/s]9251 examples [00:10, 893.13 examples/s]9341 examples [00:10, 889.15 examples/s]9431 examples [00:10, 889.00 examples/s]9520 examples [00:10, 883.55 examples/s]9611 examples [00:11, 889.02 examples/s]9705 examples [00:11, 903.54 examples/s]9796 examples [00:11, 902.34 examples/s]9887 examples [00:11, 893.16 examples/s]9977 examples [00:11, 883.27 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete36QEQN/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete36QEQN/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf99e8488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf99e8488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf99e8488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b7c477fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1b7ee4aac8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1bf99e8488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1bf99e8488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1bf99e8488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1be0afb400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1be0afb128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1b71808158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1b71808158> 

  function with postional parmater data_info <function split_train_valid at 0x7f1b71808158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1b71808268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1b71808268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1b71808268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=a644a2d289c3727b3e6c30f1d795ab79e5db9296286920e0a0e586511324a8fd
  Stored in directory: /tmp/pip-ephem-wheel-cache-7u59zese/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:49:12, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:54:27, 14.2kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:53:31, 20.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<8:19:57, 28.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.53M/862M [00:01<5:49:06, 41.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.52M/862M [00:01<4:02:46, 58.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:01<2:48:55, 83.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:57:32, 119kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<1:21:48, 170kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:02<56:58, 243kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:02<39:41, 346kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.9M/862M [00:02<27:50, 492kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<19:26, 700kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.4M/862M [00:02<13:40, 990kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<10:18, 1.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<09:05, 1.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<10:02, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<07:48, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<05:40, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<07:59, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<07:10, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<05:24, 2.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:32, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:23, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:09<04:42, 2.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:09<03:39, 3.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:09<03:10, 4.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<09:08, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<10:32, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<08:24, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<06:22, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<04:54, 2.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:06, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<09:44, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<07:51, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<05:54, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:13<04:35, 2.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<07:47, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<09:41, 1.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<07:51, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<05:57, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<04:38, 2.81MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<07:41, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<09:41, 1.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<07:39, 1.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:17<05:44, 2.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:17<04:37, 2.80MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<03:43, 3.49MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<07:57, 1.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<09:33, 1.36MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<07:32, 1.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:19<05:44, 2.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<04:21, 2.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<03:36, 3.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<12:02, 1.07MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<17:36, 731kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<14:18, 900kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<10:30, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:21<07:47, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<13:27, 951kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<13:22, 958kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:23<10:13, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<07:37, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<05:48, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<08:34, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<13:56, 914kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<11:42, 1.09MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<08:46, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<06:33, 1.93MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:58, 2.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:25, 878kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:03, 901kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<10:47, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<07:59, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<05:59, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:19, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:24, 939kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<11:15, 1.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<08:24, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:17, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:45, 2.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<23:58, 523kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<20:24, 614kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<15:04, 830kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<10:59, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<07:58, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:44, 979kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<15:44, 792kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:43, 979kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<09:23, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:53, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:05, 1.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:27, 994kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<10:21, 1.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<07:42, 1.60MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:49, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<12:14, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:04, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:25, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:27, 2.25MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:19, 1.47MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:51, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:44, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<07:08, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:13, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:48, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:18, 989kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:58, 1.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:16, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:17, 2.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:42, 1.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:46, 948kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<10:09, 1.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:22, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<05:20, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<18:06, 665kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<16:57, 710kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<12:50, 938kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<09:16, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:19, 1.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:44, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:21, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:06, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:29, 2.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:10, 978kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:28, 954kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:41, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<07:02, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:06, 1.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:19, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:25, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:06, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:13, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:28, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:43, 2.48MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<03:30, 3.34MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:58:43, 17.8kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<7:44:23, 25.2kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<5:25:31, 35.9kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<3:47:25, 51.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<2:39:04, 73.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<7:18:27, 26.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<5:09:42, 37.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<3:37:19, 53.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<2:31:49, 76.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<1:50:50, 104kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<1:20:16, 144kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<56:42, 204kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<39:46, 290kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<31:45, 362kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<25:33, 450kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<18:39, 615kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<13:14, 865kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<13:32, 844kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<12:40, 901kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<09:36, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<06:54, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:25, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:23, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<07:10, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<05:09, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:21, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:56, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<07:00, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<05:03, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:58, 1.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:38, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:39, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:49, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:17, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:08, 1.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:27, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:42, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:37, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:55, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:11, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:30, 2.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:23, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:43, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:14<06:41, 1.64MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<04:49, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:13, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:27, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:42, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:41, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:10, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:25, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:40, 2.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:51, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:20, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:34, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:45, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:25, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:05, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<03:42, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<15:29, 688kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<13:10, 809kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:41, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:59, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:58, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:06, 1.49MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:20, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:41, 1.85MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:59, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<04:40, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:10, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:50, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:42, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:56, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:24, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:11, 2.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:03, 3.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<08:46, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<08:05, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:08, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:59, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:46, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:00, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:26, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:17, 2.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:40, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:06, 1.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:03, 2.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:29, 2.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:58, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:57, 2.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:25, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:58, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<03:56, 2.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:24, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:57, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<03:55, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:22, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:55, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:54, 2.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:20, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:53, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:52, 2.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:19, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:51, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:47, 2.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<02:45, 3.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<18:38, 517kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<14:51, 648kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<10:50, 887kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<09:08, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<08:12, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<06:11, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<04:33, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:38, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:10, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:49, 2.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:57<02:47, 3.38MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:19:46, 118kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<57:41, 163kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<40:42, 231kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<28:47, 326kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<22:21, 419kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<17:59, 520kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<13:01, 717kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<09:22, 994kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:47, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:36, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:22, 1.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:42, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:33, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:11, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:45, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<03:24, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<1:16:19, 120kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<55:34, 165kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<39:15, 233kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<27:42, 329kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<21:30, 422kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<16:22, 554kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<13:13, 686kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<09:25, 960kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<08:42, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:48, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:48, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:11, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:25, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:36, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:23, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:35, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:19, 2.05MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:28, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:46, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:03, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:33, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:55, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:25, 2.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:28, 3.48MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<24:48, 347kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<19:00, 452kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<13:38, 629kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<09:40, 884kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<09:35, 889kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<08:21, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:11, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:25, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<07:33, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:53, 1.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:12, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:01, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:06, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:57, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:08, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:29, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:31, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:49, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:14, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:21, 2.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:41, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:07, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:12, 2.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:20, 3.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<07:06, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<06:27, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:08, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:52, 2.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<08:15, 974kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<11:34, 696kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<09:33, 841kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:00, 1.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:08, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:52, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:39, 826kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:34, 834kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:15, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:20, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:54, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:03, 2.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<13:02, 607kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<13:59, 566kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<11:00, 719kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<07:58, 989kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:46, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<04:18, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:02, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:23, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:41, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:15, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:09, 2.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<06:36, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<08:54, 872kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:23, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:28, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:01, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:23, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<08:02, 958kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:43, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:59, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:41, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:09, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<07:46, 983kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:30, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:48, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:31, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:41, 2.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<08:31, 887kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<08:03, 939kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:08, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:29, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:18, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<11:34, 648kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<12:11, 615kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<09:35, 781kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:59, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:02, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:18, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<08:28, 877kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:58, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<05:07, 1.44MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:46, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<05:13, 1.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:42, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:29, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<03:18, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<12:31, 582kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<12:47, 570kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<09:56, 732kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<07:11, 1.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:15, 1.38MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:49, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<10:00, 721kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<10:57, 660kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:38, 835kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:19, 1.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:35, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:45, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<07:58, 897kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:33, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:51, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<03:33, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:05, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:26, 953kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:10, 1.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:35, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<03:23, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:50, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:12, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:05, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:03, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:16, 3.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<10:26, 666kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<10:43, 648kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:25, 824kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:08, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:27, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:52, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<07:29, 918kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:09, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:31, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:34, 2.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<10:18, 660kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<11:00, 618kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<08:35, 793kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<06:13, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<04:30, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:56, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<07:52, 856kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:23, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:41, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<03:26, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:25, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<07:30, 888kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:07, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:31, 1.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<03:18, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:05, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:47, 971kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:34, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<04:06, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<03:01, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:16, 2.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<42:28, 154kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<32:58, 198kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<23:52, 273kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<16:53, 385kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<11:55, 544kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<11:41, 553kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<11:24, 567kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:37, 748kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<06:14, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:29, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:46, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<06:54, 926kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<05:32, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:02, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:55, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:50, 925kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:38, 828kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:01, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:23, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<03:11, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<09:32, 656kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<09:14, 677kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<07:06, 880kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:08, 1.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:42, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<08:22, 739kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<08:24, 736kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:29, 953kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:40, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<03:20, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<12:30, 489kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<14:46, 414kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<11:46, 520kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<10:23, 589kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<09:24, 650kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<08:35, 711kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<08:09, 749kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<07:29, 815kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<07:14, 842kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<07:04, 863kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<06:49, 894kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<06:40, 914kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<06:32, 931kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<06:04, 1.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<06:31, 933kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:54, 1.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<06:24, 949kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<06:14, 974kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<06:02, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<05:48, 1.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<05:38, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<05:31, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<05:21, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:14, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:09, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:01, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<04:56, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:52, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:45, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:41, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:34, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<04:30, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:26, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:20, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:16, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:13, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<04:08, 1.45MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<04:05, 1.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:59, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:58, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:52, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:51, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:47, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:44, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:40, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<03:23, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:28, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:55, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:45, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:24, 1.75MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:44, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:35, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:28, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<03:10, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<03:32, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:41<03:15, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:27, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:10, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:09, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:15, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:06, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<03:06, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<02:57, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:47, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<03:06, 1.90MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:59, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<02:50, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:42, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:34, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:18, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:29, 2.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:13, 2.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:25, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:08, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:18, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:02, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<02:05, 2.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<01:51, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<01:59, 2.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<01:48, 3.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<01:47, 3.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<01:38, 3.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:44<01:38, 3.54MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:44<01:30, 3.87MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<01:32, 3.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<01:23, 4.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<01:24, 4.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:38, 2.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<02:12, 2.62MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<01:52, 3.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<01:37, 3.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<01:26, 3.99MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<01:17, 4.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<05:30, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<06:40, 858kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<05:25, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<04:06, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<03:06, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:24, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<01:54, 2.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<23:09, 244kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<18:31, 305kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<13:26, 420kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<09:38, 583kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<06:55, 810kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<05:01, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<09:01, 619kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<08:16, 675kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:51<06:10, 902kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<04:29, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:19, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:28, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<05:25, 1.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<05:23, 1.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<04:06, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<03:01, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:14, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:59, 909kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<07:09, 760kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<05:42, 952kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<04:10, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:02, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<04:15, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<03:16, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:23, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<03:16, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<04:29, 1.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<03:35, 1.48MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:39, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<01:57, 2.69MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<04:26, 1.18MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<05:07, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<03:59, 1.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:55, 1.78MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:09, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<04:19, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<04:38, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<03:34, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<02:35, 1.97MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<01:54, 2.67MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<05:29, 927kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<05:26, 936kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:08, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<03:00, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:11, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<04:33, 1.10MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<05:28, 917kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<04:19, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<03:09, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:16, 2.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<08:00, 619kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<07:39, 648kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<05:51, 844kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<04:12, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<04:00, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<04:41, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:42, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<01:59, 2.43MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<03:31, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<04:19, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:12<03:26, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:30, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<01:51, 2.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<03:28, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<04:15, 1.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<03:23, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<02:29, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:16<03:37, 1.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<02:52, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:05, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<01:34, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<03:38, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<04:09, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<03:18, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:24, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:46, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:33, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:52, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:06, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:33, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<03:22, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:44, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:00, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<01:32, 2.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<04:22, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<05:13, 842kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<04:14, 1.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<03:04, 1.42MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:13, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:41, 2.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<32:42, 133kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<25:16, 171kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<18:09, 238kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<12:47, 337kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<08:58, 477kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<08:26, 506kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<08:00, 533kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<06:08, 694kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<04:24, 960kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:09, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<04:49, 869kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<05:27, 769kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<04:19, 968kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:09, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:16, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<04:20, 953kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<04:53, 844kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<03:54, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:51, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:04, 1.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<04:34, 889kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<05:13, 778kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<04:07, 982kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:00, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:10, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<04:36, 868kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<05:01, 794kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<03:57, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:51, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:04, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<04:56, 795kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<05:14, 749kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<04:06, 954kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:58, 1.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:09, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<04:51, 792kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<05:11, 743kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<04:03, 946kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:56, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:05, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<04:38, 814kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<04:59, 758kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<03:55, 961kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:51, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:03, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<04:50, 767kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<05:03, 735kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<03:56, 940kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:50, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:02, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<03:44, 975kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<04:16, 854kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<03:23, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:28, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:47, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<04:39, 768kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<04:53, 732kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<03:49, 935kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:45, 1.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:59, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<04:40, 750kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<04:51, 722kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<03:47, 925kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:44, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:58, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<04:43, 728kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<04:52, 706kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<03:47, 905kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:44, 1.24MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<01:57, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<04:40, 720kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<04:47, 703kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<03:43, 903kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:41, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<04:37, 714kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<04:43, 698kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<03:40, 895kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:38, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:53, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:58<03:34, 906kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:58<03:56, 820kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<03:06, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:14, 1.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:36, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<03:20, 946kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<03:45, 842kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:00<02:57, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<02:09, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:02<04:28, 691kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:02<04:31, 684kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<03:29, 885kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<02:31, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:52, 1.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<02:15, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<02:55, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:04<02:22, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:44, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:15, 2.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<04:46, 619kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<04:32, 650kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<03:29, 845kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:29, 1.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<05:00, 576kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<04:40, 616kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<03:33, 808kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:33, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:49, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<06:03, 464kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<05:23, 522kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<04:03, 693kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<02:52, 967kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:02, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<03:42, 741kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<03:38, 756kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:48, 976kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:01, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:26, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<07:50, 342kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<06:34, 407kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<04:51, 550kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<03:26, 768kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:25, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<19:01, 137kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<14:17, 182kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<10:11, 255kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<07:08, 361kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<04:58, 512kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<08:38, 294kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<06:56, 366kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<05:04, 498kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<03:34, 701kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<03:02, 811kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<03:04, 805kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<02:22, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:41, 1.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:14, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<02:23, 1.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<02:33, 940kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<02:01, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:26, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:03, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<02:32, 917kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:57, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:24, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:02, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<02:41, 843kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<02:38, 854kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<02:01, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:27, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:35, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:48, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:23, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:00, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:13, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:24, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:05, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:30<00:47, 2.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:32, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:53, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:13, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:21, 1.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:45, 2.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:02, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:09, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:54, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:39, 2.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:59, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:06, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:52, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:37, 2.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:26, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:25, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:46, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:24, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:19, 1.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:59, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:03, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:06, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:50, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:36, 2.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:57, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:03, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:35, 2.59MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:16, 1.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<01:09, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:51, 1.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:36, 2.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:02, 1.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<01:01, 1.39MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:46, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:32, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<02:55, 467kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<02:15, 601kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:37, 830kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<01:19, 981kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:11, 1.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:37, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<02:06, 581kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:55<01:44, 704kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:15, 955kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:51, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<04:04, 284kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<03:05, 373kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<02:11, 521kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:30, 736kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:33, 700kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:19, 824kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:57, 1.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:39, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:47, 570kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<01:28, 693kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<01:03, 949kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:43, 1.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:54, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:03<00:46, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:33, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:32, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:05<00:32, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:24, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:07<00:24, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:26, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:20, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:20, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:22, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:17, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:18, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:20, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:15, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:16, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:18, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:09, 3.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:15<00:25, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:23, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:17<00:16, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:08, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:17, 1.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:19<00:16, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:12, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:10, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:08, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<00:07, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:08, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:06, 2.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:05, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:04, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:03, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:01, 1.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:00, 2.51MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<20:08:31,  5.52it/s]  0%|          | 740/400000 [00:00<14:04:40,  7.88it/s]  0%|          | 1436/400000 [00:00<9:50:31, 11.25it/s]  1%|          | 2129/400000 [00:00<6:52:56, 16.06it/s]  1%|          | 2859/400000 [00:00<4:48:48, 22.92it/s]  1%|          | 3600/400000 [00:00<3:22:03, 32.70it/s]  1%|          | 4342/400000 [00:00<2:21:26, 46.62it/s]  1%|▏         | 5101/400000 [00:00<1:39:04, 66.43it/s]  1%|▏         | 5780/400000 [00:00<1:09:31, 94.50it/s]  2%|▏         | 6532/400000 [00:01<48:50, 134.28it/s]   2%|▏         | 7273/400000 [00:01<34:23, 190.35it/s]  2%|▏         | 7993/400000 [00:01<24:17, 268.88it/s]  2%|▏         | 8708/400000 [00:01<17:15, 378.01it/s]  2%|▏         | 9435/400000 [00:01<12:19, 528.23it/s]  3%|▎         | 10182/400000 [00:01<08:52, 732.41it/s]  3%|▎         | 10918/400000 [00:01<06:27, 1003.46it/s]  3%|▎         | 11681/400000 [00:01<04:46, 1356.95it/s]  3%|▎         | 12441/400000 [00:01<03:35, 1800.68it/s]  3%|▎         | 13187/400000 [00:01<02:46, 2324.02it/s]  3%|▎         | 13926/400000 [00:02<02:12, 2924.42it/s]  4%|▎         | 14664/400000 [00:02<01:48, 3551.00it/s]  4%|▍         | 15440/400000 [00:02<01:30, 4240.70it/s]  4%|▍         | 16214/400000 [00:02<01:18, 4905.12it/s]  4%|▍         | 16967/400000 [00:02<01:10, 5430.84it/s]  4%|▍         | 17711/400000 [00:02<01:04, 5905.78it/s]  5%|▍         | 18464/400000 [00:02<01:00, 6311.81it/s]  5%|▍         | 19210/400000 [00:02<00:57, 6579.14it/s]  5%|▍         | 19952/400000 [00:02<00:55, 6798.19it/s]  5%|▌         | 20692/400000 [00:03<00:54, 6910.64it/s]  5%|▌         | 21431/400000 [00:03<00:53, 7045.92it/s]  6%|▌         | 22166/400000 [00:03<00:52, 7130.26it/s]  6%|▌         | 22901/400000 [00:03<00:52, 7169.58it/s]  6%|▌         | 23634/400000 [00:03<00:52, 7210.91it/s]  6%|▌         | 24366/400000 [00:03<00:51, 7239.45it/s]  6%|▋         | 25109/400000 [00:03<00:51, 7295.46it/s]  6%|▋         | 25867/400000 [00:03<00:50, 7378.36it/s]  7%|▋         | 26609/400000 [00:03<00:50, 7371.56it/s]  7%|▋         | 27356/400000 [00:03<00:50, 7400.60it/s]  7%|▋         | 28099/400000 [00:04<00:51, 7274.13it/s]  7%|▋         | 28829/400000 [00:04<00:51, 7166.08it/s]  7%|▋         | 29591/400000 [00:04<00:50, 7294.51it/s]  8%|▊         | 30323/400000 [00:04<00:51, 7228.54it/s]  8%|▊         | 31051/400000 [00:04<00:50, 7241.52it/s]  8%|▊         | 31777/400000 [00:04<00:50, 7226.73it/s]  8%|▊         | 32508/400000 [00:04<00:50, 7249.43it/s]  8%|▊         | 33282/400000 [00:04<00:49, 7387.90it/s]  9%|▊         | 34046/400000 [00:04<00:49, 7460.77it/s]  9%|▊         | 34793/400000 [00:04<00:49, 7384.20it/s]  9%|▉         | 35533/400000 [00:05<00:49, 7306.70it/s]  9%|▉         | 36282/400000 [00:05<00:49, 7358.60it/s]  9%|▉         | 37031/400000 [00:05<00:49, 7395.14it/s]  9%|▉         | 37794/400000 [00:05<00:48, 7462.69it/s] 10%|▉         | 38541/400000 [00:05<00:48, 7454.10it/s] 10%|▉         | 39287/400000 [00:05<00:49, 7326.94it/s] 10%|█         | 40021/400000 [00:05<00:49, 7229.14it/s] 10%|█         | 40745/400000 [00:05<00:49, 7210.38it/s] 10%|█         | 41489/400000 [00:05<00:49, 7276.15it/s] 11%|█         | 42244/400000 [00:05<00:48, 7353.68it/s] 11%|█         | 42982/400000 [00:06<00:48, 7359.77it/s] 11%|█         | 43724/400000 [00:06<00:48, 7376.99it/s] 11%|█         | 44482/400000 [00:06<00:47, 7436.45it/s] 11%|█▏        | 45226/400000 [00:06<00:48, 7341.82it/s] 11%|█▏        | 45992/400000 [00:06<00:47, 7431.16it/s] 12%|█▏        | 46736/400000 [00:06<00:47, 7377.28it/s] 12%|█▏        | 47484/400000 [00:06<00:47, 7405.01it/s] 12%|█▏        | 48243/400000 [00:06<00:47, 7458.57it/s] 12%|█▏        | 48994/400000 [00:06<00:46, 7471.12it/s] 12%|█▏        | 49742/400000 [00:06<00:47, 7404.04it/s] 13%|█▎        | 50483/400000 [00:07<00:47, 7362.20it/s] 13%|█▎        | 51239/400000 [00:07<00:47, 7420.41it/s] 13%|█▎        | 52004/400000 [00:07<00:46, 7486.52it/s] 13%|█▎        | 52754/400000 [00:07<00:46, 7462.14it/s] 13%|█▎        | 53501/400000 [00:07<00:46, 7456.04it/s] 14%|█▎        | 54254/400000 [00:07<00:46, 7474.95it/s] 14%|█▍        | 55005/400000 [00:07<00:46, 7484.15it/s] 14%|█▍        | 55778/400000 [00:07<00:45, 7554.91it/s] 14%|█▍        | 56534/400000 [00:07<00:45, 7505.63it/s] 14%|█▍        | 57285/400000 [00:07<00:46, 7449.68it/s] 15%|█▍        | 58031/400000 [00:08<00:46, 7365.08it/s] 15%|█▍        | 58778/400000 [00:08<00:46, 7395.03it/s] 15%|█▍        | 59552/400000 [00:08<00:45, 7493.16it/s] 15%|█▌        | 60325/400000 [00:08<00:44, 7559.83it/s] 15%|█▌        | 61082/400000 [00:08<00:44, 7551.78it/s] 15%|█▌        | 61838/400000 [00:08<00:45, 7476.39it/s] 16%|█▌        | 62587/400000 [00:08<00:46, 7317.14it/s] 16%|█▌        | 63340/400000 [00:08<00:45, 7378.39it/s] 16%|█▌        | 64085/400000 [00:08<00:45, 7399.23it/s] 16%|█▌        | 64826/400000 [00:08<00:45, 7393.39it/s] 16%|█▋        | 65566/400000 [00:09<00:45, 7391.56it/s] 17%|█▋        | 66306/400000 [00:09<00:45, 7381.23it/s] 17%|█▋        | 67045/400000 [00:09<00:45, 7288.12it/s] 17%|█▋        | 67795/400000 [00:09<00:45, 7350.11it/s] 17%|█▋        | 68539/400000 [00:09<00:44, 7375.84it/s] 17%|█▋        | 69278/400000 [00:09<00:44, 7378.54it/s] 18%|█▊        | 70017/400000 [00:09<00:44, 7343.32it/s] 18%|█▊        | 70759/400000 [00:09<00:44, 7365.89it/s] 18%|█▊        | 71498/400000 [00:09<00:44, 7369.93it/s] 18%|█▊        | 72262/400000 [00:09<00:44, 7446.61it/s] 18%|█▊        | 73020/400000 [00:10<00:43, 7483.44it/s] 18%|█▊        | 73769/400000 [00:10<00:44, 7349.03it/s] 19%|█▊        | 74537/400000 [00:10<00:43, 7443.37it/s] 19%|█▉        | 75283/400000 [00:10<00:43, 7412.19it/s] 19%|█▉        | 76028/400000 [00:10<00:43, 7422.24it/s] 19%|█▉        | 76778/400000 [00:10<00:43, 7445.36it/s] 19%|█▉        | 77527/400000 [00:10<00:43, 7457.10it/s] 20%|█▉        | 78276/400000 [00:10<00:43, 7466.78it/s] 20%|█▉        | 79025/400000 [00:10<00:42, 7471.94it/s] 20%|█▉        | 79778/400000 [00:10<00:42, 7486.89it/s] 20%|██        | 80552/400000 [00:11<00:42, 7559.32it/s] 20%|██        | 81309/400000 [00:11<00:42, 7550.32it/s] 21%|██        | 82069/400000 [00:11<00:42, 7562.86it/s] 21%|██        | 82826/400000 [00:11<00:42, 7487.84it/s] 21%|██        | 83584/400000 [00:11<00:42, 7512.84it/s] 21%|██        | 84336/400000 [00:11<00:42, 7441.35it/s] 21%|██▏       | 85081/400000 [00:11<00:42, 7334.14it/s] 21%|██▏       | 85815/400000 [00:11<00:43, 7272.61it/s] 22%|██▏       | 86543/400000 [00:11<00:43, 7242.36it/s] 22%|██▏       | 87278/400000 [00:12<00:42, 7273.69it/s] 22%|██▏       | 88051/400000 [00:12<00:42, 7402.97it/s] 22%|██▏       | 88793/400000 [00:12<00:43, 7193.95it/s] 22%|██▏       | 89547/400000 [00:12<00:42, 7293.94it/s] 23%|██▎       | 90279/400000 [00:12<00:42, 7296.96it/s] 23%|██▎       | 91023/400000 [00:12<00:42, 7337.30it/s] 23%|██▎       | 91771/400000 [00:12<00:41, 7377.15it/s] 23%|██▎       | 92510/400000 [00:12<00:41, 7331.75it/s] 23%|██▎       | 93244/400000 [00:12<00:41, 7322.53it/s] 23%|██▎       | 93977/400000 [00:12<00:42, 7281.05it/s] 24%|██▎       | 94717/400000 [00:13<00:41, 7315.01it/s] 24%|██▍       | 95449/400000 [00:13<00:41, 7310.70it/s] 24%|██▍       | 96181/400000 [00:13<00:41, 7269.28it/s] 24%|██▍       | 96909/400000 [00:13<00:43, 7008.01it/s] 24%|██▍       | 97629/400000 [00:13<00:42, 7062.35it/s] 25%|██▍       | 98357/400000 [00:13<00:42, 7124.19it/s] 25%|██▍       | 99094/400000 [00:13<00:41, 7194.89it/s] 25%|██▍       | 99836/400000 [00:13<00:41, 7260.45it/s] 25%|██▌       | 100588/400000 [00:13<00:40, 7335.01it/s] 25%|██▌       | 101323/400000 [00:13<00:40, 7321.56it/s] 26%|██▌       | 102056/400000 [00:14<00:40, 7313.56it/s] 26%|██▌       | 102796/400000 [00:14<00:40, 7337.65it/s] 26%|██▌       | 103532/400000 [00:14<00:40, 7342.54it/s] 26%|██▌       | 104286/400000 [00:14<00:39, 7399.19it/s] 26%|██▋       | 105059/400000 [00:14<00:39, 7495.24it/s] 26%|██▋       | 105815/400000 [00:14<00:39, 7511.28it/s] 27%|██▋       | 106567/400000 [00:14<00:39, 7507.69it/s] 27%|██▋       | 107319/400000 [00:14<00:39, 7489.35it/s] 27%|██▋       | 108087/400000 [00:14<00:38, 7542.51it/s] 27%|██▋       | 108854/400000 [00:14<00:38, 7579.24it/s] 27%|██▋       | 109613/400000 [00:15<00:38, 7564.67it/s] 28%|██▊       | 110370/400000 [00:15<00:38, 7516.36it/s] 28%|██▊       | 111122/400000 [00:15<00:38, 7486.20it/s] 28%|██▊       | 111898/400000 [00:15<00:38, 7565.48it/s] 28%|██▊       | 112664/400000 [00:15<00:37, 7591.25it/s] 28%|██▊       | 113424/400000 [00:15<00:39, 7226.09it/s] 29%|██▊       | 114155/400000 [00:15<00:39, 7250.66it/s] 29%|██▊       | 114897/400000 [00:15<00:39, 7298.46it/s] 29%|██▉       | 115677/400000 [00:15<00:38, 7440.25it/s] 29%|██▉       | 116424/400000 [00:15<00:38, 7427.40it/s] 29%|██▉       | 117186/400000 [00:16<00:37, 7482.58it/s] 29%|██▉       | 117936/400000 [00:16<00:37, 7486.99it/s] 30%|██▉       | 118686/400000 [00:16<00:38, 7347.38it/s] 30%|██▉       | 119422/400000 [00:16<00:38, 7245.61it/s] 30%|███       | 120149/400000 [00:16<00:38, 7252.05it/s] 30%|███       | 120910/400000 [00:16<00:37, 7355.25it/s] 30%|███       | 121647/400000 [00:16<00:37, 7351.82it/s] 31%|███       | 122384/400000 [00:16<00:37, 7356.86it/s] 31%|███       | 123121/400000 [00:16<00:38, 7175.47it/s] 31%|███       | 123883/400000 [00:16<00:37, 7302.34it/s] 31%|███       | 124646/400000 [00:17<00:37, 7396.75it/s] 31%|███▏      | 125387/400000 [00:17<00:37, 7347.71it/s] 32%|███▏      | 126134/400000 [00:17<00:37, 7382.35it/s] 32%|███▏      | 126873/400000 [00:17<00:37, 7336.67it/s] 32%|███▏      | 127608/400000 [00:17<00:37, 7324.34it/s] 32%|███▏      | 128373/400000 [00:17<00:36, 7417.34it/s] 32%|███▏      | 129126/400000 [00:17<00:36, 7449.32it/s] 32%|███▏      | 129872/400000 [00:17<00:36, 7407.26it/s] 33%|███▎      | 130618/400000 [00:17<00:36, 7420.20it/s] 33%|███▎      | 131361/400000 [00:17<00:36, 7407.61it/s] 33%|███▎      | 132102/400000 [00:18<00:36, 7309.86it/s] 33%|███▎      | 132848/400000 [00:18<00:36, 7351.50it/s] 33%|███▎      | 133584/400000 [00:18<00:36, 7315.53it/s] 34%|███▎      | 134348/400000 [00:18<00:35, 7409.52it/s] 34%|███▍      | 135116/400000 [00:18<00:35, 7487.01it/s] 34%|███▍      | 135866/400000 [00:18<00:35, 7376.33it/s] 34%|███▍      | 136605/400000 [00:18<00:36, 7191.77it/s] 34%|███▍      | 137326/400000 [00:18<00:37, 7006.83it/s] 35%|███▍      | 138080/400000 [00:18<00:36, 7158.18it/s] 35%|███▍      | 138847/400000 [00:19<00:35, 7302.18it/s] 35%|███▍      | 139590/400000 [00:19<00:35, 7338.22it/s] 35%|███▌      | 140326/400000 [00:19<00:35, 7303.39it/s] 35%|███▌      | 141058/400000 [00:19<00:36, 7126.73it/s] 35%|███▌      | 141780/400000 [00:19<00:36, 7152.51it/s] 36%|███▌      | 142500/400000 [00:19<00:35, 7164.19it/s] 36%|███▌      | 143228/400000 [00:19<00:35, 7196.18it/s] 36%|███▌      | 143949/400000 [00:19<00:35, 7142.58it/s] 36%|███▌      | 144664/400000 [00:19<00:36, 6964.05it/s] 36%|███▋      | 145408/400000 [00:19<00:35, 7098.01it/s] 37%|███▋      | 146160/400000 [00:20<00:35, 7217.72it/s] 37%|███▋      | 146917/400000 [00:20<00:34, 7319.28it/s] 37%|███▋      | 147676/400000 [00:20<00:34, 7397.45it/s] 37%|███▋      | 148420/400000 [00:20<00:33, 7407.88it/s] 37%|███▋      | 149162/400000 [00:20<00:34, 7331.57it/s] 37%|███▋      | 149935/400000 [00:20<00:33, 7446.45it/s] 38%|███▊      | 150700/400000 [00:20<00:33, 7502.46it/s] 38%|███▊      | 151452/400000 [00:20<00:33, 7475.84it/s] 38%|███▊      | 152201/400000 [00:20<00:33, 7402.04it/s] 38%|███▊      | 152942/400000 [00:20<00:33, 7394.18it/s] 38%|███▊      | 153716/400000 [00:21<00:32, 7494.47it/s] 39%|███▊      | 154467/400000 [00:21<00:32, 7460.79it/s] 39%|███▉      | 155227/400000 [00:21<00:32, 7501.40it/s] 39%|███▉      | 155978/400000 [00:21<00:33, 7388.35it/s] 39%|███▉      | 156721/400000 [00:21<00:32, 7399.81it/s] 39%|███▉      | 157462/400000 [00:21<00:33, 7329.38it/s] 40%|███▉      | 158209/400000 [00:21<00:32, 7367.84it/s] 40%|███▉      | 158953/400000 [00:21<00:32, 7388.98it/s] 40%|███▉      | 159693/400000 [00:21<00:32, 7372.38it/s] 40%|████      | 160431/400000 [00:21<00:32, 7319.65it/s] 40%|████      | 161164/400000 [00:22<00:32, 7315.87it/s] 40%|████      | 161896/400000 [00:22<00:33, 7143.07it/s] 41%|████      | 162626/400000 [00:22<00:33, 7187.96it/s] 41%|████      | 163346/400000 [00:22<00:33, 7121.34it/s] 41%|████      | 164059/400000 [00:22<00:33, 7062.27it/s] 41%|████      | 164809/400000 [00:22<00:32, 7185.83it/s] 41%|████▏     | 165582/400000 [00:22<00:31, 7337.69it/s] 42%|████▏     | 166318/400000 [00:22<00:32, 7239.63it/s] 42%|████▏     | 167044/400000 [00:22<00:32, 7244.71it/s] 42%|████▏     | 167807/400000 [00:22<00:31, 7353.72it/s] 42%|████▏     | 168575/400000 [00:23<00:31, 7446.35it/s] 42%|████▏     | 169330/400000 [00:23<00:30, 7477.00it/s] 43%|████▎     | 170079/400000 [00:23<00:31, 7389.30it/s] 43%|████▎     | 170833/400000 [00:23<00:30, 7432.80it/s] 43%|████▎     | 171595/400000 [00:23<00:30, 7485.88it/s] 43%|████▎     | 172345/400000 [00:23<00:30, 7402.84it/s] 43%|████▎     | 173104/400000 [00:23<00:30, 7456.58it/s] 43%|████▎     | 173862/400000 [00:23<00:30, 7491.93it/s] 44%|████▎     | 174612/400000 [00:23<00:30, 7389.61it/s] 44%|████▍     | 175370/400000 [00:24<00:30, 7445.47it/s] 44%|████▍     | 176116/400000 [00:24<00:30, 7420.84it/s] 44%|████▍     | 176859/400000 [00:24<00:30, 7368.74it/s] 44%|████▍     | 177606/400000 [00:24<00:30, 7396.13it/s] 45%|████▍     | 178349/400000 [00:24<00:29, 7404.88it/s] 45%|████▍     | 179090/400000 [00:24<00:30, 7329.14it/s] 45%|████▍     | 179824/400000 [00:24<00:30, 7320.33it/s] 45%|████▌     | 180557/400000 [00:24<00:29, 7316.57it/s] 45%|████▌     | 181289/400000 [00:24<00:29, 7315.21it/s] 46%|████▌     | 182037/400000 [00:24<00:29, 7362.78it/s] 46%|████▌     | 182774/400000 [00:25<00:29, 7337.06it/s] 46%|████▌     | 183508/400000 [00:25<00:29, 7283.95it/s] 46%|████▌     | 184237/400000 [00:25<00:29, 7282.53it/s] 46%|████▌     | 184974/400000 [00:25<00:29, 7307.39it/s] 46%|████▋     | 185718/400000 [00:25<00:29, 7345.00it/s] 47%|████▋     | 186453/400000 [00:25<00:29, 7336.36it/s] 47%|████▋     | 187187/400000 [00:25<00:29, 7106.60it/s] 47%|████▋     | 187909/400000 [00:25<00:29, 7138.29it/s] 47%|████▋     | 188649/400000 [00:25<00:29, 7213.86it/s] 47%|████▋     | 189372/400000 [00:25<00:29, 7200.41it/s] 48%|████▊     | 190093/400000 [00:26<00:29, 7200.46it/s] 48%|████▊     | 190814/400000 [00:26<00:29, 7162.29it/s] 48%|████▊     | 191531/400000 [00:26<00:29, 7149.33it/s] 48%|████▊     | 192247/400000 [00:26<00:30, 6887.80it/s] 48%|████▊     | 192939/400000 [00:26<00:30, 6889.74it/s] 48%|████▊     | 193681/400000 [00:26<00:29, 7038.26it/s] 49%|████▊     | 194402/400000 [00:26<00:29, 7086.30it/s] 49%|████▉     | 195153/400000 [00:26<00:28, 7208.03it/s] 49%|████▉     | 195876/400000 [00:26<00:28, 7205.35it/s] 49%|████▉     | 196615/400000 [00:26<00:28, 7257.96it/s] 49%|████▉     | 197377/400000 [00:27<00:27, 7362.02it/s] 50%|████▉     | 198115/400000 [00:27<00:27, 7330.39it/s] 50%|████▉     | 198849/400000 [00:27<00:27, 7286.59it/s] 50%|████▉     | 199579/400000 [00:27<00:27, 7281.64it/s] 50%|█████     | 200308/400000 [00:27<00:27, 7219.06it/s] 50%|█████     | 201092/400000 [00:27<00:26, 7393.21it/s] 50%|█████     | 201846/400000 [00:27<00:26, 7435.08it/s] 51%|█████     | 202635/400000 [00:27<00:26, 7563.36it/s] 51%|█████     | 203393/400000 [00:27<00:26, 7490.29it/s] 51%|█████     | 204144/400000 [00:27<00:26, 7457.46it/s] 51%|█████     | 204891/400000 [00:28<00:26, 7420.45it/s] 51%|█████▏    | 205634/400000 [00:28<00:26, 7413.99it/s] 52%|█████▏    | 206376/400000 [00:28<00:26, 7373.01it/s] 52%|█████▏    | 207115/400000 [00:28<00:26, 7375.69it/s] 52%|█████▏    | 207857/400000 [00:28<00:26, 7369.20it/s] 52%|█████▏    | 208595/400000 [00:28<00:26, 7338.65it/s] 52%|█████▏    | 209330/400000 [00:28<00:26, 7313.80it/s] 53%|█████▎    | 210062/400000 [00:28<00:26, 7304.98it/s] 53%|█████▎    | 210801/400000 [00:28<00:25, 7328.23it/s] 53%|█████▎    | 211534/400000 [00:28<00:25, 7323.23it/s] 53%|█████▎    | 212274/400000 [00:29<00:25, 7346.09it/s] 53%|█████▎    | 213009/400000 [00:29<00:25, 7276.66it/s] 53%|█████▎    | 213759/400000 [00:29<00:25, 7340.39it/s] 54%|█████▎    | 214494/400000 [00:29<00:25, 7315.70it/s] 54%|█████▍    | 215226/400000 [00:29<00:25, 7256.35it/s] 54%|█████▍    | 215983/400000 [00:29<00:25, 7344.70it/s] 54%|█████▍    | 216748/400000 [00:29<00:24, 7431.90it/s] 54%|█████▍    | 217492/400000 [00:29<00:24, 7413.34it/s] 55%|█████▍    | 218255/400000 [00:29<00:24, 7472.69it/s] 55%|█████▍    | 219003/400000 [00:29<00:24, 7397.16it/s] 55%|█████▍    | 219744/400000 [00:30<00:24, 7325.19it/s] 55%|█████▌    | 220479/400000 [00:30<00:24, 7331.85it/s] 55%|█████▌    | 221229/400000 [00:30<00:24, 7381.32it/s] 55%|█████▌    | 221968/400000 [00:30<00:24, 7299.96it/s] 56%|█████▌    | 222716/400000 [00:30<00:24, 7349.59it/s] 56%|█████▌    | 223469/400000 [00:30<00:23, 7401.39it/s] 56%|█████▌    | 224223/400000 [00:30<00:23, 7442.28it/s] 56%|█████▌    | 224983/400000 [00:30<00:23, 7488.78it/s] 56%|█████▋    | 225733/400000 [00:30<00:23, 7385.16it/s] 57%|█████▋    | 226486/400000 [00:30<00:23, 7425.95it/s] 57%|█████▋    | 227273/400000 [00:31<00:22, 7553.70it/s] 57%|█████▋    | 228066/400000 [00:31<00:22, 7661.84it/s] 57%|█████▋    | 228834/400000 [00:31<00:22, 7612.75it/s] 57%|█████▋    | 229597/400000 [00:31<00:22, 7612.43it/s] 58%|█████▊    | 230359/400000 [00:31<00:22, 7451.23it/s] 58%|█████▊    | 231106/400000 [00:31<00:22, 7449.29it/s] 58%|█████▊    | 231852/400000 [00:31<00:23, 7226.06it/s] 58%|█████▊    | 232652/400000 [00:31<00:22, 7440.65it/s] 58%|█████▊    | 233422/400000 [00:31<00:22, 7515.36it/s] 59%|█████▊    | 234176/400000 [00:32<00:22, 7351.81it/s] 59%|█████▊    | 234915/400000 [00:32<00:22, 7361.61it/s] 59%|█████▉    | 235658/400000 [00:32<00:22, 7381.63it/s] 59%|█████▉    | 236398/400000 [00:32<00:22, 7346.51it/s] 59%|█████▉    | 237134/400000 [00:32<00:22, 7102.71it/s] 59%|█████▉    | 237847/400000 [00:32<00:22, 7092.57it/s] 60%|█████▉    | 238568/400000 [00:32<00:22, 7125.09it/s] 60%|█████▉    | 239327/400000 [00:32<00:22, 7255.99it/s] 60%|██████    | 240086/400000 [00:32<00:21, 7352.58it/s] 60%|██████    | 240823/400000 [00:32<00:21, 7320.98it/s] 60%|██████    | 241557/400000 [00:33<00:22, 7185.12it/s] 61%|██████    | 242298/400000 [00:33<00:21, 7248.47it/s] 61%|██████    | 243024/400000 [00:33<00:21, 7241.67it/s] 61%|██████    | 243783/400000 [00:33<00:21, 7340.83it/s] 61%|██████    | 244526/400000 [00:33<00:21, 7365.50it/s] 61%|██████▏   | 245267/400000 [00:33<00:20, 7378.67it/s] 62%|██████▏   | 246029/400000 [00:33<00:20, 7447.47it/s] 62%|██████▏   | 246793/400000 [00:33<00:20, 7503.56it/s] 62%|██████▏   | 247544/400000 [00:33<00:20, 7426.60it/s] 62%|██████▏   | 248288/400000 [00:33<00:20, 7267.43it/s] 62%|██████▏   | 249016/400000 [00:34<00:20, 7265.68it/s] 62%|██████▏   | 249744/400000 [00:34<00:20, 7264.49it/s] 63%|██████▎   | 250482/400000 [00:34<00:20, 7297.55it/s] 63%|██████▎   | 251228/400000 [00:34<00:20, 7345.18it/s] 63%|██████▎   | 251963/400000 [00:34<00:20, 7343.71it/s] 63%|██████▎   | 252698/400000 [00:34<00:20, 7276.76it/s] 63%|██████▎   | 253426/400000 [00:34<00:20, 7232.26it/s] 64%|██████▎   | 254171/400000 [00:34<00:19, 7296.05it/s] 64%|██████▎   | 254953/400000 [00:34<00:19, 7444.23it/s] 64%|██████▍   | 255699/400000 [00:34<00:19, 7322.47it/s] 64%|██████▍   | 256433/400000 [00:35<00:19, 7260.79it/s] 64%|██████▍   | 257161/400000 [00:35<00:19, 7266.10it/s] 64%|██████▍   | 257889/400000 [00:35<00:19, 7262.69it/s] 65%|██████▍   | 258619/400000 [00:35<00:19, 7248.35it/s] 65%|██████▍   | 259353/400000 [00:35<00:19, 7272.75it/s] 65%|██████▌   | 260081/400000 [00:35<00:19, 7248.56it/s] 65%|██████▌   | 260834/400000 [00:35<00:18, 7329.00it/s] 65%|██████▌   | 261585/400000 [00:35<00:18, 7380.47it/s] 66%|██████▌   | 262347/400000 [00:35<00:18, 7450.59it/s] 66%|██████▌   | 263093/400000 [00:35<00:18, 7307.59it/s] 66%|██████▌   | 263834/400000 [00:36<00:18, 7335.31it/s] 66%|██████▌   | 264580/400000 [00:36<00:18, 7371.22it/s] 66%|██████▋   | 265342/400000 [00:36<00:18, 7442.17it/s] 67%|██████▋   | 266087/400000 [00:36<00:18, 7320.34it/s] 67%|██████▋   | 266831/400000 [00:36<00:18, 7355.75it/s] 67%|██████▋   | 267569/400000 [00:36<00:17, 7361.63it/s] 67%|██████▋   | 268318/400000 [00:36<00:17, 7399.24it/s] 67%|██████▋   | 269059/400000 [00:36<00:17, 7331.08it/s] 67%|██████▋   | 269815/400000 [00:36<00:17, 7398.24it/s] 68%|██████▊   | 270556/400000 [00:36<00:17, 7316.65it/s] 68%|██████▊   | 271300/400000 [00:37<00:17, 7352.60it/s] 68%|██████▊   | 272045/400000 [00:37<00:17, 7379.69it/s] 68%|██████▊   | 272784/400000 [00:37<00:17, 7365.60it/s] 68%|██████▊   | 273521/400000 [00:37<00:17, 7362.27it/s] 69%|██████▊   | 274258/400000 [00:37<00:17, 7359.33it/s] 69%|██████▊   | 274995/400000 [00:37<00:17, 7230.50it/s] 69%|██████▉   | 275736/400000 [00:37<00:17, 7281.78it/s] 69%|██████▉   | 276474/400000 [00:37<00:16, 7310.92it/s] 69%|██████▉   | 277206/400000 [00:37<00:17, 7209.80it/s] 69%|██████▉   | 277928/400000 [00:38<00:17, 7013.39it/s] 70%|██████▉   | 278653/400000 [00:38<00:17, 7082.51it/s] 70%|██████▉   | 279392/400000 [00:38<00:16, 7170.92it/s] 70%|███████   | 280147/400000 [00:38<00:16, 7278.44it/s] 70%|███████   | 280922/400000 [00:38<00:16, 7410.83it/s] 70%|███████   | 281665/400000 [00:38<00:16, 7374.18it/s] 71%|███████   | 282413/400000 [00:38<00:15, 7404.83it/s] 71%|███████   | 283170/400000 [00:38<00:15, 7452.18it/s] 71%|███████   | 283927/400000 [00:38<00:15, 7484.76it/s] 71%|███████   | 284676/400000 [00:38<00:15, 7485.45it/s] 71%|███████▏  | 285425/400000 [00:39<00:15, 7371.19it/s] 72%|███████▏  | 286163/400000 [00:39<00:15, 7200.54it/s] 72%|███████▏  | 286922/400000 [00:39<00:15, 7310.82it/s] 72%|███████▏  | 287681/400000 [00:39<00:15, 7390.26it/s] 72%|███████▏  | 288429/400000 [00:39<00:15, 7413.98it/s] 72%|███████▏  | 289182/400000 [00:39<00:14, 7445.34it/s] 72%|███████▏  | 289928/400000 [00:39<00:15, 7282.80it/s] 73%|███████▎  | 290691/400000 [00:39<00:14, 7382.98it/s] 73%|███████▎  | 291463/400000 [00:39<00:14, 7479.34it/s] 73%|███████▎  | 292229/400000 [00:39<00:14, 7532.22it/s] 73%|███████▎  | 292984/400000 [00:40<00:14, 7461.25it/s] 73%|███████▎  | 293731/400000 [00:40<00:14, 7344.68it/s] 74%|███████▎  | 294467/400000 [00:40<00:14, 7316.50it/s] 74%|███████▍  | 295225/400000 [00:40<00:14, 7393.04it/s] 74%|███████▍  | 296004/400000 [00:40<00:13, 7505.07it/s] 74%|███████▍  | 296756/400000 [00:40<00:13, 7460.82it/s] 74%|███████▍  | 297503/400000 [00:40<00:13, 7431.40it/s] 75%|███████▍  | 298247/400000 [00:40<00:14, 7169.67it/s] 75%|███████▍  | 298967/400000 [00:40<00:14, 7169.77it/s] 75%|███████▍  | 299706/400000 [00:40<00:13, 7231.82it/s] 75%|███████▌  | 300431/400000 [00:41<00:13, 7228.69it/s] 75%|███████▌  | 301155/400000 [00:41<00:13, 7205.06it/s] 75%|███████▌  | 301898/400000 [00:41<00:13, 7270.13it/s] 76%|███████▌  | 302626/400000 [00:41<00:13, 7218.47it/s] 76%|███████▌  | 303377/400000 [00:41<00:13, 7302.62it/s] 76%|███████▌  | 304118/400000 [00:41<00:13, 7332.81it/s] 76%|███████▌  | 304854/400000 [00:41<00:12, 7339.49it/s] 76%|███████▋  | 305591/400000 [00:41<00:12, 7347.50it/s] 77%|███████▋  | 306343/400000 [00:41<00:12, 7398.25it/s] 77%|███████▋  | 307084/400000 [00:41<00:12, 7286.76it/s] 77%|███████▋  | 307814/400000 [00:42<00:12, 7290.39it/s] 77%|███████▋  | 308544/400000 [00:42<00:12, 7156.09it/s] 77%|███████▋  | 309268/400000 [00:42<00:12, 7179.00it/s] 78%|███████▊  | 310051/400000 [00:42<00:12, 7362.25it/s] 78%|███████▊  | 310789/400000 [00:42<00:12, 7186.99it/s] 78%|███████▊  | 311516/400000 [00:42<00:12, 7210.09it/s] 78%|███████▊  | 312268/400000 [00:42<00:12, 7297.47it/s] 78%|███████▊  | 313017/400000 [00:42<00:11, 7351.41it/s] 78%|███████▊  | 313754/400000 [00:42<00:11, 7353.98it/s] 79%|███████▊  | 314491/400000 [00:42<00:11, 7293.41it/s] 79%|███████▉  | 315222/400000 [00:43<00:11, 7296.49it/s] 79%|███████▉  | 315953/400000 [00:43<00:11, 7192.24it/s] 79%|███████▉  | 316694/400000 [00:43<00:11, 7253.63it/s] 79%|███████▉  | 317420/400000 [00:43<00:11, 7245.89it/s] 80%|███████▉  | 318164/400000 [00:43<00:11, 7302.24it/s] 80%|███████▉  | 318895/400000 [00:43<00:11, 7286.64it/s] 80%|███████▉  | 319624/400000 [00:43<00:11, 7279.99it/s] 80%|████████  | 320398/400000 [00:43<00:10, 7408.96it/s] 80%|████████  | 321158/400000 [00:43<00:10, 7464.00it/s] 80%|████████  | 321905/400000 [00:44<00:10, 7434.14it/s] 81%|████████  | 322657/400000 [00:44<00:10, 7457.37it/s] 81%|████████  | 323404/400000 [00:44<00:10, 7315.29it/s] 81%|████████  | 324148/400000 [00:44<00:10, 7351.01it/s] 81%|████████  | 324923/400000 [00:44<00:10, 7465.73it/s] 81%|████████▏ | 325692/400000 [00:44<00:09, 7529.73it/s] 82%|████████▏ | 326455/400000 [00:44<00:09, 7557.94it/s] 82%|████████▏ | 327212/400000 [00:44<00:09, 7497.98it/s] 82%|████████▏ | 327963/400000 [00:44<00:09, 7465.62it/s] 82%|████████▏ | 328710/400000 [00:44<00:09, 7426.69it/s] 82%|████████▏ | 329453/400000 [00:45<00:09, 7353.86it/s] 83%|████████▎ | 330219/400000 [00:45<00:09, 7442.44it/s] 83%|████████▎ | 330964/400000 [00:45<00:09, 7406.99it/s] 83%|████████▎ | 331721/400000 [00:45<00:09, 7454.85it/s] 83%|████████▎ | 332467/400000 [00:45<00:09, 7446.95it/s] 83%|████████▎ | 333224/400000 [00:45<00:08, 7481.82it/s] 83%|████████▎ | 333988/400000 [00:45<00:08, 7526.50it/s] 84%|████████▎ | 334741/400000 [00:45<00:08, 7497.47it/s] 84%|████████▍ | 335491/400000 [00:45<00:08, 7477.83it/s] 84%|████████▍ | 336242/400000 [00:45<00:08, 7487.00it/s] 84%|████████▍ | 336991/400000 [00:46<00:08, 7392.83it/s] 84%|████████▍ | 337751/400000 [00:46<00:08, 7453.15it/s] 85%|████████▍ | 338497/400000 [00:46<00:08, 7396.60it/s] 85%|████████▍ | 339245/400000 [00:46<00:08, 7419.10it/s] 85%|████████▍ | 339988/400000 [00:46<00:08, 7404.10it/s] 85%|████████▌ | 340731/400000 [00:46<00:07, 7409.69it/s] 85%|████████▌ | 341473/400000 [00:46<00:08, 6893.82it/s] 86%|████████▌ | 342212/400000 [00:46<00:08, 7033.37it/s] 86%|████████▌ | 342922/400000 [00:46<00:08, 6911.28it/s] 86%|████████▌ | 343653/400000 [00:46<00:08, 7023.10it/s] 86%|████████▌ | 344364/400000 [00:47<00:07, 7046.07it/s] 86%|████████▋ | 345113/400000 [00:47<00:07, 7171.99it/s] 86%|████████▋ | 345833/400000 [00:47<00:07, 7170.21it/s] 87%|████████▋ | 346600/400000 [00:47<00:07, 7311.72it/s] 87%|████████▋ | 347351/400000 [00:47<00:07, 7370.00it/s] 87%|████████▋ | 348097/400000 [00:47<00:07, 7395.36it/s] 87%|████████▋ | 348838/400000 [00:47<00:06, 7350.69it/s] 87%|████████▋ | 349574/400000 [00:47<00:06, 7236.02it/s] 88%|████████▊ | 350320/400000 [00:47<00:06, 7300.85it/s] 88%|████████▊ | 351051/400000 [00:47<00:06, 7296.91it/s] 88%|████████▊ | 351791/400000 [00:48<00:06, 7323.36it/s] 88%|████████▊ | 352544/400000 [00:48<00:06, 7381.38it/s] 88%|████████▊ | 353283/400000 [00:48<00:06, 7339.67it/s] 89%|████████▊ | 354018/400000 [00:48<00:06, 7297.56it/s] 89%|████████▊ | 354759/400000 [00:48<00:06, 7328.67it/s] 89%|████████▉ | 355502/400000 [00:48<00:06, 7356.86it/s] 89%|████████▉ | 356247/400000 [00:48<00:05, 7383.25it/s] 89%|████████▉ | 356986/400000 [00:48<00:05, 7346.35it/s] 89%|████████▉ | 357721/400000 [00:48<00:05, 7345.97it/s] 90%|████████▉ | 358456/400000 [00:48<00:05, 7276.77it/s] 90%|████████▉ | 359184/400000 [00:49<00:05, 7245.47it/s] 90%|████████▉ | 359924/400000 [00:49<00:05, 7291.03it/s] 90%|█████████ | 360654/400000 [00:49<00:05, 7231.82it/s] 90%|█████████ | 361397/400000 [00:49<00:05, 7289.67it/s] 91%|█████████ | 362127/400000 [00:49<00:05, 7187.43it/s] 91%|█████████ | 362847/400000 [00:49<00:05, 7171.88it/s] 91%|█████████ | 363565/400000 [00:49<00:05, 7126.03it/s] 91%|█████████ | 364293/400000 [00:49<00:04, 7168.84it/s] 91%|█████████▏| 365011/400000 [00:49<00:04, 7088.84it/s] 91%|█████████▏| 365750/400000 [00:49<00:04, 7173.69it/s] 92%|█████████▏| 366468/400000 [00:50<00:04, 7120.65it/s] 92%|█████████▏| 367199/400000 [00:50<00:04, 7175.77it/s] 92%|█████████▏| 367918/400000 [00:50<00:04, 7068.43it/s] 92%|█████████▏| 368664/400000 [00:50<00:04, 7180.55it/s] 92%|█████████▏| 369389/400000 [00:50<00:04, 7198.61it/s] 93%|█████████▎| 370110/400000 [00:50<00:04, 7125.40it/s] 93%|█████████▎| 370824/400000 [00:50<00:04, 7101.75it/s] 93%|█████████▎| 371553/400000 [00:50<00:03, 7155.85it/s] 93%|█████████▎| 372314/400000 [00:50<00:03, 7284.71it/s] 93%|█████████▎| 373055/400000 [00:51<00:03, 7320.09it/s] 93%|█████████▎| 373795/400000 [00:51<00:03, 7342.01it/s] 94%|█████████▎| 374535/400000 [00:51<00:03, 7355.75it/s] 94%|█████████▍| 375271/400000 [00:51<00:03, 7225.29it/s] 94%|█████████▍| 376007/400000 [00:51<00:03, 7264.22it/s] 94%|█████████▍| 376745/400000 [00:51<00:03, 7296.85it/s] 94%|█████████▍| 377494/400000 [00:51<00:03, 7352.16it/s] 95%|█████████▍| 378230/400000 [00:51<00:02, 7328.62it/s] 95%|█████████▍| 378964/400000 [00:51<00:02, 7322.67it/s] 95%|█████████▍| 379697/400000 [00:51<00:02, 7253.38it/s] 95%|█████████▌| 380423/400000 [00:52<00:02, 7204.50it/s] 95%|█████████▌| 381199/400000 [00:52<00:02, 7361.58it/s] 95%|█████████▌| 381963/400000 [00:52<00:02, 7440.73it/s] 96%|█████████▌| 382717/400000 [00:52<00:02, 7470.05it/s] 96%|█████████▌| 383465/400000 [00:52<00:02, 7434.36it/s] 96%|█████████▌| 384209/400000 [00:52<00:02, 7288.69it/s] 96%|█████████▌| 384940/400000 [00:52<00:02, 7294.75it/s] 96%|█████████▋| 385702/400000 [00:52<00:01, 7389.10it/s] 97%|█████████▋| 386442/400000 [00:52<00:01, 7383.27it/s] 97%|█████████▋| 387181/400000 [00:52<00:01, 7382.09it/s] 97%|█████████▋| 387920/400000 [00:53<00:01, 7381.85it/s] 97%|█████████▋| 388665/400000 [00:53<00:01, 7400.66it/s] 97%|█████████▋| 389406/400000 [00:53<00:01, 7378.89it/s] 98%|█████████▊| 390145/400000 [00:53<00:01, 7334.08it/s] 98%|█████████▊| 390893/400000 [00:53<00:01, 7375.60it/s] 98%|█████████▊| 391657/400000 [00:53<00:01, 7451.40it/s] 98%|█████████▊| 392403/400000 [00:53<00:01, 7427.34it/s] 98%|█████████▊| 393156/400000 [00:53<00:00, 7455.60it/s] 98%|█████████▊| 393902/400000 [00:53<00:00, 7434.93it/s] 99%|█████████▊| 394652/400000 [00:53<00:00, 7452.82it/s] 99%|█████████▉| 395406/400000 [00:54<00:00, 7476.49it/s] 99%|█████████▉| 396154/400000 [00:54<00:00, 7440.10it/s] 99%|█████████▉| 396902/400000 [00:54<00:00, 7451.31it/s] 99%|█████████▉| 397648/400000 [00:54<00:00, 7451.37it/s]100%|█████████▉| 398421/400000 [00:54<00:00, 7531.26it/s]100%|█████████▉| 399177/400000 [00:54<00:00, 7538.49it/s]100%|█████████▉| 399932/400000 [00:54<00:00, 7498.86it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7318.81it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1b71826320>, <torchtext.data.dataset.TabularDataset object at 0x7f1b71826470>, <torchtext.vocab.Vocab object at 0x7f1b71826390>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 41.97 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 45.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.48 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.48 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.51 file/s]2020-06-27 00:21:56.313714: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-27 00:21:56.318547: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-27 00:21:56.318750: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556d1b6f2660 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-27 00:21:56.318769: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3719168/9912422 [00:00<00:00, 36846904.42it/s]9920512it [00:00, 34751938.82it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1679772.10it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475562.92it/s]1654784it [00:00, 11907969.42it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207162.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
