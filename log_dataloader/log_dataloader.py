
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3fc653a7999aa2761b089e07ab0db303a5051c0b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3fc653a7999aa2761b089e07ab0db303a5051c0b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3fc653a7999aa2761b089e07ab0db303a5051c0b

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f105b1456a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f105b1456a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f10c5e6e488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f10c5e6e488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f10e508de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f10e508de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f10731a6510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f10731a6510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f10731a6510> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2924544/9912422 [00:00<00:00, 29239472.62it/s]9920512it [00:00, 30190397.15it/s]                             
0it [00:00, ?it/s]32768it [00:00, 638956.73it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486208.95it/s]1654784it [00:00, 10986709.81it/s]                         
0it [00:00, ?it/s]8192it [00:00, 228788.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1071209198>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1071362a20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f10731a61e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f10731a61e0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f10731a61e0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:30,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:59,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:59,  2.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:40,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:39,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:39,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:38,  3.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:26,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:25,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:25,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:25,  4.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:17,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:11,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:10,  9.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:07, 12.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 17.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 23.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 30.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 55.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 64.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 64.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 73.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.64 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.64 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.40s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.41s/ url]
0 examples [00:00, ? examples/s]2020-09-03 12:08:39.693550: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-03 12:08:39.714939: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-03 12:08:39.716697: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bb728ff950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 12:08:39.716778: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.03 examples/s]107 examples [00:00,  4.33 examples/s]204 examples [00:00,  6.17 examples/s]301 examples [00:00,  8.79 examples/s]397 examples [00:00, 12.51 examples/s]486 examples [00:00, 17.76 examples/s]586 examples [00:00, 25.18 examples/s]687 examples [00:01, 35.58 examples/s]782 examples [00:01, 50.03 examples/s]883 examples [00:01, 69.98 examples/s]977 examples [00:01, 96.51 examples/s]1080 examples [00:01, 132.53 examples/s]1185 examples [00:01, 179.58 examples/s]1290 examples [00:01, 238.97 examples/s]1398 examples [00:01, 311.57 examples/s]1500 examples [00:01, 391.97 examples/s]1603 examples [00:01, 480.92 examples/s]1709 examples [00:02, 575.03 examples/s]1812 examples [00:02, 641.73 examples/s]1917 examples [00:02, 726.13 examples/s]2021 examples [00:02, 796.56 examples/s]2127 examples [00:02, 858.92 examples/s]2233 examples [00:02, 909.88 examples/s]2339 examples [00:02, 950.10 examples/s]2444 examples [00:02, 977.23 examples/s]2549 examples [00:02, 993.46 examples/s]2655 examples [00:02, 1012.13 examples/s]2760 examples [00:03, 1019.05 examples/s]2868 examples [00:03, 1034.13 examples/s]2974 examples [00:03, 1041.54 examples/s]3080 examples [00:03, 964.98 examples/s] 3187 examples [00:03, 992.99 examples/s]3291 examples [00:03, 1005.17 examples/s]3398 examples [00:03, 1023.44 examples/s]3502 examples [00:03, 1024.30 examples/s]3608 examples [00:03, 1032.32 examples/s]3714 examples [00:04, 1038.32 examples/s]3821 examples [00:04, 1046.34 examples/s]3926 examples [00:04, 1012.34 examples/s]4028 examples [00:04, 994.74 examples/s] 4130 examples [00:04, 1000.50 examples/s]4237 examples [00:04, 1018.76 examples/s]4342 examples [00:04, 1025.71 examples/s]4450 examples [00:04, 1040.63 examples/s]4559 examples [00:04, 1053.37 examples/s]4665 examples [00:04, 1052.49 examples/s]4772 examples [00:05, 1045.01 examples/s]4877 examples [00:05, 1038.21 examples/s]4981 examples [00:05, 1033.79 examples/s]5088 examples [00:05, 1044.18 examples/s]5193 examples [00:05, 1035.65 examples/s]5300 examples [00:05, 1044.69 examples/s]5405 examples [00:05, 1031.86 examples/s]5509 examples [00:05, 1028.64 examples/s]5612 examples [00:05, 994.76 examples/s] 5717 examples [00:05, 1009.65 examples/s]5825 examples [00:06, 1027.84 examples/s]5929 examples [00:06, 992.02 examples/s] 6033 examples [00:06, 1005.06 examples/s]6134 examples [00:06, 969.58 examples/s] 6238 examples [00:06, 988.62 examples/s]6338 examples [00:06, 968.38 examples/s]6438 examples [00:06, 976.17 examples/s]6543 examples [00:06, 993.92 examples/s]6643 examples [00:06, 940.72 examples/s]6748 examples [00:07, 970.09 examples/s]6854 examples [00:07, 993.36 examples/s]6958 examples [00:07, 1004.15 examples/s]7059 examples [00:07, 981.78 examples/s] 7158 examples [00:07, 970.28 examples/s]7256 examples [00:07, 953.34 examples/s]7362 examples [00:07, 980.41 examples/s]7463 examples [00:07, 986.73 examples/s]7570 examples [00:07, 1009.48 examples/s]7675 examples [00:07, 1021.17 examples/s]7781 examples [00:08, 1030.60 examples/s]7887 examples [00:08, 1037.41 examples/s]7992 examples [00:08, 1039.41 examples/s]8097 examples [00:08, 1031.71 examples/s]8201 examples [00:08, 1025.69 examples/s]8304 examples [00:08, 1013.30 examples/s]8406 examples [00:08, 1010.04 examples/s]8512 examples [00:08, 1023.61 examples/s]8616 examples [00:08, 1025.69 examples/s]8721 examples [00:08, 1030.45 examples/s]8825 examples [00:09, 993.23 examples/s] 8928 examples [00:09, 1003.74 examples/s]9029 examples [00:09, 985.99 examples/s] 9128 examples [00:09, 933.85 examples/s]9223 examples [00:09, 905.13 examples/s]9323 examples [00:09, 929.49 examples/s]9428 examples [00:09, 961.03 examples/s]9530 examples [00:09, 975.96 examples/s]9636 examples [00:09, 998.37 examples/s]9737 examples [00:10, 993.77 examples/s]9844 examples [00:10, 1013.30 examples/s]9950 examples [00:10, 1026.10 examples/s]10053 examples [00:10, 952.67 examples/s]10155 examples [00:10, 971.19 examples/s]10260 examples [00:10, 991.04 examples/s]10364 examples [00:10, 1004.67 examples/s]10469 examples [00:10, 1015.24 examples/s]10573 examples [00:10, 1020.62 examples/s]10681 examples [00:10, 1035.59 examples/s]10787 examples [00:11, 1042.67 examples/s]10892 examples [00:11, 1014.35 examples/s]10994 examples [00:11, 1005.67 examples/s]11097 examples [00:11, 1010.32 examples/s]11199 examples [00:11, 1009.98 examples/s]11304 examples [00:11, 1021.41 examples/s]11412 examples [00:11, 1036.10 examples/s]11516 examples [00:11, 1036.43 examples/s]11623 examples [00:11, 1044.54 examples/s]11729 examples [00:11, 1048.85 examples/s]11836 examples [00:12, 1054.25 examples/s]11942 examples [00:12, 1028.38 examples/s]12047 examples [00:12, 1032.24 examples/s]12151 examples [00:12, 1025.26 examples/s]12254 examples [00:12, 1021.87 examples/s]12357 examples [00:12, 1023.51 examples/s]12460 examples [00:12, 994.10 examples/s] 12560 examples [00:12, 984.57 examples/s]12665 examples [00:12, 1001.97 examples/s]12767 examples [00:12, 1004.20 examples/s]12868 examples [00:13, 1001.44 examples/s]12976 examples [00:13, 1022.54 examples/s]13084 examples [00:13, 1036.61 examples/s]13188 examples [00:13, 1008.93 examples/s]13294 examples [00:13, 1021.38 examples/s]13397 examples [00:13, 996.13 examples/s] 13504 examples [00:13, 1015.42 examples/s]13606 examples [00:13, 1013.68 examples/s]13708 examples [00:13, 1009.60 examples/s]13810 examples [00:14, 984.26 examples/s] 13909 examples [00:14, 965.47 examples/s]14019 examples [00:14, 999.97 examples/s]14128 examples [00:14, 1024.70 examples/s]14236 examples [00:14, 1040.09 examples/s]14341 examples [00:14, 1040.63 examples/s]14446 examples [00:14, 977.45 examples/s] 14545 examples [00:14, 937.46 examples/s]14652 examples [00:14, 971.67 examples/s]14760 examples [00:14, 999.80 examples/s]14864 examples [00:15, 1009.10 examples/s]14966 examples [00:15, 995.83 examples/s] 15074 examples [00:15, 1018.40 examples/s]15181 examples [00:15, 1016.46 examples/s]15286 examples [00:15, 1023.87 examples/s]15391 examples [00:15, 1029.92 examples/s]15499 examples [00:15, 1043.32 examples/s]15604 examples [00:15, 1040.18 examples/s]15712 examples [00:15, 1051.20 examples/s]15821 examples [00:15, 1061.26 examples/s]15928 examples [00:16, 1026.84 examples/s]16031 examples [00:16, 991.30 examples/s] 16131 examples [00:16, 978.02 examples/s]16236 examples [00:16, 996.68 examples/s]16342 examples [00:16, 1014.85 examples/s]16444 examples [00:16, 971.10 examples/s] 16546 examples [00:16, 983.46 examples/s]16645 examples [00:16, 984.96 examples/s]16750 examples [00:16, 1001.94 examples/s]16857 examples [00:17, 1018.81 examples/s]16960 examples [00:17, 985.82 examples/s] 17064 examples [00:17, 1000.61 examples/s]17165 examples [00:17, 969.74 examples/s] 17268 examples [00:17, 986.51 examples/s]17368 examples [00:17, 982.49 examples/s]17469 examples [00:17, 988.35 examples/s]17572 examples [00:17, 999.41 examples/s]17677 examples [00:17, 1012.40 examples/s]17779 examples [00:17, 989.94 examples/s] 17879 examples [00:18, 951.59 examples/s]17975 examples [00:18, 912.47 examples/s]18067 examples [00:18, 894.58 examples/s]18163 examples [00:18, 911.26 examples/s]18265 examples [00:18, 941.12 examples/s]18360 examples [00:18, 936.40 examples/s]18455 examples [00:18, 928.76 examples/s]18551 examples [00:18, 935.69 examples/s]18646 examples [00:18, 939.00 examples/s]18752 examples [00:19, 971.36 examples/s]18858 examples [00:19, 993.74 examples/s]18962 examples [00:19, 1007.12 examples/s]19064 examples [00:19, 1000.18 examples/s]19165 examples [00:19, 987.41 examples/s] 19267 examples [00:19, 996.16 examples/s]19367 examples [00:19, 929.15 examples/s]19462 examples [00:19, 933.61 examples/s]19557 examples [00:19, 936.38 examples/s]19659 examples [00:19, 958.97 examples/s]19756 examples [00:20, 961.40 examples/s]19856 examples [00:20, 971.33 examples/s]19962 examples [00:20, 994.13 examples/s]20062 examples [00:20, 951.82 examples/s]20165 examples [00:20, 972.13 examples/s]20263 examples [00:20, 971.98 examples/s]20366 examples [00:20, 986.98 examples/s]20472 examples [00:20, 1007.28 examples/s]20578 examples [00:20, 1020.32 examples/s]20686 examples [00:20, 1036.84 examples/s]20793 examples [00:21, 1043.88 examples/s]20898 examples [00:21, 1044.59 examples/s]21003 examples [00:21, 1017.24 examples/s]21110 examples [00:21, 1031.60 examples/s]21214 examples [00:21, 1027.56 examples/s]21317 examples [00:21, 992.80 examples/s] 21420 examples [00:21, 1003.52 examples/s]21521 examples [00:21, 994.23 examples/s] 21628 examples [00:21, 1014.84 examples/s]21733 examples [00:22, 1024.87 examples/s]21836 examples [00:22, 1023.36 examples/s]21939 examples [00:22, 1013.47 examples/s]22042 examples [00:22, 1017.68 examples/s]22146 examples [00:22, 1021.60 examples/s]22249 examples [00:22, 1000.09 examples/s]22350 examples [00:22, 966.19 examples/s] 22448 examples [00:22, 969.23 examples/s]22551 examples [00:22, 985.70 examples/s]22657 examples [00:22, 1004.72 examples/s]22759 examples [00:23, 1007.96 examples/s]22860 examples [00:23, 1001.37 examples/s]22963 examples [00:23, 1009.55 examples/s]23070 examples [00:23, 1026.02 examples/s]23177 examples [00:23, 1036.25 examples/s]23281 examples [00:23, 1026.69 examples/s]23385 examples [00:23, 1024.54 examples/s]23488 examples [00:23, 1023.01 examples/s]23591 examples [00:23, 1019.55 examples/s]23699 examples [00:23, 1035.25 examples/s]23803 examples [00:24, 1023.30 examples/s]23906 examples [00:24, 995.30 examples/s] 24006 examples [00:24, 984.61 examples/s]24109 examples [00:24, 995.51 examples/s]24214 examples [00:24, 1010.87 examples/s]24323 examples [00:24, 1031.17 examples/s]24427 examples [00:24, 1031.04 examples/s]24531 examples [00:24, 949.26 examples/s] 24628 examples [00:24, 920.59 examples/s]24722 examples [00:25, 910.81 examples/s]24814 examples [00:25, 849.73 examples/s]24901 examples [00:25, 810.91 examples/s]24988 examples [00:25, 826.72 examples/s]25072 examples [00:25, 813.61 examples/s]25155 examples [00:25, 815.34 examples/s]25238 examples [00:25, 806.33 examples/s]25332 examples [00:25, 841.44 examples/s]25421 examples [00:25, 852.72 examples/s]25507 examples [00:25, 845.16 examples/s]25596 examples [00:26, 857.06 examples/s]25687 examples [00:26, 870.87 examples/s]25788 examples [00:26, 907.32 examples/s]25886 examples [00:26, 926.00 examples/s]25993 examples [00:26, 964.12 examples/s]26091 examples [00:26, 967.08 examples/s]26189 examples [00:26, 966.24 examples/s]26296 examples [00:26, 992.78 examples/s]26402 examples [00:26, 1010.40 examples/s]26504 examples [00:27, 994.76 examples/s] 26611 examples [00:27, 1012.55 examples/s]26716 examples [00:27, 1020.95 examples/s]26824 examples [00:27, 1035.35 examples/s]26932 examples [00:27, 1046.46 examples/s]27037 examples [00:27, 1046.43 examples/s]27142 examples [00:27, 1016.08 examples/s]27244 examples [00:27, 1006.14 examples/s]27345 examples [00:27, 988.93 examples/s] 27445 examples [00:27, 987.57 examples/s]27544 examples [00:28, 957.67 examples/s]27643 examples [00:28, 965.64 examples/s]27745 examples [00:28, 979.25 examples/s]27844 examples [00:28, 978.57 examples/s]27948 examples [00:28, 992.74 examples/s]28048 examples [00:28, 990.86 examples/s]28153 examples [00:28, 1007.78 examples/s]28254 examples [00:28, 1002.69 examples/s]28360 examples [00:28, 1017.07 examples/s]28462 examples [00:28, 993.03 examples/s] 28562 examples [00:29, 970.16 examples/s]28660 examples [00:29, 971.20 examples/s]28764 examples [00:29, 989.62 examples/s]28872 examples [00:29, 1013.76 examples/s]28974 examples [00:29, 994.44 examples/s] 29074 examples [00:29, 973.13 examples/s]29172 examples [00:29, 956.49 examples/s]29268 examples [00:29, 942.02 examples/s]29364 examples [00:29, 946.64 examples/s]29460 examples [00:29, 949.38 examples/s]29561 examples [00:30, 964.59 examples/s]29658 examples [00:30, 965.22 examples/s]29755 examples [00:30, 935.87 examples/s]29855 examples [00:30, 954.15 examples/s]29952 examples [00:30, 956.28 examples/s]30048 examples [00:30, 915.64 examples/s]30143 examples [00:30, 923.69 examples/s]30242 examples [00:30, 941.44 examples/s]30337 examples [00:30, 914.43 examples/s]30444 examples [00:31, 955.95 examples/s]30549 examples [00:31, 982.35 examples/s]30653 examples [00:31, 997.66 examples/s]30761 examples [00:31, 1020.43 examples/s]30866 examples [00:31, 1026.65 examples/s]30970 examples [00:31, 1029.25 examples/s]31074 examples [00:31, 1024.33 examples/s]31177 examples [00:31, 994.75 examples/s] 31277 examples [00:31, 973.97 examples/s]31379 examples [00:31, 985.80 examples/s]31485 examples [00:32, 1006.01 examples/s]31591 examples [00:32, 1021.12 examples/s]31694 examples [00:32, 987.67 examples/s] 31794 examples [00:32, 970.76 examples/s]31900 examples [00:32, 995.54 examples/s]32007 examples [00:32, 1015.87 examples/s]32112 examples [00:32, 1024.35 examples/s]32217 examples [00:32, 1030.21 examples/s]32321 examples [00:32, 1030.98 examples/s]32430 examples [00:32, 1047.14 examples/s]32538 examples [00:33, 1056.51 examples/s]32644 examples [00:33, 1020.49 examples/s]32747 examples [00:33, 1018.10 examples/s]32853 examples [00:33, 1029.91 examples/s]32957 examples [00:33, 1025.34 examples/s]33063 examples [00:33, 1035.38 examples/s]33167 examples [00:33, 1018.78 examples/s]33270 examples [00:33, 986.81 examples/s] 33375 examples [00:33, 1002.72 examples/s]33478 examples [00:34, 1008.47 examples/s]33580 examples [00:34, 1007.17 examples/s]33681 examples [00:34, 1002.58 examples/s]33782 examples [00:34, 1004.29 examples/s]33885 examples [00:34, 1009.12 examples/s]33991 examples [00:34, 1021.32 examples/s]34099 examples [00:34, 1035.76 examples/s]34203 examples [00:34, 1032.99 examples/s]34307 examples [00:34, 1018.31 examples/s]34415 examples [00:34, 1034.88 examples/s]34523 examples [00:35, 1045.78 examples/s]34631 examples [00:35, 1053.44 examples/s]34737 examples [00:35, 1048.16 examples/s]34842 examples [00:35, 1029.43 examples/s]34946 examples [00:35, 1015.95 examples/s]35050 examples [00:35, 1022.74 examples/s]35153 examples [00:35, 1023.86 examples/s]35256 examples [00:35, 1014.71 examples/s]35363 examples [00:35, 1027.94 examples/s]35466 examples [00:35, 1024.94 examples/s]35569 examples [00:36, 1008.60 examples/s]35670 examples [00:36, 973.44 examples/s] 35768 examples [00:36, 894.56 examples/s]35868 examples [00:36, 922.99 examples/s]35970 examples [00:36, 948.04 examples/s]36075 examples [00:36, 976.39 examples/s]36174 examples [00:36, 976.36 examples/s]36273 examples [00:36, 975.88 examples/s]36372 examples [00:36, 968.63 examples/s]36471 examples [00:37, 974.39 examples/s]36576 examples [00:37, 995.40 examples/s]36676 examples [00:37, 991.89 examples/s]36776 examples [00:37, 966.47 examples/s]36878 examples [00:37, 981.20 examples/s]36984 examples [00:37, 1001.97 examples/s]37087 examples [00:37, 1009.85 examples/s]37189 examples [00:37, 977.37 examples/s] 37288 examples [00:37, 977.84 examples/s]37394 examples [00:37, 999.84 examples/s]37499 examples [00:38, 1012.11 examples/s]37601 examples [00:38, 1013.36 examples/s]37703 examples [00:38, 984.68 examples/s] 37802 examples [00:38, 947.73 examples/s]37898 examples [00:38, 943.01 examples/s]38002 examples [00:38, 967.94 examples/s]38109 examples [00:38, 994.85 examples/s]38213 examples [00:38, 1005.66 examples/s]38316 examples [00:38, 1012.21 examples/s]38421 examples [00:38, 1022.03 examples/s]38527 examples [00:39, 1031.02 examples/s]38633 examples [00:39, 1038.58 examples/s]38737 examples [00:39, 1037.20 examples/s]38841 examples [00:39, 1033.34 examples/s]38945 examples [00:39, 1033.01 examples/s]39049 examples [00:39, 1011.69 examples/s]39152 examples [00:39, 1016.28 examples/s]39254 examples [00:39, 1006.23 examples/s]39355 examples [00:39, 987.46 examples/s] 39455 examples [00:39, 990.44 examples/s]39555 examples [00:40, 992.34 examples/s]39664 examples [00:40, 1018.89 examples/s]39767 examples [00:40, 1019.81 examples/s]39870 examples [00:40, 1010.42 examples/s]39972 examples [00:40, 997.73 examples/s] 40072 examples [00:40, 891.26 examples/s]40170 examples [00:40, 915.39 examples/s]40272 examples [00:40, 941.91 examples/s]40377 examples [00:40, 969.63 examples/s]40482 examples [00:41, 991.80 examples/s]40583 examples [00:41, 954.11 examples/s]40688 examples [00:41, 980.62 examples/s]40789 examples [00:41, 987.99 examples/s]40896 examples [00:41, 1010.39 examples/s]41004 examples [00:41, 1030.12 examples/s]41108 examples [00:41, 1017.55 examples/s]41211 examples [00:41, 1013.27 examples/s]41313 examples [00:41, 1000.97 examples/s]41416 examples [00:41, 1008.05 examples/s]41519 examples [00:42, 1013.60 examples/s]41621 examples [00:42, 1004.65 examples/s]41725 examples [00:42, 1013.29 examples/s]41827 examples [00:42, 1005.17 examples/s]41928 examples [00:42, 1004.80 examples/s]42030 examples [00:42, 1008.30 examples/s]42131 examples [00:42, 1001.68 examples/s]42232 examples [00:42, 1003.56 examples/s]42333 examples [00:42, 988.77 examples/s] 42432 examples [00:42, 973.63 examples/s]42535 examples [00:43, 989.12 examples/s]42635 examples [00:43, 984.54 examples/s]42741 examples [00:43, 1003.70 examples/s]42842 examples [00:43, 1000.36 examples/s]42949 examples [00:43, 1019.52 examples/s]43054 examples [00:43, 1027.06 examples/s]43157 examples [00:43, 1008.16 examples/s]43265 examples [00:43, 1027.90 examples/s]43369 examples [00:43, 1021.08 examples/s]43472 examples [00:44, 1022.05 examples/s]43575 examples [00:44, 1022.52 examples/s]43678 examples [00:44, 1014.98 examples/s]43781 examples [00:44, 1017.39 examples/s]43883 examples [00:44, 994.43 examples/s] 43985 examples [00:44, 1000.54 examples/s]44090 examples [00:44, 1013.85 examples/s]44196 examples [00:44, 1024.67 examples/s]44303 examples [00:44, 1035.46 examples/s]44407 examples [00:44, 1026.68 examples/s]44512 examples [00:45, 1033.46 examples/s]44616 examples [00:45, 1028.42 examples/s]44719 examples [00:45, 983.43 examples/s] 44818 examples [00:45, 961.81 examples/s]44915 examples [00:45, 957.62 examples/s]45012 examples [00:45, 959.37 examples/s]45116 examples [00:45, 981.84 examples/s]45223 examples [00:45, 1005.23 examples/s]45324 examples [00:45, 982.92 examples/s] 45423 examples [00:45, 957.23 examples/s]45520 examples [00:46, 944.36 examples/s]45622 examples [00:46, 965.06 examples/s]45724 examples [00:46, 979.53 examples/s]45823 examples [00:46, 955.44 examples/s]45924 examples [00:46, 970.30 examples/s]46022 examples [00:46, 970.42 examples/s]46127 examples [00:46, 991.68 examples/s]46233 examples [00:46, 1009.13 examples/s]46336 examples [00:46, 1012.79 examples/s]46438 examples [00:47, 1004.96 examples/s]46539 examples [00:47, 972.74 examples/s] 46637 examples [00:47, 970.03 examples/s]46735 examples [00:47, 938.42 examples/s]46833 examples [00:47, 949.34 examples/s]46933 examples [00:47, 963.12 examples/s]47035 examples [00:47, 979.27 examples/s]47135 examples [00:47, 984.05 examples/s]47241 examples [00:47, 1003.96 examples/s]47347 examples [00:47, 1018.01 examples/s]47450 examples [00:48, 1008.27 examples/s]47556 examples [00:48, 1020.72 examples/s]47659 examples [00:48, 998.81 examples/s] 47760 examples [00:48, 988.02 examples/s]47859 examples [00:48, 980.52 examples/s]47958 examples [00:48, 981.38 examples/s]48057 examples [00:48, 960.60 examples/s]48159 examples [00:48, 975.88 examples/s]48264 examples [00:48, 996.26 examples/s]48368 examples [00:48, 1006.54 examples/s]48469 examples [00:49, 994.58 examples/s] 48575 examples [00:49, 1011.55 examples/s]48682 examples [00:49, 1026.10 examples/s]48785 examples [00:49, 1021.10 examples/s]48890 examples [00:49, 1027.10 examples/s]48993 examples [00:49, 1026.61 examples/s]49096 examples [00:49, 1024.95 examples/s]49202 examples [00:49, 1033.84 examples/s]49306 examples [00:49, 1016.48 examples/s]49410 examples [00:49, 1022.93 examples/s]49514 examples [00:50, 1027.54 examples/s]49617 examples [00:50, 1021.21 examples/s]49720 examples [00:50, 1023.56 examples/s]49823 examples [00:50, 1018.73 examples/s]49925 examples [00:50, 1008.40 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s]  6%|▋         | 3153/50000 [00:00<00:01, 31528.23 examples/s] 28%|██▊       | 14076/50000 [00:00<00:00, 40081.84 examples/s] 52%|█████▏    | 25868/50000 [00:00<00:00, 49979.07 examples/s] 76%|███████▌  | 37857/50000 [00:00<00:00, 60575.84 examples/s] 98%|█████████▊| 49205/50000 [00:00<00:00, 70425.40 examples/s]                                                               0 examples [00:00, ? examples/s]79 examples [00:00, 789.86 examples/s]179 examples [00:00, 842.84 examples/s]284 examples [00:00, 894.97 examples/s]390 examples [00:00, 937.47 examples/s]491 examples [00:00, 956.44 examples/s]598 examples [00:00, 987.55 examples/s]705 examples [00:00, 1008.48 examples/s]810 examples [00:00, 1019.47 examples/s]909 examples [00:00, 989.41 examples/s] 1009 examples [00:01, 992.32 examples/s]1116 examples [00:01, 1014.25 examples/s]1219 examples [00:01, 1018.86 examples/s]1321 examples [00:01, 1017.59 examples/s]1428 examples [00:01, 1030.86 examples/s]1539 examples [00:01, 1051.75 examples/s]1646 examples [00:01, 1054.70 examples/s]1757 examples [00:01, 1068.41 examples/s]1867 examples [00:01, 1077.40 examples/s]1977 examples [00:01, 1082.59 examples/s]2086 examples [00:02, 1080.39 examples/s]2195 examples [00:02, 1077.55 examples/s]2304 examples [00:02, 1079.01 examples/s]2414 examples [00:02, 1082.47 examples/s]2525 examples [00:02, 1088.23 examples/s]2634 examples [00:02, 1087.97 examples/s]2745 examples [00:02, 1092.37 examples/s]2855 examples [00:02, 1092.81 examples/s]2967 examples [00:02, 1098.15 examples/s]3077 examples [00:02, 1095.96 examples/s]3187 examples [00:03, 1091.15 examples/s]3297 examples [00:03, 1080.13 examples/s]3406 examples [00:03, 1076.63 examples/s]3514 examples [00:03, 1067.09 examples/s]3625 examples [00:03, 1077.29 examples/s]3734 examples [00:03, 1077.87 examples/s]3842 examples [00:03, 1049.66 examples/s]3948 examples [00:03, 1036.12 examples/s]4052 examples [00:03, 1035.39 examples/s]4161 examples [00:03, 1048.59 examples/s]4266 examples [00:04, 1010.61 examples/s]4368 examples [00:04, 1009.34 examples/s]4476 examples [00:04, 1026.96 examples/s]4581 examples [00:04, 1033.49 examples/s]4691 examples [00:04, 1051.29 examples/s]4797 examples [00:04, 726.81 examples/s] 4898 examples [00:04, 792.54 examples/s]4993 examples [00:04, 832.92 examples/s]5098 examples [00:05, 885.56 examples/s]5204 examples [00:05, 927.62 examples/s]5303 examples [00:05, 941.04 examples/s]5406 examples [00:05, 963.47 examples/s]5513 examples [00:05, 990.65 examples/s]5618 examples [00:05, 1007.16 examples/s]5726 examples [00:05, 1025.27 examples/s]5830 examples [00:05, 1014.39 examples/s]5939 examples [00:05, 1034.38 examples/s]6049 examples [00:05, 1053.01 examples/s]6158 examples [00:06, 1062.58 examples/s]6268 examples [00:06, 1072.12 examples/s]6376 examples [00:06, 1058.30 examples/s]6484 examples [00:06, 1063.88 examples/s]6591 examples [00:06, 1052.87 examples/s]6697 examples [00:06, 1037.75 examples/s]6801 examples [00:06, 1027.42 examples/s]6904 examples [00:06, 998.92 examples/s] 7005 examples [00:06, 994.53 examples/s]7105 examples [00:07, 884.94 examples/s]7209 examples [00:07, 926.11 examples/s]7317 examples [00:07, 966.72 examples/s]7420 examples [00:07, 983.60 examples/s]7526 examples [00:07, 1004.65 examples/s]7628 examples [00:07, 1006.48 examples/s]7730 examples [00:07, 996.92 examples/s] 7832 examples [00:07, 1002.30 examples/s]7933 examples [00:07, 997.49 examples/s] 8034 examples [00:07, 995.27 examples/s]8142 examples [00:08, 1016.93 examples/s]8246 examples [00:08, 1023.28 examples/s]8349 examples [00:08, 1011.33 examples/s]8455 examples [00:08, 1023.03 examples/s]8561 examples [00:08, 1033.39 examples/s]8669 examples [00:08, 1044.36 examples/s]8774 examples [00:08, 1025.63 examples/s]8880 examples [00:08, 1034.41 examples/s]8985 examples [00:08, 1036.65 examples/s]9094 examples [00:08, 1049.91 examples/s]9204 examples [00:09, 1061.79 examples/s]9311 examples [00:09, 1037.92 examples/s]9416 examples [00:09, 1038.44 examples/s]9520 examples [00:09, 1014.06 examples/s]9622 examples [00:09, 1008.13 examples/s]9725 examples [00:09, 1012.13 examples/s]9827 examples [00:09, 1003.82 examples/s]9928 examples [00:09, 986.50 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZLFNU6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteZLFNU6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f10731a6510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f10731a6510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f10731a6510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1071209198>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0ff94155c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f10731a6510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f10731a6510> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f10731a6510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0ff9415fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1071369160>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.60 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.60 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.01 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]2020-09-03 12:09:49.093595: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-03 12:09:49.098838: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-09-03 12:09:49.099071: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e0dbfa0400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 12:09:49.099087: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 145162.47it/s] 61%|██████▏   | 6094848/9912422 [00:00<00:18, 207162.91it/s]9920512it [00:00, 40040342.04it/s]                           
0it [00:00, ?it/s]32768it [00:00, 641108.67it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478314.73it/s]1654784it [00:00, 11916740.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 223071.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
