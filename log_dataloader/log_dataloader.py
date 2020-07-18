
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f75d72219d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f75d72219d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7641e8c510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7641e8c510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f76610dbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f76610dbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f75ef1bc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f75ef1bc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f75ef1bc950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3260416/9912422 [00:00<00:00, 30052425.41it/s]9920512it [00:00, 31548053.89it/s]                             
0it [00:00, ?it/s]32768it [00:00, 575839.02it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162977.92it/s]1654784it [00:00, 11680967.16it/s]                         
0it [00:00, ?it/s]8192it [00:00, 133313.69it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f75ed29a208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f75ed72f908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f75ef1bc598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f75ef1bc598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f75ef1bc598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:09,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:08,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:07,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:44,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:19,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:19,  6.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 35.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 52.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 61.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 68.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 74.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 79.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.34 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.83s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.16s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 83.34 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.83s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.83s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 42.27 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.83s/ url]
0 examples [00:00, ? examples/s]2020-07-18 18:08:06.473693: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-18 18:08:06.531703: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-18 18:08:06.532009: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea7e003d90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 18:08:06.532024: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.66 examples/s]118 examples [00:00,  2.37 examples/s]248 examples [00:00,  3.39 examples/s]379 examples [00:00,  4.84 examples/s]513 examples [00:01,  6.90 examples/s]652 examples [00:01,  9.83 examples/s]786 examples [00:01, 14.00 examples/s]910 examples [00:01, 19.91 examples/s]1047 examples [00:01, 28.27 examples/s]1176 examples [00:01, 40.00 examples/s]1315 examples [00:01, 56.45 examples/s]1454 examples [00:01, 79.25 examples/s]1593 examples [00:01, 110.51 examples/s]1734 examples [00:01, 152.72 examples/s]1870 examples [00:02, 208.10 examples/s]2006 examples [00:02, 278.36 examples/s]2146 examples [00:02, 366.38 examples/s]2283 examples [00:02, 469.39 examples/s]2420 examples [00:02, 582.67 examples/s]2556 examples [00:02, 699.40 examples/s]2691 examples [00:02, 811.60 examples/s]2824 examples [00:02, 909.92 examples/s]2956 examples [00:02, 998.54 examples/s]3092 examples [00:02, 1084.57 examples/s]3225 examples [00:03, 1142.98 examples/s]3363 examples [00:03, 1203.35 examples/s]3497 examples [00:03, 1227.11 examples/s]3634 examples [00:03, 1266.31 examples/s]3774 examples [00:03, 1301.96 examples/s]3910 examples [00:03, 1305.00 examples/s]4049 examples [00:03, 1327.62 examples/s]4186 examples [00:03, 1339.33 examples/s]4322 examples [00:03, 1338.86 examples/s]4458 examples [00:03, 1315.41 examples/s]4591 examples [00:04, 1278.88 examples/s]4720 examples [00:04, 1280.54 examples/s]4858 examples [00:04, 1307.88 examples/s]4992 examples [00:04, 1316.22 examples/s]5125 examples [00:04, 1277.24 examples/s]5258 examples [00:04, 1291.19 examples/s]5396 examples [00:04, 1316.29 examples/s]5535 examples [00:04, 1337.07 examples/s]5672 examples [00:04, 1345.11 examples/s]5811 examples [00:04, 1356.87 examples/s]5947 examples [00:05, 1334.19 examples/s]6081 examples [00:05, 1328.04 examples/s]6216 examples [00:05, 1333.00 examples/s]6351 examples [00:05, 1336.54 examples/s]6485 examples [00:05, 1335.80 examples/s]6619 examples [00:05, 1271.95 examples/s]6747 examples [00:05, 1233.37 examples/s]6876 examples [00:05, 1220.37 examples/s]7012 examples [00:05, 1256.95 examples/s]7144 examples [00:06, 1272.19 examples/s]7272 examples [00:06, 1265.78 examples/s]7399 examples [00:06, 1226.92 examples/s]7527 examples [00:06, 1240.70 examples/s]7656 examples [00:06, 1254.56 examples/s]7782 examples [00:06, 1226.79 examples/s]7906 examples [00:06, 1186.77 examples/s]8039 examples [00:06, 1224.75 examples/s]8171 examples [00:06, 1249.64 examples/s]8302 examples [00:06, 1265.31 examples/s]8431 examples [00:07, 1272.52 examples/s]8565 examples [00:07, 1291.33 examples/s]8695 examples [00:07, 1286.69 examples/s]8830 examples [00:07, 1303.71 examples/s]8963 examples [00:07, 1309.34 examples/s]9095 examples [00:07, 1303.90 examples/s]9226 examples [00:07, 1278.08 examples/s]9355 examples [00:07, 1280.63 examples/s]9493 examples [00:07, 1307.54 examples/s]9627 examples [00:07, 1314.05 examples/s]9759 examples [00:08, 1307.76 examples/s]9890 examples [00:08, 1302.06 examples/s]10021 examples [00:08, 1255.83 examples/s]10159 examples [00:08, 1289.93 examples/s]10289 examples [00:08, 1272.59 examples/s]10417 examples [00:08, 1274.45 examples/s]10551 examples [00:08, 1292.01 examples/s]10688 examples [00:08, 1312.64 examples/s]10823 examples [00:08, 1321.97 examples/s]10956 examples [00:08, 1321.01 examples/s]11089 examples [00:09, 1264.90 examples/s]11217 examples [00:09, 1244.28 examples/s]11345 examples [00:09, 1252.39 examples/s]11471 examples [00:09, 1245.45 examples/s]11596 examples [00:09, 1187.16 examples/s]11716 examples [00:09, 1175.27 examples/s]11843 examples [00:09, 1201.49 examples/s]11964 examples [00:09, 1193.00 examples/s]12088 examples [00:09, 1205.39 examples/s]12212 examples [00:10, 1214.94 examples/s]12334 examples [00:10, 1215.66 examples/s]12462 examples [00:10, 1232.90 examples/s]12597 examples [00:10, 1263.89 examples/s]12735 examples [00:10, 1294.83 examples/s]12870 examples [00:10, 1310.00 examples/s]13002 examples [00:10, 1303.38 examples/s]13134 examples [00:10, 1305.82 examples/s]13265 examples [00:10, 1298.58 examples/s]13401 examples [00:10, 1313.65 examples/s]13536 examples [00:11, 1323.75 examples/s]13669 examples [00:11, 1320.54 examples/s]13804 examples [00:11, 1326.69 examples/s]13937 examples [00:11, 1316.85 examples/s]14070 examples [00:11, 1318.39 examples/s]14203 examples [00:11, 1319.99 examples/s]14336 examples [00:11, 1262.35 examples/s]14466 examples [00:11, 1271.13 examples/s]14602 examples [00:11, 1295.40 examples/s]14739 examples [00:11, 1315.87 examples/s]14876 examples [00:12, 1330.04 examples/s]15010 examples [00:12, 1302.82 examples/s]15147 examples [00:12, 1321.49 examples/s]15280 examples [00:12, 1303.43 examples/s]15416 examples [00:12, 1317.82 examples/s]15553 examples [00:12, 1332.97 examples/s]15687 examples [00:12, 1308.28 examples/s]15825 examples [00:12, 1326.43 examples/s]15961 examples [00:12, 1336.17 examples/s]16098 examples [00:12, 1343.70 examples/s]16236 examples [00:13, 1351.95 examples/s]16372 examples [00:13, 1335.44 examples/s]16507 examples [00:13, 1338.06 examples/s]16642 examples [00:13, 1339.77 examples/s]16777 examples [00:13, 1339.30 examples/s]16912 examples [00:13, 1340.80 examples/s]17047 examples [00:13, 1331.72 examples/s]17183 examples [00:13, 1338.76 examples/s]17321 examples [00:13, 1348.35 examples/s]17456 examples [00:14, 1321.05 examples/s]17594 examples [00:14, 1336.61 examples/s]17728 examples [00:14, 1299.22 examples/s]17859 examples [00:14, 1232.43 examples/s]17995 examples [00:14, 1266.15 examples/s]18127 examples [00:14, 1279.12 examples/s]18256 examples [00:14, 1278.07 examples/s]18389 examples [00:14, 1289.91 examples/s]18522 examples [00:14, 1301.51 examples/s]18659 examples [00:14, 1321.30 examples/s]18792 examples [00:15, 1317.25 examples/s]18924 examples [00:15, 1264.49 examples/s]19056 examples [00:15, 1279.67 examples/s]19192 examples [00:15, 1302.31 examples/s]19324 examples [00:15, 1305.91 examples/s]19455 examples [00:15, 1279.21 examples/s]19588 examples [00:15, 1290.60 examples/s]19720 examples [00:15, 1297.23 examples/s]19855 examples [00:15, 1311.10 examples/s]19987 examples [00:15, 1309.98 examples/s]20119 examples [00:16, 1263.51 examples/s]20251 examples [00:16, 1278.16 examples/s]20383 examples [00:16, 1288.56 examples/s]20515 examples [00:16, 1296.26 examples/s]20648 examples [00:16, 1304.35 examples/s]20780 examples [00:16, 1306.31 examples/s]20914 examples [00:16, 1313.58 examples/s]21048 examples [00:16, 1319.33 examples/s]21180 examples [00:16, 1309.53 examples/s]21318 examples [00:16, 1328.18 examples/s]21451 examples [00:17, 1305.80 examples/s]21587 examples [00:17, 1318.28 examples/s]21719 examples [00:17, 1305.94 examples/s]21854 examples [00:17, 1315.99 examples/s]21988 examples [00:17, 1322.38 examples/s]22123 examples [00:17, 1328.14 examples/s]22256 examples [00:17, 1322.11 examples/s]22389 examples [00:17, 1302.57 examples/s]22528 examples [00:17, 1326.32 examples/s]22667 examples [00:18, 1344.60 examples/s]22802 examples [00:18, 1302.25 examples/s]22937 examples [00:18, 1314.96 examples/s]23072 examples [00:18, 1323.99 examples/s]23212 examples [00:18, 1343.58 examples/s]23351 examples [00:18, 1355.79 examples/s]23487 examples [00:18, 1325.41 examples/s]23621 examples [00:18, 1329.13 examples/s]23755 examples [00:18, 1321.36 examples/s]23891 examples [00:18, 1330.21 examples/s]24025 examples [00:19, 1313.53 examples/s]24157 examples [00:19, 1307.52 examples/s]24293 examples [00:19, 1321.42 examples/s]24429 examples [00:19, 1331.47 examples/s]24563 examples [00:19, 1305.83 examples/s]24694 examples [00:19, 1279.89 examples/s]24826 examples [00:19, 1289.55 examples/s]24963 examples [00:19, 1311.11 examples/s]25095 examples [00:19, 1311.95 examples/s]25232 examples [00:19, 1328.38 examples/s]25369 examples [00:20, 1339.15 examples/s]25504 examples [00:20, 1335.99 examples/s]25638 examples [00:20, 1331.22 examples/s]25772 examples [00:20, 1295.35 examples/s]25911 examples [00:20, 1321.52 examples/s]26044 examples [00:20, 1322.90 examples/s]26177 examples [00:20, 1294.29 examples/s]26307 examples [00:20, 1279.61 examples/s]26446 examples [00:20, 1308.58 examples/s]26586 examples [00:20, 1333.40 examples/s]26725 examples [00:21, 1348.41 examples/s]26865 examples [00:21, 1361.66 examples/s]27002 examples [00:21, 1352.17 examples/s]27138 examples [00:21, 1333.93 examples/s]27272 examples [00:21, 1321.84 examples/s]27407 examples [00:21, 1327.86 examples/s]27542 examples [00:21, 1334.42 examples/s]27676 examples [00:21, 1332.49 examples/s]27815 examples [00:21, 1346.98 examples/s]27953 examples [00:22, 1356.40 examples/s]28089 examples [00:22, 1350.00 examples/s]28225 examples [00:22, 1347.23 examples/s]28360 examples [00:22, 1325.73 examples/s]28500 examples [00:22, 1345.85 examples/s]28638 examples [00:22, 1354.70 examples/s]28775 examples [00:22, 1359.21 examples/s]28913 examples [00:22, 1365.33 examples/s]29050 examples [00:22, 1344.96 examples/s]29185 examples [00:22, 1344.53 examples/s]29323 examples [00:23, 1353.30 examples/s]29461 examples [00:23, 1358.23 examples/s]29597 examples [00:23, 1349.56 examples/s]29733 examples [00:23, 1345.40 examples/s]29869 examples [00:23, 1349.41 examples/s]30004 examples [00:23, 1274.70 examples/s]30139 examples [00:23, 1295.66 examples/s]30273 examples [00:23, 1308.12 examples/s]30405 examples [00:23, 1251.97 examples/s]30540 examples [00:23, 1278.71 examples/s]30672 examples [00:24, 1289.89 examples/s]30807 examples [00:24, 1306.67 examples/s]30939 examples [00:24, 1306.56 examples/s]31070 examples [00:24, 1274.97 examples/s]31208 examples [00:24, 1303.01 examples/s]31339 examples [00:24, 1298.34 examples/s]31470 examples [00:24, 1225.87 examples/s]31604 examples [00:24, 1255.45 examples/s]31732 examples [00:24, 1260.74 examples/s]31869 examples [00:24, 1290.55 examples/s]31999 examples [00:25, 1283.24 examples/s]32132 examples [00:25, 1296.04 examples/s]32269 examples [00:25, 1315.27 examples/s]32401 examples [00:25, 1309.32 examples/s]32533 examples [00:25, 1304.11 examples/s]32664 examples [00:25, 1304.53 examples/s]32795 examples [00:25, 1305.75 examples/s]32926 examples [00:25, 1293.22 examples/s]33056 examples [00:25, 1261.45 examples/s]33184 examples [00:26, 1263.88 examples/s]33319 examples [00:26, 1287.34 examples/s]33457 examples [00:26, 1310.44 examples/s]33593 examples [00:26, 1322.75 examples/s]33726 examples [00:26, 1314.14 examples/s]33858 examples [00:26, 1312.73 examples/s]33993 examples [00:26, 1321.14 examples/s]34126 examples [00:26, 1295.16 examples/s]34261 examples [00:26, 1308.79 examples/s]34393 examples [00:26, 1237.09 examples/s]34522 examples [00:27, 1252.10 examples/s]34648 examples [00:27, 1252.39 examples/s]34780 examples [00:27, 1269.99 examples/s]34910 examples [00:27, 1278.63 examples/s]35039 examples [00:27, 1244.77 examples/s]35169 examples [00:27, 1259.49 examples/s]35297 examples [00:27, 1263.31 examples/s]35433 examples [00:27, 1290.58 examples/s]35569 examples [00:27, 1310.21 examples/s]35702 examples [00:27, 1315.43 examples/s]35836 examples [00:28, 1322.65 examples/s]35976 examples [00:28, 1343.85 examples/s]36114 examples [00:28, 1352.32 examples/s]36253 examples [00:28, 1362.00 examples/s]36390 examples [00:28, 1355.97 examples/s]36529 examples [00:28, 1365.18 examples/s]36666 examples [00:28, 1363.13 examples/s]36803 examples [00:28, 1351.97 examples/s]36943 examples [00:28, 1363.86 examples/s]37080 examples [00:28, 1332.48 examples/s]37215 examples [00:29, 1336.09 examples/s]37349 examples [00:29, 1336.29 examples/s]37489 examples [00:29, 1352.54 examples/s]37629 examples [00:29, 1364.94 examples/s]37766 examples [00:29, 1346.94 examples/s]37901 examples [00:29, 1341.23 examples/s]38041 examples [00:29, 1356.24 examples/s]38177 examples [00:29, 1277.01 examples/s]38306 examples [00:29, 1257.42 examples/s]38440 examples [00:30, 1278.12 examples/s]38569 examples [00:30, 1276.86 examples/s]38700 examples [00:30, 1283.52 examples/s]38831 examples [00:30, 1290.10 examples/s]38967 examples [00:30, 1309.06 examples/s]39099 examples [00:30, 1301.43 examples/s]39235 examples [00:30, 1315.71 examples/s]39367 examples [00:30, 1314.60 examples/s]39500 examples [00:30, 1317.17 examples/s]39632 examples [00:30, 1242.75 examples/s]39772 examples [00:31, 1283.34 examples/s]39902 examples [00:31, 1278.34 examples/s]40031 examples [00:31, 1186.18 examples/s]40161 examples [00:31, 1216.89 examples/s]40286 examples [00:31, 1224.72 examples/s]40422 examples [00:31, 1260.83 examples/s]40559 examples [00:31, 1291.68 examples/s]40697 examples [00:31, 1316.88 examples/s]40833 examples [00:31, 1327.47 examples/s]40967 examples [00:31, 1311.04 examples/s]41099 examples [00:32, 1293.22 examples/s]41233 examples [00:32, 1304.42 examples/s]41371 examples [00:32, 1326.19 examples/s]41504 examples [00:32, 1309.94 examples/s]41636 examples [00:32, 1308.43 examples/s]41772 examples [00:32, 1321.91 examples/s]41905 examples [00:32, 1306.27 examples/s]42036 examples [00:32, 1300.66 examples/s]42172 examples [00:32, 1315.06 examples/s]42304 examples [00:33, 1313.61 examples/s]42441 examples [00:33, 1328.33 examples/s]42578 examples [00:33, 1337.75 examples/s]42714 examples [00:33, 1344.02 examples/s]42850 examples [00:33, 1346.74 examples/s]42985 examples [00:33, 1323.47 examples/s]43118 examples [00:33, 1296.91 examples/s]43255 examples [00:33, 1317.73 examples/s]43392 examples [00:33, 1331.97 examples/s]43533 examples [00:33, 1353.83 examples/s]43669 examples [00:34, 1339.74 examples/s]43806 examples [00:34, 1347.95 examples/s]43941 examples [00:34, 1347.98 examples/s]44076 examples [00:34, 1348.01 examples/s]44211 examples [00:34, 1343.03 examples/s]44346 examples [00:34, 1342.19 examples/s]44484 examples [00:34, 1350.46 examples/s]44621 examples [00:34, 1354.91 examples/s]44759 examples [00:34, 1359.46 examples/s]44895 examples [00:34, 1346.72 examples/s]45030 examples [00:35, 1295.72 examples/s]45161 examples [00:35, 1245.58 examples/s]45293 examples [00:35, 1266.25 examples/s]45427 examples [00:35, 1286.35 examples/s]45563 examples [00:35, 1307.27 examples/s]45695 examples [00:35, 1302.93 examples/s]45830 examples [00:35, 1315.11 examples/s]45962 examples [00:35, 1312.52 examples/s]46094 examples [00:35, 1309.95 examples/s]46228 examples [00:35, 1318.30 examples/s]46360 examples [00:36, 1305.06 examples/s]46491 examples [00:36, 1294.49 examples/s]46621 examples [00:36, 1277.49 examples/s]46753 examples [00:36, 1286.86 examples/s]46887 examples [00:36, 1301.91 examples/s]47018 examples [00:36, 1198.34 examples/s]47146 examples [00:36, 1218.20 examples/s]47280 examples [00:36, 1250.76 examples/s]47413 examples [00:36, 1268.77 examples/s]47555 examples [00:37, 1309.59 examples/s]47687 examples [00:37, 1270.73 examples/s]47825 examples [00:37, 1301.64 examples/s]47964 examples [00:37, 1326.68 examples/s]48103 examples [00:37, 1343.33 examples/s]48243 examples [00:37, 1359.22 examples/s]48380 examples [00:37, 1340.55 examples/s]48517 examples [00:37, 1347.16 examples/s]48656 examples [00:37, 1358.40 examples/s]48793 examples [00:37, 1357.94 examples/s]48933 examples [00:38, 1368.26 examples/s]49070 examples [00:38, 1351.36 examples/s]49206 examples [00:38, 1276.21 examples/s]49335 examples [00:38, 1271.01 examples/s]49463 examples [00:38, 1255.13 examples/s]49594 examples [00:38, 1268.62 examples/s]49722 examples [00:38, 1243.47 examples/s]49847 examples [00:38, 1227.85 examples/s]49974 examples [00:38, 1238.77 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8689/50000 [00:00<00:00, 86887.60 examples/s] 47%|████▋     | 23568/50000 [00:00<00:00, 99277.71 examples/s] 74%|███████▍  | 37168/50000 [00:00<00:00, 108027.80 examples/s]                                                                0 examples [00:00, ? examples/s]75 examples [00:00, 747.92 examples/s]195 examples [00:00, 842.53 examples/s]317 examples [00:00, 927.95 examples/s]445 examples [00:00, 1009.84 examples/s]577 examples [00:00, 1084.77 examples/s]707 examples [00:00, 1139.56 examples/s]837 examples [00:00, 1183.09 examples/s]973 examples [00:00, 1228.35 examples/s]1095 examples [00:00, 1223.51 examples/s]1230 examples [00:01, 1256.94 examples/s]1370 examples [00:01, 1294.57 examples/s]1511 examples [00:01, 1324.73 examples/s]1649 examples [00:01, 1338.20 examples/s]1783 examples [00:01, 1330.10 examples/s]1917 examples [00:01, 1318.80 examples/s]2058 examples [00:01, 1344.13 examples/s]2193 examples [00:01, 1344.85 examples/s]2328 examples [00:01, 1312.60 examples/s]2467 examples [00:01, 1334.70 examples/s]2611 examples [00:02, 1361.29 examples/s]2748 examples [00:02, 1307.55 examples/s]2886 examples [00:02, 1327.17 examples/s]3020 examples [00:02, 1307.26 examples/s]3156 examples [00:02, 1319.10 examples/s]3296 examples [00:02, 1339.63 examples/s]3441 examples [00:02, 1368.25 examples/s]3581 examples [00:02, 1377.41 examples/s]3720 examples [00:02, 1363.68 examples/s]3863 examples [00:02, 1380.41 examples/s]4002 examples [00:03, 1351.55 examples/s]4139 examples [00:03, 1356.97 examples/s]4276 examples [00:03, 1359.76 examples/s]4413 examples [00:03, 1337.00 examples/s]4553 examples [00:03, 1353.27 examples/s]4690 examples [00:03, 1356.44 examples/s]4831 examples [00:03, 1369.76 examples/s]4969 examples [00:03, 1367.03 examples/s]5106 examples [00:03, 1351.86 examples/s]5243 examples [00:03, 1357.04 examples/s]5379 examples [00:04, 1337.14 examples/s]5519 examples [00:04, 1353.63 examples/s]5655 examples [00:04, 1273.26 examples/s]5784 examples [00:04, 1232.21 examples/s]5909 examples [00:04, 1228.50 examples/s]6047 examples [00:04, 1268.76 examples/s]6185 examples [00:04, 1299.33 examples/s]6317 examples [00:04, 1302.59 examples/s]6450 examples [00:04, 1308.71 examples/s]6589 examples [00:05, 1330.19 examples/s]6727 examples [00:05, 1344.59 examples/s]6866 examples [00:05, 1357.21 examples/s]7002 examples [00:05, 1335.87 examples/s]7139 examples [00:05, 1343.90 examples/s]7274 examples [00:05, 1304.40 examples/s]7408 examples [00:05, 1312.54 examples/s]7540 examples [00:05, 1307.19 examples/s]7671 examples [00:05, 1293.66 examples/s]7807 examples [00:05, 1310.38 examples/s]7939 examples [00:06, 1304.42 examples/s]8070 examples [00:06, 1295.95 examples/s]8205 examples [00:06, 1310.09 examples/s]8337 examples [00:06, 1281.92 examples/s]8475 examples [00:06, 1308.45 examples/s]8614 examples [00:06, 1330.22 examples/s]8748 examples [00:06, 1332.33 examples/s]8882 examples [00:06, 1308.81 examples/s]9015 examples [00:06, 1314.39 examples/s]9150 examples [00:06, 1323.39 examples/s]9283 examples [00:07, 1323.64 examples/s]9416 examples [00:07, 1307.46 examples/s]9547 examples [00:07, 1299.77 examples/s]9678 examples [00:07, 1275.96 examples/s]9816 examples [00:07, 1304.49 examples/s]9953 examples [00:07, 1323.23 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete934UTX/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete934UTX/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f75ef1bc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f75ef1bc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f75ef1bc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f765a6e6f98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f75ad575470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f75ef1bc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f75ef1bc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f75ef1bc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f75ad575e80>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f75ed72f3c8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7567b7de18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7567b7de18> 

  function with postional parmater data_info <function split_train_valid at 0x7f7567b7de18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7567b7df28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7567b7df28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7567b7df28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=7ce958b1299845e04d634a772e3d78f3393270df4695c237d42e3cb70a41852e
  Stored in directory: /tmp/pip-ephem-wheel-cache-sl62h7ij/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:55:04, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:10:19, 16.9kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<9:58:35, 24.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<6:59:33, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.53M/862M [00:01<4:52:59, 48.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<3:23:48, 69.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:21:48, 99.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.3M/862M [00:01<1:38:44, 142kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:08:58, 203kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.8M/862M [00:01<48:03, 289kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:01<33:38, 411kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.6M/862M [00:01<23:27, 586kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<16:30, 830kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.1M/862M [00:02<11:32, 1.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:02<08:11, 1.65MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<06:09, 2.19MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:04<04:43, 2.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<05:51, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<07:20, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:49, 2.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:06<04:14, 3.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<09:31, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<08:01, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:08<05:54, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<07:13, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<07:42, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:57, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:10<04:19, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<11:24, 1.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<09:20, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<06:48, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<07:50, 1.67MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<08:14, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<04:36, 2.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<50:27, 258kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<36:25, 357kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:16<25:46, 504kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<18:45, 690kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<10:23:34, 20.8kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<7:16:43, 29.6kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:18<5:04:38, 42.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<4:02:53, 53.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<2:52:46, 74.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<2:02:27, 105kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<1:26:10, 149kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<1:00:16, 213kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<55:56, 229kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<40:28, 317kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<28:35, 447kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<22:57, 555kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<17:20, 735kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<12:26, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:40, 1.09MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:47, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:05, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:53, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:02, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:55, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:06, 2.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:31, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:05, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:36, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<04:03, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:56:45, 17.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<8:22:45, 24.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<5:51:28, 35.4kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<4:08:13, 50.0kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<2:56:13, 70.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<2:03:51, 100kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<1:28:20, 140kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:03:02, 196kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<44:18, 278kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<33:49, 363kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<26:09, 469kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<18:54, 648kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<13:20, 915kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<17:31, 696kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<13:33, 900kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:16, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:33, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:32, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<10:24, 1.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<09:24, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:01, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:04, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<10:56, 1.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<10:33, 1.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:08, 1.48MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<05:50, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<10:41, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:57, 1.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:34, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:00, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<07:56, 1.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:11, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:35, 2.59MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:05, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:44, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:19, 2.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:46, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:06, 2.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:10, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:55, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<02:54, 4.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<14:03, 828kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<11:02, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<08:01, 1.45MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:51, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<08:20, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<08:47, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<06:46, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:53, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:19, 1.38MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:00, 1.64MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<05:15, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:49, 3.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<20:44, 552kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<15:55, 718kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<11:25, 999kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:17, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:57, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:40, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<05:32, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<10:25, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:41, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<06:22, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:43, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:06, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<04:37, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:28, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:40, 1.67MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:14, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<03:58, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:10, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:00, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:47, 2.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<02:46, 3.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<27:23, 402kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<21:58, 502kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:03, 686kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<11:22, 965kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<14:11, 772kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<11:15, 973kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<08:12, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:55, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:22, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:32, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<04:43, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:23, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:54, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<05:51, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:15, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:59, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:33, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<04:01, 2.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<08:51, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:32, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:34, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:01, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:58, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<04:00, 2.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<08:44, 1.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:26, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:28, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:56, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:27, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:05, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:57, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:10, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:32, 2.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:22, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:43, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:15, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:06, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:03, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:52, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:32, 2.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:56, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:49, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:04, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:35, 1.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:29, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:10, 1.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:45, 2.68MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:18, 1.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:04, 1.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:14, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<04:09, 2.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<7:48:51, 21.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<5:28:28, 30.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<3:49:04, 43.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<2:45:24, 60.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<1:58:13, 83.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<1:24:04, 118kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<59:12, 167kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<41:31, 238kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<32:04, 307kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<23:56, 411kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<17:06, 574kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<13:40, 715kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:03, 810kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:02, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<06:27, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<10:03, 966kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<08:12, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:59, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:07, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:44, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:19, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:50, 2.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:48, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:39, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:53, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:20, 1.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:02, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:42, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:26, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:24, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:55, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:40, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:28, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:31, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:27, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:14, 2.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:34, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:45, 2.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:44, 3.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<17:44, 520kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<14:26, 638kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<10:31, 874kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<07:28, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:06, 1.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:24, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<05:25, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:44, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:14, 1.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:55, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:34, 2.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:38, 1.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:27, 1.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:45, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:07, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:51, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:33, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:23, 2.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:30, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:08, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:09, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<02:20, 3.77MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<12:03, 730kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<09:41, 908kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<07:02, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<05:00, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<32:26, 269kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<23:46, 367kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<16:50, 517kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<13:29, 643kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<10:30, 824kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<07:34, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:01, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:59, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:25, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:55, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:36, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:22, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:41, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:58, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:31, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:23, 2.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:03, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:56, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:51, 2.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:27, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:31, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:06, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:10, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:04, 2.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:58, 2.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:50, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:23, 1.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:17, 2.48MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:56, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<03:45, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<02:50, 2.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:36, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:37, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:38, 3.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<06:07, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:16, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:55, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:19, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:02, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:56, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:50, 2.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:23, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<04:42, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:31, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:01, 1.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:47, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:52, 2.69MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:33, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<03:26, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:36, 2.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:21, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:12, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:26, 3.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:29, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:57, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:59, 2.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:36, 2.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:19, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:27, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:31, 2.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:01, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:08, 1.44MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<03:48, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:08, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:48, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:51, 2.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:27, 2.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:10, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:18, 2.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:23, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<03:08, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:38, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:06, 1.74MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:16, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:22, 2.99MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<10:15, 689kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<08:02, 880kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:49, 1.21MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:27, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:41, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:29, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:26, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:27, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:33, 2.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:28, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:29, 2.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:06, 2.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:15, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:59, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:04, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:15, 2.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:47, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:28, 1.93MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:38, 2.54MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:10, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:49, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<03:05, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:14, 2.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:06, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:16, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:09, 2.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:19, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:12, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:20, 1.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:08, 1.57MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<02:57, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:31, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:39, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:27, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:41, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:21, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:32, 2.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:02, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:50, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:08, 2.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:51, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:22, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:41, 2.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<01:56, 3.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<08:00, 772kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<06:16, 983kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:27, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:35, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:32, 1.72MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<02:32, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:18, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:45, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:48, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:09, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<01:58, 2.98MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:12, 2.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<4:22:36, 22.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<3:03:47, 32.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<2:07:41, 45.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<1:34:34, 61.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<1:07:30, 86.3kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<47:25, 123kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<33:02, 175kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<26:20, 219kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<19:03, 302kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<13:25, 427kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<10:33, 539kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<08:00, 710kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<05:43, 989kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:11, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:51, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:42, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<08:16, 671kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<06:23, 867kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<04:35, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:23, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:29, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:32, 1.54MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:38, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:52, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:41, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:02, 2.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:29, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:02, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:23, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<01:44, 3.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:21, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:59, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:14, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:36, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:09, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:31, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:49, 2.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:01, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:23, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:50, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:57, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:22, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:55, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:18, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:40, 2.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:45, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:04, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:26, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:57, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:18, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:40, 2.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:52, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:30, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:54, 2.50MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:16, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:47, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:12, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:35, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:00, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:40, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:00, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:19, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:48, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:11, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:35, 2.85MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:42, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:27, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:49, 2.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:10, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:35, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:02, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:29, 2.96MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:42, 1.61MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:26, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:49, 2.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:08, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:27, 1.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:23, 3.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:37, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:18, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:42, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:05, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:29, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:57, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:25, 2.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:12, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:03, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:33, 2.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:53, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<01:49, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:22, 2.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:45, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:11, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:43, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:16, 3.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:05, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:57, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:28, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:47, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:11, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:43, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:15, 2.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:10, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:56, 1.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:26, 2.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:47, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:45, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:16, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:49, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:24, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:46, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:56, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:13, 1.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:44, 2.03MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:14, 2.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:29, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:10, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:36, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:48, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:07, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:41, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:13, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:37, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:14, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:38, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:35, 2.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:09, 2.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:51, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:41, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:16, 2.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:30, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:48, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:25, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:01, 2.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:15, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:57, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:26, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:35, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:52, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:28, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:03, 2.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:12, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:51, 1.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:21, 2.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:33, 1.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:46, 1.60MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:23, 2.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<00:59, 2.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:00, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:45, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:17, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:26, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:41, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<00:58, 2.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:11, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:51, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<01:21, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:27, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:44, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:36, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:13, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:53, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:31, 1.63MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:22, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:01, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:11, 2.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:27, 1.67MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:08, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:19<00:48, 2.91MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:42, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:29, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:06, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:13, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:07, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:51, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:16, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:00, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:43, 3.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:16, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:07, 1.92MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:50, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:59, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:13, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:41, 2.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:27, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:15, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:55, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:01, 1.88MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:40, 2.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:50, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:37, 2.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:47, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:00, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:49, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:35, 3.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:22, 1.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:09, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:51, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:55, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:04, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:50, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:36, 2.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:19, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:07, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:49, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:52, 1.75MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:57, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:52, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:40, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:29, 2.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:46, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:44, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:32, 2.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:47, 1.76MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:27, 2.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:04, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:53, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:38, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:41, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:48, 1.55MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:26, 2.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:51, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:44, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:35, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:41, 1.59MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:33, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:23, 2.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:51, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:42, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:30, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:38, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:20, 2.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:45, 1.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:38, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:27, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:28, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:32, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:25, 1.93MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:17, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:33, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:28, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:21, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:26, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:20, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:13, 2.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:29, 1.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:18, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:18, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:21, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:16, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:11, 2.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:07, 3.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:16, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:10, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.63MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:09, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.98MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.64MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 3.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.28MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<75:55:19,  1.46it/s]  0%|          | 1108/400000 [00:00<53:00:05,  2.09it/s]  1%|          | 2230/400000 [00:00<36:59:58,  2.99it/s]  1%|          | 3290/400000 [00:00<25:50:01,  4.27it/s]  1%|          | 4403/400000 [00:01<18:02:09,  6.09it/s]  1%|▏         | 5430/400000 [00:01<12:35:43,  8.70it/s]  2%|▏         | 6525/400000 [00:01<8:47:43, 12.43it/s]   2%|▏         | 7637/400000 [00:01<6:08:32, 17.74it/s]  2%|▏         | 8685/400000 [00:01<4:17:28, 25.33it/s]  2%|▏         | 9801/400000 [00:01<2:59:53, 36.15it/s]  3%|▎         | 10952/400000 [00:01<2:05:43, 51.57it/s]  3%|▎         | 12029/400000 [00:01<1:27:56, 73.52it/s]  3%|▎         | 13168/400000 [00:01<1:01:33, 104.74it/s]  4%|▎         | 14266/400000 [00:01<43:08, 149.03it/s]    4%|▍         | 15359/400000 [00:02<30:17, 211.58it/s]  4%|▍         | 16485/400000 [00:02<21:19, 299.84it/s]  4%|▍         | 17617/400000 [00:02<15:02, 423.53it/s]  5%|▍         | 18743/400000 [00:02<10:40, 595.44it/s]  5%|▍         | 19852/400000 [00:02<07:37, 830.81it/s]  5%|▌         | 20953/400000 [00:02<05:29, 1149.67it/s]  6%|▌         | 22095/400000 [00:02<04:00, 1574.45it/s]  6%|▌         | 23233/400000 [00:02<02:57, 2123.26it/s]  6%|▌         | 24385/400000 [00:02<02:13, 2811.14it/s]  6%|▋         | 25514/400000 [00:02<01:43, 3617.87it/s]  7%|▋         | 26633/400000 [00:03<01:22, 4532.73it/s]  7%|▋         | 27789/400000 [00:03<01:07, 5543.24it/s]  7%|▋         | 28916/400000 [00:03<00:56, 6533.61it/s]  8%|▊         | 30042/400000 [00:03<00:49, 7472.35it/s]  8%|▊         | 31167/400000 [00:03<00:44, 8299.02it/s]  8%|▊         | 32290/400000 [00:03<00:41, 8960.87it/s]  8%|▊         | 33407/400000 [00:03<00:39, 9375.76it/s]  9%|▊         | 34508/400000 [00:03<00:37, 9811.80it/s]  9%|▉         | 35628/400000 [00:03<00:35, 10190.22it/s]  9%|▉         | 36733/400000 [00:04<00:36, 9976.90it/s]   9%|▉         | 37792/400000 [00:04<00:35, 10084.18it/s] 10%|▉         | 38944/400000 [00:04<00:34, 10475.12it/s] 10%|█         | 40061/400000 [00:04<00:33, 10672.80it/s] 10%|█         | 41153/400000 [00:04<00:33, 10712.20it/s] 11%|█         | 42242/400000 [00:04<00:33, 10704.59it/s] 11%|█         | 43339/400000 [00:04<00:33, 10782.27it/s] 11%|█         | 44468/400000 [00:04<00:32, 10929.16it/s] 11%|█▏        | 45615/400000 [00:04<00:31, 11082.93it/s] 12%|█▏        | 46762/400000 [00:04<00:31, 11196.25it/s] 12%|█▏        | 47886/400000 [00:05<00:31, 11154.40it/s] 12%|█▏        | 49005/400000 [00:05<00:32, 10944.19it/s] 13%|█▎        | 50103/400000 [00:05<00:32, 10895.13it/s] 13%|█▎        | 51195/400000 [00:05<00:33, 10538.43it/s] 13%|█▎        | 52302/400000 [00:05<00:32, 10691.01it/s] 13%|█▎        | 53459/400000 [00:05<00:31, 10938.06it/s] 14%|█▎        | 54557/400000 [00:05<00:32, 10776.71it/s] 14%|█▍        | 55721/400000 [00:05<00:31, 11020.80it/s] 14%|█▍        | 56827/400000 [00:05<00:31, 10991.57it/s] 14%|█▍        | 57929/400000 [00:05<00:31, 10972.45it/s] 15%|█▍        | 59098/400000 [00:06<00:30, 11175.79it/s] 15%|█▌        | 60218/400000 [00:06<00:30, 11146.38it/s] 15%|█▌        | 61335/400000 [00:06<00:30, 11108.79it/s] 16%|█▌        | 62462/400000 [00:06<00:30, 11154.14it/s] 16%|█▌        | 63579/400000 [00:06<00:30, 11075.09it/s] 16%|█▌        | 64688/400000 [00:06<00:31, 10786.26it/s] 16%|█▋        | 65769/400000 [00:06<00:31, 10509.85it/s] 17%|█▋        | 66824/400000 [00:06<00:32, 10205.61it/s] 17%|█▋        | 67926/400000 [00:06<00:31, 10435.49it/s] 17%|█▋        | 69036/400000 [00:06<00:31, 10625.93it/s] 18%|█▊        | 70165/400000 [00:07<00:30, 10816.63it/s] 18%|█▊        | 71323/400000 [00:07<00:29, 11033.26it/s] 18%|█▊        | 72469/400000 [00:07<00:29, 11157.46it/s] 18%|█▊        | 73588/400000 [00:07<00:29, 10925.26it/s] 19%|█▊        | 74684/400000 [00:07<00:29, 10849.20it/s] 19%|█▉        | 75772/400000 [00:07<00:30, 10790.04it/s] 19%|█▉        | 76853/400000 [00:07<00:29, 10794.33it/s] 19%|█▉        | 77934/400000 [00:07<00:30, 10510.06it/s] 20%|█▉        | 79096/400000 [00:07<00:29, 10817.14it/s] 20%|██        | 80267/400000 [00:08<00:28, 11069.91it/s] 20%|██        | 81379/400000 [00:08<00:28, 11073.72it/s] 21%|██        | 82493/400000 [00:08<00:28, 11092.25it/s] 21%|██        | 83634/400000 [00:08<00:28, 11184.41it/s] 21%|██        | 84785/400000 [00:08<00:27, 11276.33it/s] 21%|██▏       | 85914/400000 [00:08<00:28, 11058.58it/s] 22%|██▏       | 87022/400000 [00:08<00:28, 10916.65it/s] 22%|██▏       | 88116/400000 [00:08<00:28, 10859.01it/s] 22%|██▏       | 89264/400000 [00:08<00:28, 11038.08it/s] 23%|██▎       | 90429/400000 [00:08<00:27, 11214.45it/s] 23%|██▎       | 91594/400000 [00:09<00:27, 11340.72it/s] 23%|██▎       | 92730/400000 [00:09<00:28, 10924.55it/s] 23%|██▎       | 93827/400000 [00:09<00:28, 10723.44it/s] 24%|██▎       | 94997/400000 [00:09<00:27, 10998.56it/s] 24%|██▍       | 96149/400000 [00:09<00:27, 11147.97it/s] 24%|██▍       | 97268/400000 [00:09<00:27, 11112.21it/s] 25%|██▍       | 98382/400000 [00:09<00:27, 10969.74it/s] 25%|██▍       | 99482/400000 [00:09<00:27, 10966.63it/s] 25%|██▌       | 100612/400000 [00:09<00:27, 11062.29it/s] 25%|██▌       | 101781/400000 [00:09<00:26, 11243.02it/s] 26%|██▌       | 102907/400000 [00:10<00:26, 11188.11it/s] 26%|██▌       | 104028/400000 [00:10<00:26, 10974.00it/s] 26%|██▋       | 105128/400000 [00:10<00:26, 10972.54it/s] 27%|██▋       | 106249/400000 [00:10<00:26, 11042.27it/s] 27%|██▋       | 107355/400000 [00:10<00:27, 10760.29it/s] 27%|██▋       | 108473/400000 [00:10<00:26, 10881.06it/s] 27%|██▋       | 109564/400000 [00:10<00:27, 10647.08it/s] 28%|██▊       | 110632/400000 [00:10<00:27, 10558.67it/s] 28%|██▊       | 111747/400000 [00:10<00:26, 10727.54it/s] 28%|██▊       | 112822/400000 [00:10<00:27, 10300.31it/s] 28%|██▊       | 113918/400000 [00:11<00:27, 10487.07it/s] 29%|██▊       | 114997/400000 [00:11<00:26, 10575.04it/s] 29%|██▉       | 116058/400000 [00:11<00:27, 10379.43it/s] 29%|██▉       | 117099/400000 [00:11<00:27, 10212.15it/s] 30%|██▉       | 118176/400000 [00:11<00:27, 10372.91it/s] 30%|██▉       | 119316/400000 [00:11<00:26, 10658.62it/s] 30%|███       | 120386/400000 [00:11<00:26, 10567.63it/s] 30%|███       | 121446/400000 [00:11<00:26, 10504.81it/s] 31%|███       | 122583/400000 [00:11<00:25, 10749.92it/s] 31%|███       | 123736/400000 [00:12<00:25, 10971.20it/s] 31%|███       | 124880/400000 [00:12<00:24, 11106.77it/s] 31%|███▏      | 125994/400000 [00:12<00:24, 10964.49it/s] 32%|███▏      | 127145/400000 [00:12<00:24, 11122.53it/s] 32%|███▏      | 128281/400000 [00:12<00:24, 11190.20it/s] 32%|███▏      | 129428/400000 [00:12<00:24, 11270.66it/s] 33%|███▎      | 130557/400000 [00:12<00:23, 11255.82it/s] 33%|███▎      | 131684/400000 [00:12<00:24, 11160.03it/s] 33%|███▎      | 132808/400000 [00:12<00:23, 11183.84it/s] 33%|███▎      | 133936/400000 [00:12<00:23, 11209.83it/s] 34%|███▍      | 135058/400000 [00:13<00:24, 10910.71it/s] 34%|███▍      | 136205/400000 [00:13<00:23, 11069.71it/s] 34%|███▍      | 137314/400000 [00:13<00:23, 10967.66it/s] 35%|███▍      | 138413/400000 [00:13<00:24, 10874.91it/s] 35%|███▍      | 139524/400000 [00:13<00:23, 10944.05it/s] 35%|███▌      | 140631/400000 [00:13<00:23, 10979.07it/s] 35%|███▌      | 141730/400000 [00:13<00:24, 10721.60it/s] 36%|███▌      | 142805/400000 [00:13<00:25, 10282.42it/s] 36%|███▌      | 143839/400000 [00:13<00:25, 10166.44it/s] 36%|███▌      | 144952/400000 [00:13<00:24, 10436.35it/s] 37%|███▋      | 146001/400000 [00:14<00:24, 10401.62it/s] 37%|███▋      | 147098/400000 [00:14<00:23, 10564.48it/s] 37%|███▋      | 148158/400000 [00:14<00:23, 10552.51it/s] 37%|███▋      | 149216/400000 [00:14<00:24, 10359.62it/s] 38%|███▊      | 150255/400000 [00:14<00:24, 10173.12it/s] 38%|███▊      | 151346/400000 [00:14<00:23, 10381.43it/s] 38%|███▊      | 152478/400000 [00:14<00:23, 10644.45it/s] 38%|███▊      | 153567/400000 [00:14<00:22, 10715.55it/s] 39%|███▊      | 154681/400000 [00:14<00:22, 10838.22it/s] 39%|███▉      | 155767/400000 [00:14<00:22, 10779.68it/s] 39%|███▉      | 156847/400000 [00:15<00:22, 10694.07it/s] 39%|███▉      | 157918/400000 [00:15<00:23, 10286.03it/s] 40%|███▉      | 158951/400000 [00:15<00:23, 10098.34it/s] 40%|████      | 160074/400000 [00:15<00:23, 10412.15it/s] 40%|████      | 161193/400000 [00:15<00:22, 10632.64it/s] 41%|████      | 162329/400000 [00:15<00:21, 10839.34it/s] 41%|████      | 163470/400000 [00:15<00:21, 11003.51it/s] 41%|████      | 164574/400000 [00:15<00:21, 10960.54it/s] 41%|████▏     | 165711/400000 [00:15<00:21, 11078.28it/s] 42%|████▏     | 166836/400000 [00:16<00:20, 11126.45it/s] 42%|████▏     | 167993/400000 [00:16<00:20, 11254.43it/s] 42%|████▏     | 169144/400000 [00:16<00:20, 11328.78it/s] 43%|████▎     | 170278/400000 [00:16<00:20, 10996.37it/s] 43%|████▎     | 171439/400000 [00:16<00:20, 11172.69it/s] 43%|████▎     | 172559/400000 [00:16<00:20, 10961.23it/s] 43%|████▎     | 173658/400000 [00:16<00:20, 10790.49it/s] 44%|████▎     | 174773/400000 [00:16<00:20, 10894.96it/s] 44%|████▍     | 175869/400000 [00:16<00:20, 10913.95it/s] 44%|████▍     | 177028/400000 [00:16<00:20, 11108.12it/s] 45%|████▍     | 178146/400000 [00:17<00:19, 11127.11it/s] 45%|████▍     | 179293/400000 [00:17<00:19, 11226.50it/s] 45%|████▌     | 180468/400000 [00:17<00:19, 11376.03it/s] 45%|████▌     | 181607/400000 [00:17<00:19, 11190.17it/s] 46%|████▌     | 182767/400000 [00:17<00:19, 11307.94it/s] 46%|████▌     | 183911/400000 [00:17<00:19, 11346.78it/s] 46%|████▋     | 185047/400000 [00:17<00:19, 10851.27it/s] 47%|████▋     | 186165/400000 [00:17<00:19, 10943.67it/s] 47%|████▋     | 187264/400000 [00:17<00:19, 10778.49it/s] 47%|████▋     | 188397/400000 [00:17<00:19, 10935.48it/s] 47%|████▋     | 189573/400000 [00:18<00:18, 11169.06it/s] 48%|████▊     | 190694/400000 [00:18<00:19, 10962.38it/s] 48%|████▊     | 191828/400000 [00:18<00:18, 11071.65it/s] 48%|████▊     | 192938/400000 [00:18<00:18, 11027.30it/s] 49%|████▊     | 194078/400000 [00:18<00:18, 11134.11it/s] 49%|████▉     | 195223/400000 [00:18<00:18, 11225.80it/s] 49%|████▉     | 196347/400000 [00:18<00:18, 10966.94it/s] 49%|████▉     | 197515/400000 [00:18<00:18, 11171.46it/s] 50%|████▉     | 198635/400000 [00:18<00:18, 11081.94it/s] 50%|████▉     | 199766/400000 [00:18<00:17, 11148.36it/s] 50%|█████     | 200904/400000 [00:19<00:17, 11214.87it/s] 51%|█████     | 202047/400000 [00:19<00:17, 11278.28it/s] 51%|█████     | 203199/400000 [00:19<00:17, 11347.51it/s] 51%|█████     | 204335/400000 [00:19<00:17, 10961.38it/s] 51%|█████▏    | 205435/400000 [00:19<00:17, 10868.26it/s] 52%|█████▏    | 206525/400000 [00:19<00:18, 10673.76it/s] 52%|█████▏    | 207616/400000 [00:19<00:17, 10740.77it/s] 52%|█████▏    | 208692/400000 [00:19<00:17, 10702.81it/s] 52%|█████▏    | 209826/400000 [00:19<00:17, 10884.76it/s] 53%|█████▎    | 210986/400000 [00:20<00:17, 11089.81it/s] 53%|█████▎    | 212098/400000 [00:20<00:17, 11010.80it/s] 53%|█████▎    | 213288/400000 [00:20<00:16, 11262.46it/s] 54%|█████▎    | 214457/400000 [00:20<00:16, 11384.79it/s] 54%|█████▍    | 215598/400000 [00:20<00:16, 11155.33it/s] 54%|█████▍    | 216754/400000 [00:20<00:16, 11271.46it/s] 54%|█████▍    | 217885/400000 [00:20<00:16, 11281.55it/s] 55%|█████▍    | 219023/400000 [00:20<00:16, 11308.64it/s] 55%|█████▌    | 220166/400000 [00:20<00:15, 11343.68it/s] 55%|█████▌    | 221302/400000 [00:20<00:16, 11127.33it/s] 56%|█████▌    | 222468/400000 [00:21<00:15, 11279.56it/s] 56%|█████▌    | 223598/400000 [00:21<00:15, 11267.59it/s] 56%|█████▌    | 224726/400000 [00:21<00:15, 11171.84it/s] 56%|█████▋    | 225845/400000 [00:21<00:15, 11134.61it/s] 57%|█████▋    | 226960/400000 [00:21<00:15, 11012.87it/s] 57%|█████▋    | 228115/400000 [00:21<00:15, 11167.18it/s] 57%|█████▋    | 229272/400000 [00:21<00:15, 11284.11it/s] 58%|█████▊    | 230425/400000 [00:21<00:14, 11355.50it/s] 58%|█████▊    | 231594/400000 [00:21<00:14, 11450.89it/s] 58%|█████▊    | 232740/400000 [00:21<00:14, 11339.26it/s] 58%|█████▊    | 233901/400000 [00:22<00:14, 11417.02it/s] 59%|█████▉    | 235044/400000 [00:22<00:14, 11407.45it/s] 59%|█████▉    | 236204/400000 [00:22<00:14, 11462.63it/s] 59%|█████▉    | 237371/400000 [00:22<00:14, 11523.24it/s] 60%|█████▉    | 238524/400000 [00:22<00:14, 11344.04it/s] 60%|█████▉    | 239712/400000 [00:22<00:13, 11499.05it/s] 60%|██████    | 240864/400000 [00:22<00:13, 11479.60it/s] 61%|██████    | 242028/400000 [00:22<00:13, 11527.09it/s] 61%|██████    | 243191/400000 [00:22<00:13, 11550.66it/s] 61%|██████    | 244347/400000 [00:22<00:13, 11120.19it/s] 61%|██████▏   | 245475/400000 [00:23<00:13, 11166.04it/s] 62%|██████▏   | 246595/400000 [00:23<00:13, 10957.71it/s] 62%|██████▏   | 247776/400000 [00:23<00:13, 11200.06it/s] 62%|██████▏   | 248916/400000 [00:23<00:13, 11258.82it/s] 63%|██████▎   | 250045/400000 [00:23<00:13, 10986.83it/s] 63%|██████▎   | 251195/400000 [00:23<00:13, 11135.47it/s] 63%|██████▎   | 252324/400000 [00:23<00:13, 11179.29it/s] 63%|██████▎   | 253464/400000 [00:23<00:13, 11244.26it/s] 64%|██████▎   | 254629/400000 [00:23<00:12, 11361.08it/s] 64%|██████▍   | 255767/400000 [00:23<00:12, 11190.59it/s] 64%|██████▍   | 256927/400000 [00:24<00:12, 11310.15it/s] 65%|██████▍   | 258082/400000 [00:24<00:12, 11380.66it/s] 65%|██████▍   | 259223/400000 [00:24<00:12, 11387.34it/s] 65%|██████▌   | 260383/400000 [00:24<00:12, 11449.29it/s] 65%|██████▌   | 261529/400000 [00:24<00:12, 11243.99it/s] 66%|██████▌   | 262655/400000 [00:24<00:12, 11034.06it/s] 66%|██████▌   | 263761/400000 [00:24<00:12, 10851.58it/s] 66%|██████▌   | 264849/400000 [00:24<00:12, 10827.79it/s] 66%|██████▋   | 266000/400000 [00:24<00:12, 11021.22it/s] 67%|██████▋   | 267104/400000 [00:24<00:12, 10957.03it/s] 67%|██████▋   | 268244/400000 [00:25<00:11, 11084.32it/s] 67%|██████▋   | 269406/400000 [00:25<00:11, 11238.65it/s] 68%|██████▊   | 270557/400000 [00:25<00:11, 11317.13it/s] 68%|██████▊   | 271727/400000 [00:25<00:11, 11427.67it/s] 68%|██████▊   | 272871/400000 [00:25<00:11, 11337.98it/s] 69%|██████▊   | 274011/400000 [00:25<00:11, 11353.63it/s] 69%|██████▉   | 275147/400000 [00:25<00:11, 11235.64it/s] 69%|██████▉   | 276303/400000 [00:25<00:10, 11329.21it/s] 69%|██████▉   | 277475/400000 [00:25<00:10, 11441.77it/s] 70%|██████▉   | 278620/400000 [00:26<00:10, 11097.00it/s] 70%|██████▉   | 279733/400000 [00:26<00:10, 10955.97it/s] 70%|███████   | 280831/400000 [00:26<00:10, 10877.35it/s] 70%|███████   | 281960/400000 [00:26<00:10, 10996.50it/s] 71%|███████   | 283062/400000 [00:26<00:10, 10963.19it/s] 71%|███████   | 284214/400000 [00:26<00:10, 11122.75it/s] 71%|███████▏  | 285328/400000 [00:26<00:10, 10603.33it/s] 72%|███████▏  | 286395/400000 [00:26<00:10, 10588.95it/s] 72%|███████▏  | 287536/400000 [00:26<00:10, 10822.31it/s] 72%|███████▏  | 288639/400000 [00:26<00:10, 10883.57it/s] 72%|███████▏  | 289768/400000 [00:27<00:10, 11002.38it/s] 73%|███████▎  | 290925/400000 [00:27<00:09, 11166.31it/s] 73%|███████▎  | 292103/400000 [00:27<00:09, 11342.30it/s] 73%|███████▎  | 293268/400000 [00:27<00:09, 11432.46it/s] 74%|███████▎  | 294414/400000 [00:27<00:09, 11373.41it/s] 74%|███████▍  | 295553/400000 [00:27<00:09, 11317.49it/s] 74%|███████▍  | 296686/400000 [00:27<00:09, 11303.08it/s] 74%|███████▍  | 297840/400000 [00:27<00:08, 11370.65it/s] 75%|███████▍  | 298978/400000 [00:27<00:09, 11081.96it/s] 75%|███████▌  | 300089/400000 [00:27<00:09, 10780.69it/s] 75%|███████▌  | 301171/400000 [00:28<00:09, 10463.00it/s] 76%|███████▌  | 302279/400000 [00:28<00:09, 10638.62it/s] 76%|███████▌  | 303423/400000 [00:28<00:08, 10866.21it/s] 76%|███████▌  | 304587/400000 [00:28<00:08, 11086.27it/s] 76%|███████▋  | 305709/400000 [00:28<00:08, 11124.60it/s] 77%|███████▋  | 306869/400000 [00:28<00:08, 11261.04it/s] 77%|███████▋  | 307998/400000 [00:28<00:08, 11220.05it/s] 77%|███████▋  | 309140/400000 [00:28<00:08, 11276.63it/s] 78%|███████▊  | 310269/400000 [00:28<00:07, 11230.74it/s] 78%|███████▊  | 311393/400000 [00:28<00:07, 11131.69it/s] 78%|███████▊  | 312518/400000 [00:29<00:07, 11165.03it/s] 78%|███████▊  | 313648/400000 [00:29<00:07, 11203.42it/s] 79%|███████▊  | 314789/400000 [00:29<00:07, 11262.14it/s] 79%|███████▉  | 315916/400000 [00:29<00:07, 11181.19it/s] 79%|███████▉  | 317035/400000 [00:29<00:07, 10975.87it/s] 80%|███████▉  | 318135/400000 [00:29<00:07, 10981.77it/s] 80%|███████▉  | 319254/400000 [00:29<00:07, 11043.04it/s] 80%|████████  | 320371/400000 [00:29<00:07, 11078.79it/s] 80%|████████  | 321480/400000 [00:29<00:07, 10774.14it/s] 81%|████████  | 322560/400000 [00:30<00:07, 10662.39it/s] 81%|████████  | 323629/400000 [00:30<00:07, 10619.98it/s] 81%|████████  | 324734/400000 [00:30<00:07, 10745.16it/s] 81%|████████▏ | 325904/400000 [00:30<00:06, 11013.11it/s] 82%|████████▏ | 327073/400000 [00:30<00:06, 11207.33it/s] 82%|████████▏ | 328229/400000 [00:30<00:06, 11307.27it/s] 82%|████████▏ | 329362/400000 [00:30<00:06, 11249.18it/s] 83%|████████▎ | 330542/400000 [00:30<00:06, 11406.47it/s] 83%|████████▎ | 331712/400000 [00:30<00:05, 11491.67it/s] 83%|████████▎ | 332863/400000 [00:30<00:05, 11484.35it/s] 84%|████████▎ | 334013/400000 [00:31<00:06, 10991.02it/s] 84%|████████▍ | 335120/400000 [00:31<00:05, 11014.08it/s] 84%|████████▍ | 336248/400000 [00:31<00:05, 11090.79it/s] 84%|████████▍ | 337406/400000 [00:31<00:05, 11231.19it/s] 85%|████████▍ | 338546/400000 [00:31<00:05, 11280.65it/s] 85%|████████▍ | 339676/400000 [00:31<00:05, 11010.61it/s] 85%|████████▌ | 340780/400000 [00:31<00:05, 10941.51it/s] 85%|████████▌ | 341899/400000 [00:31<00:05, 11012.36it/s] 86%|████████▌ | 343037/400000 [00:31<00:05, 11119.31it/s] 86%|████████▌ | 344204/400000 [00:31<00:04, 11277.99it/s] 86%|████████▋ | 345334/400000 [00:32<00:04, 11220.61it/s] 87%|████████▋ | 346458/400000 [00:32<00:04, 11063.53it/s] 87%|████████▋ | 347632/400000 [00:32<00:04, 11255.54it/s] 87%|████████▋ | 348805/400000 [00:32<00:04, 11391.49it/s] 87%|████████▋ | 349957/400000 [00:32<00:04, 11428.53it/s] 88%|████████▊ | 351101/400000 [00:32<00:04, 11274.26it/s] 88%|████████▊ | 352230/400000 [00:32<00:04, 10742.32it/s] 88%|████████▊ | 353375/400000 [00:32<00:04, 10942.98it/s] 89%|████████▊ | 354475/400000 [00:32<00:04, 10931.79it/s] 89%|████████▉ | 355572/400000 [00:32<00:04, 10875.70it/s] 89%|████████▉ | 356712/400000 [00:33<00:03, 11027.00it/s] 89%|████████▉ | 357818/400000 [00:33<00:03, 11000.35it/s] 90%|████████▉ | 358942/400000 [00:33<00:03, 11069.18it/s] 90%|█████████ | 360099/400000 [00:33<00:03, 11213.53it/s] 90%|█████████ | 361267/400000 [00:33<00:03, 11348.90it/s] 91%|█████████ | 362438/400000 [00:33<00:03, 11453.64it/s] 91%|█████████ | 363585/400000 [00:33<00:03, 11213.13it/s] 91%|█████████ | 364744/400000 [00:33<00:03, 11323.46it/s] 91%|█████████▏| 365900/400000 [00:33<00:02, 11392.01it/s] 92%|█████████▏| 367041/400000 [00:33<00:02, 11360.34it/s] 92%|█████████▏| 368201/400000 [00:34<00:02, 11430.59it/s] 92%|█████████▏| 369345/400000 [00:34<00:02, 11237.35it/s] 93%|█████████▎| 370470/400000 [00:34<00:02, 10968.44it/s] 93%|█████████▎| 371581/400000 [00:34<00:02, 11008.13it/s] 93%|█████████▎| 372722/400000 [00:34<00:02, 11125.06it/s] 93%|█████████▎| 373836/400000 [00:34<00:02, 10970.81it/s] 94%|█████████▎| 374935/400000 [00:34<00:02, 10670.89it/s] 94%|█████████▍| 376005/400000 [00:34<00:02, 10469.16it/s] 94%|█████████▍| 377055/400000 [00:34<00:02, 10471.96it/s] 95%|█████████▍| 378135/400000 [00:35<00:02, 10566.13it/s] 95%|█████████▍| 379194/400000 [00:35<00:02, 10363.35it/s] 95%|█████████▌| 380312/400000 [00:35<00:01, 10593.71it/s] 95%|█████████▌| 381429/400000 [00:35<00:01, 10759.01it/s] 96%|█████████▌| 382520/400000 [00:35<00:01, 10802.77it/s] 96%|█████████▌| 383603/400000 [00:35<00:01, 10786.14it/s] 96%|█████████▌| 384683/400000 [00:35<00:01, 10694.27it/s] 96%|█████████▋| 385754/400000 [00:35<00:01, 10512.79it/s] 97%|█████████▋| 386838/400000 [00:35<00:01, 10607.34it/s] 97%|█████████▋| 387920/400000 [00:35<00:01, 10668.11it/s] 97%|█████████▋| 389027/400000 [00:36<00:01, 10785.42it/s] 98%|█████████▊| 390150/400000 [00:36<00:00, 10914.83it/s] 98%|█████████▊| 391243/400000 [00:36<00:00, 10891.28it/s] 98%|█████████▊| 392376/400000 [00:36<00:00, 11018.61it/s] 98%|█████████▊| 393479/400000 [00:36<00:00, 10953.45it/s] 99%|█████████▊| 394620/400000 [00:36<00:00, 11084.47it/s] 99%|█████████▉| 395730/400000 [00:36<00:00, 10937.43it/s] 99%|█████████▉| 396825/400000 [00:36<00:00, 10920.63it/s] 99%|█████████▉| 397983/400000 [00:36<00:00, 11107.96it/s]100%|█████████▉| 399155/400000 [00:36<00:00, 11282.53it/s]100%|█████████▉| 399999/400000 [00:37<00:00, 10808.95it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7567e66c50>, <torchtext.data.dataset.TabularDataset object at 0x7f7567e66da0>, <torchtext.vocab.Vocab object at 0x7f7567e66cc0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.63 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.37 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 16.37 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 16.37 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00, 10.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.08 file/s]2020-07-18 18:16:47.629916: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-18 18:16:47.633264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-18 18:16:47.633396: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d00fb42d70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-18 18:16:47.633408: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 145853.85it/s] 67%|██████▋   | 6635520/9912422 [00:00<00:15, 208164.16it/s]9920512it [00:00, 41166771.04it/s]                           
0it [00:00, ?it/s]32768it [00:00, 565503.97it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 471761.66it/s]1654784it [00:00, 12007368.34it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191499.17it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
