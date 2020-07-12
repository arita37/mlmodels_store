
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f04205ea048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f04205ea048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f048c1921e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f048c1921e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f04aa4d9ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f04aa4d9ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f04394c0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f04394c0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f04394c0620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4071424/9912422 [00:00<00:00, 40713308.15it/s]9920512it [00:00, 36764163.22it/s]                             
0it [00:00, ?it/s]32768it [00:00, 944273.13it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485123.18it/s]1654784it [00:00, 12163293.43it/s]                         
0it [00:00, ?it/s]8192it [00:00, 219659.12it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04368b5940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04204f2be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f04394c0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f04394c0268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f04394c0268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:33,  1.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:33,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:32,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:32,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:30,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:29,  1.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:02,  2.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:02,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:02,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:01,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:01,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:00,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:00,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:00,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:59,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:59,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:41,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:41,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:40,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:40,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:40,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:39,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:39,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:38,  3.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:26,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:25,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:25,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:25,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:18,  6.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:18,  6.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 12.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 12.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 30.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 37.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 46.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 54.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.05 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.05 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.18s/ url]
0 examples [00:00, ? examples/s]2020-07-12 18:10:23.711589: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-12 18:10:23.725524: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-07-12 18:10:23.725721: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555d72073060 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 18:10:23.725737: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
26 examples [00:00, 258.51 examples/s]148 examples [00:00, 338.35 examples/s]260 examples [00:00, 427.63 examples/s]367 examples [00:00, 521.51 examples/s]480 examples [00:00, 621.01 examples/s]601 examples [00:00, 726.85 examples/s]726 examples [00:00, 830.78 examples/s]849 examples [00:00, 919.17 examples/s]971 examples [00:00, 991.61 examples/s]1093 examples [00:01, 1048.96 examples/s]1214 examples [00:01, 1091.20 examples/s]1334 examples [00:01, 1119.00 examples/s]1457 examples [00:01, 1147.67 examples/s]1577 examples [00:01, 1152.07 examples/s]1696 examples [00:01, 1147.26 examples/s]1813 examples [00:01, 1118.28 examples/s]1927 examples [00:01, 1107.61 examples/s]2044 examples [00:01, 1125.56 examples/s]2164 examples [00:01, 1145.21 examples/s]2287 examples [00:02, 1168.19 examples/s]2413 examples [00:02, 1194.28 examples/s]2538 examples [00:02, 1209.58 examples/s]2660 examples [00:02, 1201.03 examples/s]2781 examples [00:02, 1195.35 examples/s]2901 examples [00:02, 1118.22 examples/s]3014 examples [00:02, 1103.49 examples/s]3126 examples [00:02, 1106.62 examples/s]3251 examples [00:02, 1144.34 examples/s]3375 examples [00:02, 1169.94 examples/s]3493 examples [00:03, 1160.21 examples/s]3610 examples [00:03, 1121.73 examples/s]3723 examples [00:03, 1115.23 examples/s]3835 examples [00:03, 1102.58 examples/s]3948 examples [00:03, 1110.18 examples/s]4060 examples [00:03, 1105.86 examples/s]4171 examples [00:03, 1101.78 examples/s]4284 examples [00:03, 1110.03 examples/s]4403 examples [00:03, 1130.87 examples/s]4517 examples [00:03, 1122.97 examples/s]4630 examples [00:04, 1101.20 examples/s]4746 examples [00:04, 1118.12 examples/s]4859 examples [00:04, 1109.22 examples/s]4971 examples [00:04, 1072.38 examples/s]5080 examples [00:04, 1076.29 examples/s]5188 examples [00:04, 1073.84 examples/s]5301 examples [00:04, 1089.65 examples/s]5411 examples [00:04, 1088.73 examples/s]5521 examples [00:04, 1078.99 examples/s]5633 examples [00:05, 1090.68 examples/s]5743 examples [00:05, 1085.30 examples/s]5858 examples [00:05, 1102.29 examples/s]5970 examples [00:05, 1104.21 examples/s]6081 examples [00:05, 1078.76 examples/s]6191 examples [00:05, 1082.45 examples/s]6303 examples [00:05, 1091.70 examples/s]6414 examples [00:05, 1095.82 examples/s]6524 examples [00:05, 1094.05 examples/s]6640 examples [00:05, 1111.61 examples/s]6752 examples [00:06, 1099.71 examples/s]6864 examples [00:06, 1103.13 examples/s]6985 examples [00:06, 1131.60 examples/s]7102 examples [00:06, 1142.36 examples/s]7217 examples [00:06, 1139.44 examples/s]7332 examples [00:06, 1124.05 examples/s]7445 examples [00:06, 1119.04 examples/s]7560 examples [00:06, 1125.59 examples/s]7673 examples [00:06, 1114.95 examples/s]7785 examples [00:06, 1108.94 examples/s]7906 examples [00:07, 1136.44 examples/s]8021 examples [00:07, 1138.26 examples/s]8135 examples [00:07, 1105.86 examples/s]8253 examples [00:07, 1125.45 examples/s]8373 examples [00:07, 1145.69 examples/s]8494 examples [00:07, 1162.07 examples/s]8611 examples [00:07, 1129.49 examples/s]8725 examples [00:07, 1118.08 examples/s]8843 examples [00:07, 1134.42 examples/s]8959 examples [00:07, 1140.54 examples/s]9074 examples [00:08, 1138.61 examples/s]9188 examples [00:08, 1133.51 examples/s]9303 examples [00:08, 1136.39 examples/s]9417 examples [00:08, 1125.84 examples/s]9535 examples [00:08, 1139.73 examples/s]9650 examples [00:08, 1103.71 examples/s]9763 examples [00:08, 1110.29 examples/s]9878 examples [00:08, 1120.77 examples/s]9996 examples [00:08, 1136.31 examples/s]10110 examples [00:09, 1083.43 examples/s]10222 examples [00:09, 1092.42 examples/s]10344 examples [00:09, 1126.72 examples/s]10471 examples [00:09, 1164.21 examples/s]10589 examples [00:09, 1164.79 examples/s]10711 examples [00:09, 1180.48 examples/s]10830 examples [00:09, 1180.00 examples/s]10955 examples [00:09, 1199.27 examples/s]11076 examples [00:09, 1196.31 examples/s]11197 examples [00:09, 1198.89 examples/s]11318 examples [00:10, 1201.40 examples/s]11441 examples [00:10, 1207.67 examples/s]11563 examples [00:10, 1209.82 examples/s]11685 examples [00:10, 1184.06 examples/s]11805 examples [00:10, 1185.92 examples/s]11924 examples [00:10, 1160.18 examples/s]12044 examples [00:10, 1169.65 examples/s]12162 examples [00:10, 1172.66 examples/s]12280 examples [00:10, 1166.72 examples/s]12399 examples [00:10, 1173.49 examples/s]12519 examples [00:11, 1180.38 examples/s]12638 examples [00:11, 1150.50 examples/s]12754 examples [00:11, 1124.51 examples/s]12867 examples [00:11, 1109.30 examples/s]12986 examples [00:11, 1129.82 examples/s]13104 examples [00:11, 1143.41 examples/s]13219 examples [00:11, 1136.39 examples/s]13343 examples [00:11, 1165.25 examples/s]13460 examples [00:11, 1161.56 examples/s]13581 examples [00:11, 1172.64 examples/s]13699 examples [00:12, 1146.70 examples/s]13814 examples [00:12, 1135.58 examples/s]13939 examples [00:12, 1167.57 examples/s]14062 examples [00:12, 1185.54 examples/s]14181 examples [00:12, 1178.02 examples/s]14300 examples [00:12, 1174.77 examples/s]14418 examples [00:12, 1165.78 examples/s]14535 examples [00:12, 1147.80 examples/s]14658 examples [00:12, 1171.18 examples/s]14779 examples [00:13, 1181.77 examples/s]14899 examples [00:13, 1186.45 examples/s]15018 examples [00:13, 1186.61 examples/s]15144 examples [00:13, 1207.28 examples/s]15265 examples [00:13, 1207.58 examples/s]15386 examples [00:13, 1195.70 examples/s]15506 examples [00:13, 1179.36 examples/s]15625 examples [00:13, 1169.41 examples/s]15743 examples [00:13, 1147.21 examples/s]15858 examples [00:13, 1097.91 examples/s]15972 examples [00:14, 1108.23 examples/s]16089 examples [00:14, 1123.50 examples/s]16204 examples [00:14, 1129.78 examples/s]16327 examples [00:14, 1156.14 examples/s]16449 examples [00:14, 1172.48 examples/s]16574 examples [00:14, 1194.45 examples/s]16700 examples [00:14, 1211.15 examples/s]16822 examples [00:14, 1202.99 examples/s]16947 examples [00:14, 1215.22 examples/s]17076 examples [00:14, 1235.03 examples/s]17205 examples [00:15, 1249.22 examples/s]17332 examples [00:15, 1254.51 examples/s]17458 examples [00:15, 1244.46 examples/s]17584 examples [00:15, 1248.40 examples/s]17709 examples [00:15, 1231.97 examples/s]17833 examples [00:15, 1205.97 examples/s]17956 examples [00:15, 1210.36 examples/s]18078 examples [00:15, 1205.78 examples/s]18202 examples [00:15, 1213.12 examples/s]18324 examples [00:15, 1205.66 examples/s]18451 examples [00:16, 1221.78 examples/s]18577 examples [00:16, 1231.44 examples/s]18701 examples [00:16, 1210.39 examples/s]18823 examples [00:16, 1212.53 examples/s]18945 examples [00:16, 1210.25 examples/s]19067 examples [00:16, 1187.87 examples/s]19186 examples [00:16, 1177.93 examples/s]19305 examples [00:16, 1178.14 examples/s]19427 examples [00:16, 1188.74 examples/s]19548 examples [00:17, 1193.65 examples/s]19671 examples [00:17, 1203.12 examples/s]19792 examples [00:17, 1189.25 examples/s]19912 examples [00:17, 1142.93 examples/s]20027 examples [00:17, 1104.03 examples/s]20145 examples [00:17, 1123.81 examples/s]20263 examples [00:17, 1137.83 examples/s]20379 examples [00:17, 1144.33 examples/s]20494 examples [00:17, 1130.27 examples/s]20611 examples [00:17, 1141.31 examples/s]20730 examples [00:18, 1154.54 examples/s]20853 examples [00:18, 1175.59 examples/s]20981 examples [00:18, 1201.82 examples/s]21102 examples [00:18, 1186.65 examples/s]21223 examples [00:18, 1193.07 examples/s]21344 examples [00:18, 1196.40 examples/s]21464 examples [00:18, 1194.69 examples/s]21589 examples [00:18, 1208.94 examples/s]21710 examples [00:18, 1181.30 examples/s]21829 examples [00:18, 1179.12 examples/s]21952 examples [00:19, 1193.21 examples/s]22074 examples [00:19, 1198.77 examples/s]22195 examples [00:19, 1199.99 examples/s]22316 examples [00:19, 1172.46 examples/s]22434 examples [00:19, 1166.55 examples/s]22551 examples [00:19, 1160.62 examples/s]22669 examples [00:19, 1165.35 examples/s]22791 examples [00:19, 1180.52 examples/s]22910 examples [00:19, 1151.94 examples/s]23035 examples [00:19, 1178.84 examples/s]23160 examples [00:20, 1198.36 examples/s]23281 examples [00:20, 1200.81 examples/s]23406 examples [00:20, 1213.01 examples/s]23528 examples [00:20, 1179.70 examples/s]23647 examples [00:20, 1157.25 examples/s]23764 examples [00:20, 1142.09 examples/s]23887 examples [00:20, 1165.72 examples/s]24007 examples [00:20, 1174.84 examples/s]24126 examples [00:20, 1179.00 examples/s]24246 examples [00:21, 1183.14 examples/s]24366 examples [00:21, 1187.79 examples/s]24491 examples [00:21, 1202.82 examples/s]24612 examples [00:21, 1200.58 examples/s]24733 examples [00:21, 1200.15 examples/s]24854 examples [00:21, 1201.21 examples/s]24975 examples [00:21, 1192.06 examples/s]25095 examples [00:21, 1184.94 examples/s]25216 examples [00:21, 1190.31 examples/s]25336 examples [00:21, 1189.86 examples/s]25460 examples [00:22, 1203.69 examples/s]25582 examples [00:22, 1207.97 examples/s]25703 examples [00:22, 1200.83 examples/s]25826 examples [00:22, 1207.95 examples/s]25947 examples [00:22, 1206.54 examples/s]26068 examples [00:22, 1164.66 examples/s]26185 examples [00:22, 1138.04 examples/s]26300 examples [00:22, 1126.55 examples/s]26418 examples [00:22, 1140.85 examples/s]26539 examples [00:22, 1159.94 examples/s]26662 examples [00:23, 1178.49 examples/s]26790 examples [00:23, 1202.70 examples/s]26911 examples [00:23, 1185.76 examples/s]27030 examples [00:23, 1170.77 examples/s]27148 examples [00:23, 1160.42 examples/s]27265 examples [00:23, 1149.99 examples/s]27383 examples [00:23, 1158.37 examples/s]27505 examples [00:23, 1173.80 examples/s]27623 examples [00:23, 1168.47 examples/s]27740 examples [00:23, 1142.71 examples/s]27865 examples [00:24, 1171.89 examples/s]27991 examples [00:24, 1196.17 examples/s]28116 examples [00:24, 1210.11 examples/s]28238 examples [00:24, 1203.01 examples/s]28361 examples [00:24, 1210.91 examples/s]28484 examples [00:24, 1215.85 examples/s]28606 examples [00:24, 1190.74 examples/s]28726 examples [00:24, 1173.95 examples/s]28844 examples [00:24, 1157.07 examples/s]28960 examples [00:25, 1143.08 examples/s]29090 examples [00:25, 1185.54 examples/s]29210 examples [00:25, 1170.52 examples/s]29329 examples [00:25, 1174.90 examples/s]29447 examples [00:25, 1153.39 examples/s]29563 examples [00:25, 1154.26 examples/s]29679 examples [00:25, 1148.66 examples/s]29799 examples [00:25, 1162.53 examples/s]29922 examples [00:25, 1180.17 examples/s]30041 examples [00:25, 1103.71 examples/s]30167 examples [00:26, 1145.32 examples/s]30290 examples [00:26, 1166.75 examples/s]30417 examples [00:26, 1194.67 examples/s]30541 examples [00:26, 1207.87 examples/s]30663 examples [00:26, 1174.44 examples/s]30782 examples [00:26, 1158.40 examples/s]30901 examples [00:26, 1166.90 examples/s]31023 examples [00:26, 1182.25 examples/s]31150 examples [00:26, 1205.29 examples/s]31271 examples [00:26, 1188.33 examples/s]31401 examples [00:27, 1217.37 examples/s]31524 examples [00:27, 1218.34 examples/s]31647 examples [00:27, 1197.92 examples/s]31768 examples [00:27, 1185.11 examples/s]31889 examples [00:27, 1191.41 examples/s]32009 examples [00:27, 1190.32 examples/s]32129 examples [00:27, 1175.93 examples/s]32247 examples [00:27, 1160.13 examples/s]32364 examples [00:27, 1130.05 examples/s]32478 examples [00:28, 1128.15 examples/s]32598 examples [00:28, 1148.61 examples/s]32720 examples [00:28, 1166.76 examples/s]32846 examples [00:28, 1192.72 examples/s]32971 examples [00:28, 1209.24 examples/s]33093 examples [00:28, 1203.95 examples/s]33214 examples [00:28, 1190.96 examples/s]33334 examples [00:28, 1176.75 examples/s]33452 examples [00:28, 1174.03 examples/s]33570 examples [00:28, 1162.61 examples/s]33690 examples [00:29, 1171.60 examples/s]33812 examples [00:29, 1184.78 examples/s]33935 examples [00:29, 1195.51 examples/s]34055 examples [00:29, 1176.63 examples/s]34178 examples [00:29, 1191.09 examples/s]34298 examples [00:29, 1192.38 examples/s]34421 examples [00:29, 1202.49 examples/s]34545 examples [00:29, 1210.81 examples/s]34669 examples [00:29, 1217.75 examples/s]34791 examples [00:29, 1197.79 examples/s]34911 examples [00:30, 1172.91 examples/s]35029 examples [00:30, 1172.53 examples/s]35147 examples [00:30, 1164.61 examples/s]35264 examples [00:30, 1160.59 examples/s]35382 examples [00:30, 1163.92 examples/s]35500 examples [00:30, 1167.50 examples/s]35618 examples [00:30, 1170.43 examples/s]35738 examples [00:30, 1177.10 examples/s]35858 examples [00:30, 1181.88 examples/s]35978 examples [00:30, 1185.00 examples/s]36102 examples [00:31, 1200.05 examples/s]36223 examples [00:31, 1198.55 examples/s]36344 examples [00:31, 1199.66 examples/s]36465 examples [00:31, 1200.82 examples/s]36586 examples [00:31, 1179.73 examples/s]36705 examples [00:31, 1166.30 examples/s]36822 examples [00:31, 1151.61 examples/s]36938 examples [00:31, 1110.81 examples/s]37055 examples [00:31, 1126.27 examples/s]37170 examples [00:31, 1131.45 examples/s]37296 examples [00:32, 1166.80 examples/s]37416 examples [00:32, 1176.06 examples/s]37544 examples [00:32, 1205.42 examples/s]37665 examples [00:32, 1199.02 examples/s]37786 examples [00:32, 1171.79 examples/s]37904 examples [00:32, 1126.04 examples/s]38024 examples [00:32, 1145.75 examples/s]38142 examples [00:32, 1153.97 examples/s]38258 examples [00:32, 1146.46 examples/s]38373 examples [00:33, 1129.66 examples/s]38497 examples [00:33, 1159.72 examples/s]38614 examples [00:33, 1156.92 examples/s]38740 examples [00:33, 1184.53 examples/s]38860 examples [00:33, 1188.21 examples/s]38980 examples [00:33, 1186.93 examples/s]39099 examples [00:33, 1178.82 examples/s]39218 examples [00:33, 1156.72 examples/s]39338 examples [00:33, 1169.27 examples/s]39456 examples [00:33, 1170.20 examples/s]39574 examples [00:34, 1118.53 examples/s]39691 examples [00:34, 1131.98 examples/s]39806 examples [00:34, 1135.66 examples/s]39920 examples [00:34, 1119.30 examples/s]40033 examples [00:34, 1068.69 examples/s]40148 examples [00:34, 1090.22 examples/s]40269 examples [00:34, 1123.09 examples/s]40389 examples [00:34, 1144.24 examples/s]40509 examples [00:34, 1159.26 examples/s]40635 examples [00:34, 1185.21 examples/s]40754 examples [00:35, 1174.04 examples/s]40873 examples [00:35, 1176.54 examples/s]40991 examples [00:35, 1163.15 examples/s]41108 examples [00:35, 1130.55 examples/s]41222 examples [00:35, 1116.89 examples/s]41334 examples [00:35, 1111.02 examples/s]41446 examples [00:35, 1111.16 examples/s]41562 examples [00:35, 1122.82 examples/s]41679 examples [00:35, 1135.79 examples/s]41793 examples [00:36, 1093.01 examples/s]41910 examples [00:36, 1112.02 examples/s]42031 examples [00:36, 1138.97 examples/s]42153 examples [00:36, 1160.93 examples/s]42272 examples [00:36, 1168.64 examples/s]42396 examples [00:36, 1188.77 examples/s]42516 examples [00:36, 1167.85 examples/s]42636 examples [00:36, 1175.67 examples/s]42756 examples [00:36, 1181.77 examples/s]42878 examples [00:36, 1192.43 examples/s]42998 examples [00:37, 1191.82 examples/s]43118 examples [00:37, 1182.52 examples/s]43241 examples [00:37, 1196.28 examples/s]43361 examples [00:37, 1159.17 examples/s]43480 examples [00:37, 1168.25 examples/s]43598 examples [00:37, 1148.39 examples/s]43714 examples [00:37, 1121.12 examples/s]43828 examples [00:37, 1123.62 examples/s]43948 examples [00:37, 1143.61 examples/s]44065 examples [00:37, 1150.81 examples/s]44181 examples [00:38, 1115.00 examples/s]44293 examples [00:38, 1110.18 examples/s]44418 examples [00:38, 1147.94 examples/s]44539 examples [00:38, 1163.25 examples/s]44663 examples [00:38, 1184.53 examples/s]44782 examples [00:38, 1176.98 examples/s]44900 examples [00:38, 1172.49 examples/s]45018 examples [00:38, 1173.41 examples/s]45138 examples [00:38, 1178.23 examples/s]45268 examples [00:38, 1211.12 examples/s]45390 examples [00:39, 1205.28 examples/s]45515 examples [00:39, 1216.45 examples/s]45637 examples [00:39, 1207.18 examples/s]45758 examples [00:39, 1179.53 examples/s]45877 examples [00:39, 1172.20 examples/s]45995 examples [00:39, 1166.63 examples/s]46115 examples [00:39, 1173.56 examples/s]46242 examples [00:39, 1198.64 examples/s]46363 examples [00:39, 1178.96 examples/s]46486 examples [00:40, 1191.27 examples/s]46609 examples [00:40, 1201.54 examples/s]46733 examples [00:40, 1211.24 examples/s]46855 examples [00:40, 1202.36 examples/s]46979 examples [00:40, 1213.05 examples/s]47101 examples [00:40, 1211.63 examples/s]47223 examples [00:40, 1174.40 examples/s]47341 examples [00:40, 1161.49 examples/s]47458 examples [00:40, 1151.54 examples/s]47574 examples [00:40, 1148.69 examples/s]47690 examples [00:41, 1126.98 examples/s]47803 examples [00:41, 1112.32 examples/s]47916 examples [00:41, 1116.85 examples/s]48037 examples [00:41, 1142.38 examples/s]48152 examples [00:41, 1136.15 examples/s]48271 examples [00:41, 1150.02 examples/s]48387 examples [00:41, 1129.91 examples/s]48510 examples [00:41, 1156.28 examples/s]48630 examples [00:41, 1168.22 examples/s]48751 examples [00:41, 1179.61 examples/s]48871 examples [00:42, 1184.83 examples/s]48990 examples [00:42, 1181.75 examples/s]49109 examples [00:42, 1168.95 examples/s]49232 examples [00:42, 1185.52 examples/s]49361 examples [00:42, 1213.66 examples/s]49483 examples [00:42, 1154.89 examples/s]49600 examples [00:42, 1106.05 examples/s]49712 examples [00:42, 1098.98 examples/s]49831 examples [00:42, 1123.25 examples/s]49950 examples [00:43, 1142.35 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8693/50000 [00:00<00:00, 86929.46 examples/s] 47%|████▋     | 23545/50000 [00:00<00:00, 99280.44 examples/s] 75%|███████▍  | 37314/50000 [00:00<00:00, 108346.49 examples/s]                                                                0 examples [00:00, ? examples/s]104 examples [00:00, 1035.93 examples/s]220 examples [00:00, 1069.45 examples/s]334 examples [00:00, 1087.80 examples/s]457 examples [00:00, 1124.88 examples/s]576 examples [00:00, 1143.02 examples/s]703 examples [00:00, 1177.71 examples/s]827 examples [00:00, 1194.65 examples/s]947 examples [00:00, 1196.01 examples/s]1069 examples [00:00, 1199.65 examples/s]1190 examples [00:01, 1201.04 examples/s]1309 examples [00:01, 1195.51 examples/s]1427 examples [00:01, 1187.19 examples/s]1545 examples [00:01, 1161.61 examples/s]1661 examples [00:01, 1150.42 examples/s]1787 examples [00:01, 1180.38 examples/s]1905 examples [00:01, 1178.46 examples/s]2024 examples [00:01, 1180.21 examples/s]2146 examples [00:01, 1187.55 examples/s]2269 examples [00:01, 1196.71 examples/s]2395 examples [00:02, 1212.88 examples/s]2517 examples [00:02, 1202.88 examples/s]2638 examples [00:02, 1196.66 examples/s]2760 examples [00:02, 1202.65 examples/s]2881 examples [00:02, 1196.78 examples/s]3002 examples [00:02, 1200.07 examples/s]3127 examples [00:02, 1213.93 examples/s]3249 examples [00:02, 1209.44 examples/s]3370 examples [00:02, 1180.88 examples/s]3489 examples [00:02, 1162.30 examples/s]3606 examples [00:03, 1162.14 examples/s]3723 examples [00:03, 1156.09 examples/s]3840 examples [00:03, 1158.92 examples/s]3962 examples [00:03, 1175.26 examples/s]4084 examples [00:03, 1186.70 examples/s]4203 examples [00:03, 1183.74 examples/s]4328 examples [00:03, 1200.73 examples/s]4455 examples [00:03, 1220.27 examples/s]4578 examples [00:03, 1207.72 examples/s]4699 examples [00:03, 1206.14 examples/s]4821 examples [00:04, 1208.25 examples/s]4942 examples [00:04, 1204.94 examples/s]5063 examples [00:04, 1197.68 examples/s]5183 examples [00:04, 1198.27 examples/s]5303 examples [00:04, 1186.47 examples/s]5422 examples [00:04, 1180.47 examples/s]5541 examples [00:04, 1162.17 examples/s]5658 examples [00:04, 1158.59 examples/s]5781 examples [00:04, 1179.05 examples/s]5904 examples [00:04, 1192.99 examples/s]6024 examples [00:05, 1178.83 examples/s]6143 examples [00:05, 1158.27 examples/s]6259 examples [00:05, 1150.49 examples/s]6379 examples [00:05, 1164.07 examples/s]6502 examples [00:05, 1181.18 examples/s]6626 examples [00:05, 1196.76 examples/s]6751 examples [00:05, 1210.49 examples/s]6873 examples [00:05, 1209.27 examples/s]6995 examples [00:05, 1203.02 examples/s]7119 examples [00:05, 1213.37 examples/s]7241 examples [00:06, 1197.22 examples/s]7361 examples [00:06, 1184.71 examples/s]7480 examples [00:06, 1180.21 examples/s]7603 examples [00:06, 1193.56 examples/s]7723 examples [00:06, 1186.40 examples/s]7842 examples [00:06, 1123.42 examples/s]7956 examples [00:06, 1114.56 examples/s]8068 examples [00:06, 1111.74 examples/s]8184 examples [00:06, 1124.74 examples/s]8303 examples [00:07, 1141.54 examples/s]8419 examples [00:07, 1144.81 examples/s]8534 examples [00:07, 1132.23 examples/s]8648 examples [00:07, 1129.36 examples/s]8762 examples [00:07, 1121.21 examples/s]8875 examples [00:07, 1113.53 examples/s]8999 examples [00:07, 1146.12 examples/s]9119 examples [00:07, 1161.36 examples/s]9236 examples [00:07, 1159.18 examples/s]9356 examples [00:07, 1170.52 examples/s]9483 examples [00:08, 1196.17 examples/s]9603 examples [00:08, 1194.33 examples/s]9723 examples [00:08, 1170.98 examples/s]9841 examples [00:08, 1157.29 examples/s]9958 examples [00:08, 1159.68 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUC3Y39/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUC3Y39/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f04394c0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f04394c0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f04394c0620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f03d7a4bfd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04366e5668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f04394c0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f04394c0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f04394c0620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04204f25c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f04204f2320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f03b8add598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f03b8add598> 

  function with postional parmater data_info <function split_train_valid at 0x7f03b8add598> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f03b8add6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f03b8add6a8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f03b8add6a8> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=e1142f3c18ce576c8360d7a83c10b5d98415a85ff54815e8fb2cf93f03e70931
  Stored in directory: /tmp/pip-ephem-wheel-cache-_qj3c_35/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:10:04, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:45:12, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:04:51, 21.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:45:52, 30.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:25:17, 44.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.67M/862M [00:01<3:46:11, 62.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.73M/862M [00:01<2:47:36, 84.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:56:33, 121kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<1:21:11, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:01<56:38, 246kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<39:32, 351kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:02<27:38, 499kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:02<19:18, 710kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.2M/862M [00:02<13:39, 1.00MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<09:36, 1.41MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<16:58, 796kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<13:44, 978kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.0M/862M [00:04<11:27, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<08:23, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<21:12, 630kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<16:12, 825kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:06<11:39, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<11:17, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<09:15, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:08<06:48, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<07:53, 1.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<08:15, 1.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<06:27, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<04:40, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<1:37:20, 135kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<1:09:26, 189kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<48:46, 269kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:12<34:14, 382kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<1:12:38, 180kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<51:56, 252kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.0M/862M [00:14<36:36, 357kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:14<25:43, 507kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<1:07:20, 193kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<48:29, 269kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<34:20, 379kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.6M/862M [00:16<24:12, 536kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<23:04, 562kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<17:42, 732kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.2M/862M [00:18<12:46, 1.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<11:33, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<11:48, 1.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<09:02, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<06:38, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<07:23, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<07:00, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:17, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<03:50, 3.32MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<27:15, 468kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<20:52, 610kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<15:03, 845kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<12:52, 985kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:32, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:03, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:49, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:51, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:00, 1.57MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:03, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:34, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:21, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:54, 2.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:31, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:13, 2.95MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:32<03:12, 3.88MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:56, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:40, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:07, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:52, 2.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:29, 2.74MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:26, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:32, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:16, 2.86MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:16, 2.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:27, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:14, 2.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:14, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:24, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:10, 2.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:18, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<04:05, 2.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<02:58, 4.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:56:53, 16.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:23:19, 23.8kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<5:52:12, 34.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<4:07:50, 48.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:56:56, 67.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:04:29, 95.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<1:27:21, 136kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:03:21, 187kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<46:00, 258kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<32:31, 364kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<22:51, 516kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<25:23, 464kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<21:14, 555kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<15:33, 756kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<11:02, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<11:32, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:46, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:15, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<07:14, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:41, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:05, 2.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:43, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:39, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:18, 2.69MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:09, 3.65MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<12:11, 944kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:52, 969kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<09:07, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:32, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:03, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:00, 1.43MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:59, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:19, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:01, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:33, 2.49MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:18, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:06, 2.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:54, 2.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:29, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:13, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:50, 2.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:19, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:42, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:05, 2.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:37, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:30, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:11, 2.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:02, 3.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:35:51, 115kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<1:08:39, 160kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<48:19, 228kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<35:45, 306kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<26:32, 412kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<18:53, 579kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<13:17, 819kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<52:35, 207kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<38:23, 283kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<27:13, 399kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<20:58, 515kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<16:15, 664kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<11:42, 922kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<10:11, 1.05MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:26, 1.27MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:16, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:23, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:59, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:32, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:09, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:09, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<03:57, 2.67MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:17, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:08, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:45, 2.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:42, 1.56MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<06:09, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<04:37, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:19, 3.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:22:55, 16.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<7:17:22, 23.8kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<5:05:55, 33.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<3:33:23, 48.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<2:53:54, 59.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<2:03:10, 83.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<1:26:25, 119kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<1:02:09, 165kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<44:56, 228kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<31:42, 323kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<24:01, 424kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<18:15, 558kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<13:14, 769kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<09:23, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<10:02, 1.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<09:55, 1.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:34, 1.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:25, 1.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<07:12, 1.39MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<06:29, 1.55MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:50, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:15, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:05, 1.96MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<03:51, 2.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<02:48, 3.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:42, 925kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<10:12, 970kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:50, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<05:37, 1.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:47, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<06:50, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<05:04, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:38, 2.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<38:59, 251kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<28:41, 340kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<20:22, 478kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<16:01, 605kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<12:37, 768kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<09:09, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:11, 1.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:14, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:27, 1.75MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:09, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<03:56, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:31, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:32, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<03:27, 2.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<02:33, 3.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:28, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<07:01, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:11, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:34, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:26, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<05:09, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:45, 2.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<06:20, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:43, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:38, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:34, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:11, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:13, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:22, 3.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:00, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<08:12, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:18, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:33, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:44, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:18, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:00, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:28, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:23, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:20, 2.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:26, 3.64MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<31:03, 286kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<22:58, 386kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<16:19, 542kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<13:01, 676kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<10:22, 847kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<07:33, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:54, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:02, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:45, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:32, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:26, 2.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:30, 3.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<14:13, 604kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:11, 767kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<08:07, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:15, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:17, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:41, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:51, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:34, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:30, 2.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:00, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:00, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:05, 2.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<03:43, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:45, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<02:55, 2.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:34, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:41, 2.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:50, 2.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<02:04, 3.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<16:11, 505kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<12:31, 652kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<09:02, 902kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:48, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:36, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:53, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:54, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:36, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:21, 2.38MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:29, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:59, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:45, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:06, 1.92MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:58, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:01, 2.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<03:34, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:36, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<02:47, 2.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:24, 2.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:30, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:40, 2.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<01:57, 3.92MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<12:33, 612kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<09:51, 779kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:07, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:33, 1.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:55, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:33, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:31, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:16, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:15, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:40, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:45, 2.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<03:18, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:21, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:34, 2.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:10, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:17, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:32, 2.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:07, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:10, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:26, 2.11MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:31, 2.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:38, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:15, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:11, 2.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:33, 2.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:32, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:40, 2.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<01:57, 3.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<08:57, 789kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<07:56, 889kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:58, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<04:15, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<08:02, 870kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<06:36, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<04:51, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<04:40, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:31, 1.94MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:27, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:38, 2.59MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:07, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:08, 2.16MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:26, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<02:57, 2.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:02, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:18, 2.89MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<01:41, 3.94MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<22:05, 301kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<16:24, 405kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<11:39, 569kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<09:20, 704kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<07:29, 879kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:26, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:00, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:24, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:18, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:30, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:22, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:35, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:59, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:01, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:19, 2.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:47, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:50, 2.22MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:10, 2.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:41, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:46, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:09, 2.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:39, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:38, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:58, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:10, 2.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:57, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:37, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:42, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:01, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:57, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:15, 2.67MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<02:41, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:44, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:06, 2.83MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:34, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:03, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:31, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:34, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<01:58, 2.94MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:27, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:32, 2.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:57, 2.93MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:25, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:31, 2.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<01:57, 2.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:25, 2.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:28, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<01:54, 2.94MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:22, 4.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<42:17, 131kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<30:21, 183kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<21:21, 259kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<15:51, 346kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<11:52, 461kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<08:27, 646kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<05:55, 913kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<2:45:29, 32.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:56:30, 46.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<1:21:27, 66.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<57:32, 92.8kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<41:47, 128kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<29:35, 180kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<20:39, 256kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<16:26, 321kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<12:15, 430kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<08:44, 600kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<07:02, 739kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:39, 918kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:06, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:54, 1.77MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<16:36, 309kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<12:20, 416kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<08:45, 583kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<07:01, 721kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<06:22, 794kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:48, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<03:25, 1.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:23, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:45, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:47, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:51, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:41, 1.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:19, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:19, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:45, 2.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:07, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:10, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:41, 2.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:03, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:05, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:36, 2.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:59, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:02, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:35, 2.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:57, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:01, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:33, 2.93MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:07, 4.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<46:03, 97.9kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<33:26, 135kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<23:41, 190kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<16:30, 270kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<13:13, 336kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<09:51, 450kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<07:00, 630kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<04:56, 888kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<06:41, 653kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:18, 822kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:51, 1.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:28, 1.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<03:02, 1.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:21, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:52, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:18, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:40, 2.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:49, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:34, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:54, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:06, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:03, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:33, 2.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<01:07, 3.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<1:22:20, 48.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<58:45, 68.5kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<41:17, 97.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<28:42, 139kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<21:31, 184kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<15:37, 253kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<11:00, 357kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<08:20, 466kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<06:23, 607kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:34, 845kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:53, 982kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:14, 1.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:22, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:41, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<56:40, 66.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<40:08, 93.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<28:04, 133kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<20:06, 183kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<14:35, 252kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<10:17, 355kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<07:47, 464kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<05:56, 606kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<04:16, 840kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:36, 980kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:01, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:13, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:12, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<02:32, 1.37MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:59, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:25, 2.40MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:01, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:31, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:41, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:08, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:41, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:13, 2.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:51, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:46, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:31, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:56, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:32, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:07, 2.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:41, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:38, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:14, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<00:53, 3.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:50, 1.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<02:25, 1.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:48, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:48, 1.66MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:33, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:09, 2.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<00:51, 3.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:33, 1.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:34, 1.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:58, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:25, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:05, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:52, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:29, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:22, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:01, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:16, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:12, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<00:55, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:11, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:35, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:15, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<00:56, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:13, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:12, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<00:55, 2.77MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:06, 2.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:07, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<00:51, 2.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:36, 3.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<08:29, 286kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<06:17, 386kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:27, 541kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<03:29, 675kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:47, 847kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:00, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:24, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<04:05, 560kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<03:11, 719kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:16, 996kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:35, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<04:36, 484kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:32, 628kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<02:31, 869kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:08, 1.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:48, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:19, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:18, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:12, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:54, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:00, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:17, 1.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:02, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:44, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:17, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:10, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:53, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:57, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:55, 2.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:42, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:49, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:49, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:37, 2.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:45, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:01, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:50, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:36, 2.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:05, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:59, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:44, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:48, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:35, 2.64MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:41, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:40, 2.29MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:30, 2.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:52, 1.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:41, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:30, 2.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:53, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:49, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:36, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:39, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:39, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:29, 2.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:34, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:34, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:26, 2.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:31, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:30, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:22, 3.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:29, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:39, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:22, 2.80MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:40, 1.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:37, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:15, 784kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:00, 968kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:43, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:39, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:35, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:25, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:32, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:17, 2.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:30, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:27, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:21, 1.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:20, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:15, 2.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:17, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:17, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:13, 2.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:14, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:20, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:10, 2.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:18, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:09, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:07, 2.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:07, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:05, 2.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:05, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:04, 2.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.16MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.95MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.83MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:18:24,  6.42it/s]  0%|          | 972/400000 [00:00<12:05:19,  9.17it/s]  0%|          | 1999/400000 [00:00<8:26:36, 13.09it/s]  1%|          | 2981/400000 [00:00<5:53:57, 18.69it/s]  1%|          | 3908/400000 [00:00<4:07:24, 26.68it/s]  1%|          | 4842/400000 [00:00<2:52:59, 38.07it/s]  1%|▏         | 5819/400000 [00:00<2:00:59, 54.30it/s]  2%|▏         | 6837/400000 [00:00<1:24:40, 77.39it/s]  2%|▏         | 7822/400000 [00:00<59:19, 110.19it/s]   2%|▏         | 8861/400000 [00:01<41:36, 156.70it/s]  2%|▏         | 9821/400000 [00:01<29:16, 222.19it/s]  3%|▎         | 10794/400000 [00:01<20:38, 314.34it/s]  3%|▎         | 11768/400000 [00:01<14:36, 442.93it/s]  3%|▎         | 12745/400000 [00:01<10:23, 620.69it/s]  3%|▎         | 13708/400000 [00:01<07:28, 860.87it/s]  4%|▎         | 14683/400000 [00:01<05:25, 1184.94it/s]  4%|▍         | 15687/400000 [00:01<03:58, 1611.21it/s]  4%|▍         | 16676/400000 [00:01<02:58, 2151.39it/s]  4%|▍         | 17716/400000 [00:01<02:15, 2823.09it/s]  5%|▍         | 18709/400000 [00:02<01:47, 3547.50it/s]  5%|▍         | 19670/400000 [00:02<01:27, 4324.49it/s]  5%|▌         | 20666/400000 [00:02<01:12, 5208.59it/s]  5%|▌         | 21622/400000 [00:02<01:03, 5989.98it/s]  6%|▌         | 22574/400000 [00:02<00:56, 6739.43it/s]  6%|▌         | 23522/400000 [00:02<00:52, 7233.98it/s]  6%|▌         | 24447/400000 [00:02<00:49, 7653.64it/s]  6%|▋         | 25360/400000 [00:02<00:47, 7814.38it/s]  7%|▋         | 26246/400000 [00:02<00:46, 7969.18it/s]  7%|▋         | 27202/400000 [00:03<00:44, 8386.87it/s]  7%|▋         | 28099/400000 [00:03<00:43, 8548.90it/s]  7%|▋         | 29038/400000 [00:03<00:42, 8783.56it/s]  7%|▋         | 29987/400000 [00:03<00:41, 8982.47it/s]  8%|▊         | 31024/400000 [00:03<00:39, 9355.93it/s]  8%|▊         | 31980/400000 [00:03<00:39, 9229.86it/s]  8%|▊         | 32918/400000 [00:03<00:39, 9224.88it/s]  8%|▊         | 33973/400000 [00:03<00:38, 9585.95it/s]  9%|▊         | 34943/400000 [00:03<00:39, 9159.09it/s]  9%|▉         | 35877/400000 [00:03<00:39, 9209.45it/s]  9%|▉         | 36825/400000 [00:04<00:39, 9288.50it/s]  9%|▉         | 37775/400000 [00:04<00:38, 9350.95it/s] 10%|▉         | 38784/400000 [00:04<00:37, 9559.43it/s] 10%|▉         | 39799/400000 [00:04<00:37, 9727.10it/s] 10%|█         | 40827/400000 [00:04<00:36, 9884.62it/s] 10%|█         | 41819/400000 [00:04<00:36, 9818.78it/s] 11%|█         | 42804/400000 [00:04<00:38, 9247.49it/s] 11%|█         | 43761/400000 [00:04<00:38, 9341.48it/s] 11%|█         | 44775/400000 [00:04<00:37, 9565.35it/s] 11%|█▏        | 45845/400000 [00:04<00:35, 9877.36it/s] 12%|█▏        | 46874/400000 [00:05<00:35, 9996.59it/s] 12%|█▏        | 47879/400000 [00:05<00:35, 9964.42it/s] 12%|█▏        | 48879/400000 [00:05<00:35, 9806.36it/s] 12%|█▏        | 49963/400000 [00:05<00:34, 10092.98it/s] 13%|█▎        | 50977/400000 [00:05<00:34, 10093.13it/s] 13%|█▎        | 51990/400000 [00:05<00:34, 9967.42it/s]  13%|█▎        | 52990/400000 [00:05<00:35, 9791.64it/s] 13%|█▎        | 53972/400000 [00:05<00:35, 9795.71it/s] 14%|█▎        | 54954/400000 [00:05<00:35, 9756.82it/s] 14%|█▍        | 55976/400000 [00:05<00:34, 9890.90it/s] 14%|█▍        | 57038/400000 [00:06<00:33, 10097.12it/s] 15%|█▍        | 58050/400000 [00:06<00:35, 9510.41it/s]  15%|█▍        | 59010/400000 [00:06<00:36, 9469.18it/s] 15%|█▌        | 60034/400000 [00:06<00:35, 9686.36it/s] 15%|█▌        | 61032/400000 [00:06<00:34, 9770.74it/s] 16%|█▌        | 62014/400000 [00:06<00:34, 9730.67it/s] 16%|█▌        | 62990/400000 [00:06<00:35, 9424.84it/s] 16%|█▌        | 63989/400000 [00:06<00:35, 9587.06it/s] 16%|█▋        | 65040/400000 [00:06<00:34, 9845.86it/s] 17%|█▋        | 66063/400000 [00:07<00:33, 9956.26it/s] 17%|█▋        | 67121/400000 [00:07<00:32, 10135.08it/s] 17%|█▋        | 68138/400000 [00:07<00:33, 9865.94it/s]  17%|█▋        | 69129/400000 [00:07<00:33, 9855.17it/s] 18%|█▊        | 70138/400000 [00:07<00:33, 9923.71it/s] 18%|█▊        | 71155/400000 [00:07<00:32, 9994.11it/s] 18%|█▊        | 72156/400000 [00:07<00:32, 9983.06it/s] 18%|█▊        | 73156/400000 [00:07<00:33, 9854.63it/s] 19%|█▊        | 74143/400000 [00:07<00:34, 9544.99it/s] 19%|█▉        | 75101/400000 [00:07<00:34, 9499.20it/s] 19%|█▉        | 76053/400000 [00:08<00:35, 9238.37it/s] 19%|█▉        | 76980/400000 [00:08<00:35, 9066.73it/s] 19%|█▉        | 77890/400000 [00:08<00:35, 9046.53it/s] 20%|█▉        | 78835/400000 [00:08<00:35, 9161.66it/s] 20%|█▉        | 79863/400000 [00:08<00:33, 9470.61it/s] 20%|██        | 80879/400000 [00:08<00:33, 9664.68it/s] 20%|██        | 81850/400000 [00:08<00:32, 9674.00it/s] 21%|██        | 82820/400000 [00:08<00:33, 9548.10it/s] 21%|██        | 83878/400000 [00:08<00:32, 9833.80it/s] 21%|██        | 84866/400000 [00:08<00:32, 9675.93it/s] 21%|██▏       | 85861/400000 [00:09<00:32, 9756.46it/s] 22%|██▏       | 86868/400000 [00:09<00:31, 9845.91it/s] 22%|██▏       | 87855/400000 [00:09<00:32, 9584.36it/s] 22%|██▏       | 88950/400000 [00:09<00:31, 9955.17it/s] 22%|██▏       | 89952/400000 [00:09<00:31, 9972.12it/s] 23%|██▎       | 90954/400000 [00:09<00:31, 9846.86it/s] 23%|██▎       | 91942/400000 [00:09<00:32, 9459.89it/s] 23%|██▎       | 92894/400000 [00:09<00:33, 9122.37it/s] 23%|██▎       | 93924/400000 [00:09<00:32, 9444.75it/s] 24%|██▎       | 94887/400000 [00:10<00:32, 9496.78it/s] 24%|██▍       | 95884/400000 [00:10<00:31, 9631.99it/s] 24%|██▍       | 96897/400000 [00:10<00:31, 9770.28it/s] 24%|██▍       | 97878/400000 [00:10<00:31, 9698.19it/s] 25%|██▍       | 98882/400000 [00:10<00:30, 9796.86it/s] 25%|██▍       | 99880/400000 [00:10<00:30, 9848.36it/s] 25%|██▌       | 100963/400000 [00:10<00:29, 10121.17it/s] 25%|██▌       | 101979/400000 [00:10<00:29, 10031.55it/s] 26%|██▌       | 102985/400000 [00:10<00:30, 9881.95it/s]  26%|██▌       | 104055/400000 [00:10<00:29, 10113.63it/s] 26%|██▋       | 105070/400000 [00:11<00:29, 10099.21it/s] 27%|██▋       | 106089/400000 [00:11<00:29, 10125.61it/s] 27%|██▋       | 107135/400000 [00:11<00:28, 10221.48it/s] 27%|██▋       | 108159/400000 [00:11<00:29, 9765.84it/s]  27%|██▋       | 109146/400000 [00:11<00:29, 9795.46it/s] 28%|██▊       | 110130/400000 [00:11<00:30, 9407.64it/s] 28%|██▊       | 111178/400000 [00:11<00:29, 9705.17it/s] 28%|██▊       | 112167/400000 [00:11<00:29, 9759.33it/s] 28%|██▊       | 113148/400000 [00:11<00:30, 9488.86it/s] 29%|██▊       | 114102/400000 [00:11<00:30, 9462.58it/s] 29%|██▉       | 115155/400000 [00:12<00:29, 9758.04it/s] 29%|██▉       | 116136/400000 [00:12<00:29, 9502.24it/s] 29%|██▉       | 117169/400000 [00:12<00:29, 9735.14it/s] 30%|██▉       | 118148/400000 [00:12<00:29, 9554.11it/s] 30%|██▉       | 119186/400000 [00:12<00:28, 9786.88it/s] 30%|███       | 120193/400000 [00:12<00:28, 9866.83it/s] 30%|███       | 121183/400000 [00:12<00:28, 9789.09it/s] 31%|███       | 122165/400000 [00:12<00:28, 9662.36it/s] 31%|███       | 123156/400000 [00:12<00:28, 9732.79it/s] 31%|███       | 124131/400000 [00:13<00:28, 9683.47it/s] 31%|███▏      | 125132/400000 [00:13<00:28, 9775.67it/s] 32%|███▏      | 126111/400000 [00:13<00:28, 9668.75it/s] 32%|███▏      | 127125/400000 [00:13<00:27, 9804.37it/s] 32%|███▏      | 128107/400000 [00:13<00:28, 9589.00it/s] 32%|███▏      | 129068/400000 [00:13<00:29, 9165.11it/s] 32%|███▏      | 129990/400000 [00:13<00:29, 9080.83it/s] 33%|███▎      | 130902/400000 [00:13<00:30, 8958.37it/s] 33%|███▎      | 131828/400000 [00:13<00:29, 9044.13it/s] 33%|███▎      | 132784/400000 [00:13<00:29, 9191.08it/s] 33%|███▎      | 133732/400000 [00:14<00:28, 9273.24it/s] 34%|███▎      | 134725/400000 [00:14<00:28, 9457.65it/s] 34%|███▍      | 135772/400000 [00:14<00:27, 9738.61it/s] 34%|███▍      | 136785/400000 [00:14<00:26, 9851.26it/s] 34%|███▍      | 137829/400000 [00:14<00:26, 10019.71it/s] 35%|███▍      | 138852/400000 [00:14<00:25, 10080.68it/s] 35%|███▍      | 139884/400000 [00:14<00:25, 10150.75it/s] 35%|███▌      | 140901/400000 [00:14<00:26, 9769.77it/s]  35%|███▌      | 141883/400000 [00:14<00:28, 9150.72it/s] 36%|███▌      | 142878/400000 [00:14<00:27, 9374.48it/s] 36%|███▌      | 143828/400000 [00:15<00:27, 9410.30it/s] 36%|███▌      | 144877/400000 [00:15<00:26, 9709.06it/s] 36%|███▋      | 145945/400000 [00:15<00:25, 9978.82it/s] 37%|███▋      | 146950/400000 [00:15<00:26, 9626.15it/s] 37%|███▋      | 147992/400000 [00:15<00:25, 9850.19it/s] 37%|███▋      | 149010/400000 [00:15<00:25, 9946.01it/s] 38%|███▊      | 150012/400000 [00:15<00:25, 9965.83it/s] 38%|███▊      | 151012/400000 [00:15<00:25, 9828.56it/s] 38%|███▊      | 152000/400000 [00:15<00:25, 9841.42it/s] 38%|███▊      | 153013/400000 [00:16<00:24, 9925.08it/s] 39%|███▊      | 154056/400000 [00:16<00:24, 10070.00it/s] 39%|███▉      | 155088/400000 [00:16<00:24, 10143.70it/s] 39%|███▉      | 156104/400000 [00:16<00:24, 10146.70it/s] 39%|███▉      | 157120/400000 [00:16<00:24, 9947.57it/s]  40%|███▉      | 158117/400000 [00:16<00:25, 9582.52it/s] 40%|███▉      | 159133/400000 [00:16<00:24, 9746.52it/s] 40%|████      | 160177/400000 [00:16<00:24, 9942.61it/s] 40%|████      | 161203/400000 [00:16<00:23, 10033.79it/s] 41%|████      | 162209/400000 [00:16<00:24, 9807.71it/s]  41%|████      | 163193/400000 [00:17<00:24, 9779.91it/s] 41%|████      | 164174/400000 [00:17<00:24, 9751.32it/s] 41%|████▏     | 165204/400000 [00:17<00:23, 9907.53it/s] 42%|████▏     | 166224/400000 [00:17<00:23, 9992.73it/s] 42%|████▏     | 167225/400000 [00:17<00:23, 9984.56it/s] 42%|████▏     | 168225/400000 [00:17<00:23, 9978.57it/s] 42%|████▏     | 169258/400000 [00:17<00:22, 10078.88it/s] 43%|████▎     | 170307/400000 [00:17<00:22, 10196.32it/s] 43%|████▎     | 171328/400000 [00:17<00:22, 9996.75it/s]  43%|████▎     | 172330/400000 [00:17<00:23, 9890.12it/s] 43%|████▎     | 173321/400000 [00:18<00:23, 9697.13it/s] 44%|████▎     | 174346/400000 [00:18<00:22, 9856.50it/s] 44%|████▍     | 175334/400000 [00:18<00:23, 9719.67it/s] 44%|████▍     | 176351/400000 [00:18<00:22, 9846.07it/s] 44%|████▍     | 177349/400000 [00:18<00:22, 9885.21it/s] 45%|████▍     | 178370/400000 [00:18<00:22, 9978.75it/s] 45%|████▍     | 179369/400000 [00:18<00:22, 9943.96it/s] 45%|████▌     | 180365/400000 [00:18<00:22, 9898.27it/s] 45%|████▌     | 181356/400000 [00:18<00:22, 9617.27it/s] 46%|████▌     | 182320/400000 [00:18<00:22, 9499.09it/s] 46%|████▌     | 183274/400000 [00:19<00:22, 9510.70it/s] 46%|████▌     | 184343/400000 [00:19<00:21, 9836.05it/s] 46%|████▋     | 185410/400000 [00:19<00:21, 10070.42it/s] 47%|████▋     | 186430/400000 [00:19<00:21, 10106.59it/s] 47%|████▋     | 187477/400000 [00:19<00:20, 10211.77it/s] 47%|████▋     | 188501/400000 [00:19<00:21, 9957.56it/s]  47%|████▋     | 189530/400000 [00:19<00:20, 10053.72it/s] 48%|████▊     | 190538/400000 [00:19<00:20, 10027.11it/s] 48%|████▊     | 191543/400000 [00:19<00:21, 9676.52it/s]  48%|████▊     | 192515/400000 [00:20<00:21, 9640.70it/s] 48%|████▊     | 193482/400000 [00:20<00:21, 9480.70it/s] 49%|████▊     | 194454/400000 [00:20<00:21, 9548.72it/s] 49%|████▉     | 195517/400000 [00:20<00:20, 9847.12it/s] 49%|████▉     | 196535/400000 [00:20<00:20, 9943.77it/s] 49%|████▉     | 197533/400000 [00:20<00:20, 9854.50it/s] 50%|████▉     | 198560/400000 [00:20<00:20, 9973.96it/s] 50%|████▉     | 199560/400000 [00:20<00:20, 9919.70it/s] 50%|█████     | 200568/400000 [00:20<00:20, 9965.46it/s] 50%|█████     | 201597/400000 [00:20<00:19, 10059.58it/s] 51%|█████     | 202604/400000 [00:21<00:20, 9823.77it/s]  51%|█████     | 203622/400000 [00:21<00:19, 9925.58it/s] 51%|█████     | 204617/400000 [00:21<00:19, 9931.36it/s] 51%|█████▏    | 205685/400000 [00:21<00:19, 10143.19it/s] 52%|█████▏    | 206702/400000 [00:21<00:19, 10028.55it/s] 52%|█████▏    | 207707/400000 [00:21<00:20, 9604.31it/s]  52%|█████▏    | 208673/400000 [00:21<00:20, 9386.47it/s] 52%|█████▏    | 209697/400000 [00:21<00:19, 9626.24it/s] 53%|█████▎    | 210665/400000 [00:21<00:19, 9584.32it/s] 53%|█████▎    | 211700/400000 [00:21<00:19, 9799.21it/s] 53%|█████▎    | 212684/400000 [00:22<00:19, 9750.26it/s] 53%|█████▎    | 213775/400000 [00:22<00:18, 10070.42it/s] 54%|█████▎    | 214787/400000 [00:22<00:18, 9968.98it/s]  54%|█████▍    | 215788/400000 [00:22<00:19, 9525.80it/s] 54%|█████▍    | 216782/400000 [00:22<00:18, 9645.36it/s] 54%|█████▍    | 217752/400000 [00:22<00:19, 9534.14it/s] 55%|█████▍    | 218783/400000 [00:22<00:18, 9752.31it/s] 55%|█████▍    | 219763/400000 [00:22<00:18, 9732.05it/s] 55%|█████▌    | 220801/400000 [00:22<00:18, 9915.38it/s] 55%|█████▌    | 221815/400000 [00:22<00:17, 9978.57it/s] 56%|█████▌    | 222836/400000 [00:23<00:17, 10044.90it/s] 56%|█████▌    | 223888/400000 [00:23<00:17, 10179.83it/s] 56%|█████▌    | 224908/400000 [00:23<00:17, 9970.23it/s]  56%|█████▋    | 225908/400000 [00:23<00:17, 9773.99it/s] 57%|█████▋    | 226888/400000 [00:23<00:17, 9775.04it/s] 57%|█████▋    | 227913/400000 [00:23<00:17, 9911.64it/s] 57%|█████▋    | 228906/400000 [00:23<00:17, 9873.11it/s] 57%|█████▋    | 229895/400000 [00:23<00:17, 9692.13it/s] 58%|█████▊    | 230866/400000 [00:23<00:17, 9517.65it/s] 58%|█████▊    | 231820/400000 [00:24<00:18, 9189.77it/s] 58%|█████▊    | 232754/400000 [00:24<00:18, 9232.61it/s] 58%|█████▊    | 233680/400000 [00:24<00:18, 9097.10it/s] 59%|█████▊    | 234639/400000 [00:24<00:17, 9238.53it/s] 59%|█████▉    | 235634/400000 [00:24<00:17, 9440.35it/s] 59%|█████▉    | 236581/400000 [00:24<00:17, 9405.36it/s] 59%|█████▉    | 237590/400000 [00:24<00:16, 9599.62it/s] 60%|█████▉    | 238581/400000 [00:24<00:16, 9690.18it/s] 60%|█████▉    | 239590/400000 [00:24<00:16, 9803.81it/s] 60%|██████    | 240572/400000 [00:24<00:16, 9614.00it/s] 60%|██████    | 241536/400000 [00:25<00:16, 9350.70it/s] 61%|██████    | 242565/400000 [00:25<00:16, 9612.10it/s] 61%|██████    | 243607/400000 [00:25<00:15, 9839.39it/s] 61%|██████    | 244698/400000 [00:25<00:15, 10136.39it/s] 61%|██████▏   | 245753/400000 [00:25<00:15, 10255.68it/s] 62%|██████▏   | 246783/400000 [00:25<00:15, 10170.26it/s] 62%|██████▏   | 247803/400000 [00:25<00:15, 10103.36it/s] 62%|██████▏   | 248854/400000 [00:25<00:14, 10219.65it/s] 62%|██████▏   | 249878/400000 [00:25<00:14, 10070.83it/s] 63%|██████▎   | 250967/400000 [00:25<00:14, 10303.18it/s] 63%|██████▎   | 252000/400000 [00:26<00:14, 10095.51it/s] 63%|██████▎   | 253070/400000 [00:26<00:14, 10269.43it/s] 64%|██████▎   | 254112/400000 [00:26<00:14, 10313.49it/s] 64%|██████▍   | 255167/400000 [00:26<00:13, 10382.17it/s] 64%|██████▍   | 256250/400000 [00:26<00:13, 10511.37it/s] 64%|██████▍   | 257303/400000 [00:26<00:14, 10131.79it/s] 65%|██████▍   | 258321/400000 [00:26<00:14, 10004.23it/s] 65%|██████▍   | 259325/400000 [00:26<00:14, 9760.17it/s]  65%|██████▌   | 260330/400000 [00:26<00:14, 9845.00it/s] 65%|██████▌   | 261401/400000 [00:26<00:13, 10086.88it/s] 66%|██████▌   | 262413/400000 [00:27<00:13, 9958.07it/s]  66%|██████▌   | 263436/400000 [00:27<00:13, 10037.46it/s] 66%|██████▌   | 264442/400000 [00:27<00:13, 9922.00it/s]  66%|██████▋   | 265470/400000 [00:27<00:13, 10024.95it/s] 67%|██████▋   | 266474/400000 [00:27<00:13, 9959.06it/s]  67%|██████▋   | 267472/400000 [00:27<00:14, 9342.59it/s] 67%|██████▋   | 268526/400000 [00:27<00:13, 9670.24it/s] 67%|██████▋   | 269502/400000 [00:27<00:13, 9597.84it/s] 68%|██████▊   | 270483/400000 [00:27<00:13, 9658.49it/s] 68%|██████▊   | 271454/400000 [00:28<00:13, 9624.60it/s] 68%|██████▊   | 272420/400000 [00:28<00:13, 9581.70it/s] 68%|██████▊   | 273443/400000 [00:28<00:12, 9765.71it/s] 69%|██████▊   | 274432/400000 [00:28<00:12, 9800.39it/s] 69%|██████▉   | 275414/400000 [00:28<00:12, 9594.33it/s] 69%|██████▉   | 276449/400000 [00:28<00:12, 9808.28it/s] 69%|██████▉   | 277433/400000 [00:28<00:12, 9698.66it/s] 70%|██████▉   | 278443/400000 [00:28<00:12, 9813.02it/s] 70%|██████▉   | 279440/400000 [00:28<00:12, 9859.34it/s] 70%|███████   | 280453/400000 [00:28<00:12, 9936.78it/s] 70%|███████   | 281484/400000 [00:29<00:11, 10042.71it/s] 71%|███████   | 282496/400000 [00:29<00:11, 10063.89it/s] 71%|███████   | 283539/400000 [00:29<00:11, 10170.59it/s] 71%|███████   | 284569/400000 [00:29<00:11, 10206.71it/s] 71%|███████▏  | 285633/400000 [00:29<00:11, 10332.03it/s] 72%|███████▏  | 286667/400000 [00:29<00:11, 9956.17it/s]  72%|███████▏  | 287667/400000 [00:29<00:11, 9691.40it/s] 72%|███████▏  | 288641/400000 [00:29<00:11, 9585.90it/s] 72%|███████▏  | 289603/400000 [00:29<00:11, 9450.22it/s] 73%|███████▎  | 290551/400000 [00:29<00:11, 9318.02it/s] 73%|███████▎  | 291486/400000 [00:30<00:11, 9064.08it/s] 73%|███████▎  | 292407/400000 [00:30<00:11, 9105.35it/s] 73%|███████▎  | 293417/400000 [00:30<00:11, 9381.05it/s] 74%|███████▎  | 294452/400000 [00:30<00:10, 9650.82it/s] 74%|███████▍  | 295422/400000 [00:30<00:10, 9526.73it/s] 74%|███████▍  | 296379/400000 [00:30<00:11, 9370.26it/s] 74%|███████▍  | 297330/400000 [00:30<00:10, 9410.48it/s] 75%|███████▍  | 298316/400000 [00:30<00:10, 9540.70it/s] 75%|███████▍  | 299332/400000 [00:30<00:10, 9717.76it/s] 75%|███████▌  | 300382/400000 [00:30<00:10, 9939.26it/s] 75%|███████▌  | 301409/400000 [00:31<00:09, 10036.16it/s] 76%|███████▌  | 302432/400000 [00:31<00:09, 10092.75it/s] 76%|███████▌  | 303493/400000 [00:31<00:09, 10239.26it/s] 76%|███████▌  | 304519/400000 [00:31<00:09, 10212.78it/s] 76%|███████▋  | 305542/400000 [00:31<00:09, 10152.91it/s] 77%|███████▋  | 306559/400000 [00:31<00:09, 10041.06it/s] 77%|███████▋  | 307565/400000 [00:31<00:09, 9570.35it/s]  77%|███████▋  | 308528/400000 [00:31<00:09, 9541.99it/s] 77%|███████▋  | 309503/400000 [00:31<00:09, 9601.75it/s] 78%|███████▊  | 310540/400000 [00:32<00:09, 9818.49it/s] 78%|███████▊  | 311536/400000 [00:32<00:08, 9859.13it/s] 78%|███████▊  | 312525/400000 [00:32<00:08, 9847.16it/s] 78%|███████▊  | 313567/400000 [00:32<00:08, 10011.35it/s] 79%|███████▊  | 314620/400000 [00:32<00:08, 10038.57it/s] 79%|███████▉  | 315659/400000 [00:32<00:08, 10140.77it/s] 79%|███████▉  | 316675/400000 [00:32<00:08, 10014.91it/s] 79%|███████▉  | 317678/400000 [00:32<00:08, 9827.28it/s]  80%|███████▉  | 318663/400000 [00:32<00:08, 9607.91it/s] 80%|███████▉  | 319650/400000 [00:32<00:08, 9683.98it/s] 80%|████████  | 320641/400000 [00:33<00:08, 9747.69it/s] 80%|████████  | 321638/400000 [00:33<00:07, 9811.37it/s] 81%|████████  | 322621/400000 [00:33<00:08, 9469.94it/s] 81%|████████  | 323572/400000 [00:33<00:08, 9435.51it/s] 81%|████████  | 324518/400000 [00:33<00:08, 9357.17it/s] 81%|████████▏ | 325517/400000 [00:33<00:07, 9535.54it/s] 82%|████████▏ | 326473/400000 [00:33<00:07, 9469.08it/s] 82%|████████▏ | 327456/400000 [00:33<00:07, 9571.34it/s] 82%|████████▏ | 328415/400000 [00:33<00:07, 9377.58it/s] 82%|████████▏ | 329393/400000 [00:33<00:07, 9493.81it/s] 83%|████████▎ | 330356/400000 [00:34<00:07, 9531.59it/s] 83%|████████▎ | 331311/400000 [00:34<00:07, 9388.31it/s] 83%|████████▎ | 332252/400000 [00:34<00:07, 9299.74it/s] 83%|████████▎ | 333184/400000 [00:34<00:07, 9305.64it/s] 84%|████████▎ | 334180/400000 [00:34<00:06, 9490.26it/s] 84%|████████▍ | 335184/400000 [00:34<00:06, 9648.44it/s] 84%|████████▍ | 336151/400000 [00:34<00:06, 9635.79it/s] 84%|████████▍ | 337125/400000 [00:34<00:06, 9664.18it/s] 85%|████████▍ | 338093/400000 [00:34<00:06, 9502.76it/s] 85%|████████▍ | 339045/400000 [00:35<00:06, 9149.01it/s] 85%|████████▍ | 339964/400000 [00:35<00:06, 8803.41it/s] 85%|████████▌ | 340850/400000 [00:35<00:06, 8735.98it/s] 85%|████████▌ | 341909/400000 [00:35<00:06, 9217.67it/s] 86%|████████▌ | 342965/400000 [00:35<00:05, 9582.59it/s] 86%|████████▌ | 343996/400000 [00:35<00:05, 9788.73it/s] 86%|████████▋ | 345030/400000 [00:35<00:05, 9946.71it/s] 87%|████████▋ | 346032/400000 [00:35<00:05, 9806.64it/s] 87%|████████▋ | 347048/400000 [00:35<00:05, 9909.28it/s] 87%|████████▋ | 348043/400000 [00:35<00:05, 9774.50it/s] 87%|████████▋ | 349084/400000 [00:36<00:05, 9954.19it/s] 88%|████████▊ | 350128/400000 [00:36<00:04, 10093.14it/s] 88%|████████▊ | 351140/400000 [00:36<00:05, 9464.80it/s]  88%|████████▊ | 352139/400000 [00:36<00:04, 9616.44it/s] 88%|████████▊ | 353149/400000 [00:36<00:04, 9754.03it/s] 89%|████████▊ | 354131/400000 [00:36<00:04, 9302.75it/s] 89%|████████▉ | 355117/400000 [00:36<00:04, 9460.63it/s] 89%|████████▉ | 356070/400000 [00:36<00:04, 8845.54it/s] 89%|████████▉ | 356968/400000 [00:36<00:04, 8661.04it/s] 89%|████████▉ | 357990/400000 [00:37<00:04, 9075.26it/s] 90%|████████▉ | 359053/400000 [00:37<00:04, 9490.47it/s] 90%|█████████ | 360016/400000 [00:37<00:04, 9433.14it/s] 90%|█████████ | 361025/400000 [00:37<00:04, 9620.88it/s] 91%|█████████ | 362080/400000 [00:37<00:03, 9870.94it/s] 91%|█████████ | 363130/400000 [00:37<00:03, 10051.23it/s] 91%|█████████ | 364141/400000 [00:37<00:03, 10049.00it/s] 91%|█████████▏| 365150/400000 [00:37<00:03, 9710.36it/s]  92%|█████████▏| 366127/400000 [00:37<00:03, 9591.91it/s] 92%|█████████▏| 367091/400000 [00:37<00:03, 9502.33it/s] 92%|█████████▏| 368045/400000 [00:38<00:03, 9345.80it/s] 92%|█████████▏| 369084/400000 [00:38<00:03, 9636.11it/s] 93%|█████████▎| 370073/400000 [00:38<00:03, 9709.05it/s] 93%|█████████▎| 371047/400000 [00:38<00:03, 9644.12it/s] 93%|█████████▎| 372034/400000 [00:38<00:02, 9710.04it/s] 93%|█████████▎| 373007/400000 [00:38<00:02, 9642.60it/s] 93%|█████████▎| 373973/400000 [00:38<00:02, 9389.84it/s] 94%|█████████▎| 374915/400000 [00:38<00:02, 9349.10it/s] 94%|█████████▍| 375914/400000 [00:38<00:02, 9532.45it/s] 94%|█████████▍| 376894/400000 [00:38<00:02, 9608.70it/s] 94%|█████████▍| 377896/400000 [00:39<00:02, 9726.73it/s] 95%|█████████▍| 378880/400000 [00:39<00:02, 9758.12it/s] 95%|█████████▍| 379860/400000 [00:39<00:02, 9769.40it/s] 95%|█████████▌| 380915/400000 [00:39<00:01, 9990.02it/s] 95%|█████████▌| 381916/400000 [00:39<00:01, 9894.26it/s] 96%|█████████▌| 382934/400000 [00:39<00:01, 9978.12it/s] 96%|█████████▌| 383933/400000 [00:39<00:01, 9690.15it/s] 96%|█████████▌| 384934/400000 [00:39<00:01, 9782.78it/s] 96%|█████████▋| 385915/400000 [00:39<00:01, 9783.14it/s] 97%|█████████▋| 386895/400000 [00:39<00:01, 9710.90it/s] 97%|█████████▋| 387952/400000 [00:40<00:01, 9951.61it/s] 97%|█████████▋| 388950/400000 [00:40<00:01, 9860.96it/s] 97%|█████████▋| 389938/400000 [00:40<00:01, 9183.09it/s] 98%|█████████▊| 390922/400000 [00:40<00:00, 9369.20it/s] 98%|█████████▊| 391868/400000 [00:40<00:00, 9153.88it/s] 98%|█████████▊| 392791/400000 [00:40<00:00, 9134.44it/s] 98%|█████████▊| 393786/400000 [00:40<00:00, 9364.09it/s] 99%|█████████▊| 394728/400000 [00:40<00:00, 9373.02it/s] 99%|█████████▉| 395792/400000 [00:40<00:00, 9718.08it/s] 99%|█████████▉| 396816/400000 [00:41<00:00, 9866.50it/s] 99%|█████████▉| 397864/400000 [00:41<00:00, 10041.03it/s]100%|█████████▉| 398909/400000 [00:41<00:00, 10160.29it/s]100%|█████████▉| 399938/400000 [00:41<00:00, 10198.33it/s]100%|█████████▉| 399999/400000 [00:41<00:00, 9682.04it/s] Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f03b8af4400>, <torchtext.data.dataset.TabularDataset object at 0x7f03b8af4550>, <torchtext.vocab.Vocab object at 0x7f03b8af4470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.41 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.79 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.79 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.54 file/s]2020-07-12 18:19:18.662149: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-12 18:19:18.665691: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-07-12 18:19:18.665861: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5630cd9624d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-12 18:19:18.665876: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3432448/9912422 [00:00<00:00, 33445378.44it/s]9920512it [00:00, 34666238.30it/s]                             
0it [00:00, ?it/s]32768it [00:00, 532370.72it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481473.84it/s]1654784it [00:00, 11893440.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 167413.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
