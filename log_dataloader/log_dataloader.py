
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff416ae4ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff416ae4ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff481da41e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff481da41e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff4a00ece18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff4a00ece18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff42f0cc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff42f0cc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff42f0cc488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:12, 135713.76it/s] 21%|██        | 2048000/9912422 [00:00<00:40, 193316.99it/s] 59%|█████▉    | 5857280/9912422 [00:00<00:14, 275557.01it/s]9920512it [00:00, 25512128.12it/s]                           
0it [00:00, ?it/s]32768it [00:00, 740371.99it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 894884.43it/s]1654784it [00:00, 12578959.18it/s]                         
0it [00:00, ?it/s]8192it [00:00, 262244.04it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff42c4ba320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff42c4e65c0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff42f0cc0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff42f0cc0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff42f0cc0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:19,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:53,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:53,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:35,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:16,  7.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:16,  7.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:16,  7.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 30.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 37.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 44.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 44.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 67.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 67.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 73.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 79.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 79.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.82 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.82 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.99 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.15s/ url]
0 examples [00:00, ? examples/s]2020-06-26 06:09:57.086341: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-26 06:09:57.099812: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-26 06:09:57.100111: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dbc5f2df00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 06:09:57.100129: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
61 examples [00:00, 605.91 examples/s]187 examples [00:00, 717.36 examples/s]318 examples [00:00, 829.88 examples/s]451 examples [00:00, 934.20 examples/s]584 examples [00:00, 1025.29 examples/s]699 examples [00:00, 1057.44 examples/s]824 examples [00:00, 1107.09 examples/s]955 examples [00:00, 1160.79 examples/s]1078 examples [00:00, 1179.97 examples/s]1198 examples [00:01, 1170.84 examples/s]1327 examples [00:01, 1202.42 examples/s]1449 examples [00:01, 1206.94 examples/s]1583 examples [00:01, 1243.27 examples/s]1710 examples [00:01, 1249.91 examples/s]1843 examples [00:01, 1269.75 examples/s]1975 examples [00:01, 1283.03 examples/s]2104 examples [00:01, 1275.51 examples/s]2232 examples [00:01, 1276.59 examples/s]2364 examples [00:01, 1288.97 examples/s]2494 examples [00:02, 1285.01 examples/s]2623 examples [00:02, 1251.52 examples/s]2749 examples [00:02, 1209.61 examples/s]2874 examples [00:02, 1219.55 examples/s]3010 examples [00:02, 1257.29 examples/s]3139 examples [00:02, 1265.39 examples/s]3270 examples [00:02, 1278.36 examples/s]3399 examples [00:02, 1256.59 examples/s]3525 examples [00:02, 1251.65 examples/s]3651 examples [00:02, 1223.63 examples/s]3774 examples [00:03, 1176.34 examples/s]3909 examples [00:03, 1221.00 examples/s]4040 examples [00:03, 1246.23 examples/s]4176 examples [00:03, 1277.42 examples/s]4310 examples [00:03, 1294.18 examples/s]4440 examples [00:03, 1250.35 examples/s]4566 examples [00:03, 1203.90 examples/s]4688 examples [00:03, 1193.35 examples/s]4808 examples [00:03, 1156.77 examples/s]4925 examples [00:04, 1142.11 examples/s]5042 examples [00:04, 1148.33 examples/s]5158 examples [00:04, 1144.80 examples/s]5273 examples [00:04, 1128.26 examples/s]5387 examples [00:04, 1113.82 examples/s]5499 examples [00:04, 1100.50 examples/s]5611 examples [00:04, 1104.38 examples/s]5722 examples [00:04, 1092.79 examples/s]5832 examples [00:04, 1064.94 examples/s]5942 examples [00:04, 1073.51 examples/s]6051 examples [00:05, 1077.17 examples/s]6160 examples [00:05, 1079.28 examples/s]6269 examples [00:05, 1058.21 examples/s]6380 examples [00:05, 1072.78 examples/s]6490 examples [00:05, 1079.49 examples/s]6600 examples [00:05, 1085.50 examples/s]6709 examples [00:05, 1085.64 examples/s]6818 examples [00:05, 1072.02 examples/s]6926 examples [00:05, 1074.30 examples/s]7034 examples [00:05, 1075.38 examples/s]7144 examples [00:06, 1080.88 examples/s]7253 examples [00:06, 1081.79 examples/s]7362 examples [00:06, 1084.11 examples/s]7471 examples [00:06, 1070.64 examples/s]7585 examples [00:06, 1088.24 examples/s]7702 examples [00:06, 1108.22 examples/s]7818 examples [00:06, 1120.61 examples/s]7931 examples [00:06, 1120.31 examples/s]8045 examples [00:06, 1124.16 examples/s]8159 examples [00:06, 1127.10 examples/s]8276 examples [00:07, 1138.70 examples/s]8390 examples [00:07, 1138.55 examples/s]8504 examples [00:07, 1129.56 examples/s]8618 examples [00:07, 1121.46 examples/s]8731 examples [00:07, 1123.15 examples/s]8844 examples [00:07, 1120.72 examples/s]8957 examples [00:07, 1121.37 examples/s]9070 examples [00:07, 1098.57 examples/s]9183 examples [00:07, 1105.77 examples/s]9295 examples [00:07, 1107.35 examples/s]9406 examples [00:08, 1098.22 examples/s]9516 examples [00:08, 1051.22 examples/s]9622 examples [00:08, 1020.09 examples/s]9735 examples [00:08, 1049.84 examples/s]9851 examples [00:08, 1079.98 examples/s]9960 examples [00:08, 1078.46 examples/s]10069 examples [00:08, 1029.34 examples/s]10179 examples [00:08, 1049.39 examples/s]10291 examples [00:08, 1067.89 examples/s]10403 examples [00:09, 1081.69 examples/s]10512 examples [00:09, 1048.53 examples/s]10629 examples [00:09, 1081.12 examples/s]10739 examples [00:09, 1084.58 examples/s]10848 examples [00:09, 1076.79 examples/s]10956 examples [00:09, 1075.11 examples/s]11070 examples [00:09, 1090.98 examples/s]11190 examples [00:09, 1118.33 examples/s]11303 examples [00:09, 1103.70 examples/s]11423 examples [00:09, 1128.13 examples/s]11537 examples [00:10, 1122.95 examples/s]11656 examples [00:10, 1141.20 examples/s]11779 examples [00:10, 1164.10 examples/s]11896 examples [00:10, 1133.74 examples/s]12020 examples [00:10, 1161.58 examples/s]12152 examples [00:10, 1202.58 examples/s]12276 examples [00:10, 1189.81 examples/s]12396 examples [00:10, 1172.94 examples/s]12524 examples [00:10, 1202.24 examples/s]12658 examples [00:10, 1238.61 examples/s]12784 examples [00:11, 1242.12 examples/s]12920 examples [00:11, 1273.45 examples/s]13048 examples [00:11, 1240.76 examples/s]13173 examples [00:11, 1221.26 examples/s]13300 examples [00:11, 1235.46 examples/s]13433 examples [00:11, 1260.89 examples/s]13562 examples [00:11, 1268.44 examples/s]13690 examples [00:11, 1265.45 examples/s]13818 examples [00:11, 1266.88 examples/s]13945 examples [00:12, 1262.49 examples/s]14072 examples [00:12, 1260.80 examples/s]14201 examples [00:12, 1268.56 examples/s]14334 examples [00:12, 1283.90 examples/s]14463 examples [00:12, 1282.06 examples/s]14598 examples [00:12, 1299.60 examples/s]14729 examples [00:12, 1286.35 examples/s]14858 examples [00:12, 1262.26 examples/s]14985 examples [00:12, 1257.90 examples/s]15111 examples [00:12, 1249.92 examples/s]15245 examples [00:13, 1274.24 examples/s]15382 examples [00:13, 1299.22 examples/s]15523 examples [00:13, 1328.27 examples/s]15657 examples [00:13, 1309.82 examples/s]15789 examples [00:13, 1298.73 examples/s]15920 examples [00:13, 1299.70 examples/s]16054 examples [00:13, 1309.83 examples/s]16189 examples [00:13, 1319.31 examples/s]16322 examples [00:13, 1313.65 examples/s]16454 examples [00:13, 1297.13 examples/s]16584 examples [00:14, 1279.70 examples/s]16717 examples [00:14, 1293.29 examples/s]16847 examples [00:14, 1270.20 examples/s]16975 examples [00:14, 1209.36 examples/s]17098 examples [00:14, 1214.98 examples/s]17235 examples [00:14, 1255.60 examples/s]17367 examples [00:14, 1272.91 examples/s]17500 examples [00:14, 1289.07 examples/s]17630 examples [00:14, 1256.39 examples/s]17763 examples [00:14, 1275.31 examples/s]17896 examples [00:15, 1290.50 examples/s]18026 examples [00:15, 1271.07 examples/s]18154 examples [00:15, 1270.80 examples/s]18282 examples [00:15, 1227.85 examples/s]18407 examples [00:15, 1233.56 examples/s]18543 examples [00:15, 1266.87 examples/s]18677 examples [00:15, 1286.11 examples/s]18806 examples [00:15, 1254.09 examples/s]18932 examples [00:15, 1211.39 examples/s]19054 examples [00:16, 1204.34 examples/s]19191 examples [00:16, 1249.20 examples/s]19321 examples [00:16, 1262.17 examples/s]19450 examples [00:16, 1268.48 examples/s]19578 examples [00:16, 1247.02 examples/s]19708 examples [00:16, 1261.10 examples/s]19835 examples [00:16, 1245.26 examples/s]19971 examples [00:16, 1276.30 examples/s]20099 examples [00:16, 1212.92 examples/s]20222 examples [00:16, 1189.53 examples/s]20342 examples [00:17, 1172.13 examples/s]20475 examples [00:17, 1213.86 examples/s]20608 examples [00:17, 1245.46 examples/s]20735 examples [00:17, 1249.26 examples/s]20861 examples [00:17, 1214.85 examples/s]20991 examples [00:17, 1238.36 examples/s]21119 examples [00:17, 1248.94 examples/s]21245 examples [00:17, 1239.22 examples/s]21371 examples [00:17, 1242.31 examples/s]21496 examples [00:18, 1237.82 examples/s]21633 examples [00:18, 1271.53 examples/s]21761 examples [00:18, 1242.19 examples/s]21886 examples [00:18, 1235.26 examples/s]22011 examples [00:18, 1237.83 examples/s]22135 examples [00:18, 1228.67 examples/s]22259 examples [00:18, 1219.10 examples/s]22392 examples [00:18, 1248.66 examples/s]22520 examples [00:18, 1254.70 examples/s]22646 examples [00:18, 1227.31 examples/s]22773 examples [00:19, 1237.89 examples/s]22903 examples [00:19, 1255.19 examples/s]23038 examples [00:19, 1279.70 examples/s]23167 examples [00:19, 1272.68 examples/s]23295 examples [00:19, 1248.17 examples/s]23426 examples [00:19, 1265.82 examples/s]23553 examples [00:19, 1258.73 examples/s]23680 examples [00:19, 1247.75 examples/s]23807 examples [00:19, 1252.63 examples/s]23933 examples [00:19, 1229.30 examples/s]24060 examples [00:20, 1239.65 examples/s]24190 examples [00:20, 1254.65 examples/s]24326 examples [00:20, 1283.14 examples/s]24455 examples [00:20, 1273.71 examples/s]24583 examples [00:20, 1268.02 examples/s]24710 examples [00:20, 1250.65 examples/s]24836 examples [00:20, 1250.40 examples/s]24962 examples [00:20, 1245.49 examples/s]25087 examples [00:20, 1241.86 examples/s]25212 examples [00:20, 1216.20 examples/s]25334 examples [00:21, 1197.64 examples/s]25466 examples [00:21, 1228.08 examples/s]25590 examples [00:21, 1231.14 examples/s]25714 examples [00:21, 1208.64 examples/s]25837 examples [00:21, 1212.49 examples/s]25960 examples [00:21, 1213.68 examples/s]26082 examples [00:21, 1213.88 examples/s]26213 examples [00:21, 1240.55 examples/s]26338 examples [00:21, 1230.12 examples/s]26462 examples [00:22, 1209.42 examples/s]26585 examples [00:22, 1214.90 examples/s]26707 examples [00:22, 1207.54 examples/s]26828 examples [00:22, 1186.51 examples/s]26947 examples [00:22, 1178.82 examples/s]27066 examples [00:22, 1147.37 examples/s]27182 examples [00:22, 1119.70 examples/s]27302 examples [00:22, 1141.43 examples/s]27419 examples [00:22, 1148.63 examples/s]27539 examples [00:22, 1163.20 examples/s]27656 examples [00:23, 1121.86 examples/s]27775 examples [00:23, 1140.97 examples/s]27897 examples [00:23, 1161.50 examples/s]28014 examples [00:23, 1162.88 examples/s]28131 examples [00:23, 1143.95 examples/s]28248 examples [00:23, 1150.66 examples/s]28364 examples [00:23, 1153.39 examples/s]28495 examples [00:23, 1195.22 examples/s]28622 examples [00:23, 1215.03 examples/s]28744 examples [00:23, 1212.31 examples/s]28866 examples [00:24, 1186.44 examples/s]28993 examples [00:24, 1209.22 examples/s]29119 examples [00:24, 1221.13 examples/s]29242 examples [00:24, 1202.08 examples/s]29363 examples [00:24, 1166.00 examples/s]29481 examples [00:24, 1154.59 examples/s]29607 examples [00:24, 1181.23 examples/s]29731 examples [00:24, 1193.01 examples/s]29856 examples [00:24, 1208.33 examples/s]29978 examples [00:24, 1197.87 examples/s]30098 examples [00:25, 1126.60 examples/s]30222 examples [00:25, 1156.64 examples/s]30343 examples [00:25, 1171.35 examples/s]30468 examples [00:25, 1193.34 examples/s]30591 examples [00:25, 1203.74 examples/s]30713 examples [00:25, 1205.83 examples/s]30836 examples [00:25, 1210.74 examples/s]30971 examples [00:25, 1248.05 examples/s]31104 examples [00:25, 1270.18 examples/s]31232 examples [00:26, 1214.81 examples/s]31355 examples [00:26, 1190.52 examples/s]31475 examples [00:26, 1172.77 examples/s]31596 examples [00:26, 1181.90 examples/s]31724 examples [00:26, 1209.57 examples/s]31849 examples [00:26, 1220.31 examples/s]31978 examples [00:26, 1239.51 examples/s]32112 examples [00:26, 1267.53 examples/s]32240 examples [00:26, 1221.00 examples/s]32366 examples [00:26, 1231.66 examples/s]32490 examples [00:27, 1208.20 examples/s]32612 examples [00:27, 1192.49 examples/s]32736 examples [00:27, 1204.55 examples/s]32871 examples [00:27, 1242.47 examples/s]32996 examples [00:27, 1229.44 examples/s]33121 examples [00:27, 1233.34 examples/s]33256 examples [00:27, 1264.74 examples/s]33391 examples [00:27, 1286.90 examples/s]33521 examples [00:27, 1269.27 examples/s]33649 examples [00:27, 1263.90 examples/s]33776 examples [00:28, 1247.46 examples/s]33901 examples [00:28, 1230.29 examples/s]34025 examples [00:28, 1185.30 examples/s]34145 examples [00:28, 1176.58 examples/s]34264 examples [00:28, 1163.12 examples/s]34381 examples [00:28, 1160.09 examples/s]34498 examples [00:28, 1153.74 examples/s]34614 examples [00:28, 1143.73 examples/s]34729 examples [00:28, 1138.60 examples/s]34843 examples [00:29, 1127.06 examples/s]34956 examples [00:29, 1116.06 examples/s]35084 examples [00:29, 1158.54 examples/s]35213 examples [00:29, 1195.07 examples/s]35340 examples [00:29, 1214.69 examples/s]35462 examples [00:29, 1199.83 examples/s]35588 examples [00:29, 1216.12 examples/s]35715 examples [00:29, 1229.77 examples/s]35839 examples [00:29, 1206.65 examples/s]35974 examples [00:29, 1245.66 examples/s]36100 examples [00:30, 1200.44 examples/s]36221 examples [00:30, 1175.35 examples/s]36347 examples [00:30, 1197.47 examples/s]36478 examples [00:30, 1228.49 examples/s]36602 examples [00:30, 1206.27 examples/s]36724 examples [00:30, 1209.96 examples/s]36846 examples [00:30, 1208.45 examples/s]36978 examples [00:30, 1238.80 examples/s]37111 examples [00:30, 1262.55 examples/s]37238 examples [00:30, 1260.93 examples/s]37365 examples [00:31, 1168.82 examples/s]37484 examples [00:31, 1152.37 examples/s]37601 examples [00:31, 1146.35 examples/s]37717 examples [00:31, 1118.01 examples/s]37836 examples [00:31, 1138.01 examples/s]37960 examples [00:31, 1166.76 examples/s]38084 examples [00:31, 1186.44 examples/s]38219 examples [00:31, 1229.39 examples/s]38344 examples [00:31, 1233.09 examples/s]38468 examples [00:32, 1191.65 examples/s]38588 examples [00:32, 1166.60 examples/s]38722 examples [00:32, 1211.85 examples/s]38852 examples [00:32, 1235.03 examples/s]38985 examples [00:32, 1260.75 examples/s]39113 examples [00:32, 1263.13 examples/s]39240 examples [00:32, 1264.76 examples/s]39373 examples [00:32, 1282.83 examples/s]39502 examples [00:32, 1261.70 examples/s]39629 examples [00:32, 1242.31 examples/s]39754 examples [00:33, 1187.17 examples/s]39877 examples [00:33, 1197.71 examples/s]39998 examples [00:33, 1200.97 examples/s]40119 examples [00:33, 1178.08 examples/s]40246 examples [00:33, 1203.71 examples/s]40367 examples [00:33, 1201.28 examples/s]40488 examples [00:33, 1162.96 examples/s]40623 examples [00:33, 1212.96 examples/s]40754 examples [00:33, 1238.29 examples/s]40879 examples [00:34, 1241.74 examples/s]41004 examples [00:34, 1224.69 examples/s]41127 examples [00:34, 1212.93 examples/s]41260 examples [00:34, 1245.61 examples/s]41395 examples [00:34, 1273.28 examples/s]41523 examples [00:34, 1261.04 examples/s]41654 examples [00:34, 1274.45 examples/s]41784 examples [00:34, 1279.80 examples/s]41913 examples [00:34, 1268.98 examples/s]42041 examples [00:34, 1262.91 examples/s]42168 examples [00:35, 1229.70 examples/s]42292 examples [00:35, 1202.28 examples/s]42413 examples [00:35, 1177.27 examples/s]42532 examples [00:35, 1174.70 examples/s]42666 examples [00:35, 1218.09 examples/s]42789 examples [00:35, 1183.42 examples/s]42908 examples [00:35, 1173.28 examples/s]43026 examples [00:35, 1169.69 examples/s]43148 examples [00:35, 1182.19 examples/s]43267 examples [00:35, 1155.57 examples/s]43396 examples [00:36, 1191.11 examples/s]43516 examples [00:36, 1189.65 examples/s]43637 examples [00:36, 1194.66 examples/s]43757 examples [00:36, 1182.34 examples/s]43876 examples [00:36, 1146.42 examples/s]43998 examples [00:36, 1166.70 examples/s]44123 examples [00:36, 1189.10 examples/s]44249 examples [00:36, 1208.99 examples/s]44371 examples [00:36, 1189.21 examples/s]44491 examples [00:37, 1160.29 examples/s]44611 examples [00:37, 1171.87 examples/s]44737 examples [00:37, 1194.90 examples/s]44863 examples [00:37, 1213.08 examples/s]44985 examples [00:37, 1199.23 examples/s]45106 examples [00:37, 1193.56 examples/s]45227 examples [00:37, 1196.82 examples/s]45347 examples [00:37, 1195.55 examples/s]45467 examples [00:37, 1178.98 examples/s]45592 examples [00:37, 1197.53 examples/s]45720 examples [00:38, 1220.75 examples/s]45852 examples [00:38, 1247.26 examples/s]45978 examples [00:38, 1211.65 examples/s]46100 examples [00:38, 1203.09 examples/s]46221 examples [00:38, 1199.50 examples/s]46342 examples [00:38, 1190.10 examples/s]46462 examples [00:38, 1174.62 examples/s]46580 examples [00:38, 1175.32 examples/s]46710 examples [00:38, 1209.08 examples/s]46838 examples [00:38, 1227.03 examples/s]46961 examples [00:39, 1227.68 examples/s]47088 examples [00:39, 1238.95 examples/s]47213 examples [00:39, 1185.46 examples/s]47336 examples [00:39, 1198.44 examples/s]47463 examples [00:39, 1218.36 examples/s]47586 examples [00:39, 1215.74 examples/s]47708 examples [00:39, 1203.52 examples/s]47829 examples [00:39, 1200.66 examples/s]47960 examples [00:39, 1231.43 examples/s]48091 examples [00:39, 1251.01 examples/s]48217 examples [00:40, 1239.69 examples/s]48346 examples [00:40, 1251.86 examples/s]48474 examples [00:40, 1258.83 examples/s]48601 examples [00:40, 1259.21 examples/s]48728 examples [00:40, 1249.40 examples/s]48855 examples [00:40, 1254.21 examples/s]48981 examples [00:40, 1251.59 examples/s]49107 examples [00:40, 1249.51 examples/s]49233 examples [00:40, 1251.79 examples/s]49361 examples [00:40, 1257.90 examples/s]49487 examples [00:41, 1228.37 examples/s]49613 examples [00:41, 1237.61 examples/s]49737 examples [00:41, 1234.62 examples/s]49863 examples [00:41, 1239.70 examples/s]49988 examples [00:41, 1230.70 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 20%|█▉        | 9899/50000 [00:00<00:00, 98987.73 examples/s] 52%|█████▏    | 25935/50000 [00:00<00:00, 111826.43 examples/s] 81%|████████▏ | 40730/50000 [00:00<00:00, 120663.61 examples/s]                                                                0 examples [00:00, ? examples/s]110 examples [00:00, 1092.90 examples/s]238 examples [00:00, 1141.91 examples/s]370 examples [00:00, 1189.50 examples/s]494 examples [00:00, 1204.12 examples/s]631 examples [00:00, 1248.55 examples/s]767 examples [00:00, 1278.57 examples/s]899 examples [00:00, 1287.66 examples/s]1032 examples [00:00, 1297.54 examples/s]1156 examples [00:00, 1266.71 examples/s]1279 examples [00:01, 1245.05 examples/s]1412 examples [00:01, 1268.61 examples/s]1546 examples [00:01, 1287.94 examples/s]1682 examples [00:01, 1306.06 examples/s]1813 examples [00:01, 1304.57 examples/s]1943 examples [00:01, 1282.05 examples/s]2072 examples [00:01, 1283.32 examples/s]2201 examples [00:01, 1278.10 examples/s]2336 examples [00:01, 1297.74 examples/s]2466 examples [00:01, 1295.67 examples/s]2596 examples [00:02, 1259.68 examples/s]2726 examples [00:02, 1270.86 examples/s]2854 examples [00:02, 1260.08 examples/s]2987 examples [00:02, 1278.92 examples/s]3120 examples [00:02, 1292.86 examples/s]3250 examples [00:02, 1284.04 examples/s]3379 examples [00:02, 1260.91 examples/s]3506 examples [00:02, 1256.61 examples/s]3638 examples [00:02, 1274.32 examples/s]3766 examples [00:02, 1262.93 examples/s]3893 examples [00:03, 1247.61 examples/s]4027 examples [00:03, 1271.36 examples/s]4155 examples [00:03, 1258.53 examples/s]4283 examples [00:03, 1263.32 examples/s]4410 examples [00:03, 1236.81 examples/s]4539 examples [00:03, 1250.39 examples/s]4674 examples [00:03, 1276.04 examples/s]4802 examples [00:03, 1268.52 examples/s]4930 examples [00:03, 1262.74 examples/s]5057 examples [00:03, 1246.39 examples/s]5183 examples [00:04, 1249.82 examples/s]5312 examples [00:04, 1260.72 examples/s]5439 examples [00:04, 1246.22 examples/s]5574 examples [00:04, 1274.33 examples/s]5702 examples [00:04, 1272.79 examples/s]5830 examples [00:04, 1263.33 examples/s]5957 examples [00:04, 1245.04 examples/s]6085 examples [00:04, 1254.44 examples/s]6220 examples [00:04, 1279.66 examples/s]6349 examples [00:04, 1275.17 examples/s]6477 examples [00:05, 1229.73 examples/s]6601 examples [00:05, 1225.96 examples/s]6736 examples [00:05, 1258.92 examples/s]6874 examples [00:05, 1292.08 examples/s]7004 examples [00:05, 1291.87 examples/s]7134 examples [00:05, 1278.59 examples/s]7263 examples [00:05, 1273.99 examples/s]7397 examples [00:05, 1291.30 examples/s]7533 examples [00:05, 1307.99 examples/s]7668 examples [00:06, 1318.58 examples/s]7801 examples [00:06, 1284.10 examples/s]7930 examples [00:06, 1264.11 examples/s]8057 examples [00:06, 1215.95 examples/s]8180 examples [00:06, 1207.69 examples/s]8302 examples [00:06, 1201.47 examples/s]8434 examples [00:06, 1232.64 examples/s]8566 examples [00:06, 1256.08 examples/s]8696 examples [00:06, 1266.26 examples/s]8823 examples [00:06, 1242.77 examples/s]8948 examples [00:07, 1223.81 examples/s]9079 examples [00:07, 1244.22 examples/s]9204 examples [00:07, 1242.73 examples/s]9337 examples [00:07, 1266.14 examples/s]9466 examples [00:07, 1272.13 examples/s]9594 examples [00:07, 1234.63 examples/s]9721 examples [00:07, 1243.91 examples/s]9849 examples [00:07, 1251.78 examples/s]9985 examples [00:07, 1280.79 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteG9YVLR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteG9YVLR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff42f0cc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff42f0cc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff42f0cc488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff416279550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3b44f8208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff42f0cc488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff42f0cc488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff42f0cc488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3b44f8a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff42c4e6358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff3b0d38e18> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff3b0d38e18> 

  function with postional parmater data_info <function split_train_valid at 0x7ff3b0d38e18> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff3b0d38f28> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff3b0d38f28> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff3b0d38f28> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=931bb26000c7dace0fb780e8929a216008235f8d034b9ca408685a1b0a082a16
  Stored in directory: /tmp/pip-ephem-wheel-cache-udo32m4g/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:20:37, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:35:17, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:40:06, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:10:35, 29.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:42:32, 41.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.45M/862M [00:01<3:58:14, 59.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:01<2:46:18, 85.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:55:50, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.7M/862M [00:01<1:20:47, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<56:21, 247kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:02<39:22, 352kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:02<27:30, 501kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:02<19:16, 712kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:02<13:31, 1.01MB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.9M/862M [00:02<09:35, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<07:08, 1.89MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<06:54, 1.95MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<05:11, 2.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<03:50, 3.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<09:45, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<08:27, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<06:18, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<07:14, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:09<06:30, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<04:54, 2.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<06:31, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<06:00, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<04:32, 2.90MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<06:16, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<05:49, 2.26MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<04:21, 3.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<03:14, 4.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<1:38:10, 133kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<1:10:07, 187kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<49:20, 265kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<37:29, 347kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<27:38, 471kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<19:40, 661kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<16:46, 773kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<13:08, 986kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<09:32, 1.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<09:40, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<08:09, 1.58MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<07:13, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<06:27, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<04:47, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<06:22, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<05:50, 2.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:24<04:25, 2.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:03, 2.09MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:36, 2.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:16, 2.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:56, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:30, 2.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:11, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:28, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:05, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:26, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:05, 3.04MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<05:44, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:22, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<04:01, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:44, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:04, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:43, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:19, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:03, 3.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:41, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:18, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:02, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:16, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:00, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<05:38, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<03:58, 3.02MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<05:35, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:13, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:58, 3.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<05:34, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:11, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<03:57, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<05:32, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<05:09, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<03:55, 3.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:30, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:07, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<03:50, 3.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:27, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:05, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:52, 3.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:26, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:03, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:47, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<05:02, 2.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:49, 3.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:23, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<04:59, 2.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:47, 3.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:20, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<04:58, 2.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<03:46, 3.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:56, 2.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<03:45, 3.02MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<05:16, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<04:55, 2.30MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<03:44, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:14, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<03:43, 3.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:13, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:38, 3.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:10, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<04:50, 2.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:41, 3.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:08, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:47, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:38, 3.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:45, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:37, 3.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:44, 2.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:36, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:03, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:42, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<03:34, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:01, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:40, 2.30MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:33, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<04:59, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:38, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<03:32, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<04:57, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:37, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<03:30, 3.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:56, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:34, 2.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:28, 3.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:53, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:33, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:28, 3.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:31, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:26, 3.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:49, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:30, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:22, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:47, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:28, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:23, 3.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:46, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:26, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<03:22, 3.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:44, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:24, 2.30MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:18, 3.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<07:59, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<06:39, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:12, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:45, 2.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<20:13, 492kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<16:13, 613kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<11:46, 843kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:25, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<08:42, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<07:11, 1.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:14, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:55, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<05:12, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:51, 2.54MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:59, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<04:33, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<03:26, 2.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:40, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<04:18, 2.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:16, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:32, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:01, 2.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<03:21, 2.86MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:26, 3.91MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<37:21, 255kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<27:10, 351kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<19:13, 494kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<15:36, 606kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<11:56, 792kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:35, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<08:11, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:44, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:57, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:38, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:58, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:41, 2.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:45, 1.94MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:18, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:13, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:05, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:03, 2.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:17, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:59, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:59, 3.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:55, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:58, 3.02MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:11, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:44, 2.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<02:56, 3.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:09, 4.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<37:01, 241kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<26:51, 332kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<18:59, 468kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<15:18, 578kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<11:30, 768kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<08:20, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<05:55, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<36:54, 238kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<28:12, 311kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<20:06, 436kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<14:04, 619kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<1:13:50, 118kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<52:24, 166kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<37:03, 235kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<25:53, 334kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<50:26, 171kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<36:13, 238kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<25:31, 337kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<19:47, 433kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<14:46, 579kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<10:32, 810kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<09:20, 910kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<07:27, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:25, 1.56MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<05:45, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:56, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:38, 2.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:31, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:04, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:02, 2.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:14, 3.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<55:36, 149kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<39:47, 208kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<27:59, 295kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<21:26, 384kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<15:53, 517kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<11:16, 727kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<09:46, 834kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<07:43, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:36, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<05:48, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:56, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:40, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:26, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:58, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:59, 2.67MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:56, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<03:37, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:44, 2.89MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:45, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<03:29, 2.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<02:38, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:40, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:24, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<02:35, 3.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:37, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:22, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<02:33, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:42, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:49, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:49, 2.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:28, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:15, 2.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:27, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:29, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:15, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:28, 3.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:28, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:14, 2.30MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:27, 3.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:27, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:12, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:26, 3.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:11, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:25, 3.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:23, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:09, 2.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:23, 3.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:21, 2.14MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:07, 2.30MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:20, 3.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:19, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:57, 2.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<02:21, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<01:43, 4.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<29:19, 241kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<21:15, 332kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<15:01, 468kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<12:05, 578kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<09:05, 768kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<06:34, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<04:39, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<29:08, 238kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<21:07, 327kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<14:53, 463kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<11:58, 573kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:06, 752kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<06:32, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<06:10, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:03, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:42, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:09, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:38, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:42, 2.46MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:27, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:08, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:21, 2.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:11, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:54, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:46, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<02:54, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:44, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:03, 3.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:56, 2.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:45, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:05, 3.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:57, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:45, 2.30MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:05, 3.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:56, 2.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:43, 2.31MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:04, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:54, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:42, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:03, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:52, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:00, 3.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:50, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:39, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:00, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:48, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:37, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:59, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:47, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:35, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<01:58, 3.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:45, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:34, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<01:57, 3.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:43, 2.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:31, 2.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<01:53, 3.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:41, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:29, 2.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<01:51, 3.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:38, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:28, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<01:52, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:37, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:26, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:51, 3.01MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:35, 2.14MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:24, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:48, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:33, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:22, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:46, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:21, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:47, 3.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:19, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<01:45, 3.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:37, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:56, 2.71MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<02:30, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:18, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:45, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:25, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:41, 3.03MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:22, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:12, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:39, 3.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:19, 2.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:11, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:39, 3.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:18, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:08, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:37, 3.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:16, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:06, 2.30MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:36, 3.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:05, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:33, 3.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:03, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:33, 3.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:01, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:32, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:59, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:29, 3.06MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<01:58, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:29, 3.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:04, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<01:56, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:28, 3.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:02, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<01:54, 2.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:25, 3.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:03, 4.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<4:14:07, 17.0kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 604M/862M [04:26<2:58:04, 24.2kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<2:04:00, 34.6kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<1:27:04, 48.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<1:01:19, 69.2kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<42:45, 98.6kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<29:44, 141kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<1:42:41, 40.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<1:12:13, 57.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<50:20, 82.4kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<35:54, 115kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<25:32, 161kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<17:52, 228kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<13:21, 303kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<09:46, 413kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<06:54, 581kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:43, 694kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:26, 895kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:11, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:07, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:36, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:55, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:13, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:58, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:28, 2.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:54, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:44, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:17, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<00:56, 3.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<27:51, 133kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<19:51, 186kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<13:54, 264kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:29, 346kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<07:43, 469kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<05:27, 659kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:37, 771kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:37, 983kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:35, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:37, 1.33MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:12, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:37, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:56, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:43, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:17, 2.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:40, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:32, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:09, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:34, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:27, 2.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:05, 3.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:30, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:24, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:02, 3.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:28, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:22, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:02, 3.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:26, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:20, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:00, 3.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:24, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:18, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<00:59, 3.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:22, 2.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:16, 2.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:57, 3.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<01:20, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:56, 3.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:18, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:13, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:55, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:16, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:11, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:53, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:14, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:09, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:52, 3.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:12, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:07, 2.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:51, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:06, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:49, 3.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:09, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:04, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:47, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:07, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:02, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:46, 3.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:33, 4.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<34:36, 67.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<24:21, 95.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<17:07, 135kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<11:45, 193kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<12:00, 188kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<08:37, 261kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:25<06:01, 370kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:39, 470kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:29, 626kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<02:27, 877kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:11, 965kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:46, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:16, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:21, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:10, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:51, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:03, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:57, 2.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:42, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:56, 2.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:38, 2.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:52, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:48, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:36, 3.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:26, 4.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<17:43, 100kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<12:32, 141kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<08:42, 201kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<06:22, 269kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<04:37, 369kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<03:13, 521kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:35, 632kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:59, 822kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:24, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:20, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:06, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:47, 1.94MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:53, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:47, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:35, 2.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:44, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:39, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:29, 2.84MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:39, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:36, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:26, 3.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:19, 4.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<09:45, 133kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<06:56, 186kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<04:47, 264kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<03:32, 347kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<02:36, 470kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:48, 660kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<01:30, 772kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:10, 985kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:50, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:49, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:41, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:30, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:34, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:22, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:28, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:19, 2.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:25, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:23, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:23, 2.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:21, 2.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 3.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:19, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:14, 3.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:17, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:12, 3.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:15, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:11, 3.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:14, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:10, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:06, 4.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<09:26, 50.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<06:33, 71.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<04:21, 102kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<02:53, 141kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<02:02, 197kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:20, 279kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:55, 365kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:40, 494kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:26, 695kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:16, 979kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<03:22, 80.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<02:20, 113kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<01:28, 161kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:55, 218kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:39, 301kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:23, 426kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:15, 532kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:10, 702kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:06, 977kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.76MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<13:19:57,  8.33it/s]  0%|          | 1014/400000 [00:00<9:18:44, 11.90it/s]  1%|          | 2096/400000 [00:00<6:30:14, 16.99it/s]  1%|          | 3144/400000 [00:00<4:32:38, 24.26it/s]  1%|          | 4252/400000 [00:00<3:10:29, 34.62it/s]  1%|▏         | 5332/400000 [00:00<2:13:09, 49.40it/s]  2%|▏         | 6326/400000 [00:00<1:33:10, 70.41it/s]  2%|▏         | 7481/400000 [00:00<1:05:12, 100.33it/s]  2%|▏         | 8546/400000 [00:00<45:42, 142.75it/s]    2%|▏         | 9669/400000 [00:01<32:04, 202.83it/s]  3%|▎         | 10773/400000 [00:01<22:33, 287.49it/s]  3%|▎         | 11908/400000 [00:01<15:55, 406.28it/s]  3%|▎         | 12994/400000 [00:01<11:18, 570.16it/s]  4%|▎         | 14045/400000 [00:01<08:05, 795.46it/s]  4%|▍         | 15087/400000 [00:01<05:50, 1098.80it/s]  4%|▍         | 16157/400000 [00:01<04:15, 1503.50it/s]  4%|▍         | 17198/400000 [00:01<03:10, 2004.87it/s]  5%|▍         | 18198/400000 [00:01<02:25, 2626.49it/s]  5%|▍         | 19183/400000 [00:01<01:53, 3359.78it/s]  5%|▌         | 20197/400000 [00:02<01:30, 4202.56it/s]  5%|▌         | 21242/400000 [00:02<01:13, 5120.87it/s]  6%|▌         | 22352/400000 [00:02<01:01, 6107.05it/s]  6%|▌         | 23389/400000 [00:02<00:54, 6897.09it/s]  6%|▌         | 24412/400000 [00:02<00:50, 7489.24it/s]  6%|▋         | 25409/400000 [00:02<00:46, 8008.64it/s]  7%|▋         | 26480/400000 [00:02<00:43, 8663.97it/s]  7%|▋         | 27555/400000 [00:02<00:40, 9199.30it/s]  7%|▋         | 28585/400000 [00:02<00:39, 9490.39it/s]  7%|▋         | 29614/400000 [00:02<00:39, 9494.11it/s]  8%|▊         | 30619/400000 [00:03<00:38, 9535.98it/s]  8%|▊         | 31612/400000 [00:03<00:38, 9636.53it/s]  8%|▊         | 32604/400000 [00:03<00:37, 9698.24it/s]  8%|▊         | 33725/400000 [00:03<00:36, 10106.87it/s]  9%|▊         | 34784/400000 [00:03<00:35, 10246.52it/s]  9%|▉         | 35822/400000 [00:03<00:35, 10249.13it/s]  9%|▉         | 36899/400000 [00:03<00:34, 10398.42it/s]  9%|▉         | 37946/400000 [00:03<00:34, 10386.90it/s] 10%|▉         | 38990/400000 [00:03<00:36, 9896.67it/s]  10%|█         | 40088/400000 [00:04<00:35, 10198.50it/s] 10%|█         | 41143/400000 [00:04<00:34, 10301.37it/s] 11%|█         | 42209/400000 [00:04<00:34, 10405.92it/s] 11%|█         | 43346/400000 [00:04<00:33, 10677.19it/s] 11%|█         | 44455/400000 [00:04<00:32, 10796.85it/s] 11%|█▏        | 45539/400000 [00:04<00:34, 10276.94it/s] 12%|█▏        | 46626/400000 [00:04<00:33, 10447.30it/s] 12%|█▏        | 47677/400000 [00:04<00:33, 10444.10it/s] 12%|█▏        | 48746/400000 [00:04<00:33, 10514.56it/s] 12%|█▏        | 49801/400000 [00:04<00:33, 10327.78it/s] 13%|█▎        | 50854/400000 [00:05<00:33, 10387.11it/s] 13%|█▎        | 51906/400000 [00:05<00:33, 10425.36it/s] 13%|█▎        | 53005/400000 [00:05<00:32, 10585.91it/s] 14%|█▎        | 54066/400000 [00:05<00:33, 10462.38it/s] 14%|█▍        | 55114/400000 [00:05<00:33, 10160.07it/s] 14%|█▍        | 56189/400000 [00:05<00:33, 10328.40it/s] 14%|█▍        | 57225/400000 [00:05<00:34, 9924.34it/s]  15%|█▍        | 58337/400000 [00:05<00:33, 10253.86it/s] 15%|█▍        | 59471/400000 [00:05<00:32, 10557.12it/s] 15%|█▌        | 60535/400000 [00:05<00:32, 10580.58it/s] 15%|█▌        | 61598/400000 [00:06<00:32, 10513.14it/s] 16%|█▌        | 62683/400000 [00:06<00:31, 10610.62it/s] 16%|█▌        | 63780/400000 [00:06<00:31, 10713.41it/s] 16%|█▌        | 64854/400000 [00:06<00:31, 10648.23it/s] 16%|█▋        | 65949/400000 [00:06<00:31, 10735.25it/s] 17%|█▋        | 67044/400000 [00:06<00:30, 10796.12it/s] 17%|█▋        | 68182/400000 [00:06<00:30, 10962.58it/s] 17%|█▋        | 69280/400000 [00:06<00:30, 10815.92it/s] 18%|█▊        | 70406/400000 [00:06<00:30, 10944.30it/s] 18%|█▊        | 71502/400000 [00:06<00:30, 10670.66it/s] 18%|█▊        | 72572/400000 [00:07<00:30, 10609.86it/s] 18%|█▊        | 73681/400000 [00:07<00:30, 10747.15it/s] 19%|█▊        | 74817/400000 [00:07<00:29, 10921.87it/s] 19%|█▉        | 75912/400000 [00:07<00:29, 10912.61it/s] 19%|█▉        | 77005/400000 [00:07<00:29, 10886.99it/s] 20%|█▉        | 78095/400000 [00:07<00:29, 10817.13it/s] 20%|█▉        | 79178/400000 [00:07<00:30, 10656.59it/s] 20%|██        | 80268/400000 [00:07<00:29, 10728.20it/s] 20%|██        | 81364/400000 [00:07<00:29, 10795.87it/s] 21%|██        | 82461/400000 [00:07<00:29, 10845.70it/s] 21%|██        | 83547/400000 [00:08<00:29, 10818.16it/s] 21%|██        | 84630/400000 [00:08<00:29, 10763.63it/s] 21%|██▏       | 85715/400000 [00:08<00:29, 10787.96it/s] 22%|██▏       | 86867/400000 [00:08<00:28, 10996.63it/s] 22%|██▏       | 87968/400000 [00:08<00:29, 10691.49it/s] 22%|██▏       | 89040/400000 [00:08<00:29, 10674.83it/s] 23%|██▎       | 90163/400000 [00:08<00:28, 10834.52it/s] 23%|██▎       | 91249/400000 [00:08<00:28, 10800.56it/s] 23%|██▎       | 92331/400000 [00:08<00:29, 10527.42it/s] 23%|██▎       | 93387/400000 [00:09<00:30, 10207.76it/s] 24%|██▎       | 94412/400000 [00:09<00:30, 9976.19it/s]  24%|██▍       | 95537/400000 [00:09<00:29, 10326.00it/s] 24%|██▍       | 96576/400000 [00:09<00:29, 10326.06it/s] 24%|██▍       | 97622/400000 [00:09<00:29, 10362.73it/s] 25%|██▍       | 98669/400000 [00:09<00:28, 10393.32it/s] 25%|██▍       | 99711/400000 [00:09<00:28, 10371.38it/s] 25%|██▌       | 100843/400000 [00:09<00:28, 10637.47it/s] 25%|██▌       | 101949/400000 [00:09<00:27, 10758.98it/s] 26%|██▌       | 103129/400000 [00:09<00:26, 11049.39it/s] 26%|██▌       | 104238/400000 [00:10<00:27, 10790.44it/s] 26%|██▋       | 105321/400000 [00:10<00:27, 10733.71it/s] 27%|██▋       | 106398/400000 [00:10<00:27, 10611.39it/s] 27%|██▋       | 107462/400000 [00:10<00:27, 10618.14it/s] 27%|██▋       | 108531/400000 [00:10<00:27, 10638.36it/s] 27%|██▋       | 109596/400000 [00:10<00:28, 10194.35it/s] 28%|██▊       | 110621/400000 [00:10<00:29, 9971.66it/s]  28%|██▊       | 111637/400000 [00:10<00:28, 10026.15it/s] 28%|██▊       | 112643/400000 [00:10<00:28, 9995.47it/s]  28%|██▊       | 113710/400000 [00:10<00:28, 10187.93it/s] 29%|██▊       | 114732/400000 [00:11<00:29, 9714.88it/s]  29%|██▉       | 115751/400000 [00:11<00:28, 9850.13it/s] 29%|██▉       | 116741/400000 [00:11<00:29, 9749.61it/s] 29%|██▉       | 117720/400000 [00:11<00:29, 9513.35it/s] 30%|██▉       | 118677/400000 [00:11<00:29, 9529.55it/s] 30%|██▉       | 119633/400000 [00:11<00:30, 9235.95it/s] 30%|███       | 120718/400000 [00:11<00:28, 9666.47it/s] 30%|███       | 121811/400000 [00:11<00:27, 10013.17it/s] 31%|███       | 122827/400000 [00:11<00:27, 10054.96it/s] 31%|███       | 123887/400000 [00:12<00:27, 10211.36it/s] 31%|███       | 124914/400000 [00:12<00:27, 10050.92it/s] 31%|███▏      | 125965/400000 [00:12<00:26, 10182.16it/s] 32%|███▏      | 127100/400000 [00:12<00:25, 10504.53it/s] 32%|███▏      | 128156/400000 [00:12<00:25, 10485.10it/s] 32%|███▏      | 129300/400000 [00:12<00:25, 10753.03it/s] 33%|███▎      | 130380/400000 [00:12<00:25, 10637.95it/s] 33%|███▎      | 131522/400000 [00:12<00:24, 10860.09it/s] 33%|███▎      | 132630/400000 [00:12<00:24, 10923.77it/s] 33%|███▎      | 133725/400000 [00:12<00:24, 10850.85it/s] 34%|███▎      | 134854/400000 [00:13<00:24, 10976.44it/s] 34%|███▍      | 135954/400000 [00:13<00:24, 10621.09it/s] 34%|███▍      | 137066/400000 [00:13<00:24, 10764.21it/s] 35%|███▍      | 138160/400000 [00:13<00:24, 10812.49it/s] 35%|███▍      | 139244/400000 [00:13<00:24, 10656.73it/s] 35%|███▌      | 140374/400000 [00:13<00:23, 10841.80it/s] 35%|███▌      | 141461/400000 [00:13<00:24, 10595.73it/s] 36%|███▌      | 142599/400000 [00:13<00:23, 10818.14it/s] 36%|███▌      | 143754/400000 [00:13<00:23, 11024.84it/s] 36%|███▌      | 144860/400000 [00:13<00:23, 10702.39it/s] 36%|███▋      | 145994/400000 [00:14<00:23, 10884.51it/s] 37%|███▋      | 147087/400000 [00:14<00:24, 10475.79it/s] 37%|███▋      | 148188/400000 [00:14<00:23, 10629.38it/s] 37%|███▋      | 149312/400000 [00:14<00:23, 10805.13it/s] 38%|███▊      | 150451/400000 [00:14<00:22, 10971.77it/s] 38%|███▊      | 151552/400000 [00:14<00:23, 10760.22it/s] 38%|███▊      | 152632/400000 [00:14<00:23, 10684.28it/s] 38%|███▊      | 153771/400000 [00:14<00:22, 10885.92it/s] 39%|███▊      | 154918/400000 [00:14<00:22, 11052.71it/s] 39%|███▉      | 156026/400000 [00:14<00:22, 10988.46it/s] 39%|███▉      | 157127/400000 [00:15<00:22, 10699.59it/s] 40%|███▉      | 158200/400000 [00:15<00:23, 10505.51it/s] 40%|███▉      | 159320/400000 [00:15<00:22, 10702.88it/s] 40%|████      | 160427/400000 [00:15<00:22, 10808.75it/s] 40%|████      | 161552/400000 [00:15<00:21, 10936.42it/s] 41%|████      | 162677/400000 [00:15<00:21, 11027.43it/s] 41%|████      | 163782/400000 [00:15<00:22, 10659.00it/s] 41%|████      | 164887/400000 [00:15<00:21, 10772.94it/s] 41%|████▏     | 165968/400000 [00:15<00:21, 10724.87it/s] 42%|████▏     | 167111/400000 [00:16<00:21, 10926.87it/s] 42%|████▏     | 168247/400000 [00:16<00:20, 11051.98it/s] 42%|████▏     | 169355/400000 [00:16<00:21, 10930.79it/s] 43%|████▎     | 170459/400000 [00:16<00:20, 10962.98it/s] 43%|████▎     | 171557/400000 [00:16<00:20, 10894.65it/s] 43%|████▎     | 172693/400000 [00:16<00:20, 11029.47it/s] 43%|████▎     | 173798/400000 [00:16<00:20, 10850.59it/s] 44%|████▎     | 174885/400000 [00:16<00:21, 10716.59it/s] 44%|████▍     | 176017/400000 [00:16<00:20, 10890.70it/s] 44%|████▍     | 177153/400000 [00:16<00:20, 11024.34it/s] 45%|████▍     | 178257/400000 [00:17<00:20, 10851.89it/s] 45%|████▍     | 179377/400000 [00:17<00:20, 10952.08it/s] 45%|████▌     | 180474/400000 [00:17<00:20, 10827.73it/s] 45%|████▌     | 181559/400000 [00:17<00:20, 10681.37it/s] 46%|████▌     | 182640/400000 [00:17<00:20, 10718.83it/s] 46%|████▌     | 183751/400000 [00:17<00:19, 10832.00it/s] 46%|████▌     | 184836/400000 [00:17<00:20, 10525.55it/s] 46%|████▋     | 185892/400000 [00:17<00:20, 10434.23it/s] 47%|████▋     | 187036/400000 [00:17<00:19, 10715.85it/s] 47%|████▋     | 188149/400000 [00:17<00:19, 10836.49it/s] 47%|████▋     | 189281/400000 [00:18<00:19, 10977.07it/s] 48%|████▊     | 190402/400000 [00:18<00:18, 11044.29it/s] 48%|████▊     | 191509/400000 [00:18<00:19, 10743.54it/s] 48%|████▊     | 192587/400000 [00:18<00:20, 10239.07it/s] 48%|████▊     | 193739/400000 [00:18<00:19, 10592.26it/s] 49%|████▊     | 194875/400000 [00:18<00:18, 10811.35it/s] 49%|████▉     | 196024/400000 [00:18<00:18, 11004.79it/s] 49%|████▉     | 197131/400000 [00:18<00:18, 10924.80it/s] 50%|████▉     | 198240/400000 [00:18<00:18, 10972.69it/s] 50%|████▉     | 199383/400000 [00:19<00:18, 11104.60it/s] 50%|█████     | 200496/400000 [00:19<00:17, 11099.88it/s] 50%|█████     | 201637/400000 [00:19<00:17, 11190.77it/s] 51%|█████     | 202758/400000 [00:19<00:17, 11031.57it/s] 51%|█████     | 203879/400000 [00:19<00:17, 11082.52it/s] 51%|█████     | 204989/400000 [00:19<00:17, 10873.84it/s] 52%|█████▏    | 206097/400000 [00:19<00:17, 10933.87it/s] 52%|█████▏    | 207192/400000 [00:19<00:18, 10416.48it/s] 52%|█████▏    | 208304/400000 [00:19<00:18, 10616.63it/s] 52%|█████▏    | 209381/400000 [00:19<00:17, 10661.13it/s] 53%|█████▎    | 210451/400000 [00:20<00:17, 10670.69it/s] 53%|█████▎    | 211564/400000 [00:20<00:17, 10802.78it/s] 53%|█████▎    | 212647/400000 [00:20<00:17, 10611.11it/s] 53%|█████▎    | 213711/400000 [00:20<00:17, 10405.98it/s] 54%|█████▎    | 214775/400000 [00:20<00:17, 10473.64it/s] 54%|█████▍    | 215871/400000 [00:20<00:17, 10614.91it/s] 54%|█████▍    | 217026/400000 [00:20<00:16, 10876.36it/s] 55%|█████▍    | 218117/400000 [00:20<00:17, 10477.12it/s] 55%|█████▍    | 219239/400000 [00:20<00:16, 10687.44it/s] 55%|█████▌    | 220381/400000 [00:20<00:16, 10895.35it/s] 55%|█████▌    | 221513/400000 [00:21<00:16, 11016.91it/s] 56%|█████▌    | 222654/400000 [00:21<00:15, 11128.22it/s] 56%|█████▌    | 223770/400000 [00:21<00:17, 10167.16it/s] 56%|█████▌    | 224850/400000 [00:21<00:16, 10338.12it/s] 56%|█████▋    | 225898/400000 [00:21<00:16, 10287.30it/s] 57%|█████▋    | 227058/400000 [00:21<00:16, 10646.99it/s] 57%|█████▋    | 228133/400000 [00:21<00:16, 10393.80it/s] 57%|█████▋    | 229196/400000 [00:21<00:16, 10462.46it/s] 58%|█████▊    | 230272/400000 [00:21<00:16, 10548.19it/s] 58%|█████▊    | 231398/400000 [00:22<00:15, 10751.38it/s] 58%|█████▊    | 232565/400000 [00:22<00:15, 11009.93it/s] 58%|█████▊    | 233719/400000 [00:22<00:14, 11162.15it/s] 59%|█████▊    | 234839/400000 [00:22<00:15, 10989.12it/s] 59%|█████▉    | 235987/400000 [00:22<00:14, 11130.54it/s] 59%|█████▉    | 237103/400000 [00:22<00:14, 11072.78it/s] 60%|█████▉    | 238213/400000 [00:22<00:14, 10817.09it/s] 60%|█████▉    | 239310/400000 [00:22<00:14, 10862.37it/s] 60%|██████    | 240399/400000 [00:22<00:14, 10868.38it/s] 60%|██████    | 241509/400000 [00:22<00:14, 10935.60it/s] 61%|██████    | 242632/400000 [00:23<00:14, 11020.04it/s] 61%|██████    | 243735/400000 [00:23<00:14, 10843.94it/s] 61%|██████    | 244879/400000 [00:23<00:14, 11015.32it/s] 61%|██████▏   | 245983/400000 [00:23<00:14, 10902.26it/s] 62%|██████▏   | 247075/400000 [00:23<00:14, 10906.37it/s] 62%|██████▏   | 248184/400000 [00:23<00:13, 10960.28it/s] 62%|██████▏   | 249315/400000 [00:23<00:13, 11059.89it/s] 63%|██████▎   | 250442/400000 [00:23<00:13, 11119.42it/s] 63%|██████▎   | 251555/400000 [00:23<00:13, 10679.23it/s] 63%|██████▎   | 252715/400000 [00:23<00:13, 10938.78it/s] 63%|██████▎   | 253814/400000 [00:24<00:14, 10419.05it/s] 64%|██████▎   | 254904/400000 [00:24<00:13, 10555.34it/s] 64%|██████▍   | 255966/400000 [00:24<00:14, 10231.82it/s] 64%|██████▍   | 256996/400000 [00:24<00:14, 10150.76it/s] 65%|██████▍   | 258128/400000 [00:24<00:13, 10475.20it/s] 65%|██████▍   | 259271/400000 [00:24<00:13, 10744.00it/s] 65%|██████▌   | 260352/400000 [00:24<00:12, 10747.55it/s] 65%|██████▌   | 261431/400000 [00:24<00:12, 10705.15it/s] 66%|██████▌   | 262505/400000 [00:24<00:12, 10668.44it/s] 66%|██████▌   | 263650/400000 [00:24<00:12, 10889.82it/s] 66%|██████▌   | 264811/400000 [00:25<00:12, 11094.17it/s] 66%|██████▋   | 265924/400000 [00:25<00:12, 11102.34it/s] 67%|██████▋   | 267037/400000 [00:25<00:11, 11107.31it/s] 67%|██████▋   | 268150/400000 [00:25<00:12, 10911.40it/s] 67%|██████▋   | 269243/400000 [00:25<00:12, 10644.58it/s] 68%|██████▊   | 270343/400000 [00:25<00:12, 10746.24it/s] 68%|██████▊   | 271469/400000 [00:25<00:11, 10894.81it/s] 68%|██████▊   | 272561/400000 [00:25<00:11, 10883.79it/s] 68%|██████▊   | 273651/400000 [00:25<00:11, 10743.73it/s] 69%|██████▊   | 274796/400000 [00:25<00:11, 10946.27it/s] 69%|██████▉   | 275917/400000 [00:26<00:11, 11021.73it/s] 69%|██████▉   | 277021/400000 [00:26<00:11, 10991.34it/s] 70%|██████▉   | 278122/400000 [00:26<00:11, 10910.58it/s] 70%|██████▉   | 279214/400000 [00:26<00:11, 10787.21it/s] 70%|███████   | 280354/400000 [00:26<00:10, 10963.22it/s] 70%|███████   | 281469/400000 [00:26<00:10, 11018.00it/s] 71%|███████   | 282572/400000 [00:26<00:10, 10949.01it/s] 71%|███████   | 283668/400000 [00:26<00:10, 10759.53it/s] 71%|███████   | 284746/400000 [00:26<00:10, 10619.84it/s] 71%|███████▏  | 285855/400000 [00:27<00:10, 10755.91it/s] 72%|███████▏  | 286932/400000 [00:27<00:10, 10751.09it/s] 72%|███████▏  | 288009/400000 [00:27<00:10, 10650.89it/s] 72%|███████▏  | 289075/400000 [00:27<00:10, 10601.91it/s] 73%|███████▎  | 290136/400000 [00:27<00:10, 10349.52it/s] 73%|███████▎  | 291221/400000 [00:27<00:10, 10494.59it/s] 73%|███████▎  | 292273/400000 [00:27<00:10, 10430.34it/s] 73%|███████▎  | 293318/400000 [00:27<00:10, 10367.77it/s] 74%|███████▎  | 294398/400000 [00:27<00:10, 10493.22it/s] 74%|███████▍  | 295449/400000 [00:27<00:10, 10404.16it/s] 74%|███████▍  | 296555/400000 [00:28<00:09, 10591.02it/s] 74%|███████▍  | 297658/400000 [00:28<00:09, 10716.53it/s] 75%|███████▍  | 298788/400000 [00:28<00:09, 10883.44it/s] 75%|███████▍  | 299937/400000 [00:28<00:09, 11056.62it/s] 75%|███████▌  | 301045/400000 [00:28<00:09, 10970.85it/s] 76%|███████▌  | 302172/400000 [00:28<00:08, 11055.93it/s] 76%|███████▌  | 303279/400000 [00:28<00:08, 10865.62it/s] 76%|███████▌  | 304436/400000 [00:28<00:08, 11064.81it/s] 76%|███████▋  | 305559/400000 [00:28<00:08, 11112.94it/s] 77%|███████▋  | 306672/400000 [00:28<00:08, 10962.90it/s] 77%|███████▋  | 307806/400000 [00:29<00:08, 11072.89it/s] 77%|███████▋  | 308915/400000 [00:29<00:08, 11033.05it/s] 78%|███████▊  | 310025/400000 [00:29<00:08, 11051.27it/s] 78%|███████▊  | 311149/400000 [00:29<00:08, 11105.72it/s] 78%|███████▊  | 312261/400000 [00:29<00:08, 10861.54it/s] 78%|███████▊  | 313349/400000 [00:29<00:08, 10770.62it/s] 79%|███████▊  | 314428/400000 [00:29<00:08, 10690.23it/s] 79%|███████▉  | 315499/400000 [00:29<00:08, 10036.74it/s] 79%|███████▉  | 316512/400000 [00:29<00:08, 9929.18it/s]  79%|███████▉  | 317512/400000 [00:29<00:08, 9905.89it/s] 80%|███████▉  | 318634/400000 [00:30<00:07, 10265.28it/s] 80%|███████▉  | 319744/400000 [00:30<00:07, 10499.67it/s] 80%|████████  | 320800/400000 [00:30<00:07, 10349.37it/s] 80%|████████  | 321870/400000 [00:30<00:07, 10450.84it/s] 81%|████████  | 322919/400000 [00:30<00:07, 10340.34it/s] 81%|████████  | 324061/400000 [00:30<00:07, 10640.81it/s] 81%|████████▏ | 325181/400000 [00:30<00:06, 10801.77it/s] 82%|████████▏ | 326265/400000 [00:30<00:06, 10788.66it/s] 82%|████████▏ | 327397/400000 [00:30<00:06, 10940.94it/s] 82%|████████▏ | 328494/400000 [00:31<00:06, 10730.19it/s] 82%|████████▏ | 329633/400000 [00:31<00:06, 10919.70it/s] 83%|████████▎ | 330728/400000 [00:31<00:06, 10592.84it/s] 83%|████████▎ | 331819/400000 [00:31<00:06, 10684.33it/s] 83%|████████▎ | 332923/400000 [00:31<00:06, 10788.02it/s] 84%|████████▎ | 334005/400000 [00:31<00:06, 10548.96it/s] 84%|████████▍ | 335121/400000 [00:31<00:06, 10722.95it/s] 84%|████████▍ | 336200/400000 [00:31<00:05, 10742.58it/s] 84%|████████▍ | 337286/400000 [00:31<00:05, 10777.38it/s] 85%|████████▍ | 338403/400000 [00:31<00:05, 10890.49it/s] 85%|████████▍ | 339494/400000 [00:32<00:05, 10816.34it/s] 85%|████████▌ | 340651/400000 [00:32<00:05, 11029.24it/s] 85%|████████▌ | 341756/400000 [00:32<00:05, 10970.15it/s] 86%|████████▌ | 342855/400000 [00:32<00:05, 10893.39it/s] 86%|████████▌ | 344019/400000 [00:32<00:05, 11104.32it/s] 86%|████████▋ | 345132/400000 [00:32<00:05, 10859.78it/s] 87%|████████▋ | 346221/400000 [00:32<00:05, 10566.35it/s] 87%|████████▋ | 347285/400000 [00:32<00:04, 10587.63it/s] 87%|████████▋ | 348371/400000 [00:32<00:04, 10666.73it/s] 87%|████████▋ | 349440/400000 [00:32<00:04, 10296.97it/s] 88%|████████▊ | 350474/400000 [00:33<00:04, 9930.99it/s]  88%|████████▊ | 351473/400000 [00:33<00:04, 9908.99it/s] 88%|████████▊ | 352468/400000 [00:33<00:04, 9783.06it/s] 88%|████████▊ | 353548/400000 [00:33<00:04, 10066.89it/s] 89%|████████▊ | 354559/400000 [00:33<00:04, 9755.02it/s]  89%|████████▉ | 355692/400000 [00:33<00:04, 10179.21it/s] 89%|████████▉ | 356824/400000 [00:33<00:04, 10495.81it/s] 89%|████████▉ | 357935/400000 [00:33<00:03, 10670.61it/s] 90%|████████▉ | 359095/400000 [00:33<00:03, 10933.07it/s] 90%|█████████ | 360195/400000 [00:34<00:03, 10690.77it/s] 90%|█████████ | 361298/400000 [00:34<00:03, 10788.63it/s] 91%|█████████ | 362381/400000 [00:34<00:03, 10737.52it/s] 91%|█████████ | 363458/400000 [00:34<00:03, 10502.87it/s] 91%|█████████ | 364536/400000 [00:34<00:03, 10582.82it/s] 91%|█████████▏| 365622/400000 [00:34<00:03, 10662.25it/s] 92%|█████████▏| 366700/400000 [00:34<00:03, 10696.54it/s] 92%|█████████▏| 367819/400000 [00:34<00:02, 10839.50it/s] 92%|█████████▏| 368961/400000 [00:34<00:02, 11005.06it/s] 93%|█████████▎| 370143/400000 [00:34<00:02, 11234.96it/s] 93%|█████████▎| 371269/400000 [00:35<00:02, 11052.45it/s] 93%|█████████▎| 372377/400000 [00:35<00:02, 10898.69it/s] 93%|█████████▎| 373469/400000 [00:35<00:02, 10364.69it/s] 94%|█████████▎| 374553/400000 [00:35<00:02, 10501.56it/s] 94%|█████████▍| 375610/400000 [00:35<00:02, 10521.24it/s] 94%|█████████▍| 376666/400000 [00:35<00:02, 10221.26it/s] 94%|█████████▍| 377805/400000 [00:35<00:02, 10543.72it/s] 95%|█████████▍| 378914/400000 [00:35<00:01, 10700.01it/s] 95%|█████████▌| 380022/400000 [00:35<00:01, 10810.50it/s] 95%|█████████▌| 381121/400000 [00:35<00:01, 10862.90it/s] 96%|█████████▌| 382210/400000 [00:36<00:01, 10397.27it/s] 96%|█████████▌| 383306/400000 [00:36<00:01, 10557.57it/s] 96%|█████████▌| 384421/400000 [00:36<00:01, 10728.25it/s] 96%|█████████▋| 385579/400000 [00:36<00:01, 10968.04it/s] 97%|█████████▋| 386698/400000 [00:36<00:01, 11032.01it/s] 97%|█████████▋| 387805/400000 [00:36<00:01, 10899.42it/s] 97%|█████████▋| 388923/400000 [00:36<00:01, 10981.81it/s] 98%|█████████▊| 390062/400000 [00:36<00:00, 11098.75it/s] 98%|█████████▊| 391185/400000 [00:36<00:00, 11136.06it/s] 98%|█████████▊| 392300/400000 [00:36<00:00, 11103.72it/s] 98%|█████████▊| 393412/400000 [00:37<00:00, 10871.10it/s] 99%|█████████▊| 394552/400000 [00:37<00:00, 11022.11it/s] 99%|█████████▉| 395656/400000 [00:37<00:00, 10813.29it/s] 99%|█████████▉| 396780/400000 [00:37<00:00, 10937.49it/s] 99%|█████████▉| 397934/400000 [00:37<00:00, 11111.01it/s]100%|█████████▉| 399048/400000 [00:37<00:00, 10600.14it/s]100%|█████████▉| 399999/400000 [00:37<00:00, 10611.17it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff3b0d58e80>, <torchtext.data.dataset.TabularDataset object at 0x7ff3b0d58fd0>, <torchtext.vocab.Vocab object at 0x7ff3b0d58ef0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 34.23 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 24.25 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 33.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.81 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.81 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.78 file/s]2020-06-26 06:18:38.089473: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-26 06:18:38.093528: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-06-26 06:18:38.093670: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5564b54275b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 06:18:38.093681: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3309568/9912422 [00:00<00:00, 32729887.20it/s]9920512it [00:00, 34694073.49it/s]                             
0it [00:00, ?it/s]32768it [00:00, 724552.70it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486728.96it/s]1654784it [00:00, 12361731.38it/s]                         
0it [00:00, ?it/s]8192it [00:00, 260482.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
