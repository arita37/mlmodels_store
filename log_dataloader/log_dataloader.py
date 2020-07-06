
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1a47d4e048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1a47d4e048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1ab38f61e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1ab38f61e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1ad1c3dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1ad1c3dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1a60c23620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1a60c23620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1a60c23620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  4%|▍         | 393216/9912422 [00:00<00:02, 3932154.38it/s]9920512it [00:00, 29553730.69it/s]                           
0it [00:00, ?it/s]32768it [00:00, 559395.31it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456882.05it/s]1654784it [00:00, 11086530.76it/s]                         
0it [00:00, ?it/s]8192it [00:00, 154457.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1a47c63908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1a47c56ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1a60c23268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1a60c23268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1a60c23268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:46,  1.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:46,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:45,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:45,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:44,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:43,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:43,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:42,  1.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:11,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:10,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:09,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:09,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:08,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:47,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:46,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:46,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:45,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:31,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:31,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:30,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:21,  5.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:21,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:21,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:09, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 11.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 15.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 26.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 26.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 34.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 42.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 50.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 66.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.08 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.08 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.26 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.47s/ url]
0 examples [00:00, ? examples/s]2020-07-06 00:08:46.631505: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-06 00:08:46.668390: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-06 00:08:46.668618: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f85c6ecd00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 00:08:46.668635: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.88 examples/s]103 examples [00:00,  4.12 examples/s]211 examples [00:00,  5.87 examples/s]315 examples [00:00,  8.36 examples/s]417 examples [00:00, 11.91 examples/s]529 examples [00:00, 16.93 examples/s]644 examples [00:00, 24.04 examples/s]750 examples [00:01, 34.01 examples/s]858 examples [00:01, 47.93 examples/s]967 examples [00:01, 67.20 examples/s]1072 examples [00:01, 93.44 examples/s]1183 examples [00:01, 128.79 examples/s]1289 examples [00:01, 173.83 examples/s]1394 examples [00:01, 231.74 examples/s]1498 examples [00:01, 301.98 examples/s]1603 examples [00:01, 384.06 examples/s]1714 examples [00:01, 477.25 examples/s]1820 examples [00:02, 562.25 examples/s]1924 examples [00:02, 651.00 examples/s]2027 examples [00:02, 712.58 examples/s]2140 examples [00:02, 800.40 examples/s]2248 examples [00:02, 866.50 examples/s]2353 examples [00:02, 888.25 examples/s]2455 examples [00:02, 889.29 examples/s]2556 examples [00:02, 922.23 examples/s]2665 examples [00:02, 966.45 examples/s]2767 examples [00:03, 976.59 examples/s]2875 examples [00:03, 1004.76 examples/s]2979 examples [00:03, 1001.11 examples/s]3091 examples [00:03, 1033.09 examples/s]3204 examples [00:03, 1057.86 examples/s]3313 examples [00:03, 1064.59 examples/s]3424 examples [00:03, 1076.13 examples/s]3533 examples [00:03, 1049.82 examples/s]3639 examples [00:03, 1021.68 examples/s]3742 examples [00:03, 1009.93 examples/s]3847 examples [00:04, 1019.43 examples/s]3950 examples [00:04, 1021.89 examples/s]4053 examples [00:04, 1012.40 examples/s]4156 examples [00:04, 1015.12 examples/s]4261 examples [00:04, 1024.07 examples/s]4370 examples [00:04, 1041.05 examples/s]4475 examples [00:04, 1029.37 examples/s]4579 examples [00:04, 1007.45 examples/s]4680 examples [00:04, 1002.80 examples/s]4782 examples [00:04, 1005.50 examples/s]4883 examples [00:05, 997.01 examples/s] 4983 examples [00:05, 973.75 examples/s]5081 examples [00:05, 963.70 examples/s]5178 examples [00:05, 960.30 examples/s]5283 examples [00:05, 984.49 examples/s]5382 examples [00:05, 982.22 examples/s]5481 examples [00:05, 968.95 examples/s]5590 examples [00:05, 999.36 examples/s]5692 examples [00:05, 1004.58 examples/s]5797 examples [00:06, 1015.54 examples/s]5899 examples [00:06, 996.95 examples/s] 5999 examples [00:06, 996.12 examples/s]6100 examples [00:06, 997.64 examples/s]6204 examples [00:06, 1008.92 examples/s]6309 examples [00:06, 1019.16 examples/s]6413 examples [00:06, 1023.32 examples/s]6516 examples [00:06, 1014.13 examples/s]6620 examples [00:06, 1019.51 examples/s]6731 examples [00:06, 1043.54 examples/s]6839 examples [00:07, 1052.14 examples/s]6945 examples [00:07, 1046.56 examples/s]7050 examples [00:07, 992.66 examples/s] 7150 examples [00:07, 951.12 examples/s]7250 examples [00:07, 963.29 examples/s]7351 examples [00:07, 975.27 examples/s]7449 examples [00:07, 974.46 examples/s]7547 examples [00:07, 968.94 examples/s]7645 examples [00:07, 970.83 examples/s]7745 examples [00:07, 977.83 examples/s]7844 examples [00:08, 979.34 examples/s]7943 examples [00:08, 969.37 examples/s]8042 examples [00:08, 973.39 examples/s]8140 examples [00:08, 956.96 examples/s]8236 examples [00:08, 947.77 examples/s]8335 examples [00:08, 959.51 examples/s]8432 examples [00:08, 956.64 examples/s]8534 examples [00:08, 954.55 examples/s]8634 examples [00:08, 965.85 examples/s]8742 examples [00:08, 995.76 examples/s]8853 examples [00:09, 1025.04 examples/s]8956 examples [00:09, 1016.38 examples/s]9059 examples [00:09, 1018.08 examples/s]9164 examples [00:09, 1027.10 examples/s]9269 examples [00:09, 1030.67 examples/s]9379 examples [00:09, 1049.27 examples/s]9485 examples [00:09, 1024.53 examples/s]9588 examples [00:09, 1018.46 examples/s]9691 examples [00:09, 1014.57 examples/s]9793 examples [00:10, 1014.32 examples/s]9902 examples [00:10, 1035.83 examples/s]10006 examples [00:10, 955.37 examples/s]10120 examples [00:10, 1003.36 examples/s]10224 examples [00:10, 1011.80 examples/s]10334 examples [00:10, 1035.86 examples/s]10445 examples [00:10, 1055.11 examples/s]10552 examples [00:10, 1009.38 examples/s]10658 examples [00:10, 1021.99 examples/s]10771 examples [00:10, 1050.88 examples/s]10877 examples [00:11, 1052.23 examples/s]10994 examples [00:11, 1084.31 examples/s]11104 examples [00:11, 1045.72 examples/s]11212 examples [00:11, 1054.08 examples/s]11326 examples [00:11, 1076.45 examples/s]11440 examples [00:11, 1093.80 examples/s]11550 examples [00:11, 1072.50 examples/s]11658 examples [00:11, 1015.17 examples/s]11761 examples [00:11, 1005.30 examples/s]11865 examples [00:12, 1011.39 examples/s]11972 examples [00:12, 1027.90 examples/s]12079 examples [00:12, 1040.11 examples/s]12184 examples [00:12, 1012.84 examples/s]12291 examples [00:12, 1027.54 examples/s]12400 examples [00:12, 1043.33 examples/s]12505 examples [00:12, 1034.81 examples/s]12609 examples [00:12, 1036.09 examples/s]12713 examples [00:12, 1020.59 examples/s]12821 examples [00:12, 1035.10 examples/s]12934 examples [00:13, 1061.02 examples/s]13041 examples [00:13, 1060.15 examples/s]13148 examples [00:13, 1044.15 examples/s]13253 examples [00:13, 1037.79 examples/s]13360 examples [00:13, 1034.14 examples/s]13464 examples [00:13, 1034.16 examples/s]13568 examples [00:13, 997.13 examples/s] 13669 examples [00:13, 899.33 examples/s]13761 examples [00:13, 875.33 examples/s]13851 examples [00:14, 844.65 examples/s]13941 examples [00:14, 855.70 examples/s]14028 examples [00:14, 847.76 examples/s]14114 examples [00:14, 843.57 examples/s]14200 examples [00:14, 848.26 examples/s]14293 examples [00:14, 869.30 examples/s]14382 examples [00:14, 874.85 examples/s]14470 examples [00:14, 844.39 examples/s]14563 examples [00:14, 867.91 examples/s]14655 examples [00:14, 881.64 examples/s]14767 examples [00:15, 940.76 examples/s]14888 examples [00:15, 1007.87 examples/s]14997 examples [00:15, 1028.39 examples/s]15102 examples [00:15, 1021.47 examples/s]15217 examples [00:15, 1055.78 examples/s]15328 examples [00:15, 1071.42 examples/s]15436 examples [00:15, 1062.17 examples/s]15543 examples [00:15, 1029.03 examples/s]15647 examples [00:15, 992.57 examples/s] 15761 examples [00:15, 1030.79 examples/s]15865 examples [00:16, 1032.87 examples/s]15979 examples [00:16, 1060.62 examples/s]16088 examples [00:16, 1068.51 examples/s]16196 examples [00:16, 1039.61 examples/s]16307 examples [00:16, 1057.52 examples/s]16421 examples [00:16, 1079.50 examples/s]16530 examples [00:16, 1052.95 examples/s]16636 examples [00:16, 1035.24 examples/s]16740 examples [00:16, 1030.14 examples/s]16855 examples [00:17, 1063.05 examples/s]16965 examples [00:17, 1073.78 examples/s]17073 examples [00:17, 1074.42 examples/s]17181 examples [00:17, 1062.71 examples/s]17288 examples [00:17, 1046.77 examples/s]17393 examples [00:17, 1037.40 examples/s]17506 examples [00:17, 1062.36 examples/s]17613 examples [00:17, 1025.51 examples/s]17717 examples [00:17, 1026.46 examples/s]17820 examples [00:17, 1012.60 examples/s]17922 examples [00:18, 1002.50 examples/s]18023 examples [00:18, 986.20 examples/s] 18122 examples [00:18, 967.82 examples/s]18220 examples [00:18, 947.61 examples/s]18319 examples [00:18, 957.66 examples/s]18419 examples [00:18, 969.37 examples/s]18526 examples [00:18, 995.25 examples/s]18631 examples [00:18, 1008.39 examples/s]18733 examples [00:18, 999.25 examples/s] 18834 examples [00:18, 987.95 examples/s]18946 examples [00:19, 1022.12 examples/s]19049 examples [00:19, 1022.14 examples/s]19152 examples [00:19, 1023.81 examples/s]19255 examples [00:19, 1025.12 examples/s]19358 examples [00:19, 1010.25 examples/s]19460 examples [00:19, 1000.80 examples/s]19564 examples [00:19, 1010.12 examples/s]19666 examples [00:19, 1003.55 examples/s]19767 examples [00:19, 993.02 examples/s] 19870 examples [00:19, 1002.83 examples/s]19979 examples [00:20, 1025.24 examples/s]20082 examples [00:20, 976.84 examples/s] 20188 examples [00:20, 997.97 examples/s]20289 examples [00:20, 999.45 examples/s]20391 examples [00:20, 1002.99 examples/s]20495 examples [00:20, 1011.97 examples/s]20601 examples [00:20, 1025.86 examples/s]20704 examples [00:20, 1015.86 examples/s]20809 examples [00:20, 1023.76 examples/s]20912 examples [00:21, 1013.36 examples/s]21017 examples [00:21, 1023.52 examples/s]21123 examples [00:21, 1033.96 examples/s]21227 examples [00:21, 1023.42 examples/s]21330 examples [00:21, 973.20 examples/s] 21428 examples [00:21, 971.30 examples/s]21538 examples [00:21, 1004.60 examples/s]21649 examples [00:21, 1032.63 examples/s]21762 examples [00:21, 1056.90 examples/s]21869 examples [00:21, 1038.16 examples/s]21974 examples [00:22, 1013.63 examples/s]22076 examples [00:22, 1002.26 examples/s]22177 examples [00:22, 996.58 examples/s] 22277 examples [00:22, 985.90 examples/s]22376 examples [00:22, 985.09 examples/s]22475 examples [00:22, 986.52 examples/s]22585 examples [00:22, 1015.62 examples/s]22687 examples [00:22, 1015.03 examples/s]22791 examples [00:22, 1021.98 examples/s]22894 examples [00:22, 1015.42 examples/s]22996 examples [00:23, 1013.36 examples/s]23104 examples [00:23, 1029.52 examples/s]23208 examples [00:23, 1032.52 examples/s]23312 examples [00:23, 1012.86 examples/s]23414 examples [00:23, 1013.59 examples/s]23518 examples [00:23, 1021.34 examples/s]23630 examples [00:23, 1048.92 examples/s]23741 examples [00:23, 1065.75 examples/s]23848 examples [00:23, 1054.08 examples/s]23955 examples [00:24, 1057.38 examples/s]24061 examples [00:24, 1054.19 examples/s]24173 examples [00:24, 1071.55 examples/s]24283 examples [00:24, 1078.37 examples/s]24393 examples [00:24, 1082.21 examples/s]24502 examples [00:24, 1069.39 examples/s]24610 examples [00:24, 1051.46 examples/s]24723 examples [00:24, 1072.65 examples/s]24836 examples [00:24, 1087.23 examples/s]24955 examples [00:24, 1113.12 examples/s]25067 examples [00:25, 1109.55 examples/s]25179 examples [00:25, 1086.15 examples/s]25288 examples [00:25, 1080.69 examples/s]25397 examples [00:25, 1081.33 examples/s]25506 examples [00:25, 1067.30 examples/s]25613 examples [00:25, 1053.77 examples/s]25719 examples [00:25, 1034.38 examples/s]25825 examples [00:25, 1041.59 examples/s]25936 examples [00:25, 1060.93 examples/s]26043 examples [00:25, 1062.17 examples/s]26150 examples [00:26, 1053.57 examples/s]26256 examples [00:26, 1038.73 examples/s]26367 examples [00:26, 1058.60 examples/s]26478 examples [00:26, 1071.40 examples/s]26586 examples [00:26, 1072.04 examples/s]26697 examples [00:26, 1082.33 examples/s]26806 examples [00:26, 1063.54 examples/s]26913 examples [00:26, 1058.51 examples/s]27019 examples [00:26, 1021.00 examples/s]27129 examples [00:26, 1041.71 examples/s]27234 examples [00:27, 1043.24 examples/s]27339 examples [00:27, 1017.63 examples/s]27446 examples [00:27, 1031.49 examples/s]27558 examples [00:27, 1054.30 examples/s]27665 examples [00:27, 1057.90 examples/s]27773 examples [00:27, 1062.73 examples/s]27880 examples [00:27, 1039.72 examples/s]27989 examples [00:27, 1053.70 examples/s]28101 examples [00:27, 1070.25 examples/s]28209 examples [00:28, 1058.55 examples/s]28316 examples [00:28, 1048.56 examples/s]28421 examples [00:28, 1048.21 examples/s]28526 examples [00:28, 1040.93 examples/s]28635 examples [00:28, 1053.69 examples/s]28741 examples [00:28, 1047.27 examples/s]28850 examples [00:28, 1058.21 examples/s]28956 examples [00:28, 1044.50 examples/s]29067 examples [00:28, 1061.21 examples/s]29180 examples [00:28, 1079.46 examples/s]29289 examples [00:29, 1081.17 examples/s]29400 examples [00:29, 1089.38 examples/s]29510 examples [00:29, 1063.94 examples/s]29620 examples [00:29, 1073.46 examples/s]29728 examples [00:29, 1067.44 examples/s]29836 examples [00:29, 1068.65 examples/s]29947 examples [00:29, 1078.57 examples/s]30055 examples [00:29, 1009.88 examples/s]30166 examples [00:29, 1035.48 examples/s]30271 examples [00:29, 1014.86 examples/s]30379 examples [00:30, 1030.52 examples/s]30490 examples [00:30, 1051.03 examples/s]30596 examples [00:30, 1047.10 examples/s]30708 examples [00:30, 1065.21 examples/s]30815 examples [00:30, 1065.55 examples/s]30922 examples [00:30, 1060.74 examples/s]31033 examples [00:30, 1074.76 examples/s]31141 examples [00:30, 1061.10 examples/s]31249 examples [00:30, 1065.01 examples/s]31357 examples [00:30, 1069.28 examples/s]31465 examples [00:31, 1048.67 examples/s]31577 examples [00:31, 1067.62 examples/s]31684 examples [00:31, 1033.36 examples/s]31789 examples [00:31, 1035.41 examples/s]31893 examples [00:31, 1016.33 examples/s]31995 examples [00:31, 991.68 examples/s] 32105 examples [00:31, 1019.23 examples/s]32208 examples [00:31, 1016.31 examples/s]32318 examples [00:31, 1039.16 examples/s]32429 examples [00:32, 1057.34 examples/s]32536 examples [00:32, 1060.23 examples/s]32643 examples [00:32, 1050.55 examples/s]32749 examples [00:32, 1028.85 examples/s]32853 examples [00:32, 1025.10 examples/s]32956 examples [00:32, 1019.82 examples/s]33067 examples [00:32, 1042.97 examples/s]33179 examples [00:32, 1062.51 examples/s]33286 examples [00:32, 1059.38 examples/s]33398 examples [00:32, 1075.45 examples/s]33506 examples [00:33, 1075.37 examples/s]33614 examples [00:33, 1034.51 examples/s]33727 examples [00:33, 1060.38 examples/s]33834 examples [00:33, 1049.82 examples/s]33947 examples [00:33, 1070.29 examples/s]34059 examples [00:33, 1082.40 examples/s]34168 examples [00:33, 1057.74 examples/s]34279 examples [00:33, 1071.63 examples/s]34387 examples [00:33, 1049.95 examples/s]34501 examples [00:33, 1072.33 examples/s]34610 examples [00:34, 1075.29 examples/s]34719 examples [00:34, 1077.53 examples/s]34833 examples [00:34, 1092.90 examples/s]34943 examples [00:34, 1073.21 examples/s]35056 examples [00:34, 1089.12 examples/s]35166 examples [00:34, 1077.51 examples/s]35274 examples [00:34, 1040.29 examples/s]35384 examples [00:34, 1055.50 examples/s]35490 examples [00:34, 1040.88 examples/s]35601 examples [00:35, 1059.56 examples/s]35708 examples [00:35, 1053.28 examples/s]35814 examples [00:35, 1044.06 examples/s]35919 examples [00:35, 1002.31 examples/s]36020 examples [00:35, 1000.85 examples/s]36130 examples [00:35, 1026.70 examples/s]36239 examples [00:35, 1044.22 examples/s]36344 examples [00:35, 1042.94 examples/s]36449 examples [00:35, 1042.76 examples/s]36556 examples [00:35, 1049.71 examples/s]36662 examples [00:36, 1016.58 examples/s]36772 examples [00:36, 1038.62 examples/s]36883 examples [00:36, 1056.67 examples/s]36989 examples [00:36, 1038.29 examples/s]37094 examples [00:36, 1039.52 examples/s]37209 examples [00:36, 1069.12 examples/s]37317 examples [00:36, 1064.02 examples/s]37424 examples [00:36, 1014.19 examples/s]37527 examples [00:36, 1010.57 examples/s]37638 examples [00:36, 1037.35 examples/s]37753 examples [00:37, 1066.56 examples/s]37863 examples [00:37, 1074.74 examples/s]37971 examples [00:37, 1071.93 examples/s]38079 examples [00:37, 1019.98 examples/s]38189 examples [00:37, 1042.02 examples/s]38302 examples [00:37, 1065.39 examples/s]38410 examples [00:37, 1060.74 examples/s]38517 examples [00:37, 1049.01 examples/s]38623 examples [00:37, 1020.95 examples/s]38726 examples [00:38, 1011.91 examples/s]38828 examples [00:38, 997.82 examples/s] 38932 examples [00:38, 1009.62 examples/s]39034 examples [00:38, 995.67 examples/s] 39134 examples [00:38, 992.50 examples/s]39246 examples [00:38, 1026.17 examples/s]39350 examples [00:38, 1028.28 examples/s]39454 examples [00:38, 1002.82 examples/s]39560 examples [00:38, 1017.19 examples/s]39662 examples [00:38, 983.14 examples/s] 39761 examples [00:39, 968.11 examples/s]39864 examples [00:39, 983.02 examples/s]39963 examples [00:39, 980.48 examples/s]40062 examples [00:39, 920.13 examples/s]40161 examples [00:39, 938.85 examples/s]40267 examples [00:39, 971.89 examples/s]40374 examples [00:39, 997.96 examples/s]40476 examples [00:39, 1002.14 examples/s]40582 examples [00:39, 1018.38 examples/s]40685 examples [00:39, 1009.72 examples/s]40793 examples [00:40, 1028.71 examples/s]40901 examples [00:40, 1042.92 examples/s]41006 examples [00:40, 1034.51 examples/s]41110 examples [00:40, 1030.97 examples/s]41214 examples [00:40, 1030.57 examples/s]41323 examples [00:40, 1047.21 examples/s]41431 examples [00:40, 1053.37 examples/s]41537 examples [00:40, 1022.19 examples/s]41641 examples [00:40, 1026.94 examples/s]41744 examples [00:41, 1022.86 examples/s]41855 examples [00:41, 1046.49 examples/s]41961 examples [00:41, 1050.13 examples/s]42067 examples [00:41, 1040.55 examples/s]42172 examples [00:41, 1026.60 examples/s]42277 examples [00:41, 1031.97 examples/s]42387 examples [00:41, 1049.10 examples/s]42495 examples [00:41, 1056.18 examples/s]42601 examples [00:41, 1040.87 examples/s]42710 examples [00:41, 1050.18 examples/s]42816 examples [00:42, 1041.71 examples/s]42921 examples [00:42, 1025.59 examples/s]43024 examples [00:42, 1015.08 examples/s]43129 examples [00:42, 1023.52 examples/s]43232 examples [00:42, 1025.16 examples/s]43343 examples [00:42, 1047.40 examples/s]43453 examples [00:42, 1061.70 examples/s]43560 examples [00:42, 1061.69 examples/s]43667 examples [00:42, 978.18 examples/s] 43767 examples [00:42, 976.13 examples/s]43876 examples [00:43, 1006.14 examples/s]43981 examples [00:43, 1017.79 examples/s]44087 examples [00:43, 1029.80 examples/s]44191 examples [00:43, 993.15 examples/s] 44291 examples [00:43, 989.35 examples/s]44401 examples [00:43, 1018.68 examples/s]44509 examples [00:43, 1034.94 examples/s]44615 examples [00:43, 1042.17 examples/s]44722 examples [00:43, 1048.89 examples/s]44828 examples [00:44, 1010.96 examples/s]44938 examples [00:44, 1035.95 examples/s]45049 examples [00:44, 1056.76 examples/s]45156 examples [00:44, 1051.75 examples/s]45264 examples [00:44, 1059.17 examples/s]45371 examples [00:44, 1039.93 examples/s]45486 examples [00:44, 1068.54 examples/s]45602 examples [00:44, 1092.19 examples/s]45713 examples [00:44, 1096.05 examples/s]45823 examples [00:44, 1092.34 examples/s]45933 examples [00:45, 1076.22 examples/s]46048 examples [00:45, 1096.32 examples/s]46158 examples [00:45, 1069.24 examples/s]46266 examples [00:45, 1064.20 examples/s]46373 examples [00:45, 1057.53 examples/s]46479 examples [00:45, 1048.78 examples/s]46588 examples [00:45, 1059.13 examples/s]46695 examples [00:45, 1055.54 examples/s]46801 examples [00:45, 1035.56 examples/s]46911 examples [00:45, 1052.21 examples/s]47017 examples [00:46, 1034.74 examples/s]47131 examples [00:46, 1063.42 examples/s]47244 examples [00:46, 1081.51 examples/s]47353 examples [00:46, 1077.91 examples/s]47466 examples [00:46, 1091.58 examples/s]47576 examples [00:46, 1072.18 examples/s]47689 examples [00:46, 1087.42 examples/s]47798 examples [00:46, 1070.13 examples/s]47906 examples [00:46, 1036.85 examples/s]48021 examples [00:46, 1066.12 examples/s]48129 examples [00:47, 1030.47 examples/s]48233 examples [00:47, 1030.70 examples/s]48337 examples [00:47, 1021.91 examples/s]48440 examples [00:47, 1018.38 examples/s]48543 examples [00:47, 1021.62 examples/s]48646 examples [00:47, 1011.05 examples/s]48756 examples [00:47, 1033.70 examples/s]48863 examples [00:47, 1041.95 examples/s]48968 examples [00:47, 1038.14 examples/s]49077 examples [00:48, 1051.36 examples/s]49183 examples [00:48, 1029.26 examples/s]49293 examples [00:48, 1046.07 examples/s]49399 examples [00:48, 1048.32 examples/s]49512 examples [00:48, 1070.61 examples/s]49627 examples [00:48, 1093.01 examples/s]49737 examples [00:48, 1078.19 examples/s]49848 examples [00:48, 1084.72 examples/s]49957 examples [00:48, 1075.19 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8785/50000 [00:00<00:00, 87843.80 examples/s] 45%|████▍     | 22262/50000 [00:00<00:00, 98086.74 examples/s] 70%|██████▉   | 34761/50000 [00:00<00:00, 104856.67 examples/s] 97%|█████████▋| 48593/50000 [00:00<00:00, 113061.83 examples/s]                                                                0 examples [00:00, ? examples/s]87 examples [00:00, 867.23 examples/s]200 examples [00:00, 930.87 examples/s]303 examples [00:00, 957.71 examples/s]406 examples [00:00, 976.46 examples/s]519 examples [00:00, 1015.60 examples/s]627 examples [00:00, 1033.43 examples/s]739 examples [00:00, 1055.36 examples/s]846 examples [00:00, 1058.62 examples/s]955 examples [00:00, 1067.39 examples/s]1066 examples [00:01, 1078.06 examples/s]1172 examples [00:01, 1066.38 examples/s]1278 examples [00:01, 1000.47 examples/s]1378 examples [00:01, 996.14 examples/s] 1482 examples [00:01, 1007.06 examples/s]1583 examples [00:01, 1001.17 examples/s]1687 examples [00:01, 1010.64 examples/s]1797 examples [00:01, 1035.70 examples/s]1901 examples [00:01, 1036.76 examples/s]2010 examples [00:01, 1051.09 examples/s]2116 examples [00:02, 1016.65 examples/s]2228 examples [00:02, 1045.53 examples/s]2344 examples [00:02, 1075.94 examples/s]2453 examples [00:02, 1044.93 examples/s]2559 examples [00:02, 1018.56 examples/s]2662 examples [00:02, 1019.76 examples/s]2767 examples [00:02, 1028.50 examples/s]2879 examples [00:02, 1051.97 examples/s]2985 examples [00:02, 1034.20 examples/s]3094 examples [00:02, 1049.33 examples/s]3203 examples [00:03, 1060.54 examples/s]3310 examples [00:03, 1059.17 examples/s]3418 examples [00:03, 1062.92 examples/s]3525 examples [00:03, 1024.78 examples/s]3629 examples [00:03, 1028.77 examples/s]3735 examples [00:03, 1036.47 examples/s]3839 examples [00:03, 1030.94 examples/s]3946 examples [00:03, 1041.39 examples/s]4051 examples [00:03, 1024.93 examples/s]4163 examples [00:04, 1051.47 examples/s]4269 examples [00:04, 1053.83 examples/s]4375 examples [00:04, 1004.32 examples/s]4491 examples [00:04, 1045.36 examples/s]4597 examples [00:04, 1036.96 examples/s]4710 examples [00:04, 1063.01 examples/s]4820 examples [00:04, 1072.92 examples/s]4933 examples [00:04, 1088.79 examples/s]5043 examples [00:04, 1070.69 examples/s]5151 examples [00:04, 1060.26 examples/s]5258 examples [00:05, 1051.03 examples/s]5371 examples [00:05, 1073.43 examples/s]5485 examples [00:05, 1090.54 examples/s]5600 examples [00:05, 1107.71 examples/s]5712 examples [00:05, 1075.76 examples/s]5820 examples [00:05, 1054.95 examples/s]5926 examples [00:05, 1043.88 examples/s]6031 examples [00:05, 1045.45 examples/s]6142 examples [00:05, 1063.96 examples/s]6249 examples [00:05, 1047.07 examples/s]6358 examples [00:06, 1058.38 examples/s]6466 examples [00:06, 1063.73 examples/s]6573 examples [00:06, 1062.31 examples/s]6688 examples [00:06, 1085.77 examples/s]6797 examples [00:06, 1077.73 examples/s]6909 examples [00:06, 1087.80 examples/s]7024 examples [00:06, 1102.95 examples/s]7135 examples [00:06, 1065.62 examples/s]7242 examples [00:06, 981.62 examples/s] 7345 examples [00:07, 994.79 examples/s]7455 examples [00:07, 1022.49 examples/s]7559 examples [00:07, 1022.87 examples/s]7672 examples [00:07, 1052.75 examples/s]7778 examples [00:07, 1025.63 examples/s]7882 examples [00:07, 1019.20 examples/s]7992 examples [00:07, 1040.97 examples/s]8103 examples [00:07, 1060.28 examples/s]8210 examples [00:07, 1039.50 examples/s]8315 examples [00:07, 1024.74 examples/s]8418 examples [00:08, 1023.32 examples/s]8521 examples [00:08, 1012.93 examples/s]8632 examples [00:08, 1039.28 examples/s]8743 examples [00:08, 1058.21 examples/s]8850 examples [00:08, 1035.07 examples/s]8961 examples [00:08, 1054.10 examples/s]9067 examples [00:08, 1042.70 examples/s]9176 examples [00:08, 1056.06 examples/s]9282 examples [00:08, 1048.35 examples/s]9387 examples [00:08, 1024.97 examples/s]9502 examples [00:09, 1058.39 examples/s]9614 examples [00:09, 1075.88 examples/s]9725 examples [00:09, 1084.48 examples/s]9834 examples [00:09, 1084.76 examples/s]9943 examples [00:09, 1055.06 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete22ZRLK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete22ZRLK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1a60c23620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1a60c23620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1a60c23620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f19ea0a34e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f19ea0e0c50>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1a60c23620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1a60c23620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1a60c23620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1a47c56588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1a47c562e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f19d86e9378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f19d86e9378> 

  function with postional parmater data_info <function split_train_valid at 0x7f19d86e9378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f19d86e9488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f19d86e9488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f19d86e9488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=686a65ed5c346669825b0710091eccc66e4f2483c4dceeb85ccd519cc4dc8c05
  Stored in directory: /tmp/pip-ephem-wheel-cache-ianpempu/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:55:50, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:10:56, 16.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:58:52, 24.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:59:43, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<4:53:03, 48.8kB/s].vector_cache/glove.6B.zip:   1%|          | 7.97M/862M [00:01<3:24:12, 69.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:22:19, 99.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:39:03, 142kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:09:14, 203kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<48:13, 289kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:01<33:45, 411kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:01<23:35, 585kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<16:32, 830kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.2M/862M [00:02<11:35, 1.18MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<08:11, 1.66MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<05:46, 2.34MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<07:50, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<07:23, 1.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<06:59, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:21, 2.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:17, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<07:06, 1.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<05:39, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<06:06, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<05:36, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<04:12, 3.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<06:04, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<05:35, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:10<04:11, 3.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<06:04, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<06:55, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:12<05:25, 2.42MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<03:57, 3.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<10:40, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<08:47, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:14<06:28, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<07:34, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<06:37, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<04:57, 2.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:32, 1.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<07:14, 1.79MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:18<05:38, 2.29MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:18<04:03, 3.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<50:18, 256kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<36:30, 353kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<25:47, 498kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<21:00, 610kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<17:18, 740kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<12:45, 1.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<10:58, 1.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<08:57, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<06:33, 1.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:33, 1.68MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:52, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:02, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:22, 2.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:44, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:49, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<06:25, 1.96MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:26, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:45, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:03, 2.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:14, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:38, 2.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:15, 2.92MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:53, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:22, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:01, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:44, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:15, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:56, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:27, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:02, 2.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:40, 3.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:43, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:03, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:52, 2.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:57, 1.74MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:05, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:33, 2.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:02, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:40, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:16, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<04:58, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:44, 3.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<02:47, 4.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<56:42, 210kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<40:52, 291kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<28:51, 412kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<22:56, 516kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<17:15, 687kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<12:20, 957kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<11:24, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:10, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:42, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<07:28, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:30, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:44, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:09, 2.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<10:14, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:22, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:06, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:59, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:03, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:31, 2.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:53, 1.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:26, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:01, 2.29MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:37, 3.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<20:46, 551kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<15:41, 729kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<11:15, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:30, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:30, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<06:13, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:01, 1.61MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:12, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:36, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:43, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:08, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:52, 2.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:20, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:58, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:40, 2.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:25, 3.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:20, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:16, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:40, 2.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:49, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:11, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:54, 2.82MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:19, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:56, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:42, 2.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:02, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:38, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:28, 3.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:58, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:34, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:26, 3.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:57, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:38, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:25, 2.43MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<03:11, 3.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<20:47, 514kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<15:37, 683kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<11:11, 952kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:18, 1.03MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:21, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:38, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:44, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:14, 2.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:25, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:54, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:39, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<03:21, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:04:43, 17.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<7:04:06, 24.5kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<4:56:24, 35.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<3:29:11, 49.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<2:27:23, 70.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<1:43:10, 99.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:14:24, 138kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<53:05, 193kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<37:19, 274kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<28:27, 359kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<21:58, 464kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<15:52, 642kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<11:10, 907kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<9:53:23, 17.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<7:04:16, 23.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<4:56:15, 34.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<3:28:20, 48.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<2:25:45, 68.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<1:43:51, 96.3kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<1:13:39, 136kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<51:40, 193kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<38:24, 259kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<28:53, 344kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<20:36, 481kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<14:30, 681kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<15:42, 628kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:00, 821kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:35, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<08:18, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<07:47, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:53, 1.66MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<04:12, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<33:11, 293kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<24:12, 401kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<17:08, 565kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<14:13, 679kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<11:54, 810kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:48, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:41, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:21, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<04:38, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:28, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:45, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:29, 2.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:40, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:04, 3.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:18, 4.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<24:55, 376kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<19:19, 485kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<13:54, 673kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<09:49, 949kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<11:48, 789kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<09:11, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<06:37, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:47, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:37, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:05, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:18, 2.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:27, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:58, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:56, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:12, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:52, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:54, 3.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:07, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:42, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:40, 2.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:40, 3.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<07:08, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:53, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:20, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:06, 1.73MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:28, 1.98MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<03:20, 2.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:23, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:46, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:42, 2.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<02:44, 3.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:56, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:21, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:15, 2.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:18, 2.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:45, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:45, 2.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:00, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:41, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<02:45, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:27, 1.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:32, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:50, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:31, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:39, 3.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:48, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:21, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:24, 2.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:29, 3.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:27, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:40, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:28, 2.37MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:19, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:41, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:41, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:40, 3.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<7:53:19, 17.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<5:31:54, 24.5kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<3:51:48, 35.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<2:43:25, 49.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:55:58, 69.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:21:27, 99.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<57:57, 138kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<41:20, 194kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<29:02, 275kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<22:06, 359kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<16:14, 488kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<11:29, 688kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<09:54, 795kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<08:32, 922kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:21, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:41, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:45, 1.64MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:31, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:16, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:32, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:31, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:32, 3.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:07, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:45, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:12, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:43, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:50, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:42, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:40, 2.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:25, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<05:16, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:50, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:26, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:36, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:35, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:42, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:20, 2.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<02:31, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:28, 2.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:20, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:04, 2.36MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:19, 3.10MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:17, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:45, 1.91MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:56, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:09, 3.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:31, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<03:54, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:54, 2.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:39, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:03, 1.74MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:08, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<02:17, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:27, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:50, 1.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:51, 2.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:36, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:06, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:20, 2.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:16, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:40, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:54, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:07, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:52, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:10, 3.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:05, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:31, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:48, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:01, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:31, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:44, 2.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<01:59, 3.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:00, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:05, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:01, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:36, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:50, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:00, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:08, 2.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:29, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<01:58, 3.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<17:01, 374kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<12:32, 507kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<08:54, 711kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<07:40, 820kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<06:00, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:20, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:29, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:23, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:21, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:24, 2.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<08:10, 753kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<06:21, 968kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:35, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:37, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:27, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:23, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:20, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:56, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:12, 2.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:56, 2.03MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:15, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:32, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<01:50, 3.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<43:53, 134kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<31:18, 188kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<21:56, 267kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<16:38, 350kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<12:13, 475kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<08:39, 668kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<07:23, 778kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<06:48, 843kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:10, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:42, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:41, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:08, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:06, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:10, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:04, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:19, 2.40MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<01:41, 3.28MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:17, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:29, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:18, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<02:23, 2.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:31, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:56, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<02:07, 2.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:43, 944kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:46, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:31, 1.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:26, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:11, 1.67MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:38, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:20, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:40, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:56, 2.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:51, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:09, 2.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:26, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:28, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:53, 2.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<01:22, 3.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<13:02, 388kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<10:40, 474kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<07:49, 644kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<05:31, 907kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<05:33, 899kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:36, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:22, 1.47MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:14, 1.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:59, 1.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:14, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:36, 3.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<2:25:45, 33.3kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<1:42:39, 47.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<1:11:45, 67.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<49:49, 96.0kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<45:49, 104kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<32:44, 146kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<23:00, 207kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<16:48, 280kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<12:26, 378kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<08:51, 530kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:59, 665kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<05:33, 833kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:02, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:38, 1.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:55, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:02, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:12, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:03, 1.47MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:48, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:07, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:17, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:16, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:43, 2.56MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:59, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:43, 1.60MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:10, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:35, 2.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:33, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:25, 1.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:49, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<01:18, 3.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<08:47, 481kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<06:46, 624kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:51, 865kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<03:24, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<22:58, 181kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<16:39, 250kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<11:45, 352kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<08:53, 460kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<07:27, 548kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<05:28, 745kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:51, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<04:04, 986kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:26, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:32, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:29, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:19, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:45, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:55, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:55, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:28, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:43, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:40<01:45, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:20, 2.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<00:58, 3.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<06:00, 624kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:45, 788kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:25, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:24, 1.52MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<08:20, 441kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<06:22, 577kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:34, 800kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:49, 944kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:13, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:21, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:40, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<05:45, 615kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<05:05, 694kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<03:49, 923kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<02:42, 1.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:05, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:40, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:58, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<01:24, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<08:47, 387kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<06:38, 511kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:45, 711kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<03:53, 857kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<03:12, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:21, 1.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:13, 1.47MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:29, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:56, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:23, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:02, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:54, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:26, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:33, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:12, 2.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:23, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:53, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:30, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:07, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:25, 2.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:21, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:01, 2.91MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:18, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:47, 1.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:28, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:04, 2.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:44, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:37, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:14, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:21, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:21, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:03, 2.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:12, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:37, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:20, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:57, 2.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:35, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:30, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<00:48, 3.18MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:12, 800kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:38, 974kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:55, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:46, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:37, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<00:51, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<1:13:07, 33.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<51:52, 46.9kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<36:21, 66.6kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<25:07, 95.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<18:14, 130kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<13:02, 181kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<09:09, 256kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<06:19, 364kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<06:46, 339kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<05:00, 458kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<03:32, 641kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:53, 771kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:33, 869kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:53, 1.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:20, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:43, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:31, 1.41MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:08, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:09, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:07, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:51, 2.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:29<00:36, 3.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<05:53, 342kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<04:44, 426kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<03:25, 585kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<02:23, 823kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:19, 840kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:54, 1.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:23, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:17, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:11, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:56, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:55, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:41, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:47, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:05, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:53, 1.96MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:38, 2.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:57, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:43, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:35, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:41, 2.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:55, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:44, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:31, 2.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:52, 1.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:49, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:40, 2.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:41, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:31, 2.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:21, 3.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<06:51, 194kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<04:58, 266kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<03:29, 375kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<02:35, 487kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:59, 629kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:24, 875kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:58, 1.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:31, 472kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:05, 566kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:31, 769kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:03, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:02, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:53, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:39, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:37, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:35, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:26, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:37, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:30, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:21, 2.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:33, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:28, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:21, 2.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:24, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:32, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.91MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:18, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:28, 1.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:27, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:20, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:27, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:15, 2.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:23, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:22, 1.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:16, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:12, 2.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:18, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 1.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:10, 2.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:14, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:06, 3.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:30, 719kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:24, 888kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:16, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:13, 1.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:08, 1.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:07, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.16MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<124:50:54,  1.12s/it]  0%|          | 846/400000 [00:01<87:12:47,  1.27it/s]  0%|          | 1749/400000 [00:01<60:54:53,  1.82it/s]  1%|          | 2663/400000 [00:01<42:32:46,  2.59it/s]  1%|          | 3546/400000 [00:01<29:43:11,  3.71it/s]  1%|          | 4374/400000 [00:01<20:45:51,  5.29it/s]  1%|▏         | 5248/400000 [00:01<14:30:24,  7.56it/s]  2%|▏         | 6144/400000 [00:01<10:08:07, 10.79it/s]  2%|▏         | 6950/400000 [00:01<7:05:03, 15.41it/s]   2%|▏         | 7795/400000 [00:02<4:57:07, 22.00it/s]  2%|▏         | 8655/400000 [00:02<3:27:45, 31.39it/s]  2%|▏         | 9485/400000 [00:02<2:25:22, 44.77it/s]  3%|▎         | 10308/400000 [00:02<1:41:47, 63.81it/s]  3%|▎         | 11203/400000 [00:02<1:11:18, 90.87it/s]  3%|▎         | 12136/400000 [00:02<50:00, 129.28it/s]   3%|▎         | 13026/400000 [00:02<35:08, 183.54it/s]  3%|▎         | 13899/400000 [00:02<24:45, 259.85it/s]  4%|▎         | 14815/400000 [00:02<17:30, 366.76it/s]  4%|▍         | 15768/400000 [00:02<12:25, 515.43it/s]  4%|▍         | 16726/400000 [00:03<08:52, 719.73it/s]  4%|▍         | 17648/400000 [00:03<06:24, 993.74it/s]  5%|▍         | 18560/400000 [00:03<04:42, 1350.37it/s]  5%|▍         | 19467/400000 [00:03<03:29, 1813.30it/s]  5%|▌         | 20360/400000 [00:03<02:40, 2369.47it/s]  5%|▌         | 21236/400000 [00:03<02:06, 2999.74it/s]  6%|▌         | 22086/400000 [00:03<01:42, 3694.34it/s]  6%|▌         | 22959/400000 [00:03<01:24, 4466.80it/s]  6%|▌         | 23866/400000 [00:03<01:11, 5267.91it/s]  6%|▌         | 24753/400000 [00:03<01:02, 5998.29it/s]  6%|▋         | 25693/400000 [00:04<00:55, 6727.82it/s]  7%|▋         | 26585/400000 [00:04<00:52, 7141.50it/s]  7%|▋         | 27460/400000 [00:04<00:50, 7414.58it/s]  7%|▋         | 28404/400000 [00:04<00:46, 7924.34it/s]  7%|▋         | 29336/400000 [00:04<00:44, 8295.97it/s]  8%|▊         | 30234/400000 [00:04<00:43, 8472.70it/s]  8%|▊         | 31130/400000 [00:04<00:42, 8579.10it/s]  8%|▊         | 32023/400000 [00:04<00:43, 8533.62it/s]  8%|▊         | 32960/400000 [00:04<00:41, 8767.06it/s]  8%|▊         | 33908/400000 [00:04<00:40, 8968.96it/s]  9%|▊         | 34819/400000 [00:05<00:41, 8857.32it/s]  9%|▉         | 35715/400000 [00:05<00:41, 8749.71it/s]  9%|▉         | 36598/400000 [00:05<00:41, 8670.24it/s]  9%|▉         | 37573/400000 [00:05<00:40, 8967.41it/s] 10%|▉         | 38530/400000 [00:05<00:39, 9139.74it/s] 10%|▉         | 39450/400000 [00:05<00:39, 9089.87it/s] 10%|█         | 40363/400000 [00:05<00:40, 8972.49it/s] 10%|█         | 41264/400000 [00:05<00:39, 8971.35it/s] 11%|█         | 42205/400000 [00:05<00:39, 9097.74it/s] 11%|█         | 43178/400000 [00:06<00:38, 9277.05it/s] 11%|█         | 44120/400000 [00:06<00:38, 9317.64it/s] 11%|█▏        | 45054/400000 [00:06<00:39, 9066.32it/s] 11%|█▏        | 45964/400000 [00:06<00:39, 8941.21it/s] 12%|█▏        | 46873/400000 [00:06<00:39, 8983.68it/s] 12%|█▏        | 47851/400000 [00:06<00:38, 9206.07it/s] 12%|█▏        | 48844/400000 [00:06<00:37, 9410.16it/s] 12%|█▏        | 49788/400000 [00:06<00:38, 8991.50it/s] 13%|█▎        | 50748/400000 [00:06<00:38, 9164.23it/s] 13%|█▎        | 51673/400000 [00:06<00:37, 9188.09it/s] 13%|█▎        | 52633/400000 [00:07<00:37, 9307.68it/s] 13%|█▎        | 53567/400000 [00:07<00:37, 9261.15it/s] 14%|█▎        | 54496/400000 [00:07<00:39, 8720.94it/s] 14%|█▍        | 55376/400000 [00:07<00:39, 8705.92it/s] 14%|█▍        | 56253/400000 [00:07<00:40, 8406.35it/s] 14%|█▍        | 57100/400000 [00:07<00:41, 8359.00it/s] 14%|█▍        | 57952/400000 [00:07<00:40, 8403.70it/s] 15%|█▍        | 58802/400000 [00:07<00:40, 8429.90it/s] 15%|█▍        | 59722/400000 [00:07<00:39, 8645.36it/s] 15%|█▌        | 60656/400000 [00:07<00:38, 8842.21it/s] 15%|█▌        | 61568/400000 [00:08<00:37, 8921.85it/s] 16%|█▌        | 62463/400000 [00:08<00:38, 8826.73it/s] 16%|█▌        | 63367/400000 [00:08<00:37, 8882.55it/s] 16%|█▌        | 64285/400000 [00:08<00:37, 8966.42it/s] 16%|█▋        | 65183/400000 [00:08<00:37, 8913.64it/s] 17%|█▋        | 66076/400000 [00:08<00:37, 8906.72it/s] 17%|█▋        | 66968/400000 [00:08<00:38, 8760.41it/s] 17%|█▋        | 67885/400000 [00:08<00:37, 8878.11it/s] 17%|█▋        | 68860/400000 [00:08<00:36, 9122.52it/s] 17%|█▋        | 69798/400000 [00:08<00:35, 9197.09it/s] 18%|█▊        | 70720/400000 [00:09<00:37, 8834.99it/s] 18%|█▊        | 71608/400000 [00:09<00:38, 8608.98it/s] 18%|█▊        | 72474/400000 [00:09<00:38, 8526.85it/s] 18%|█▊        | 73330/400000 [00:09<00:38, 8394.11it/s] 19%|█▊        | 74173/400000 [00:09<00:40, 8096.96it/s] 19%|█▉        | 75057/400000 [00:09<00:39, 8304.25it/s] 19%|█▉        | 75892/400000 [00:09<00:39, 8195.16it/s] 19%|█▉        | 76790/400000 [00:09<00:38, 8415.80it/s] 19%|█▉        | 77664/400000 [00:09<00:37, 8509.19it/s] 20%|█▉        | 78562/400000 [00:10<00:37, 8644.53it/s] 20%|█▉        | 79443/400000 [00:10<00:36, 8690.69it/s] 20%|██        | 80314/400000 [00:10<00:38, 8320.76it/s] 20%|██        | 81151/400000 [00:10<00:38, 8187.70it/s] 20%|██        | 81994/400000 [00:10<00:38, 8256.90it/s] 21%|██        | 82823/400000 [00:10<00:38, 8266.16it/s] 21%|██        | 83652/400000 [00:10<00:38, 8164.44it/s] 21%|██        | 84471/400000 [00:10<00:39, 7932.61it/s] 21%|██▏       | 85330/400000 [00:10<00:38, 8117.09it/s] 22%|██▏       | 86205/400000 [00:10<00:37, 8295.75it/s] 22%|██▏       | 87038/400000 [00:11<00:37, 8252.46it/s] 22%|██▏       | 87879/400000 [00:11<00:37, 8297.98it/s] 22%|██▏       | 88711/400000 [00:11<00:38, 8094.85it/s] 22%|██▏       | 89567/400000 [00:11<00:37, 8228.89it/s] 23%|██▎       | 90445/400000 [00:11<00:36, 8386.60it/s] 23%|██▎       | 91286/400000 [00:11<00:36, 8383.05it/s] 23%|██▎       | 92126/400000 [00:11<00:37, 8265.40it/s] 23%|██▎       | 92954/400000 [00:11<00:37, 8160.70it/s] 23%|██▎       | 93778/400000 [00:11<00:37, 8183.24it/s] 24%|██▎       | 94637/400000 [00:12<00:36, 8298.62it/s] 24%|██▍       | 95501/400000 [00:12<00:36, 8396.79it/s] 24%|██▍       | 96342/400000 [00:12<00:36, 8243.76it/s] 24%|██▍       | 97202/400000 [00:12<00:36, 8345.71it/s] 25%|██▍       | 98038/400000 [00:12<00:36, 8296.09it/s] 25%|██▍       | 98869/400000 [00:12<00:37, 8095.29it/s] 25%|██▍       | 99681/400000 [00:12<00:37, 7978.24it/s] 25%|██▌       | 100503/400000 [00:12<00:37, 8047.66it/s] 25%|██▌       | 101310/400000 [00:12<00:37, 8018.52it/s] 26%|██▌       | 102173/400000 [00:12<00:36, 8190.52it/s] 26%|██▌       | 103050/400000 [00:13<00:35, 8355.04it/s] 26%|██▌       | 103888/400000 [00:13<00:35, 8330.06it/s] 26%|██▌       | 104723/400000 [00:13<00:35, 8255.04it/s] 26%|██▋       | 105550/400000 [00:13<00:35, 8257.21it/s] 27%|██▋       | 106402/400000 [00:13<00:35, 8332.78it/s] 27%|██▋       | 107292/400000 [00:13<00:34, 8495.00it/s] 27%|██▋       | 108170/400000 [00:13<00:34, 8575.78it/s] 27%|██▋       | 109029/400000 [00:13<00:34, 8521.83it/s] 27%|██▋       | 109883/400000 [00:13<00:34, 8347.55it/s] 28%|██▊       | 110736/400000 [00:13<00:34, 8401.23it/s] 28%|██▊       | 111587/400000 [00:14<00:34, 8432.22it/s] 28%|██▊       | 112431/400000 [00:14<00:34, 8404.62it/s] 28%|██▊       | 113273/400000 [00:14<00:35, 8111.13it/s] 29%|██▊       | 114087/400000 [00:14<00:35, 8111.50it/s] 29%|██▊       | 114958/400000 [00:14<00:34, 8280.13it/s] 29%|██▉       | 115803/400000 [00:14<00:34, 8330.13it/s] 29%|██▉       | 116638/400000 [00:14<00:34, 8323.87it/s] 29%|██▉       | 117472/400000 [00:14<00:34, 8245.07it/s] 30%|██▉       | 118304/400000 [00:14<00:34, 8264.80it/s] 30%|██▉       | 119189/400000 [00:14<00:33, 8425.93it/s] 30%|███       | 120048/400000 [00:15<00:33, 8472.21it/s] 30%|███       | 120902/400000 [00:15<00:32, 8490.12it/s] 30%|███       | 121752/400000 [00:15<00:33, 8327.08it/s] 31%|███       | 122586/400000 [00:15<00:33, 8166.17it/s] 31%|███       | 123506/400000 [00:15<00:32, 8450.61it/s] 31%|███       | 124424/400000 [00:15<00:31, 8654.55it/s] 31%|███▏      | 125299/400000 [00:15<00:31, 8682.00it/s] 32%|███▏      | 126170/400000 [00:15<00:32, 8540.26it/s] 32%|███▏      | 127027/400000 [00:15<00:33, 8117.93it/s] 32%|███▏      | 127958/400000 [00:16<00:32, 8441.65it/s] 32%|███▏      | 128892/400000 [00:16<00:31, 8691.88it/s] 32%|███▏      | 129814/400000 [00:16<00:30, 8843.36it/s] 33%|███▎      | 130704/400000 [00:16<00:30, 8852.32it/s] 33%|███▎      | 131611/400000 [00:16<00:30, 8913.03it/s] 33%|███▎      | 132538/400000 [00:16<00:29, 9015.84it/s] 33%|███▎      | 133488/400000 [00:16<00:29, 9153.99it/s] 34%|███▎      | 134417/400000 [00:16<00:28, 9192.36it/s] 34%|███▍      | 135338/400000 [00:16<00:30, 8804.54it/s] 34%|███▍      | 136223/400000 [00:16<00:30, 8724.39it/s] 34%|███▍      | 137105/400000 [00:17<00:30, 8751.09it/s] 34%|███▍      | 137983/400000 [00:17<00:30, 8703.35it/s] 35%|███▍      | 138856/400000 [00:17<00:30, 8484.35it/s] 35%|███▍      | 139707/400000 [00:17<00:30, 8478.17it/s] 35%|███▌      | 140601/400000 [00:17<00:30, 8609.58it/s] 35%|███▌      | 141507/400000 [00:17<00:29, 8739.38it/s] 36%|███▌      | 142415/400000 [00:17<00:29, 8838.22it/s] 36%|███▌      | 143301/400000 [00:17<00:29, 8685.52it/s] 36%|███▌      | 144172/400000 [00:17<00:29, 8630.39it/s] 36%|███▋      | 145061/400000 [00:17<00:29, 8705.80it/s] 37%|███▋      | 146008/400000 [00:18<00:28, 8920.89it/s] 37%|███▋      | 146903/400000 [00:18<00:28, 8826.91it/s] 37%|███▋      | 147788/400000 [00:18<00:28, 8710.23it/s] 37%|███▋      | 148661/400000 [00:18<00:28, 8684.66it/s] 37%|███▋      | 149531/400000 [00:18<00:28, 8643.22it/s] 38%|███▊      | 150458/400000 [00:18<00:28, 8820.79it/s] 38%|███▊      | 151366/400000 [00:18<00:27, 8895.88it/s] 38%|███▊      | 152283/400000 [00:18<00:27, 8976.18it/s] 38%|███▊      | 153182/400000 [00:18<00:28, 8732.86it/s] 39%|███▊      | 154091/400000 [00:18<00:27, 8836.28it/s] 39%|███▊      | 154997/400000 [00:19<00:27, 8900.44it/s] 39%|███▉      | 155936/400000 [00:19<00:26, 9041.79it/s] 39%|███▉      | 156876/400000 [00:19<00:26, 9146.06it/s] 39%|███▉      | 157792/400000 [00:19<00:27, 8904.80it/s] 40%|███▉      | 158705/400000 [00:19<00:26, 8969.29it/s] 40%|███▉      | 159613/400000 [00:19<00:26, 8999.63it/s] 40%|████      | 160515/400000 [00:19<00:27, 8699.26it/s] 40%|████      | 161391/400000 [00:19<00:27, 8717.06it/s] 41%|████      | 162265/400000 [00:19<00:27, 8511.57it/s] 41%|████      | 163119/400000 [00:19<00:28, 8385.69it/s] 41%|████      | 163964/400000 [00:20<00:28, 8401.28it/s] 41%|████      | 164808/400000 [00:20<00:27, 8411.27it/s] 41%|████▏     | 165678/400000 [00:20<00:27, 8492.78it/s] 42%|████▏     | 166529/400000 [00:20<00:27, 8357.28it/s] 42%|████▏     | 167366/400000 [00:20<00:28, 8191.45it/s] 42%|████▏     | 168198/400000 [00:20<00:28, 8228.88it/s] 42%|████▏     | 169023/400000 [00:20<00:28, 8035.73it/s] 42%|████▏     | 169829/400000 [00:20<00:28, 7938.93it/s] 43%|████▎     | 170625/400000 [00:20<00:29, 7792.88it/s] 43%|████▎     | 171406/400000 [00:21<00:29, 7783.99it/s] 43%|████▎     | 172217/400000 [00:21<00:28, 7878.50it/s] 43%|████▎     | 173017/400000 [00:21<00:28, 7914.43it/s] 43%|████▎     | 173872/400000 [00:21<00:27, 8092.49it/s] 44%|████▎     | 174683/400000 [00:21<00:28, 7952.03it/s] 44%|████▍     | 175495/400000 [00:21<00:28, 8000.75it/s] 44%|████▍     | 176311/400000 [00:21<00:27, 8045.65it/s] 44%|████▍     | 177230/400000 [00:21<00:26, 8357.53it/s] 45%|████▍     | 178129/400000 [00:21<00:25, 8537.26it/s] 45%|████▍     | 178987/400000 [00:21<00:26, 8472.17it/s] 45%|████▍     | 179867/400000 [00:22<00:25, 8567.20it/s] 45%|████▌     | 180726/400000 [00:22<00:25, 8473.19it/s] 45%|████▌     | 181576/400000 [00:22<00:26, 8228.08it/s] 46%|████▌     | 182404/400000 [00:22<00:26, 8243.18it/s] 46%|████▌     | 183231/400000 [00:22<00:26, 8149.00it/s] 46%|████▌     | 184048/400000 [00:22<00:26, 8072.65it/s] 46%|████▌     | 184919/400000 [00:22<00:26, 8252.73it/s] 46%|████▋     | 185756/400000 [00:22<00:25, 8286.66it/s] 47%|████▋     | 186647/400000 [00:22<00:25, 8462.00it/s] 47%|████▋     | 187496/400000 [00:22<00:25, 8370.22it/s] 47%|████▋     | 188355/400000 [00:23<00:25, 8433.74it/s] 47%|████▋     | 189200/400000 [00:23<00:25, 8280.56it/s] 48%|████▊     | 190033/400000 [00:23<00:25, 8292.48it/s] 48%|████▊     | 190888/400000 [00:23<00:24, 8368.04it/s] 48%|████▊     | 191726/400000 [00:23<00:25, 8311.43it/s] 48%|████▊     | 192608/400000 [00:23<00:24, 8455.51it/s] 48%|████▊     | 193581/400000 [00:23<00:23, 8801.01it/s] 49%|████▊     | 194466/400000 [00:23<00:23, 8738.61it/s] 49%|████▉     | 195344/400000 [00:23<00:23, 8671.36it/s] 49%|████▉     | 196214/400000 [00:23<00:23, 8522.45it/s] 49%|████▉     | 197069/400000 [00:24<00:24, 8280.15it/s] 49%|████▉     | 197901/400000 [00:24<00:24, 8142.73it/s] 50%|████▉     | 198718/400000 [00:24<00:24, 8140.43it/s] 50%|████▉     | 199534/400000 [00:24<00:25, 7974.83it/s] 50%|█████     | 200397/400000 [00:24<00:24, 8158.05it/s] 50%|█████     | 201216/400000 [00:24<00:24, 8134.52it/s] 51%|█████     | 202097/400000 [00:24<00:23, 8323.26it/s] 51%|█████     | 203042/400000 [00:24<00:22, 8631.51it/s] 51%|█████     | 203919/400000 [00:24<00:22, 8672.04it/s] 51%|█████     | 204790/400000 [00:25<00:22, 8509.70it/s] 51%|█████▏    | 205673/400000 [00:25<00:22, 8602.02it/s] 52%|█████▏    | 206585/400000 [00:25<00:22, 8750.14it/s] 52%|█████▏    | 207496/400000 [00:25<00:21, 8852.62it/s] 52%|█████▏    | 208384/400000 [00:25<00:22, 8653.86it/s] 52%|█████▏    | 209259/400000 [00:25<00:21, 8678.19it/s] 53%|█████▎    | 210129/400000 [00:25<00:22, 8618.32it/s] 53%|█████▎    | 211009/400000 [00:25<00:21, 8671.03it/s] 53%|█████▎    | 211955/400000 [00:25<00:21, 8892.16it/s] 53%|█████▎    | 212847/400000 [00:25<00:21, 8686.48it/s] 53%|█████▎    | 213719/400000 [00:26<00:21, 8486.54it/s] 54%|█████▎    | 214571/400000 [00:26<00:22, 8417.52it/s] 54%|█████▍    | 215415/400000 [00:26<00:22, 8054.70it/s] 54%|█████▍    | 216226/400000 [00:26<00:23, 7877.95it/s] 54%|█████▍    | 217018/400000 [00:26<00:23, 7669.29it/s] 54%|█████▍    | 217816/400000 [00:26<00:23, 7757.95it/s] 55%|█████▍    | 218734/400000 [00:26<00:22, 8135.41it/s] 55%|█████▍    | 219666/400000 [00:26<00:21, 8455.34it/s] 55%|█████▌    | 220562/400000 [00:26<00:20, 8600.30it/s] 55%|█████▌    | 221429/400000 [00:26<00:21, 8432.41it/s] 56%|█████▌    | 222278/400000 [00:27<00:21, 8292.97it/s] 56%|█████▌    | 223121/400000 [00:27<00:21, 8332.44it/s] 56%|█████▌    | 223958/400000 [00:27<00:21, 8266.32it/s] 56%|█████▌    | 224869/400000 [00:27<00:20, 8501.82it/s] 56%|█████▋    | 225723/400000 [00:27<00:20, 8490.00it/s] 57%|█████▋    | 226578/400000 [00:27<00:20, 8505.57it/s] 57%|█████▋    | 227431/400000 [00:27<00:20, 8504.19it/s] 57%|█████▋    | 228297/400000 [00:27<00:20, 8548.07it/s] 57%|█████▋    | 229181/400000 [00:27<00:19, 8632.49it/s] 58%|█████▊    | 230046/400000 [00:28<00:20, 8356.78it/s] 58%|█████▊    | 230931/400000 [00:28<00:19, 8496.98it/s] 58%|█████▊    | 231827/400000 [00:28<00:19, 8630.53it/s] 58%|█████▊    | 232693/400000 [00:28<00:19, 8573.63it/s] 58%|█████▊    | 233611/400000 [00:28<00:19, 8745.50it/s] 59%|█████▊    | 234488/400000 [00:28<00:19, 8605.85it/s] 59%|█████▉    | 235367/400000 [00:28<00:19, 8659.86it/s] 59%|█████▉    | 236322/400000 [00:28<00:18, 8908.43it/s] 59%|█████▉    | 237269/400000 [00:28<00:17, 9069.32it/s] 60%|█████▉    | 238197/400000 [00:28<00:17, 9131.13it/s] 60%|█████▉    | 239113/400000 [00:29<00:18, 8806.32it/s] 60%|█████▉    | 239998/400000 [00:29<00:18, 8586.01it/s] 60%|██████    | 240861/400000 [00:29<00:18, 8590.65it/s] 60%|██████    | 241809/400000 [00:29<00:17, 8837.56it/s] 61%|██████    | 242739/400000 [00:29<00:17, 8969.69it/s] 61%|██████    | 243640/400000 [00:29<00:17, 8892.40it/s] 61%|██████    | 244532/400000 [00:29<00:18, 8307.39it/s] 61%|██████▏   | 245373/400000 [00:29<00:18, 8148.98it/s] 62%|██████▏   | 246196/400000 [00:29<00:18, 8154.21it/s] 62%|██████▏   | 247017/400000 [00:29<00:19, 7985.55it/s] 62%|██████▏   | 247820/400000 [00:30<00:19, 7766.30it/s] 62%|██████▏   | 248601/400000 [00:30<00:19, 7679.11it/s] 62%|██████▏   | 249400/400000 [00:30<00:19, 7767.49it/s] 63%|██████▎   | 250242/400000 [00:30<00:18, 7947.57it/s] 63%|██████▎   | 251063/400000 [00:30<00:18, 8021.17it/s] 63%|██████▎   | 251936/400000 [00:30<00:18, 8219.37it/s] 63%|██████▎   | 252837/400000 [00:30<00:17, 8439.86it/s] 63%|██████▎   | 253798/400000 [00:30<00:16, 8759.18it/s] 64%|██████▎   | 254742/400000 [00:30<00:16, 8952.03it/s] 64%|██████▍   | 255643/400000 [00:31<00:16, 8788.50it/s] 64%|██████▍   | 256526/400000 [00:31<00:16, 8761.98it/s] 64%|██████▍   | 257422/400000 [00:31<00:16, 8816.50it/s] 65%|██████▍   | 258336/400000 [00:31<00:15, 8910.27it/s] 65%|██████▍   | 259243/400000 [00:31<00:15, 8955.14it/s] 65%|██████▌   | 260140/400000 [00:31<00:15, 8755.50it/s] 65%|██████▌   | 261067/400000 [00:31<00:15, 8902.38it/s] 65%|██████▌   | 261960/400000 [00:31<00:15, 8899.16it/s] 66%|██████▌   | 262909/400000 [00:31<00:15, 9067.33it/s] 66%|██████▌   | 263818/400000 [00:31<00:15, 9068.53it/s] 66%|██████▌   | 264727/400000 [00:32<00:15, 8623.02it/s] 66%|██████▋   | 265595/400000 [00:32<00:16, 8330.55it/s] 67%|██████▋   | 266442/400000 [00:32<00:15, 8370.45it/s] 67%|██████▋   | 267353/400000 [00:32<00:15, 8576.70it/s] 67%|██████▋   | 268298/400000 [00:32<00:14, 8820.93it/s] 67%|██████▋   | 269185/400000 [00:32<00:14, 8823.02it/s] 68%|██████▊   | 270119/400000 [00:32<00:14, 8969.73it/s] 68%|██████▊   | 271019/400000 [00:32<00:14, 8884.74it/s] 68%|██████▊   | 271966/400000 [00:32<00:14, 9052.19it/s] 68%|██████▊   | 272894/400000 [00:32<00:13, 9118.56it/s] 68%|██████▊   | 273808/400000 [00:33<00:13, 9043.37it/s] 69%|██████▊   | 274764/400000 [00:33<00:13, 9191.01it/s] 69%|██████▉   | 275685/400000 [00:33<00:13, 8967.58it/s] 69%|██████▉   | 276585/400000 [00:33<00:13, 8951.89it/s] 69%|██████▉   | 277482/400000 [00:33<00:13, 8789.83it/s] 70%|██████▉   | 278363/400000 [00:33<00:14, 8635.44it/s] 70%|██████▉   | 279321/400000 [00:33<00:13, 8898.49it/s] 70%|███████   | 280215/400000 [00:33<00:13, 8824.64it/s] 70%|███████   | 281117/400000 [00:33<00:13, 8881.96it/s] 71%|███████   | 282008/400000 [00:33<00:13, 8578.12it/s] 71%|███████   | 282870/400000 [00:34<00:13, 8523.20it/s] 71%|███████   | 283734/400000 [00:34<00:13, 8557.42it/s] 71%|███████   | 284630/400000 [00:34<00:13, 8673.92it/s] 71%|███████▏  | 285529/400000 [00:34<00:13, 8765.79it/s] 72%|███████▏  | 286427/400000 [00:34<00:12, 8828.36it/s] 72%|███████▏  | 287311/400000 [00:34<00:12, 8741.93it/s] 72%|███████▏  | 288239/400000 [00:34<00:12, 8896.03it/s] 72%|███████▏  | 289130/400000 [00:34<00:12, 8893.06it/s] 73%|███████▎  | 290060/400000 [00:34<00:12, 9009.36it/s] 73%|███████▎  | 290962/400000 [00:35<00:12, 8908.57it/s] 73%|███████▎  | 291854/400000 [00:35<00:12, 8876.14it/s] 73%|███████▎  | 292811/400000 [00:35<00:11, 9072.31it/s] 73%|███████▎  | 293735/400000 [00:35<00:11, 9121.70it/s] 74%|███████▎  | 294679/400000 [00:35<00:11, 9214.37it/s] 74%|███████▍  | 295602/400000 [00:35<00:11, 9166.55it/s] 74%|███████▍  | 296520/400000 [00:35<00:11, 8946.86it/s] 74%|███████▍  | 297456/400000 [00:35<00:11, 9065.61it/s] 75%|███████▍  | 298365/400000 [00:35<00:11, 9009.09it/s] 75%|███████▍  | 299305/400000 [00:35<00:11, 9121.11it/s] 75%|███████▌  | 300261/400000 [00:36<00:10, 9246.43it/s] 75%|███████▌  | 301187/400000 [00:36<00:10, 9101.72it/s] 76%|███████▌  | 302145/400000 [00:36<00:10, 9237.03it/s] 76%|███████▌  | 303071/400000 [00:36<00:10, 9033.13it/s] 76%|███████▌  | 303977/400000 [00:36<00:10, 8998.25it/s] 76%|███████▌  | 304929/400000 [00:36<00:10, 9146.58it/s] 76%|███████▋  | 305846/400000 [00:36<00:10, 8965.89it/s] 77%|███████▋  | 306802/400000 [00:36<00:10, 9135.47it/s] 77%|███████▋  | 307722/400000 [00:36<00:10, 9152.90it/s] 77%|███████▋  | 308671/400000 [00:36<00:09, 9250.54it/s] 77%|███████▋  | 309598/400000 [00:37<00:09, 9228.50it/s] 78%|███████▊  | 310522/400000 [00:37<00:09, 9019.93it/s] 78%|███████▊  | 311426/400000 [00:37<00:10, 8787.42it/s] 78%|███████▊  | 312308/400000 [00:37<00:09, 8781.24it/s] 78%|███████▊  | 313224/400000 [00:37<00:09, 8891.06it/s] 79%|███████▊  | 314153/400000 [00:37<00:09, 9004.74it/s] 79%|███████▉  | 315055/400000 [00:37<00:09, 8673.14it/s] 79%|███████▉  | 315966/400000 [00:37<00:09, 8797.65it/s] 79%|███████▉  | 316863/400000 [00:37<00:09, 8846.32it/s] 79%|███████▉  | 317819/400000 [00:37<00:09, 9048.97it/s] 80%|███████▉  | 318740/400000 [00:38<00:08, 9094.81it/s] 80%|███████▉  | 319652/400000 [00:38<00:09, 8780.73it/s] 80%|████████  | 320557/400000 [00:38<00:08, 8857.72it/s] 80%|████████  | 321446/400000 [00:38<00:08, 8822.54it/s] 81%|████████  | 322371/400000 [00:38<00:08, 8944.64it/s] 81%|████████  | 323305/400000 [00:38<00:08, 9056.84it/s] 81%|████████  | 324213/400000 [00:38<00:08, 8886.48it/s] 81%|████████▏ | 325117/400000 [00:38<00:08, 8929.38it/s] 82%|████████▏ | 326012/400000 [00:38<00:08, 8903.94it/s] 82%|████████▏ | 326955/400000 [00:38<00:08, 9054.41it/s] 82%|████████▏ | 327862/400000 [00:39<00:08, 8871.82it/s] 82%|████████▏ | 328755/400000 [00:39<00:08, 8888.07it/s] 82%|████████▏ | 329663/400000 [00:39<00:07, 8943.62it/s] 83%|████████▎ | 330559/400000 [00:39<00:07, 8760.19it/s] 83%|████████▎ | 331482/400000 [00:39<00:07, 8894.34it/s] 83%|████████▎ | 332376/400000 [00:39<00:07, 8905.34it/s] 83%|████████▎ | 333290/400000 [00:39<00:07, 8972.23it/s] 84%|████████▎ | 334189/400000 [00:39<00:07, 8970.35it/s] 84%|████████▍ | 335087/400000 [00:39<00:07, 8844.00it/s] 84%|████████▍ | 335986/400000 [00:40<00:07, 8886.33it/s] 84%|████████▍ | 336876/400000 [00:40<00:07, 8828.32it/s] 84%|████████▍ | 337760/400000 [00:40<00:07, 8725.21it/s] 85%|████████▍ | 338720/400000 [00:40<00:06, 8967.90it/s] 85%|████████▍ | 339619/400000 [00:40<00:06, 8860.89it/s] 85%|████████▌ | 340515/400000 [00:40<00:06, 8889.38it/s] 85%|████████▌ | 341406/400000 [00:40<00:06, 8818.71it/s] 86%|████████▌ | 342289/400000 [00:40<00:06, 8753.96it/s] 86%|████████▌ | 343230/400000 [00:40<00:06, 8939.95it/s] 86%|████████▌ | 344126/400000 [00:40<00:06, 8934.75it/s] 86%|████████▋ | 345021/400000 [00:41<00:06, 8873.88it/s] 86%|████████▋ | 345910/400000 [00:41<00:06, 8769.87it/s] 87%|████████▋ | 346788/400000 [00:41<00:06, 8651.24it/s] 87%|████████▋ | 347746/400000 [00:41<00:05, 8908.50it/s] 87%|████████▋ | 348640/400000 [00:41<00:05, 8897.74it/s] 87%|████████▋ | 349532/400000 [00:41<00:05, 8863.27it/s] 88%|████████▊ | 350420/400000 [00:41<00:05, 8864.85it/s] 88%|████████▊ | 351308/400000 [00:41<00:05, 8787.98it/s] 88%|████████▊ | 352188/400000 [00:41<00:05, 8550.11it/s] 88%|████████▊ | 353045/400000 [00:41<00:05, 8442.13it/s] 88%|████████▊ | 353972/400000 [00:42<00:05, 8672.86it/s] 89%|████████▊ | 354845/400000 [00:42<00:05, 8688.45it/s] 89%|████████▉ | 355746/400000 [00:42<00:05, 8780.29it/s] 89%|████████▉ | 356693/400000 [00:42<00:04, 8976.06it/s] 89%|████████▉ | 357609/400000 [00:42<00:04, 9029.80it/s] 90%|████████▉ | 358538/400000 [00:42<00:04, 9105.32it/s] 90%|████████▉ | 359450/400000 [00:42<00:04, 8836.07it/s] 90%|█████████ | 360337/400000 [00:42<00:04, 8835.90it/s] 90%|█████████ | 361243/400000 [00:42<00:04, 8901.27it/s] 91%|█████████ | 362135/400000 [00:42<00:04, 8608.44it/s] 91%|█████████ | 363053/400000 [00:43<00:04, 8770.00it/s] 91%|█████████ | 363956/400000 [00:43<00:04, 8846.33it/s] 91%|█████████ | 364896/400000 [00:43<00:03, 9002.59it/s] 91%|█████████▏| 365819/400000 [00:43<00:03, 9067.67it/s] 92%|█████████▏| 366728/400000 [00:43<00:03, 8965.65it/s] 92%|█████████▏| 367698/400000 [00:43<00:03, 9173.73it/s] 92%|█████████▏| 368618/400000 [00:43<00:03, 9061.60it/s] 92%|█████████▏| 369555/400000 [00:43<00:03, 9150.29it/s] 93%|█████████▎| 370472/400000 [00:43<00:03, 9114.48it/s] 93%|█████████▎| 371385/400000 [00:44<00:03, 8970.11it/s] 93%|█████████▎| 372285/400000 [00:44<00:03, 8976.39it/s] 93%|█████████▎| 373184/400000 [00:44<00:03, 8807.98it/s] 94%|█████████▎| 374145/400000 [00:44<00:02, 9032.90it/s] 94%|█████████▍| 375076/400000 [00:44<00:02, 9113.47it/s] 94%|█████████▍| 375990/400000 [00:44<00:02, 9081.64it/s] 94%|█████████▍| 376949/400000 [00:44<00:02, 9225.89it/s] 94%|█████████▍| 377874/400000 [00:44<00:02, 9138.89it/s] 95%|█████████▍| 378819/400000 [00:44<00:02, 9228.76it/s] 95%|█████████▍| 379763/400000 [00:44<00:02, 9288.47it/s] 95%|█████████▌| 380693/400000 [00:45<00:02, 9184.85it/s] 95%|█████████▌| 381645/400000 [00:45<00:01, 9282.08it/s] 96%|█████████▌| 382575/400000 [00:45<00:01, 9053.08it/s] 96%|█████████▌| 383483/400000 [00:45<00:01, 8771.52it/s] 96%|█████████▌| 384370/400000 [00:45<00:01, 8799.32it/s] 96%|█████████▋| 385253/400000 [00:45<00:01, 8602.77it/s] 97%|█████████▋| 386132/400000 [00:45<00:01, 8656.62it/s] 97%|█████████▋| 387000/400000 [00:45<00:01, 8653.21it/s] 97%|█████████▋| 387904/400000 [00:45<00:01, 8764.59it/s] 97%|█████████▋| 388833/400000 [00:45<00:01, 8911.74it/s] 97%|█████████▋| 389726/400000 [00:46<00:01, 8688.75it/s] 98%|█████████▊| 390598/400000 [00:46<00:01, 8642.69it/s] 98%|█████████▊| 391464/400000 [00:46<00:00, 8549.67it/s] 98%|█████████▊| 392350/400000 [00:46<00:00, 8640.18it/s] 98%|█████████▊| 393216/400000 [00:46<00:00, 8597.23it/s] 99%|█████████▊| 394129/400000 [00:46<00:00, 8748.33it/s] 99%|█████████▉| 395078/400000 [00:46<00:00, 8958.06it/s] 99%|█████████▉| 395976/400000 [00:46<00:00, 8856.00it/s] 99%|█████████▉| 396897/400000 [00:46<00:00, 8956.90it/s] 99%|█████████▉| 397823/400000 [00:46<00:00, 9043.52it/s]100%|█████████▉| 398729/400000 [00:47<00:00, 8924.31it/s]100%|█████████▉| 399623/400000 [00:47<00:00, 8897.39it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8472.15it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f19d8704080>, <torchtext.data.dataset.TabularDataset object at 0x7f19d87041d0>, <torchtext.vocab.Vocab object at 0x7f19d87040f0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.88 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.84 file/s]2020-07-06 00:17:52.200008: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-06 00:17:52.204005: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-06 00:17:52.204742: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bf60d0c820 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 00:17:52.204759: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 131134.28it/s] 85%|████████▌ | 8445952/9912422 [00:00<00:07, 187209.84it/s]9920512it [00:00, 41557820.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 580906.34it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 152736.08it/s]1654784it [00:00, 10870746.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 163648.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
