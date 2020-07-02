
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff86f588048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff86f588048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff8db1321e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff8db1321e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff8f9478ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff8f9478ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff888460620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff888460620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff888460620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 142979.11it/s] 79%|███████▉  | 7872512/9912422 [00:00<00:09, 204096.66it/s]9920512it [00:00, 43352139.04it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1153853.51it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161084.56it/s]1654784it [00:00, 10148005.03it/s]                         
0it [00:00, ?it/s]8192it [00:00, 203902.05it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff885852978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff86f494be0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff888460268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff888460268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff888460268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:50,  3.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:50,  3.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:50,  3.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:50,  3.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:36,  4.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:36,  4.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:36,  4.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:36,  4.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:27,  5.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:27,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:26,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:26,  5.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<00:20,  7.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:20,  7.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:20,  7.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:19,  7.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:15,  9.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:15,  9.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:15,  9.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:15,  9.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:12, 11.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:12, 11.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:12, 11.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:12, 11.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:09, 14.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:09, 14.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:09, 14.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:09, 14.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:01<00:08, 16.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:08, 16.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:08, 16.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:08, 16.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 16.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:07, 19.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:07, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:07, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:06, 19.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 20.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:06, 20.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:06, 20.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:06, 20.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:05, 22.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 23.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 23.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 24.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 24.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 24.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 24.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 24.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 24.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 24.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 24.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 25.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 25.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 25.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:04, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:04, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:04, 25.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 25.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:04, 25.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:04, 25.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:04, 25.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 25.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:04, 25.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:04, 25.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:04, 25.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 26.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:04, 26.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:04, 26.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:03, 26.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 25.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:03, 25.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 25.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:03, 25.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 25.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:03, 25.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:03, 26.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 26.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:03, 26.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 26.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 26.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:03, 26.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:03<00:03, 26.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 26.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:03<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:03<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:03<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:03<00:03, 26.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:03<00:02, 28.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:03<00:02, 28.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:03<00:02, 28.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:03<00:02, 28.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:03<00:02, 28.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:03<00:02, 28.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 31.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:03<00:02, 31.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:03<00:02, 31.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:03<00:02, 31.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:02, 31.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:02, 31.79 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:02, 35.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:02, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:02, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:02, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:01, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:01, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:01, 35.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 40.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 40.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 45.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 45.32 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 49.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:01, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 49.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 51.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 51.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 54.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:04<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:04<00:00, 54.71 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 56.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:04<00:00, 56.37 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 57.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:04<00:00, 57.79 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 59.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:04<00:00, 59.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 60.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 64.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 64.27 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 64.27 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 23.45 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.91s/ url]
0 examples [00:00, ? examples/s]2020-07-02 00:08:43.674485: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-02 00:08:43.727572: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-02 00:08:43.728346: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5651da0a1eb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-02 00:08:43.728365: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.95 examples/s]111 examples [00:00,  5.63 examples/s]219 examples [00:00,  8.03 examples/s]326 examples [00:00, 11.43 examples/s]430 examples [00:00, 16.25 examples/s]536 examples [00:00, 23.06 examples/s]645 examples [00:00, 32.65 examples/s]754 examples [00:00, 46.05 examples/s]857 examples [00:01, 64.52 examples/s]959 examples [00:01, 89.72 examples/s]1059 examples [00:01, 123.16 examples/s]1169 examples [00:01, 167.86 examples/s]1271 examples [00:01, 223.93 examples/s]1377 examples [00:01, 293.15 examples/s]1480 examples [00:01, 371.43 examples/s]1582 examples [00:01, 454.10 examples/s]1682 examples [00:01, 540.75 examples/s]1781 examples [00:01, 617.59 examples/s]1890 examples [00:02, 709.49 examples/s]2000 examples [00:02, 793.52 examples/s]2107 examples [00:02, 858.25 examples/s]2217 examples [00:02, 917.95 examples/s]2330 examples [00:02, 971.34 examples/s]2438 examples [00:02, 1000.90 examples/s]2546 examples [00:02, 998.29 examples/s] 2652 examples [00:02, 1015.12 examples/s]2763 examples [00:02, 1040.59 examples/s]2875 examples [00:03, 1062.33 examples/s]2988 examples [00:03, 1067.72 examples/s]3098 examples [00:03, 1074.58 examples/s]3207 examples [00:03, 1054.94 examples/s]3314 examples [00:03, 1058.67 examples/s]3427 examples [00:03, 1078.28 examples/s]3536 examples [00:03, 1049.50 examples/s]3642 examples [00:03, 1030.00 examples/s]3746 examples [00:03, 1030.69 examples/s]3855 examples [00:03, 1047.54 examples/s]3964 examples [00:04, 1057.58 examples/s]4070 examples [00:04, 1056.97 examples/s]4176 examples [00:04, 1055.93 examples/s]4282 examples [00:04, 1048.73 examples/s]4390 examples [00:04, 1057.84 examples/s]4497 examples [00:04, 1061.02 examples/s]4604 examples [00:04, 1037.89 examples/s]4708 examples [00:04, 1013.65 examples/s]4810 examples [00:04, 988.61 examples/s] 4910 examples [00:04, 963.64 examples/s]5018 examples [00:05, 995.07 examples/s]5124 examples [00:05, 1012.50 examples/s]5233 examples [00:05, 1033.24 examples/s]5337 examples [00:05, 1024.21 examples/s]5448 examples [00:05, 1045.97 examples/s]5557 examples [00:05, 1057.90 examples/s]5666 examples [00:05, 1064.82 examples/s]5773 examples [00:05, 1058.59 examples/s]5880 examples [00:05, 1048.75 examples/s]5993 examples [00:05, 1069.38 examples/s]6102 examples [00:06, 1075.42 examples/s]6218 examples [00:06, 1096.96 examples/s]6329 examples [00:06, 1099.29 examples/s]6440 examples [00:06, 1089.84 examples/s]6553 examples [00:06, 1100.70 examples/s]6664 examples [00:06, 1078.96 examples/s]6773 examples [00:06, 1073.69 examples/s]6881 examples [00:06, 1065.30 examples/s]6988 examples [00:06, 1048.86 examples/s]7099 examples [00:07, 1065.97 examples/s]7206 examples [00:07, 1056.84 examples/s]7316 examples [00:07, 1069.36 examples/s]7424 examples [00:07, 1064.65 examples/s]7531 examples [00:07, 1056.08 examples/s]7637 examples [00:07, 1048.58 examples/s]7746 examples [00:07, 1059.66 examples/s]7853 examples [00:07, 1054.52 examples/s]7966 examples [00:07, 1075.28 examples/s]8074 examples [00:07, 1057.53 examples/s]8180 examples [00:08, 1040.43 examples/s]8293 examples [00:08, 1063.47 examples/s]8403 examples [00:08, 1072.42 examples/s]8511 examples [00:08, 1074.30 examples/s]8619 examples [00:08, 1041.11 examples/s]8727 examples [00:08, 1051.76 examples/s]8833 examples [00:08, 1052.62 examples/s]8941 examples [00:08, 1059.04 examples/s]9052 examples [00:08, 1071.80 examples/s]9160 examples [00:08, 1066.38 examples/s]9272 examples [00:09, 1079.70 examples/s]9385 examples [00:09, 1093.25 examples/s]9495 examples [00:09, 1087.97 examples/s]9611 examples [00:09, 1107.45 examples/s]9722 examples [00:09, 1090.73 examples/s]9832 examples [00:09, 1091.50 examples/s]9943 examples [00:09, 1078.36 examples/s]10051 examples [00:09, 961.96 examples/s]10150 examples [00:09, 963.62 examples/s]10257 examples [00:10, 993.00 examples/s]10360 examples [00:10, 1002.65 examples/s]10462 examples [00:10, 980.31 examples/s] 10570 examples [00:10, 1006.55 examples/s]10673 examples [00:10, 1012.39 examples/s]10785 examples [00:10, 1040.82 examples/s]10897 examples [00:10, 1062.14 examples/s]11004 examples [00:10, 1039.16 examples/s]11113 examples [00:10, 1053.67 examples/s]11219 examples [00:10, 1050.64 examples/s]11325 examples [00:11, 1034.14 examples/s]11429 examples [00:11, 1004.85 examples/s]11530 examples [00:11, 997.82 examples/s] 11640 examples [00:11, 1024.67 examples/s]11743 examples [00:11, 1020.06 examples/s]11846 examples [00:11, 1019.52 examples/s]11949 examples [00:11, 1006.76 examples/s]12059 examples [00:11, 1032.59 examples/s]12163 examples [00:11, 1028.78 examples/s]12267 examples [00:11, 1005.65 examples/s]12368 examples [00:12, 991.65 examples/s] 12469 examples [00:12, 995.81 examples/s]12574 examples [00:12, 1010.27 examples/s]12681 examples [00:12, 1026.19 examples/s]12789 examples [00:12, 1039.73 examples/s]12897 examples [00:12, 1051.44 examples/s]13003 examples [00:12, 1049.90 examples/s]13116 examples [00:12, 1069.91 examples/s]13228 examples [00:12, 1082.25 examples/s]13337 examples [00:12, 1082.49 examples/s]13446 examples [00:13, 1078.31 examples/s]13554 examples [00:13, 1064.30 examples/s]13666 examples [00:13, 1079.28 examples/s]13775 examples [00:13, 1077.44 examples/s]13883 examples [00:13, 1063.51 examples/s]13990 examples [00:13, 1058.35 examples/s]14097 examples [00:13, 1059.86 examples/s]14204 examples [00:13, 1059.96 examples/s]14311 examples [00:13, 1060.51 examples/s]14418 examples [00:14, 1053.17 examples/s]14524 examples [00:14, 1024.01 examples/s]14632 examples [00:14, 1038.18 examples/s]14737 examples [00:14, 1041.16 examples/s]14842 examples [00:14, 1003.70 examples/s]14945 examples [00:14, 1009.38 examples/s]15047 examples [00:14, 1011.02 examples/s]15151 examples [00:14, 1016.87 examples/s]15259 examples [00:14, 1033.79 examples/s]15363 examples [00:14, 1022.47 examples/s]15466 examples [00:15, 999.07 examples/s] 15567 examples [00:15, 990.44 examples/s]15667 examples [00:15, 993.23 examples/s]15774 examples [00:15, 1013.63 examples/s]15879 examples [00:15, 1022.35 examples/s]15982 examples [00:15, 1001.13 examples/s]16083 examples [00:15, 1001.18 examples/s]16189 examples [00:15, 1018.05 examples/s]16292 examples [00:15, 1016.60 examples/s]16397 examples [00:15, 1024.59 examples/s]16500 examples [00:16, 1009.16 examples/s]16602 examples [00:16, 980.88 examples/s] 16701 examples [00:16, 976.44 examples/s]16799 examples [00:16, 932.86 examples/s]16898 examples [00:16, 947.21 examples/s]16998 examples [00:16, 960.17 examples/s]17101 examples [00:16, 978.87 examples/s]17206 examples [00:16, 998.68 examples/s]17310 examples [00:16, 1008.25 examples/s]17412 examples [00:17, 1005.66 examples/s]17513 examples [00:17, 962.52 examples/s] 17610 examples [00:17, 936.59 examples/s]17717 examples [00:17, 971.14 examples/s]17822 examples [00:17, 991.16 examples/s]17925 examples [00:17, 1000.64 examples/s]18032 examples [00:17, 1019.23 examples/s]18140 examples [00:17, 1034.27 examples/s]18247 examples [00:17, 1044.21 examples/s]18352 examples [00:17, 1029.52 examples/s]18458 examples [00:18, 1037.11 examples/s]18569 examples [00:18, 1056.45 examples/s]18675 examples [00:18, 1038.13 examples/s]18781 examples [00:18, 1044.15 examples/s]18890 examples [00:18, 1054.87 examples/s]18996 examples [00:18, 1047.05 examples/s]19101 examples [00:18, 1043.69 examples/s]19210 examples [00:18, 1057.10 examples/s]19318 examples [00:18, 1061.94 examples/s]19425 examples [00:18, 1041.56 examples/s]19530 examples [00:19, 1031.85 examples/s]19634 examples [00:19, 1032.92 examples/s]19738 examples [00:19, 1016.87 examples/s]19841 examples [00:19, 1019.61 examples/s]19946 examples [00:19, 1025.76 examples/s]20049 examples [00:19, 946.78 examples/s] 20152 examples [00:19, 969.84 examples/s]20256 examples [00:19, 989.28 examples/s]20363 examples [00:19, 1010.04 examples/s]20465 examples [00:20, 995.41 examples/s] 20566 examples [00:20, 987.35 examples/s]20666 examples [00:20, 954.07 examples/s]20762 examples [00:20, 953.09 examples/s]20865 examples [00:20, 973.00 examples/s]20966 examples [00:20, 982.72 examples/s]21065 examples [00:20, 976.67 examples/s]21168 examples [00:20, 989.08 examples/s]21276 examples [00:20, 1012.81 examples/s]21384 examples [00:20, 1030.43 examples/s]21488 examples [00:21, 1023.23 examples/s]21591 examples [00:21, 1005.27 examples/s]21692 examples [00:21, 998.75 examples/s] 21799 examples [00:21, 1016.66 examples/s]21907 examples [00:21, 1034.38 examples/s]22011 examples [00:21, 1032.74 examples/s]22115 examples [00:21, 1002.72 examples/s]22216 examples [00:21, 957.26 examples/s] 22313 examples [00:21, 921.77 examples/s]22406 examples [00:21, 897.60 examples/s]22505 examples [00:22, 922.84 examples/s]22598 examples [00:22, 915.36 examples/s]22690 examples [00:22, 886.68 examples/s]22780 examples [00:22, 887.87 examples/s]22870 examples [00:22, 877.90 examples/s]22971 examples [00:22, 912.65 examples/s]23067 examples [00:22, 925.84 examples/s]23172 examples [00:22, 957.81 examples/s]23279 examples [00:22, 987.00 examples/s]23388 examples [00:23, 1013.25 examples/s]23490 examples [00:23, 1000.81 examples/s]23595 examples [00:23, 1013.07 examples/s]23697 examples [00:23, 993.43 examples/s] 23804 examples [00:23, 1014.20 examples/s]23911 examples [00:23, 1029.10 examples/s]24015 examples [00:23, 1005.15 examples/s]24116 examples [00:23, 993.48 examples/s] 24223 examples [00:23, 1015.22 examples/s]24330 examples [00:23, 1029.29 examples/s]24436 examples [00:24, 1037.45 examples/s]24543 examples [00:24, 1045.77 examples/s]24648 examples [00:24, 1036.97 examples/s]24752 examples [00:24, 1018.72 examples/s]24857 examples [00:24, 1026.18 examples/s]24960 examples [00:24, 997.78 examples/s] 25061 examples [00:24, 992.60 examples/s]25161 examples [00:24, 991.63 examples/s]25273 examples [00:24, 1026.59 examples/s]25381 examples [00:24, 1041.94 examples/s]25493 examples [00:25, 1063.40 examples/s]25600 examples [00:25, 1061.20 examples/s]25707 examples [00:25, 1030.17 examples/s]25811 examples [00:25, 1008.63 examples/s]25913 examples [00:25, 997.90 examples/s] 26014 examples [00:25, 997.30 examples/s]26114 examples [00:25, 965.17 examples/s]26218 examples [00:25, 984.50 examples/s]26319 examples [00:25, 990.88 examples/s]26428 examples [00:26, 1016.90 examples/s]26531 examples [00:26, 985.24 examples/s] 26634 examples [00:26, 996.69 examples/s]26736 examples [00:26, 1001.88 examples/s]26845 examples [00:26, 1026.10 examples/s]26948 examples [00:26, 1012.04 examples/s]27051 examples [00:26, 1014.52 examples/s]27153 examples [00:26, 1014.25 examples/s]27260 examples [00:26, 1028.80 examples/s]27365 examples [00:26, 1034.33 examples/s]27472 examples [00:27, 1043.74 examples/s]27579 examples [00:27, 1050.61 examples/s]27685 examples [00:27, 1037.41 examples/s]27793 examples [00:27, 1047.63 examples/s]27898 examples [00:27, 1013.75 examples/s]28000 examples [00:27, 1007.97 examples/s]28107 examples [00:27, 1024.85 examples/s]28210 examples [00:27, 999.72 examples/s] 28321 examples [00:27, 1028.17 examples/s]28432 examples [00:27, 1050.76 examples/s]28538 examples [00:28, 1031.63 examples/s]28645 examples [00:28, 1041.22 examples/s]28750 examples [00:28, 1037.91 examples/s]28863 examples [00:28, 1063.87 examples/s]28973 examples [00:28, 1073.49 examples/s]29085 examples [00:28, 1084.37 examples/s]29196 examples [00:28, 1090.30 examples/s]29306 examples [00:28, 1089.29 examples/s]29421 examples [00:28, 1105.03 examples/s]29533 examples [00:28, 1107.75 examples/s]29644 examples [00:29, 1092.73 examples/s]29754 examples [00:29, 1092.81 examples/s]29864 examples [00:29, 1064.23 examples/s]29976 examples [00:29, 1079.87 examples/s]30085 examples [00:29, 553.50 examples/s] 30194 examples [00:29, 648.95 examples/s]30300 examples [00:30, 733.06 examples/s]30404 examples [00:30, 803.02 examples/s]30509 examples [00:30, 862.06 examples/s]30609 examples [00:30, 892.19 examples/s]30715 examples [00:30, 935.20 examples/s]30823 examples [00:30, 972.00 examples/s]30931 examples [00:30, 1000.06 examples/s]31038 examples [00:30, 1019.87 examples/s]31143 examples [00:30, 1028.60 examples/s]31248 examples [00:30, 994.02 examples/s] 31350 examples [00:31, 977.74 examples/s]31457 examples [00:31, 1001.93 examples/s]31559 examples [00:31, 975.44 examples/s] 31664 examples [00:31, 994.03 examples/s]31774 examples [00:31, 1021.78 examples/s]31879 examples [00:31, 1027.78 examples/s]31984 examples [00:31, 1033.29 examples/s]32088 examples [00:31, 1025.40 examples/s]32195 examples [00:31, 1036.07 examples/s]32303 examples [00:31, 1047.92 examples/s]32412 examples [00:32, 1059.35 examples/s]32520 examples [00:32, 1064.59 examples/s]32627 examples [00:32, 1050.10 examples/s]32733 examples [00:32, 1047.61 examples/s]32844 examples [00:32, 1064.23 examples/s]32951 examples [00:32, 1023.21 examples/s]33054 examples [00:32, 996.54 examples/s] 33155 examples [00:32, 977.03 examples/s]33254 examples [00:32, 968.36 examples/s]33352 examples [00:33, 954.97 examples/s]33458 examples [00:33, 983.85 examples/s]33566 examples [00:33, 1010.47 examples/s]33670 examples [00:33, 1016.63 examples/s]33781 examples [00:33, 1042.70 examples/s]33895 examples [00:33, 1068.39 examples/s]34008 examples [00:33, 1083.59 examples/s]34117 examples [00:33, 1068.19 examples/s]34225 examples [00:33, 1041.54 examples/s]34333 examples [00:33, 1049.80 examples/s]34441 examples [00:34, 1057.90 examples/s]34552 examples [00:34, 1071.82 examples/s]34666 examples [00:34, 1089.86 examples/s]34776 examples [00:34, 1075.98 examples/s]34889 examples [00:34, 1090.52 examples/s]34999 examples [00:34, 1084.83 examples/s]35108 examples [00:34, 1073.97 examples/s]35216 examples [00:34, 1054.25 examples/s]35322 examples [00:34, 1028.48 examples/s]35426 examples [00:34, 1014.68 examples/s]35530 examples [00:35, 1021.48 examples/s]35641 examples [00:35, 1045.97 examples/s]35749 examples [00:35, 1054.44 examples/s]35855 examples [00:35, 1025.03 examples/s]35958 examples [00:35, 1001.53 examples/s]36059 examples [00:35, 978.81 examples/s] 36158 examples [00:35, 972.80 examples/s]36256 examples [00:35, 972.74 examples/s]36354 examples [00:35, 972.76 examples/s]36452 examples [00:36, 961.63 examples/s]36557 examples [00:36, 984.15 examples/s]36669 examples [00:36, 1020.84 examples/s]36773 examples [00:36, 1025.33 examples/s]36878 examples [00:36, 1031.05 examples/s]36990 examples [00:36, 1055.77 examples/s]37096 examples [00:36, 1030.77 examples/s]37201 examples [00:36, 1034.81 examples/s]37305 examples [00:36, 1014.22 examples/s]37407 examples [00:36, 1014.76 examples/s]37511 examples [00:37, 1020.55 examples/s]37619 examples [00:37, 1037.11 examples/s]37729 examples [00:37, 1053.49 examples/s]37835 examples [00:37, 1043.39 examples/s]37940 examples [00:37, 1044.67 examples/s]38052 examples [00:37, 1064.07 examples/s]38159 examples [00:37, 1055.78 examples/s]38265 examples [00:37, 1034.94 examples/s]38372 examples [00:37, 1043.10 examples/s]38478 examples [00:37, 1047.30 examples/s]38589 examples [00:38, 1065.20 examples/s]38696 examples [00:38, 1034.52 examples/s]38807 examples [00:38, 1054.81 examples/s]38918 examples [00:38, 1067.73 examples/s]39027 examples [00:38, 1071.01 examples/s]39135 examples [00:38, 1054.03 examples/s]39241 examples [00:38, 1050.74 examples/s]39351 examples [00:38, 1064.39 examples/s]39458 examples [00:38, 1063.75 examples/s]39567 examples [00:38, 1069.71 examples/s]39675 examples [00:39, 1044.79 examples/s]39787 examples [00:39, 1064.49 examples/s]39902 examples [00:39, 1072.21 examples/s]40010 examples [00:39, 993.68 examples/s] 40120 examples [00:39, 1021.83 examples/s]40224 examples [00:39, 1007.68 examples/s]40334 examples [00:39, 1033.10 examples/s]40445 examples [00:39, 1054.47 examples/s]40552 examples [00:39, 1053.02 examples/s]40658 examples [00:40, 1046.42 examples/s]40763 examples [00:40, 1042.23 examples/s]40868 examples [00:40, 982.32 examples/s] 40968 examples [00:40, 971.50 examples/s]41073 examples [00:40, 992.53 examples/s]41181 examples [00:40, 1013.86 examples/s]41283 examples [00:40, 999.87 examples/s] 41384 examples [00:40, 988.45 examples/s]41489 examples [00:40, 1004.57 examples/s]41590 examples [00:40, 1002.91 examples/s]41694 examples [00:41, 1013.70 examples/s]41809 examples [00:41, 1048.85 examples/s]41915 examples [00:41, 1030.84 examples/s]42019 examples [00:41, 1025.30 examples/s]42122 examples [00:41, 1025.40 examples/s]42225 examples [00:41, 1016.19 examples/s]42327 examples [00:41, 971.40 examples/s] 42425 examples [00:41, 950.41 examples/s]42521 examples [00:41, 946.13 examples/s]42616 examples [00:42, 935.87 examples/s]42710 examples [00:42, 922.13 examples/s]42819 examples [00:42, 963.94 examples/s]42931 examples [00:42, 1005.61 examples/s]43035 examples [00:42, 1014.11 examples/s]43145 examples [00:42, 1036.67 examples/s]43252 examples [00:42, 1044.98 examples/s]43359 examples [00:42, 1051.70 examples/s]43468 examples [00:42, 1061.74 examples/s]43575 examples [00:42, 1014.58 examples/s]43678 examples [00:43, 999.57 examples/s] 43779 examples [00:43, 973.87 examples/s]43877 examples [00:43, 967.92 examples/s]43982 examples [00:43, 989.96 examples/s]44083 examples [00:43, 995.68 examples/s]44190 examples [00:43, 1015.47 examples/s]44299 examples [00:43, 1034.45 examples/s]44406 examples [00:43, 1042.40 examples/s]44516 examples [00:43, 1056.94 examples/s]44622 examples [00:43, 1057.84 examples/s]44728 examples [00:44, 1056.84 examples/s]44834 examples [00:44, 1049.13 examples/s]44939 examples [00:44, 1046.68 examples/s]45050 examples [00:44, 1063.01 examples/s]45157 examples [00:44, 1061.74 examples/s]45264 examples [00:44, 999.14 examples/s] 45365 examples [00:44, 999.77 examples/s]45471 examples [00:44, 1016.66 examples/s]45576 examples [00:44, 1024.29 examples/s]45679 examples [00:44, 1015.73 examples/s]45781 examples [00:45, 975.01 examples/s] 45889 examples [00:45, 1002.50 examples/s]45990 examples [00:45, 992.79 examples/s] 46090 examples [00:45, 990.74 examples/s]46190 examples [00:45, 985.88 examples/s]46289 examples [00:45, 964.34 examples/s]46396 examples [00:45, 992.91 examples/s]46501 examples [00:45, 1007.63 examples/s]46603 examples [00:45, 980.44 examples/s] 46702 examples [00:46, 951.56 examples/s]46798 examples [00:46, 948.65 examples/s]46905 examples [00:46, 981.83 examples/s]47011 examples [00:46, 1002.49 examples/s]47116 examples [00:46, 1014.76 examples/s]47218 examples [00:46, 986.39 examples/s] 47318 examples [00:46, 963.50 examples/s]47415 examples [00:46, 959.62 examples/s]47516 examples [00:46, 973.52 examples/s]47614 examples [00:46, 963.55 examples/s]47723 examples [00:47, 996.38 examples/s]47824 examples [00:47, 958.27 examples/s]47936 examples [00:47, 1000.51 examples/s]48045 examples [00:47, 1023.91 examples/s]48152 examples [00:47, 1034.92 examples/s]48257 examples [00:47, 1029.63 examples/s]48361 examples [00:47, 1031.39 examples/s]48468 examples [00:47, 1039.94 examples/s]48573 examples [00:47, 1023.65 examples/s]48676 examples [00:48, 1014.89 examples/s]48784 examples [00:48, 1028.20 examples/s]48887 examples [00:48, 1026.10 examples/s]49003 examples [00:48, 1060.68 examples/s]49115 examples [00:48, 1077.21 examples/s]49226 examples [00:48, 1084.64 examples/s]49335 examples [00:48, 1059.15 examples/s]49447 examples [00:48, 1076.59 examples/s]49561 examples [00:48, 1092.41 examples/s]49671 examples [00:48, 1094.37 examples/s]49781 examples [00:49, 1071.18 examples/s]49889 examples [00:49, 1053.26 examples/s]50000 examples [00:49, 1069.56 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 18%|█▊        | 8771/50000 [00:00<00:00, 87708.83 examples/s] 46%|████▌     | 22891/50000 [00:00<00:00, 98953.04 examples/s] 73%|███████▎  | 36486/50000 [00:00<00:00, 107748.15 examples/s]                                                                0 examples [00:00, ? examples/s]90 examples [00:00, 898.52 examples/s]197 examples [00:00, 942.70 examples/s]312 examples [00:00, 994.02 examples/s]419 examples [00:00, 1014.74 examples/s]522 examples [00:00, 1018.69 examples/s]633 examples [00:00, 1041.87 examples/s]728 examples [00:00, 1006.62 examples/s]836 examples [00:00, 1024.23 examples/s]939 examples [00:00, 1022.53 examples/s]1039 examples [00:01, 1005.71 examples/s]1141 examples [00:01, 1008.82 examples/s]1248 examples [00:01, 1026.13 examples/s]1358 examples [00:01, 1044.89 examples/s]1462 examples [00:01, 1038.17 examples/s]1566 examples [00:01, 1036.93 examples/s]1677 examples [00:01, 1057.54 examples/s]1783 examples [00:01, 1056.76 examples/s]1893 examples [00:01, 1068.19 examples/s]2003 examples [00:01, 1070.74 examples/s]2111 examples [00:02, 1018.43 examples/s]2225 examples [00:02, 1051.34 examples/s]2335 examples [00:02, 1065.24 examples/s]2444 examples [00:02, 1072.42 examples/s]2552 examples [00:02, 1063.89 examples/s]2659 examples [00:02, 1037.68 examples/s]2764 examples [00:02, 1038.45 examples/s]2876 examples [00:02, 1059.09 examples/s]2983 examples [00:02, 1055.72 examples/s]3089 examples [00:02, 1044.37 examples/s]3194 examples [00:03, 1023.00 examples/s]3302 examples [00:03, 1038.69 examples/s]3409 examples [00:03, 1044.75 examples/s]3519 examples [00:03, 1060.70 examples/s]3626 examples [00:03, 1042.28 examples/s]3731 examples [00:03, 1021.08 examples/s]3834 examples [00:03, 1014.04 examples/s]3936 examples [00:03, 993.68 examples/s] 4048 examples [00:03, 1026.35 examples/s]4153 examples [00:03, 1031.77 examples/s]4258 examples [00:04, 1035.08 examples/s]4372 examples [00:04, 1062.48 examples/s]4487 examples [00:04, 1086.37 examples/s]4598 examples [00:04, 1092.28 examples/s]4708 examples [00:04, 1080.04 examples/s]4819 examples [00:04, 1087.39 examples/s]4928 examples [00:04, 1070.24 examples/s]5037 examples [00:04, 1075.22 examples/s]5145 examples [00:04, 1062.03 examples/s]5252 examples [00:05, 1042.89 examples/s]5360 examples [00:05, 1053.33 examples/s]5466 examples [00:05, 933.60 examples/s] 5575 examples [00:05, 974.15 examples/s]5682 examples [00:05, 998.87 examples/s]5794 examples [00:05, 1031.61 examples/s]5907 examples [00:05, 1058.45 examples/s]6016 examples [00:05, 1066.96 examples/s]6126 examples [00:05, 1074.46 examples/s]6235 examples [00:05, 1066.08 examples/s]6343 examples [00:06, 1047.30 examples/s]6449 examples [00:06, 1045.58 examples/s]6554 examples [00:06, 1040.79 examples/s]6659 examples [00:06, 1033.73 examples/s]6763 examples [00:06, 1005.34 examples/s]6864 examples [00:06, 986.36 examples/s] 6965 examples [00:06, 990.88 examples/s]7068 examples [00:06, 1001.19 examples/s]7179 examples [00:06, 1024.02 examples/s]7282 examples [00:07, 1024.36 examples/s]7393 examples [00:07, 1046.83 examples/s]7505 examples [00:07, 1067.25 examples/s]7612 examples [00:07, 1058.62 examples/s]7724 examples [00:07, 1076.16 examples/s]7832 examples [00:07, 1063.61 examples/s]7942 examples [00:07, 1072.67 examples/s]8053 examples [00:07, 1083.02 examples/s]8162 examples [00:07, 1050.36 examples/s]8275 examples [00:07, 1070.53 examples/s]8383 examples [00:08, 1072.20 examples/s]8491 examples [00:08, 1067.10 examples/s]8602 examples [00:08, 1077.27 examples/s]8715 examples [00:08, 1091.53 examples/s]8825 examples [00:08, 1091.94 examples/s]8935 examples [00:08, 1042.94 examples/s]9044 examples [00:08, 1055.66 examples/s]9157 examples [00:08, 1076.78 examples/s]9266 examples [00:08, 1061.55 examples/s]9375 examples [00:08, 1069.31 examples/s]9483 examples [00:09, 1049.78 examples/s]9589 examples [00:09, 1045.30 examples/s]9694 examples [00:09, 1013.00 examples/s]9806 examples [00:09, 1040.75 examples/s]9920 examples [00:09, 1067.53 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNJ9NL0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNJ9NL0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff888460620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff888460620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff888460620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff885852940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff80d8d3320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff888460620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff888460620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff888460620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff86f4945c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff86f4944e0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff8078ea1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff8078ea1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7ff8078ea1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff8078ea2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff8078ea2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff8078ea2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=dbc10aa72a5f7d41caad74974fcdc7484e2a98c1423749fc10d0093c7a34af23
  Stored in directory: /tmp/pip-ephem-wheel-cache-ldcahnhi/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:45:33, 11.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:27:45, 15.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:52:36, 22.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:37:18, 31.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:19:18, 44.8kB/s].vector_cache/glove.6B.zip:   1%|          | 8.04M/862M [00:01<3:42:27, 64.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:35:05, 91.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.2M/862M [00:01<1:48:00, 130kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:15:21, 186kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.1M/862M [00:01<52:31, 265kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:01<36:41, 378kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:01<25:38, 538kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<17:59, 763kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.1M/862M [00:02<12:35, 1.08MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<08:53, 1.53MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<06:16, 2.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<16:59, 795kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<13:45, 977kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<13:15, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<24:10, 556kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<18:03, 740kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<13:18, 1.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:06<09:29, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<11:34, 1.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<11:01, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<08:19, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:08<06:00, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<09:21, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<07:56, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:10<05:51, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<07:07, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<07:45, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<05:59, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.5M/862M [00:12<04:22, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:14<08:10, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<07:04, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:14<05:16, 2.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<06:44, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<07:20, 1.77MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<05:43, 2.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<04:07, 3.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<41:07, 315kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<30:05, 430kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<21:20, 605kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<17:56, 719kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<15:09, 850kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<11:14, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<09:55, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<08:14, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:13, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<06:21, 2.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<04:45, 2.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:19, 2.00MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:42, 2.22MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:18, 2.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:00, 2.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:44, 1.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:17, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<03:49, 3.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<23:50, 526kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<17:58, 698kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<12:52, 972kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:53, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:50, 1.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:51, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<22:13, 558kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<16:49, 737kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<12:03, 1.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<11:57, 1.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<08:25, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<16:42, 734kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<12:57, 946kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<09:19, 1.31MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:22, 1.30MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:01, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:55, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:47, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:58, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:27, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:56, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:23, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<04:03, 2.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:40, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:59, 2.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<03:38, 3.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:44, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:19, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:34, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:23, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:56, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<08:25, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:06, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:12, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:24, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:52, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:20, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:37, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:55, 2.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<03:41, 3.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<02:46, 4.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<55:05, 210kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<39:42, 291kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<28:01, 412kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<22:17, 517kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<17:55, 642kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<13:03, 881kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<09:13, 1.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<19:42, 581kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<14:58, 764kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<10:41, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:08, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:23, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:08, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:54, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<04:23, 2.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:42, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:15, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:51, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<03:32, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:26, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:45, 1.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:39, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:33, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:42, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:16, 2.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<05:35, 1.97MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:08, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:51, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:09, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<04:43, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:34, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:02, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:52, 1.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:23, 3.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<1:19:42, 136kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<56:53, 190kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<39:59, 270kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<30:24, 354kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<23:29, 458kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<16:53, 636kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<11:54, 899kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<16:19, 655kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<12:29, 855kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:59, 1.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:46, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<08:17, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:15, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<04:30, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:16, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:34, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:20, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:31, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:05, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<05:18, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:44, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<03:34, 2.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:33, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:21, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:08, 3.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<15:58, 643kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<12:13, 841kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<08:47, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<08:32, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<08:01, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:04, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:21, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<10:00, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<08:01, 1.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:49, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:26, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<06:12, 1.62MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:32, 2.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:44, 1.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:45, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:16, 2.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:18, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:43, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:33, 2.79MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:48, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:09, 2.37MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:04, 3.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:38, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:56, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:41, 2.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:51, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:12, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:06, 3.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:32, 1.74MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:52, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:38, 2.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:48, 2.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:18, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:08, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<03:00, 3.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<07:02, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:54, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<04:22, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:15, 1.80MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:40, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:23, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<03:10, 2.96MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:40, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:20, 1.48MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:37, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:25, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:42, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:23, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<03:13, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:18, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:38, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:28, 2.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:35, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:05, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:16, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:55, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:57, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:10, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:45, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:42, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<02:42, 3.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:43, 1.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:38, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:08, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:58, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:17, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:09, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<03:01, 2.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<06:10, 1.43MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:12, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:51, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:45, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:07, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:01, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<02:53, 3.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<24:04, 361kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<17:45, 490kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<12:36, 688kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<10:49, 798kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:27, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:04, 1.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<06:17, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:16, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:52, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:43, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:10, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:07, 2.71MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:10, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:46, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<02:50, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:58, 2.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:28, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:30, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:33, 3.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:39, 1.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:48, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:32, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:24, 1.87MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:44, 1.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:40, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:44, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:09, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:43, 2.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:48, 2.89MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:51, 2.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:20, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:23, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:29, 3.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:35, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:01, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:00, 2.65MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:57, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:23, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:24, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:28, 3.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:38, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:27, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<04:00, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:37, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:50, 1.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:42, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:43, 2.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:28, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:55, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:53, 2.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:50, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:14, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:17, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:23, 3.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<06:24, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:16, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:52, 1.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:27, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:52, 1.94MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:51, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:47, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:09, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:21, 3.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:46, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:06, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<03:03, 2.41MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:50, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:10, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:14, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:19, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<12:55, 562kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<09:47, 742kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<07:00, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<06:34, 1.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:20, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:54, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:23, 1.62MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:57, 1.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:52, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:35, 1.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:06, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:14, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:19, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:46, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:10, 3.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:28, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<05:13, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:57, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:51, 2.41MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:16, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<05:01, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:50, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:46, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:23, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:28, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:15, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:47, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:00, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:07, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:15, 2.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:34, 1.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:24, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:42, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:56, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:23, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:41, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:55, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:17, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:37, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:51, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:18, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:37, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:16, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:33, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<01:50, 3.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:59, 780kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<06:48, 915kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:03, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:31, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:24, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:51, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:53, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:00, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:04, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:38, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:50, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:48, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:19, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:28, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:43, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:51, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:11, 1.84MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:30, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:42, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:14, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:33, 1.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:37, 2.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:08, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:18, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:35, 2.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<01:52, 3.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<10:06, 562kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<08:55, 636kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<06:40, 848kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:44, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<05:08, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:43, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:31, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:29, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:22, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:14, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<03:08, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:14, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<17:43, 308kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<13:27, 406kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<09:38, 565kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<06:47, 798kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:54, 781kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:55, 910kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<04:22, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<03:08, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:51, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:31, 1.51MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<01:54, 2.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:09, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:58, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:02, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<02:10, 2.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:06, 850kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:16, 982kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:56, 1.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:47, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<08:36, 595kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:01, 730kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<05:09, 991kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:23, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:06, 988kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:05, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:12, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:14, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:30, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:48, 2.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:59, 985kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<04:26, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:21, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:06, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:09, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:24, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:45, 2.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:57, 1.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:59, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:17, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:40, 2.83MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:49, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:53, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:14, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:19, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:33, 1.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:00, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:08, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:23, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:53, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:02, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:18, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:49, 2.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:59, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:17, 1.93MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:49, 2.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:58, 2.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:13, 1.96MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:46, 2.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:55, 2.23MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:13, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:43, 2.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:26<01:14, 3.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<06:11, 683kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<05:09, 821kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:48, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:18, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:07, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:23, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:19, 1.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:27, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:54, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:25, 2.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:29, 1.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:53, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:39, 2.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:44, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:02, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:24, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:44, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:45, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<03:01, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:41, 2.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:21, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:40, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<02:07, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:32, 2.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:41, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:53, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:16, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:37, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:44, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:51, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:34, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:46, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:08, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:33, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:55, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:14, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:45, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:16, 2.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:56, 1.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:14, 1.54MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:44, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:18, 2.64MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:39, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:01, 1.68MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:37, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:10, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:27, 1.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:31, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:23, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:25, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:29, 1.31MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:56, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:22, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:33, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:33, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:57<01:23, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:27, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:28, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:55, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:22, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<02:27, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:30, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:55, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:26, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:25, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:52, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:20, 2.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:24, 1.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:23, 1.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:49, 1.60MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:18, 2.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:55, 1.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:07, 1.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:40, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:59, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:08, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:39, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<01:12, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:30, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:42, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:20, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:11<00:58, 2.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:59, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:01, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:32, 1.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<01:06, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:37, 1.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:43, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:21, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:15<00:58, 2.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:03, 1.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:01, 1.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:31, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<01:05, 2.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:33, 1.56MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:38, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:16, 1.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:55, 2.59MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:22, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:32, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:11, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<00:51, 2.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:46, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:47, 1.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:21, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<00:58, 2.30MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:31, 1.47MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:39, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:54, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:25, 1.51MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:30, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:09, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:27<00:50, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:17, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:24, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:06, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<00:47, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:35, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:34, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:13, 1.64MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:36, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:34, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:11, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:51, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:07, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:14, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:57, 1.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:41, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:01, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:09, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:53, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:39, 2.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:56, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:03, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:49, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<00:35, 2.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:15, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:16, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:58, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:41<00:40, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:14, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:18, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:00, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<00:42, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:08, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:09, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:52, 1.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<00:37, 2.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:04, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:07, 1.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:51, 1.69MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:48, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:52, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:40, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:49<00:28, 2.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:07, 1.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:04, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:48, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:34, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:15, 1.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:08, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:51, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<00:35, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<01:22, 854kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:15, 941kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:55, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:50, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:49, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:37, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:26, 2.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:05, 958kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:58, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:42, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:29, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:53, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:49, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:36, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:25, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:37, 1.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:36, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:28, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:19, 2.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:06, 760kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:56, 890kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:41, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:27, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:13, 629kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:02, 730kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:46, 978kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<00:31, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:43, 962kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:38, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:27, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:18, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:34, 1.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:30, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:22, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:15, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<02:12, 254kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:40, 334kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<01:10, 465kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:46, 658kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:42, 687kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:36, 807kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:26, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:17, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:21, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:20, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:16, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:16, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:12, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:07, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:17, 946kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:16, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:11, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:07, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:09, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:06, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.62MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.35MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 196/400000 [00:00<03:24, 1957.83it/s]  0%|          | 1073/400000 [00:00<02:36, 2552.53it/s]  0%|          | 1934/400000 [00:00<02:03, 3235.06it/s]  1%|          | 2841/400000 [00:00<01:39, 4008.58it/s]  1%|          | 3713/400000 [00:00<01:22, 4783.27it/s]  1%|          | 4611/400000 [00:00<01:11, 5563.16it/s]  1%|▏         | 5471/400000 [00:00<01:03, 6222.22it/s]  2%|▏         | 6336/400000 [00:00<00:57, 6794.30it/s]  2%|▏         | 7231/400000 [00:00<00:53, 7322.18it/s]  2%|▏         | 8140/400000 [00:01<00:50, 7774.41it/s]  2%|▏         | 9016/400000 [00:01<00:48, 8044.50it/s]  2%|▏         | 9919/400000 [00:01<00:46, 8314.80it/s]  3%|▎         | 10848/400000 [00:01<00:45, 8584.02it/s]  3%|▎         | 11820/400000 [00:01<00:43, 8895.61it/s]  3%|▎         | 12783/400000 [00:01<00:42, 9101.85it/s]  3%|▎         | 13765/400000 [00:01<00:41, 9304.70it/s]  4%|▎         | 14710/400000 [00:01<00:43, 8958.79it/s]  4%|▍         | 15619/400000 [00:01<00:43, 8918.85it/s]  4%|▍         | 16536/400000 [00:01<00:42, 8989.59it/s]  4%|▍         | 17452/400000 [00:02<00:42, 9038.28it/s]  5%|▍         | 18381/400000 [00:02<00:41, 9111.50it/s]  5%|▍         | 19296/400000 [00:02<00:42, 9018.85it/s]  5%|▌         | 20201/400000 [00:02<00:43, 8769.28it/s]  5%|▌         | 21159/400000 [00:02<00:42, 8997.56it/s]  6%|▌         | 22087/400000 [00:02<00:41, 9078.08it/s]  6%|▌         | 22998/400000 [00:02<00:41, 8995.25it/s]  6%|▌         | 23900/400000 [00:02<00:43, 8690.28it/s]  6%|▌         | 24794/400000 [00:02<00:42, 8762.66it/s]  6%|▋         | 25724/400000 [00:02<00:41, 8916.79it/s]  7%|▋         | 26621/400000 [00:03<00:41, 8929.70it/s]  7%|▋         | 27542/400000 [00:03<00:41, 9009.14it/s]  7%|▋         | 28445/400000 [00:03<00:42, 8825.97it/s]  7%|▋         | 29407/400000 [00:03<00:40, 9047.89it/s]  8%|▊         | 30315/400000 [00:03<00:41, 8887.15it/s]  8%|▊         | 31240/400000 [00:03<00:41, 8992.75it/s]  8%|▊         | 32147/400000 [00:03<00:40, 9015.67it/s]  8%|▊         | 33051/400000 [00:03<00:41, 8855.63it/s]  8%|▊         | 33949/400000 [00:03<00:41, 8869.28it/s]  9%|▊         | 34838/400000 [00:03<00:41, 8828.04it/s]  9%|▉         | 35722/400000 [00:04<00:41, 8810.48it/s]  9%|▉         | 36662/400000 [00:04<00:40, 8978.45it/s]  9%|▉         | 37575/400000 [00:04<00:40, 9022.44it/s] 10%|▉         | 38563/400000 [00:04<00:39, 9263.56it/s] 10%|▉         | 39556/400000 [00:04<00:38, 9453.00it/s] 10%|█         | 40504/400000 [00:04<00:38, 9419.32it/s] 10%|█         | 41448/400000 [00:04<00:38, 9354.13it/s] 11%|█         | 42385/400000 [00:04<00:38, 9178.16it/s] 11%|█         | 43305/400000 [00:04<00:40, 8842.84it/s] 11%|█         | 44231/400000 [00:04<00:39, 8962.61it/s] 11%|█▏        | 45197/400000 [00:05<00:38, 9157.04it/s] 12%|█▏        | 46128/400000 [00:05<00:38, 9200.59it/s] 12%|█▏        | 47051/400000 [00:05<00:39, 8948.77it/s] 12%|█▏        | 47980/400000 [00:05<00:38, 9046.86it/s] 12%|█▏        | 48888/400000 [00:05<00:39, 8839.71it/s] 12%|█▏        | 49775/400000 [00:05<00:40, 8751.55it/s] 13%|█▎        | 50724/400000 [00:05<00:38, 8957.81it/s] 13%|█▎        | 51623/400000 [00:05<00:40, 8613.59it/s] 13%|█▎        | 52497/400000 [00:05<00:40, 8649.54it/s] 13%|█▎        | 53412/400000 [00:06<00:39, 8792.82it/s] 14%|█▎        | 54327/400000 [00:06<00:38, 8896.06it/s] 14%|█▍        | 55219/400000 [00:06<00:38, 8895.45it/s] 14%|█▍        | 56153/400000 [00:06<00:38, 9021.91it/s] 14%|█▍        | 57057/400000 [00:06<00:39, 8770.30it/s] 14%|█▍        | 57937/400000 [00:06<00:42, 8127.88it/s] 15%|█▍        | 58882/400000 [00:06<00:40, 8480.93it/s] 15%|█▍        | 59742/400000 [00:06<00:40, 8327.93it/s] 15%|█▌        | 60691/400000 [00:06<00:39, 8645.48it/s] 15%|█▌        | 61582/400000 [00:06<00:38, 8722.26it/s] 16%|█▌        | 62461/400000 [00:07<00:39, 8619.07it/s] 16%|█▌        | 63421/400000 [00:07<00:37, 8891.33it/s] 16%|█▌        | 64316/400000 [00:07<00:37, 8835.63it/s] 16%|█▋        | 65303/400000 [00:07<00:36, 9122.20it/s] 17%|█▋        | 66289/400000 [00:07<00:35, 9331.59it/s] 17%|█▋        | 67227/400000 [00:07<00:35, 9344.33it/s] 17%|█▋        | 68171/400000 [00:07<00:35, 9372.17it/s] 17%|█▋        | 69111/400000 [00:07<00:36, 9082.63it/s] 18%|█▊        | 70023/400000 [00:07<00:37, 8773.44it/s] 18%|█▊        | 70919/400000 [00:08<00:37, 8826.53it/s] 18%|█▊        | 71851/400000 [00:08<00:36, 8966.82it/s] 18%|█▊        | 72751/400000 [00:08<00:36, 8879.80it/s] 18%|█▊        | 73642/400000 [00:08<00:36, 8884.10it/s] 19%|█▊        | 74601/400000 [00:08<00:35, 9083.12it/s] 19%|█▉        | 75546/400000 [00:08<00:35, 9188.19it/s] 19%|█▉        | 76495/400000 [00:08<00:34, 9273.80it/s] 19%|█▉        | 77474/400000 [00:08<00:34, 9422.60it/s] 20%|█▉        | 78418/400000 [00:08<00:35, 9158.96it/s] 20%|█▉        | 79371/400000 [00:08<00:34, 9265.97it/s] 20%|██        | 80300/400000 [00:09<00:35, 9012.04it/s] 20%|██        | 81205/400000 [00:09<00:35, 8986.19it/s] 21%|██        | 82185/400000 [00:09<00:34, 9213.77it/s] 21%|██        | 83110/400000 [00:09<00:34, 9088.56it/s] 21%|██        | 84022/400000 [00:09<00:35, 8808.15it/s] 21%|██        | 84936/400000 [00:09<00:35, 8902.69it/s] 21%|██▏       | 85901/400000 [00:09<00:34, 9112.21it/s] 22%|██▏       | 86863/400000 [00:09<00:33, 9253.73it/s] 22%|██▏       | 87792/400000 [00:09<00:34, 8992.38it/s] 22%|██▏       | 88695/400000 [00:09<00:36, 8580.49it/s] 22%|██▏       | 89560/400000 [00:10<00:36, 8530.30it/s] 23%|██▎       | 90459/400000 [00:10<00:35, 8661.30it/s] 23%|██▎       | 91414/400000 [00:10<00:34, 8908.10it/s] 23%|██▎       | 92339/400000 [00:10<00:34, 9006.48it/s] 23%|██▎       | 93243/400000 [00:10<00:34, 8992.33it/s] 24%|██▎       | 94145/400000 [00:10<00:34, 8927.35it/s] 24%|██▍       | 95094/400000 [00:10<00:33, 9084.93it/s] 24%|██▍       | 96052/400000 [00:10<00:32, 9223.76it/s] 24%|██▍       | 96977/400000 [00:10<00:34, 8850.14it/s] 24%|██▍       | 97867/400000 [00:10<00:34, 8812.23it/s] 25%|██▍       | 98796/400000 [00:11<00:33, 8949.84it/s] 25%|██▍       | 99709/400000 [00:11<00:33, 9002.24it/s] 25%|██▌       | 100673/400000 [00:11<00:32, 9183.65it/s] 25%|██▌       | 101594/400000 [00:11<00:32, 9081.90it/s] 26%|██▌       | 102518/400000 [00:11<00:32, 9127.06it/s] 26%|██▌       | 103449/400000 [00:11<00:32, 9179.71it/s] 26%|██▌       | 104388/400000 [00:11<00:31, 9240.65it/s] 26%|██▋       | 105313/400000 [00:11<00:32, 9033.28it/s] 27%|██▋       | 106218/400000 [00:11<00:32, 8947.21it/s] 27%|██▋       | 107187/400000 [00:12<00:31, 9156.69it/s] 27%|██▋       | 108138/400000 [00:12<00:31, 9259.82it/s] 27%|██▋       | 109066/400000 [00:12<00:31, 9232.41it/s] 27%|██▋       | 109991/400000 [00:12<00:31, 9142.41it/s] 28%|██▊       | 110907/400000 [00:12<00:31, 9038.74it/s] 28%|██▊       | 111836/400000 [00:12<00:31, 9110.34it/s] 28%|██▊       | 112768/400000 [00:12<00:31, 9170.17it/s] 28%|██▊       | 113690/400000 [00:12<00:31, 9182.75it/s] 29%|██▊       | 114609/400000 [00:12<00:32, 8885.55it/s] 29%|██▉       | 115500/400000 [00:12<00:33, 8566.56it/s] 29%|██▉       | 116361/400000 [00:13<00:33, 8463.03it/s] 29%|██▉       | 117211/400000 [00:13<00:33, 8425.71it/s] 30%|██▉       | 118096/400000 [00:13<00:32, 8546.61it/s] 30%|██▉       | 118953/400000 [00:13<00:33, 8464.69it/s] 30%|██▉       | 119887/400000 [00:13<00:32, 8708.31it/s] 30%|███       | 120850/400000 [00:13<00:31, 8963.92it/s] 30%|███       | 121829/400000 [00:13<00:30, 9195.55it/s] 31%|███       | 122795/400000 [00:13<00:29, 9328.92it/s] 31%|███       | 123735/400000 [00:13<00:29, 9348.43it/s] 31%|███       | 124673/400000 [00:13<00:29, 9242.30it/s] 31%|███▏      | 125600/400000 [00:14<00:29, 9239.22it/s] 32%|███▏      | 126528/400000 [00:14<00:29, 9251.33it/s] 32%|███▏      | 127455/400000 [00:14<00:29, 9241.63it/s] 32%|███▏      | 128407/400000 [00:14<00:29, 9322.20it/s] 32%|███▏      | 129340/400000 [00:14<00:29, 9278.63it/s] 33%|███▎      | 130269/400000 [00:14<00:29, 9265.69it/s] 33%|███▎      | 131229/400000 [00:14<00:28, 9363.33it/s] 33%|███▎      | 132166/400000 [00:14<00:29, 9235.18it/s] 33%|███▎      | 133091/400000 [00:14<00:30, 8869.45it/s] 34%|███▎      | 134013/400000 [00:14<00:29, 8969.45it/s] 34%|███▎      | 134970/400000 [00:15<00:28, 9139.39it/s] 34%|███▍      | 135901/400000 [00:15<00:28, 9188.72it/s] 34%|███▍      | 136851/400000 [00:15<00:28, 9278.28it/s] 34%|███▍      | 137781/400000 [00:15<00:29, 9018.43it/s] 35%|███▍      | 138686/400000 [00:15<00:29, 8883.74it/s] 35%|███▍      | 139578/400000 [00:15<00:29, 8892.33it/s] 35%|███▌      | 140514/400000 [00:15<00:28, 9025.33it/s] 35%|███▌      | 141492/400000 [00:15<00:27, 9238.50it/s] 36%|███▌      | 142419/400000 [00:15<00:28, 8930.26it/s] 36%|███▌      | 143316/400000 [00:16<00:28, 8903.58it/s] 36%|███▌      | 144221/400000 [00:16<00:28, 8944.43it/s] 36%|███▋      | 145118/400000 [00:16<00:28, 8901.44it/s] 37%|███▋      | 146040/400000 [00:16<00:28, 8993.50it/s] 37%|███▋      | 146950/400000 [00:16<00:28, 9022.75it/s] 37%|███▋      | 147854/400000 [00:16<00:28, 8958.78it/s] 37%|███▋      | 148751/400000 [00:16<00:28, 8938.12it/s] 37%|███▋      | 149706/400000 [00:16<00:27, 9111.02it/s] 38%|███▊      | 150665/400000 [00:16<00:26, 9248.77it/s] 38%|███▊      | 151602/400000 [00:16<00:26, 9282.12it/s] 38%|███▊      | 152560/400000 [00:17<00:26, 9366.13it/s] 38%|███▊      | 153498/400000 [00:17<00:26, 9244.82it/s] 39%|███▊      | 154424/400000 [00:17<00:26, 9182.39it/s] 39%|███▉      | 155364/400000 [00:17<00:26, 9246.10it/s] 39%|███▉      | 156334/400000 [00:17<00:25, 9376.49it/s] 39%|███▉      | 157273/400000 [00:17<00:26, 9283.66it/s] 40%|███▉      | 158214/400000 [00:17<00:25, 9319.85it/s] 40%|███▉      | 159177/400000 [00:17<00:25, 9409.26it/s] 40%|████      | 160119/400000 [00:17<00:25, 9360.97it/s] 40%|████      | 161056/400000 [00:17<00:25, 9332.39it/s] 40%|████      | 161990/400000 [00:18<00:25, 9263.35it/s] 41%|████      | 162966/400000 [00:18<00:25, 9405.61it/s] 41%|████      | 163920/400000 [00:18<00:24, 9444.99it/s] 41%|████      | 164866/400000 [00:18<00:25, 9208.73it/s] 41%|████▏     | 165789/400000 [00:18<00:25, 9143.24it/s] 42%|████▏     | 166755/400000 [00:18<00:25, 9290.98it/s] 42%|████▏     | 167711/400000 [00:18<00:24, 9367.60it/s] 42%|████▏     | 168698/400000 [00:18<00:24, 9512.86it/s] 42%|████▏     | 169671/400000 [00:18<00:24, 9574.40it/s] 43%|████▎     | 170630/400000 [00:18<00:24, 9262.71it/s] 43%|████▎     | 171600/400000 [00:19<00:24, 9388.69it/s] 43%|████▎     | 172549/400000 [00:19<00:24, 9416.41it/s] 43%|████▎     | 173511/400000 [00:19<00:23, 9474.24it/s] 44%|████▎     | 174460/400000 [00:19<00:25, 9019.20it/s] 44%|████▍     | 175368/400000 [00:19<00:24, 8990.10it/s] 44%|████▍     | 176271/400000 [00:19<00:24, 8982.86it/s] 44%|████▍     | 177213/400000 [00:19<00:24, 9109.16it/s] 45%|████▍     | 178144/400000 [00:19<00:24, 9168.49it/s] 45%|████▍     | 179063/400000 [00:19<00:24, 9095.04it/s] 45%|████▍     | 179976/400000 [00:19<00:24, 9103.88it/s] 45%|████▌     | 180888/400000 [00:20<00:24, 9077.48it/s] 45%|████▌     | 181797/400000 [00:20<00:24, 8887.97it/s] 46%|████▌     | 182688/400000 [00:20<00:24, 8693.57it/s] 46%|████▌     | 183574/400000 [00:20<00:24, 8741.18it/s] 46%|████▌     | 184458/400000 [00:20<00:24, 8769.29it/s] 46%|████▋     | 185416/400000 [00:20<00:23, 8996.36it/s] 47%|████▋     | 186332/400000 [00:20<00:23, 9042.84it/s] 47%|████▋     | 187281/400000 [00:20<00:23, 9172.14it/s] 47%|████▋     | 188223/400000 [00:20<00:22, 9244.86it/s] 47%|████▋     | 189149/400000 [00:21<00:23, 9073.10it/s] 48%|████▊     | 190113/400000 [00:21<00:22, 9234.83it/s] 48%|████▊     | 191051/400000 [00:21<00:22, 9275.88it/s] 48%|████▊     | 191980/400000 [00:21<00:22, 9218.11it/s] 48%|████▊     | 192903/400000 [00:21<00:22, 9195.08it/s] 48%|████▊     | 193824/400000 [00:21<00:22, 9102.45it/s] 49%|████▊     | 194773/400000 [00:21<00:22, 9214.45it/s] 49%|████▉     | 195696/400000 [00:21<00:22, 9199.18it/s] 49%|████▉     | 196646/400000 [00:21<00:21, 9282.65it/s] 49%|████▉     | 197604/400000 [00:21<00:21, 9368.75it/s] 50%|████▉     | 198542/400000 [00:22<00:22, 9134.11it/s] 50%|████▉     | 199496/400000 [00:22<00:21, 9249.85it/s] 50%|█████     | 200424/400000 [00:22<00:21, 9257.30it/s] 50%|█████     | 201351/400000 [00:22<00:21, 9234.36it/s] 51%|█████     | 202278/400000 [00:22<00:21, 9244.25it/s] 51%|█████     | 203203/400000 [00:22<00:21, 9124.87it/s] 51%|█████     | 204166/400000 [00:22<00:21, 9269.53it/s] 51%|█████▏    | 205134/400000 [00:22<00:20, 9387.73it/s] 52%|█████▏    | 206114/400000 [00:22<00:20, 9505.96it/s] 52%|█████▏    | 207082/400000 [00:22<00:20, 9556.59it/s] 52%|█████▏    | 208039/400000 [00:23<00:20, 9456.08it/s] 52%|█████▏    | 208986/400000 [00:23<00:20, 9153.89it/s] 52%|█████▏    | 209905/400000 [00:23<00:21, 8974.66it/s] 53%|█████▎    | 210806/400000 [00:23<00:21, 8852.25it/s] 53%|█████▎    | 211744/400000 [00:23<00:20, 9004.22it/s] 53%|█████▎    | 212671/400000 [00:23<00:20, 9080.77it/s] 53%|█████▎    | 213599/400000 [00:23<00:20, 9137.04it/s] 54%|█████▎    | 214542/400000 [00:23<00:20, 9220.66it/s] 54%|█████▍    | 215519/400000 [00:23<00:19, 9377.40it/s] 54%|█████▍    | 216468/400000 [00:23<00:19, 9408.32it/s] 54%|█████▍    | 217410/400000 [00:24<00:19, 9329.12it/s] 55%|█████▍    | 218344/400000 [00:24<00:19, 9270.01it/s] 55%|█████▍    | 219272/400000 [00:24<00:20, 8998.46it/s] 55%|█████▌    | 220249/400000 [00:24<00:19, 9216.70it/s] 55%|█████▌    | 221174/400000 [00:24<00:19, 9180.11it/s] 56%|█████▌    | 222095/400000 [00:24<00:19, 8919.49it/s] 56%|█████▌    | 223038/400000 [00:24<00:19, 9064.51it/s] 56%|█████▌    | 223988/400000 [00:24<00:19, 9188.80it/s] 56%|█████▌    | 224910/400000 [00:24<00:19, 8821.02it/s] 56%|█████▋    | 225797/400000 [00:24<00:19, 8786.89it/s] 57%|█████▋    | 226705/400000 [00:25<00:19, 8871.80it/s] 57%|█████▋    | 227682/400000 [00:25<00:18, 9123.24it/s] 57%|█████▋    | 228640/400000 [00:25<00:18, 9252.97it/s] 57%|█████▋    | 229569/400000 [00:25<00:18, 9212.43it/s] 58%|█████▊    | 230507/400000 [00:25<00:18, 9256.14it/s] 58%|█████▊    | 231435/400000 [00:25<00:18, 9166.48it/s] 58%|█████▊    | 232402/400000 [00:25<00:17, 9311.52it/s] 58%|█████▊    | 233376/400000 [00:25<00:17, 9434.06it/s] 59%|█████▊    | 234321/400000 [00:25<00:17, 9329.52it/s] 59%|█████▉    | 235257/400000 [00:26<00:17, 9336.51it/s] 59%|█████▉    | 236192/400000 [00:26<00:17, 9240.07it/s] 59%|█████▉    | 237175/400000 [00:26<00:17, 9407.72it/s] 60%|█████▉    | 238148/400000 [00:26<00:17, 9501.75it/s] 60%|█████▉    | 239100/400000 [00:26<00:17, 9299.14it/s] 60%|██████    | 240035/400000 [00:26<00:17, 9313.09it/s] 60%|██████    | 240968/400000 [00:26<00:17, 9315.03it/s] 60%|██████    | 241901/400000 [00:26<00:17, 9288.60it/s] 61%|██████    | 242835/400000 [00:26<00:16, 9301.66it/s] 61%|██████    | 243774/400000 [00:26<00:16, 9326.85it/s] 61%|██████    | 244708/400000 [00:27<00:17, 9107.51it/s] 61%|██████▏   | 245621/400000 [00:27<00:17, 9036.59it/s] 62%|██████▏   | 246526/400000 [00:27<00:17, 8916.53it/s] 62%|██████▏   | 247471/400000 [00:27<00:16, 9069.48it/s] 62%|██████▏   | 248462/400000 [00:27<00:16, 9305.23it/s] 62%|██████▏   | 249423/400000 [00:27<00:16, 9392.41it/s] 63%|██████▎   | 250366/400000 [00:27<00:15, 9402.37it/s] 63%|██████▎   | 251317/400000 [00:27<00:15, 9431.90it/s] 63%|██████▎   | 252262/400000 [00:27<00:15, 9409.46it/s] 63%|██████▎   | 253232/400000 [00:27<00:15, 9490.87it/s] 64%|██████▎   | 254205/400000 [00:28<00:15, 9558.91it/s] 64%|██████▍   | 255162/400000 [00:28<00:15, 9267.07it/s] 64%|██████▍   | 256126/400000 [00:28<00:15, 9375.33it/s] 64%|██████▍   | 257088/400000 [00:28<00:15, 9445.52it/s] 65%|██████▍   | 258047/400000 [00:28<00:14, 9488.22it/s] 65%|██████▍   | 259042/400000 [00:28<00:14, 9621.89it/s] 65%|██████▌   | 260006/400000 [00:28<00:14, 9475.26it/s] 65%|██████▌   | 260975/400000 [00:28<00:14, 9537.55it/s] 65%|██████▌   | 261967/400000 [00:28<00:14, 9648.36it/s] 66%|██████▌   | 262933/400000 [00:28<00:14, 9396.12it/s] 66%|██████▌   | 263875/400000 [00:29<00:14, 9369.78it/s] 66%|██████▌   | 264814/400000 [00:29<00:14, 9361.24it/s] 66%|██████▋   | 265758/400000 [00:29<00:14, 9383.00it/s] 67%|██████▋   | 266721/400000 [00:29<00:14, 9455.69it/s] 67%|██████▋   | 267668/400000 [00:29<00:14, 9397.43it/s] 67%|██████▋   | 268614/400000 [00:29<00:13, 9413.49it/s] 67%|██████▋   | 269556/400000 [00:29<00:14, 9040.16it/s] 68%|██████▊   | 270524/400000 [00:29<00:14, 9222.22it/s] 68%|██████▊   | 271495/400000 [00:29<00:13, 9362.24it/s] 68%|██████▊   | 272435/400000 [00:29<00:13, 9197.36it/s] 68%|██████▊   | 273358/400000 [00:30<00:13, 9182.25it/s] 69%|██████▊   | 274300/400000 [00:30<00:13, 9249.33it/s] 69%|██████▉   | 275254/400000 [00:30<00:13, 9332.19it/s] 69%|██████▉   | 276189/400000 [00:30<00:13, 9288.36it/s] 69%|██████▉   | 277119/400000 [00:30<00:14, 8757.89it/s] 70%|██████▉   | 278011/400000 [00:30<00:13, 8804.50it/s] 70%|██████▉   | 278901/400000 [00:30<00:13, 8831.10it/s] 70%|██████▉   | 279828/400000 [00:30<00:13, 8957.63it/s] 70%|███████   | 280768/400000 [00:30<00:13, 9084.72it/s] 70%|███████   | 281679/400000 [00:31<00:13, 8967.28it/s] 71%|███████   | 282640/400000 [00:31<00:12, 9149.86it/s] 71%|███████   | 283583/400000 [00:31<00:12, 9230.23it/s] 71%|███████   | 284532/400000 [00:31<00:12, 9306.34it/s] 71%|███████▏  | 285470/400000 [00:31<00:12, 9325.80it/s] 72%|███████▏  | 286450/400000 [00:31<00:12, 9456.70it/s] 72%|███████▏  | 287446/400000 [00:31<00:11, 9600.49it/s] 72%|███████▏  | 288408/400000 [00:31<00:11, 9375.97it/s] 72%|███████▏  | 289355/400000 [00:31<00:11, 9403.24it/s] 73%|███████▎  | 290297/400000 [00:31<00:11, 9230.94it/s] 73%|███████▎  | 291222/400000 [00:32<00:11, 9098.95it/s] 73%|███████▎  | 292172/400000 [00:32<00:11, 9214.23it/s] 73%|███████▎  | 293112/400000 [00:32<00:11, 9268.00it/s] 74%|███████▎  | 294082/400000 [00:32<00:11, 9391.73it/s] 74%|███████▍  | 295071/400000 [00:32<00:11, 9534.16it/s] 74%|███████▍  | 296026/400000 [00:32<00:11, 9250.25it/s] 74%|███████▍  | 296988/400000 [00:32<00:11, 9353.41it/s] 74%|███████▍  | 297926/400000 [00:32<00:10, 9299.95it/s] 75%|███████▍  | 298858/400000 [00:32<00:10, 9263.51it/s] 75%|███████▍  | 299788/400000 [00:32<00:10, 9273.97it/s] 75%|███████▌  | 300720/400000 [00:33<00:10, 9287.32it/s] 75%|███████▌  | 301687/400000 [00:33<00:10, 9396.79it/s] 76%|███████▌  | 302628/400000 [00:33<00:10, 9260.20it/s] 76%|███████▌  | 303585/400000 [00:33<00:10, 9350.80it/s] 76%|███████▌  | 304521/400000 [00:33<00:10, 9068.83it/s] 76%|███████▋  | 305431/400000 [00:33<00:10, 9013.27it/s] 77%|███████▋  | 306374/400000 [00:33<00:10, 9134.37it/s] 77%|███████▋  | 307290/400000 [00:33<00:10, 9131.35it/s] 77%|███████▋  | 308232/400000 [00:33<00:09, 9215.34it/s] 77%|███████▋  | 309181/400000 [00:33<00:09, 9295.54it/s] 78%|███████▊  | 310120/400000 [00:34<00:09, 9319.25it/s] 78%|███████▊  | 311053/400000 [00:34<00:09, 9317.78it/s] 78%|███████▊  | 311986/400000 [00:34<00:09, 9280.48it/s] 78%|███████▊  | 312965/400000 [00:34<00:09, 9423.93it/s] 78%|███████▊  | 313909/400000 [00:34<00:09, 9346.30it/s] 79%|███████▊  | 314869/400000 [00:34<00:09, 9419.15it/s] 79%|███████▉  | 315812/400000 [00:34<00:09, 9262.64it/s] 79%|███████▉  | 316745/400000 [00:34<00:08, 9282.56it/s] 79%|███████▉  | 317674/400000 [00:34<00:09, 9114.01it/s] 80%|███████▉  | 318587/400000 [00:35<00:09, 8699.37it/s] 80%|███████▉  | 319502/400000 [00:35<00:09, 8828.39it/s] 80%|████████  | 320437/400000 [00:35<00:08, 8976.99it/s] 80%|████████  | 321358/400000 [00:35<00:08, 9044.13it/s] 81%|████████  | 322282/400000 [00:35<00:08, 9099.92it/s] 81%|████████  | 323194/400000 [00:35<00:08, 8974.05it/s] 81%|████████  | 324094/400000 [00:35<00:08, 8771.48it/s] 81%|████████▏ | 325044/400000 [00:35<00:08, 8976.20it/s] 81%|████████▏ | 325966/400000 [00:35<00:08, 9047.60it/s] 82%|████████▏ | 326873/400000 [00:35<00:08, 9048.97it/s] 82%|████████▏ | 327780/400000 [00:36<00:08, 8838.49it/s] 82%|████████▏ | 328666/400000 [00:36<00:08, 8440.04it/s] 82%|████████▏ | 329555/400000 [00:36<00:08, 8568.42it/s] 83%|████████▎ | 330416/400000 [00:36<00:08, 8518.07it/s] 83%|████████▎ | 331286/400000 [00:36<00:08, 8571.24it/s] 83%|████████▎ | 332216/400000 [00:36<00:07, 8777.47it/s] 83%|████████▎ | 333158/400000 [00:36<00:07, 8959.78it/s] 84%|████████▎ | 334057/400000 [00:36<00:07, 8799.10it/s] 84%|████████▎ | 334940/400000 [00:36<00:07, 8548.95it/s] 84%|████████▍ | 335799/400000 [00:36<00:07, 8474.09it/s] 84%|████████▍ | 336668/400000 [00:37<00:07, 8535.63it/s] 84%|████████▍ | 337609/400000 [00:37<00:07, 8779.79it/s] 85%|████████▍ | 338491/400000 [00:37<00:07, 8638.02it/s] 85%|████████▍ | 339425/400000 [00:37<00:06, 8835.00it/s] 85%|████████▌ | 340398/400000 [00:37<00:06, 9085.62it/s] 85%|████████▌ | 341360/400000 [00:37<00:06, 9237.76it/s] 86%|████████▌ | 342288/400000 [00:37<00:06, 9233.93it/s] 86%|████████▌ | 343214/400000 [00:37<00:06, 9158.62it/s] 86%|████████▌ | 344168/400000 [00:37<00:06, 9267.50it/s] 86%|████████▋ | 345097/400000 [00:37<00:05, 9163.17it/s] 87%|████████▋ | 346015/400000 [00:38<00:05, 9039.51it/s] 87%|████████▋ | 346986/400000 [00:38<00:05, 9229.81it/s] 87%|████████▋ | 347942/400000 [00:38<00:05, 9323.77it/s] 87%|████████▋ | 348898/400000 [00:38<00:05, 9391.03it/s] 87%|████████▋ | 349839/400000 [00:38<00:05, 9259.37it/s] 88%|████████▊ | 350814/400000 [00:38<00:05, 9399.77it/s] 88%|████████▊ | 351791/400000 [00:38<00:05, 9505.93it/s] 88%|████████▊ | 352745/400000 [00:38<00:04, 9514.60it/s] 88%|████████▊ | 353698/400000 [00:38<00:05, 9226.35it/s] 89%|████████▊ | 354624/400000 [00:39<00:05, 8925.50it/s] 89%|████████▉ | 355553/400000 [00:39<00:04, 9030.30it/s] 89%|████████▉ | 356463/400000 [00:39<00:04, 9050.30it/s] 89%|████████▉ | 357371/400000 [00:39<00:04, 8968.09it/s] 90%|████████▉ | 358302/400000 [00:39<00:04, 9065.96it/s] 90%|████████▉ | 359211/400000 [00:39<00:04, 8945.66it/s] 90%|█████████ | 360122/400000 [00:39<00:04, 8993.39it/s] 90%|█████████ | 361089/400000 [00:39<00:04, 9186.11it/s] 91%|█████████ | 362010/400000 [00:39<00:04, 9108.76it/s] 91%|█████████ | 362923/400000 [00:39<00:04, 8870.12it/s] 91%|█████████ | 363813/400000 [00:40<00:04, 8663.92it/s] 91%|█████████ | 364736/400000 [00:40<00:03, 8824.66it/s] 91%|█████████▏| 365711/400000 [00:40<00:03, 9081.74it/s] 92%|█████████▏| 366639/400000 [00:40<00:03, 9138.65it/s] 92%|█████████▏| 367584/400000 [00:40<00:03, 9227.95it/s] 92%|█████████▏| 368547/400000 [00:40<00:03, 9344.19it/s] 92%|█████████▏| 369555/400000 [00:40<00:03, 9551.83it/s] 93%|█████████▎| 370513/400000 [00:40<00:03, 9393.32it/s] 93%|█████████▎| 371455/400000 [00:40<00:03, 9183.13it/s] 93%|█████████▎| 372376/400000 [00:40<00:03, 8943.62it/s] 93%|█████████▎| 373297/400000 [00:41<00:02, 9020.83it/s] 94%|█████████▎| 374273/400000 [00:41<00:02, 9228.25it/s] 94%|█████████▍| 375208/400000 [00:41<00:02, 9264.16it/s] 94%|█████████▍| 376137/400000 [00:41<00:02, 9238.37it/s] 94%|█████████▍| 377063/400000 [00:41<00:02, 9203.12it/s] 94%|█████████▍| 377985/400000 [00:41<00:02, 9161.23it/s] 95%|█████████▍| 378976/400000 [00:41<00:02, 9372.77it/s] 95%|█████████▍| 379916/400000 [00:41<00:02, 9301.18it/s] 95%|█████████▌| 380848/400000 [00:41<00:02, 9259.24it/s] 95%|█████████▌| 381775/400000 [00:41<00:02, 8767.84it/s] 96%|█████████▌| 382668/400000 [00:42<00:01, 8815.39it/s] 96%|█████████▌| 383554/400000 [00:42<00:01, 8795.37it/s] 96%|█████████▌| 384450/400000 [00:42<00:01, 8843.79it/s] 96%|█████████▋| 385366/400000 [00:42<00:01, 8935.92it/s] 97%|█████████▋| 386311/400000 [00:42<00:01, 9082.10it/s] 97%|█████████▋| 387251/400000 [00:42<00:01, 9172.44it/s] 97%|█████████▋| 388207/400000 [00:42<00:01, 9283.66it/s] 97%|█████████▋| 389195/400000 [00:42<00:01, 9454.23it/s] 98%|█████████▊| 390143/400000 [00:42<00:01, 9447.47it/s] 98%|█████████▊| 391089/400000 [00:42<00:00, 9421.31it/s] 98%|█████████▊| 392032/400000 [00:43<00:00, 9313.41it/s] 98%|█████████▊| 393012/400000 [00:43<00:00, 9453.20it/s] 98%|█████████▊| 393981/400000 [00:43<00:00, 9522.69it/s] 99%|█████████▊| 394935/400000 [00:43<00:00, 9433.96it/s] 99%|█████████▉| 395910/400000 [00:43<00:00, 9525.08it/s] 99%|█████████▉| 396864/400000 [00:43<00:00, 9503.99it/s] 99%|█████████▉| 397815/400000 [00:43<00:00, 9438.11it/s]100%|█████████▉| 398760/400000 [00:43<00:00, 9225.90it/s]100%|█████████▉| 399684/400000 [00:43<00:00, 8948.07it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9099.80it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff807906ac8>, <torchtext.data.dataset.TabularDataset object at 0x7ff807906c18>, <torchtext.vocab.Vocab object at 0x7ff807906b38>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 31.00 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.55 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.55 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.39 file/s]2020-07-02 00:17:46.319777: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-02 00:17:46.323529: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-02 00:17:46.323687: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5647d104b370 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-02 00:17:46.323702: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138377.99it/s] 84%|████████▍ | 8339456/9912422 [00:00<00:07, 197540.70it/s]9920512it [00:00, 43055222.25it/s]                           
0it [00:00, ?it/s]32768it [00:00, 575527.96it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 481965.73it/s]1654784it [00:00, 12258959.13it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205349.76it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
