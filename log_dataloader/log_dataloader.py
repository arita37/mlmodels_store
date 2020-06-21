
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f9eb92daea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f9eb92daea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f9f2459a1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f9f2459a1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f9f428e1e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f9f428e1e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ed18c1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ed18c1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ed18c1488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3432448/9912422 [00:00<00:00, 34319685.08it/s]9920512it [00:00, 33622133.72it/s]                             
0it [00:00, ?it/s]32768it [00:00, 675744.28it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 984951.74it/s]1654784it [00:00, 12390928.62it/s]                          
0it [00:00, ?it/s]8192it [00:00, 174739.56it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9eb8a6e588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9eb89d46d8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9ed18c10d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9ed18c10d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f9ed18c10d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:34,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:33,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:32,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:30,  1.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:03,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:01,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:00,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<01:00,  2.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:39,  3.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:27,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:26,  4.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:26,  4.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:25,  4.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:25,  4.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:17,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:11,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:10,  9.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:07, 12.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 17.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 23.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 30.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 38.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 47.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 47.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 55.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 63.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.95s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 78.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.95s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.95s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 41.04 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.95s/ url]
0 examples [00:00, ? examples/s]2020-06-21 18:08:28.999514: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-21 18:08:29.053692: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-06-21 18:08:29.053969: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f0da1a2220 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 18:08:29.053985: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.24 examples/s]133 examples [00:00,  1.77 examples/s]259 examples [00:01,  2.52 examples/s]354 examples [00:01,  3.60 examples/s]480 examples [00:01,  5.14 examples/s]601 examples [00:01,  7.32 examples/s]737 examples [00:01, 10.44 examples/s]871 examples [00:01, 14.86 examples/s]1001 examples [00:01, 21.13 examples/s]1135 examples [00:01, 29.98 examples/s]1260 examples [00:01, 42.38 examples/s]1388 examples [00:01, 59.69 examples/s]1516 examples [00:02, 83.59 examples/s]1645 examples [00:02, 116.17 examples/s]1776 examples [00:02, 159.87 examples/s]1910 examples [00:02, 217.20 examples/s]2040 examples [00:02, 288.00 examples/s]2168 examples [00:02, 375.18 examples/s]2301 examples [00:02, 478.05 examples/s]2440 examples [00:02, 595.00 examples/s]2572 examples [00:02, 698.70 examples/s]2700 examples [00:02, 802.80 examples/s]2839 examples [00:03, 918.16 examples/s]2976 examples [00:03, 1018.67 examples/s]3115 examples [00:03, 1105.47 examples/s]3249 examples [00:03, 1164.06 examples/s]3384 examples [00:03, 1213.69 examples/s]3518 examples [00:03, 1232.55 examples/s]3651 examples [00:03, 1250.55 examples/s]3783 examples [00:03, 1261.89 examples/s]3914 examples [00:03, 1264.22 examples/s]4049 examples [00:03, 1287.07 examples/s]4184 examples [00:04, 1304.37 examples/s]4317 examples [00:04, 1302.99 examples/s]4451 examples [00:04, 1313.09 examples/s]4584 examples [00:04, 1301.14 examples/s]4715 examples [00:04, 1247.64 examples/s]4841 examples [00:04, 1251.04 examples/s]4967 examples [00:04, 1250.85 examples/s]5093 examples [00:04, 1238.55 examples/s]5223 examples [00:04, 1254.30 examples/s]5359 examples [00:04, 1281.48 examples/s]5488 examples [00:05, 1257.43 examples/s]5615 examples [00:05, 1219.13 examples/s]5742 examples [00:05, 1230.74 examples/s]5866 examples [00:05, 1215.06 examples/s]5988 examples [00:05, 1203.49 examples/s]6121 examples [00:05, 1237.32 examples/s]6246 examples [00:05, 1238.96 examples/s]6371 examples [00:05, 1182.54 examples/s]6496 examples [00:05, 1201.49 examples/s]6622 examples [00:06, 1216.19 examples/s]6755 examples [00:06, 1247.62 examples/s]6889 examples [00:06, 1267.38 examples/s]7017 examples [00:06, 1269.29 examples/s]7145 examples [00:06, 1253.97 examples/s]7271 examples [00:06, 1248.42 examples/s]7397 examples [00:06, 1243.65 examples/s]7528 examples [00:06, 1260.59 examples/s]7655 examples [00:06, 1225.39 examples/s]7778 examples [00:06, 1223.74 examples/s]7901 examples [00:07, 1208.24 examples/s]8029 examples [00:07, 1227.77 examples/s]8153 examples [00:07, 1231.10 examples/s]8277 examples [00:07, 1208.37 examples/s]8399 examples [00:07, 1191.62 examples/s]8522 examples [00:07, 1201.85 examples/s]8643 examples [00:07, 1201.77 examples/s]8764 examples [00:07, 1190.88 examples/s]8884 examples [00:07, 1188.54 examples/s]9009 examples [00:07, 1205.36 examples/s]9140 examples [00:08, 1233.83 examples/s]9272 examples [00:08, 1255.44 examples/s]9398 examples [00:08, 1245.16 examples/s]9527 examples [00:08, 1256.38 examples/s]9658 examples [00:08, 1270.67 examples/s]9795 examples [00:08, 1298.25 examples/s]9926 examples [00:08, 1267.37 examples/s]10054 examples [00:08, 1170.80 examples/s]10173 examples [00:08, 1174.44 examples/s]10301 examples [00:09, 1202.28 examples/s]10423 examples [00:09, 1206.92 examples/s]10556 examples [00:09, 1240.45 examples/s]10689 examples [00:09, 1265.21 examples/s]10817 examples [00:09, 1252.58 examples/s]10950 examples [00:09, 1272.66 examples/s]11078 examples [00:09, 1267.04 examples/s]11205 examples [00:09, 1262.14 examples/s]11332 examples [00:09, 1250.08 examples/s]11464 examples [00:09, 1268.32 examples/s]11592 examples [00:10, 1267.55 examples/s]11719 examples [00:10, 1262.73 examples/s]11846 examples [00:10, 1218.51 examples/s]11969 examples [00:10, 1216.78 examples/s]12091 examples [00:10, 1190.34 examples/s]12221 examples [00:10, 1220.97 examples/s]12349 examples [00:10, 1237.64 examples/s]12482 examples [00:10, 1262.94 examples/s]12609 examples [00:10, 1264.16 examples/s]12741 examples [00:10, 1280.26 examples/s]12871 examples [00:11, 1285.62 examples/s]13007 examples [00:11, 1304.38 examples/s]13138 examples [00:11, 1291.03 examples/s]13268 examples [00:11, 1210.21 examples/s]13391 examples [00:11, 1186.63 examples/s]13518 examples [00:11, 1208.43 examples/s]13653 examples [00:11, 1246.21 examples/s]13779 examples [00:11, 1238.05 examples/s]13904 examples [00:11, 1219.62 examples/s]14030 examples [00:12, 1230.88 examples/s]14164 examples [00:12, 1259.34 examples/s]14291 examples [00:12, 1259.57 examples/s]14423 examples [00:12, 1276.79 examples/s]14559 examples [00:12, 1300.20 examples/s]14691 examples [00:12, 1304.15 examples/s]14827 examples [00:12, 1312.55 examples/s]14961 examples [00:12, 1319.21 examples/s]15094 examples [00:12, 1266.60 examples/s]15222 examples [00:12, 1236.40 examples/s]15347 examples [00:13, 1239.31 examples/s]15472 examples [00:13, 1187.30 examples/s]15592 examples [00:13, 1148.12 examples/s]15708 examples [00:13, 1120.97 examples/s]15835 examples [00:13, 1161.06 examples/s]15967 examples [00:13, 1203.12 examples/s]16099 examples [00:13, 1234.72 examples/s]16227 examples [00:13, 1247.40 examples/s]16353 examples [00:13, 1203.46 examples/s]16478 examples [00:13, 1216.23 examples/s]16607 examples [00:14, 1234.59 examples/s]16744 examples [00:14, 1271.64 examples/s]16872 examples [00:14, 1270.81 examples/s]17009 examples [00:14, 1298.58 examples/s]17140 examples [00:14, 1291.89 examples/s]17277 examples [00:14, 1313.87 examples/s]17409 examples [00:14, 1312.92 examples/s]17546 examples [00:14, 1327.67 examples/s]17679 examples [00:14, 1300.68 examples/s]17810 examples [00:15, 1219.87 examples/s]17936 examples [00:15, 1231.11 examples/s]18071 examples [00:15, 1263.12 examples/s]18204 examples [00:15, 1282.33 examples/s]18335 examples [00:15, 1288.98 examples/s]18465 examples [00:15, 1289.33 examples/s]18597 examples [00:15, 1296.72 examples/s]18732 examples [00:15, 1311.95 examples/s]18864 examples [00:15, 1306.42 examples/s]18995 examples [00:15, 1271.14 examples/s]19123 examples [00:16, 1257.94 examples/s]19252 examples [00:16, 1267.25 examples/s]19385 examples [00:16, 1282.03 examples/s]19514 examples [00:16, 1279.99 examples/s]19649 examples [00:16, 1298.31 examples/s]19781 examples [00:16, 1302.91 examples/s]19912 examples [00:16, 1299.45 examples/s]20043 examples [00:16, 1206.60 examples/s]20166 examples [00:16, 1209.34 examples/s]20295 examples [00:16, 1230.52 examples/s]20425 examples [00:17, 1247.98 examples/s]20561 examples [00:17, 1277.31 examples/s]20691 examples [00:17, 1283.53 examples/s]20827 examples [00:17, 1305.34 examples/s]20961 examples [00:17, 1314.97 examples/s]21093 examples [00:17, 1308.33 examples/s]21225 examples [00:17, 1299.16 examples/s]21356 examples [00:17, 1291.21 examples/s]21489 examples [00:17, 1299.00 examples/s]21619 examples [00:18, 1252.40 examples/s]21745 examples [00:18, 1251.29 examples/s]21871 examples [00:18, 1244.98 examples/s]21996 examples [00:18, 1246.47 examples/s]22124 examples [00:18, 1254.32 examples/s]22250 examples [00:18, 1254.45 examples/s]22376 examples [00:18, 1252.93 examples/s]22502 examples [00:18, 1252.23 examples/s]22632 examples [00:18, 1265.23 examples/s]22760 examples [00:18, 1268.97 examples/s]22888 examples [00:19, 1270.61 examples/s]23016 examples [00:19, 1267.09 examples/s]23147 examples [00:19, 1277.77 examples/s]23275 examples [00:19, 1261.63 examples/s]23402 examples [00:19, 1251.74 examples/s]23539 examples [00:19, 1283.70 examples/s]23670 examples [00:19, 1288.98 examples/s]23805 examples [00:19, 1304.10 examples/s]23942 examples [00:19, 1321.15 examples/s]24078 examples [00:19, 1330.45 examples/s]24212 examples [00:20, 1313.92 examples/s]24344 examples [00:20, 1274.40 examples/s]24472 examples [00:20, 1275.60 examples/s]24600 examples [00:20, 1274.03 examples/s]24729 examples [00:20, 1277.17 examples/s]24862 examples [00:20, 1292.09 examples/s]24994 examples [00:20, 1300.22 examples/s]25127 examples [00:20, 1308.94 examples/s]25258 examples [00:20, 1298.93 examples/s]25389 examples [00:20, 1301.74 examples/s]25520 examples [00:21, 1252.86 examples/s]25646 examples [00:21, 1245.57 examples/s]25783 examples [00:21, 1279.66 examples/s]25917 examples [00:21, 1294.90 examples/s]26050 examples [00:21, 1302.15 examples/s]26182 examples [00:21, 1307.27 examples/s]26313 examples [00:21, 1306.75 examples/s]26450 examples [00:21, 1324.28 examples/s]26590 examples [00:21, 1344.34 examples/s]26725 examples [00:21, 1338.09 examples/s]26861 examples [00:22, 1343.91 examples/s]26996 examples [00:22, 1291.44 examples/s]27131 examples [00:22, 1306.83 examples/s]27263 examples [00:22, 1284.93 examples/s]27395 examples [00:22, 1293.64 examples/s]27531 examples [00:22, 1310.28 examples/s]27665 examples [00:22, 1317.19 examples/s]27797 examples [00:22, 1303.37 examples/s]27929 examples [00:22, 1305.51 examples/s]28060 examples [00:22, 1291.83 examples/s]28190 examples [00:23, 1291.11 examples/s]28323 examples [00:23, 1300.32 examples/s]28458 examples [00:23, 1314.67 examples/s]28590 examples [00:23, 1308.93 examples/s]28722 examples [00:23, 1311.05 examples/s]28854 examples [00:23, 1304.37 examples/s]28985 examples [00:23, 1302.34 examples/s]29118 examples [00:23, 1308.14 examples/s]29251 examples [00:23, 1313.08 examples/s]29383 examples [00:24, 1306.74 examples/s]29514 examples [00:24, 1252.98 examples/s]29648 examples [00:24, 1276.66 examples/s]29780 examples [00:24, 1288.36 examples/s]29910 examples [00:24, 1271.53 examples/s]30038 examples [00:24, 1182.25 examples/s]30167 examples [00:24, 1211.58 examples/s]30299 examples [00:24, 1242.13 examples/s]30432 examples [00:24, 1266.40 examples/s]30565 examples [00:24, 1282.40 examples/s]30694 examples [00:25, 1240.58 examples/s]30819 examples [00:25, 1221.53 examples/s]30942 examples [00:25, 1215.84 examples/s]31064 examples [00:25, 1214.41 examples/s]31191 examples [00:25, 1227.98 examples/s]31324 examples [00:25, 1256.70 examples/s]31453 examples [00:25, 1265.56 examples/s]31583 examples [00:25, 1274.24 examples/s]31711 examples [00:25, 1274.96 examples/s]31849 examples [00:25, 1302.55 examples/s]31980 examples [00:26, 1292.96 examples/s]32110 examples [00:26, 1254.65 examples/s]32236 examples [00:26, 1255.92 examples/s]32366 examples [00:26, 1268.81 examples/s]32494 examples [00:26, 1235.13 examples/s]32618 examples [00:26, 1216.67 examples/s]32740 examples [00:26, 1216.30 examples/s]32870 examples [00:26, 1238.85 examples/s]33003 examples [00:26, 1262.93 examples/s]33136 examples [00:27, 1279.67 examples/s]33266 examples [00:27, 1284.18 examples/s]33395 examples [00:27, 1264.00 examples/s]33524 examples [00:27, 1271.11 examples/s]33659 examples [00:27, 1291.27 examples/s]33789 examples [00:27, 1288.32 examples/s]33923 examples [00:27, 1300.77 examples/s]34054 examples [00:27, 1292.26 examples/s]34184 examples [00:27, 1271.44 examples/s]34312 examples [00:27, 1269.12 examples/s]34441 examples [00:28, 1275.23 examples/s]34571 examples [00:28, 1280.03 examples/s]34700 examples [00:28, 1248.24 examples/s]34828 examples [00:28, 1257.48 examples/s]34954 examples [00:28, 1244.24 examples/s]35079 examples [00:28, 1238.09 examples/s]35211 examples [00:28, 1260.58 examples/s]35339 examples [00:28, 1263.73 examples/s]35470 examples [00:28, 1276.45 examples/s]35603 examples [00:28, 1290.29 examples/s]35738 examples [00:29, 1304.63 examples/s]35869 examples [00:29, 1301.89 examples/s]36000 examples [00:29, 1278.21 examples/s]36136 examples [00:29, 1301.02 examples/s]36267 examples [00:29, 1298.08 examples/s]36397 examples [00:29, 1254.93 examples/s]36528 examples [00:29, 1270.14 examples/s]36656 examples [00:29, 1266.77 examples/s]36790 examples [00:29, 1287.85 examples/s]36924 examples [00:29, 1301.00 examples/s]37055 examples [00:30, 1288.95 examples/s]37185 examples [00:30, 1274.85 examples/s]37313 examples [00:30, 1248.96 examples/s]37449 examples [00:30, 1279.11 examples/s]37578 examples [00:30, 1279.85 examples/s]37707 examples [00:30, 1275.95 examples/s]37838 examples [00:30, 1283.50 examples/s]37969 examples [00:30, 1290.29 examples/s]38099 examples [00:30, 1265.55 examples/s]38229 examples [00:30, 1274.25 examples/s]38357 examples [00:31, 1266.24 examples/s]38484 examples [00:31, 1254.08 examples/s]38610 examples [00:31, 1236.49 examples/s]38745 examples [00:31, 1268.09 examples/s]38881 examples [00:31, 1292.23 examples/s]39011 examples [00:31, 1281.65 examples/s]39140 examples [00:31, 1252.81 examples/s]39269 examples [00:31, 1263.07 examples/s]39400 examples [00:31, 1275.25 examples/s]39530 examples [00:32, 1280.09 examples/s]39659 examples [00:32, 1266.62 examples/s]39786 examples [00:32, 1248.49 examples/s]39918 examples [00:32, 1267.28 examples/s]40045 examples [00:32, 1182.62 examples/s]40176 examples [00:32, 1216.90 examples/s]40306 examples [00:32, 1238.46 examples/s]40435 examples [00:32, 1253.17 examples/s]40561 examples [00:32, 1233.59 examples/s]40685 examples [00:32, 1234.21 examples/s]40818 examples [00:33, 1259.29 examples/s]40954 examples [00:33, 1285.51 examples/s]41083 examples [00:33, 1170.27 examples/s]41203 examples [00:33, 1090.52 examples/s]41315 examples [00:33, 1056.93 examples/s]41423 examples [00:33, 1034.04 examples/s]41528 examples [00:33, 1000.66 examples/s]41630 examples [00:33, 986.51 examples/s] 41730 examples [00:33, 989.96 examples/s]41830 examples [00:34, 982.91 examples/s]41961 examples [00:34, 1061.42 examples/s]42088 examples [00:34, 1114.56 examples/s]42219 examples [00:34, 1165.57 examples/s]42347 examples [00:34, 1196.45 examples/s]42475 examples [00:34, 1218.63 examples/s]42599 examples [00:34, 1221.94 examples/s]42723 examples [00:34, 1214.50 examples/s]42846 examples [00:34, 1214.32 examples/s]42969 examples [00:34, 1218.23 examples/s]43093 examples [00:35, 1223.52 examples/s]43224 examples [00:35, 1245.77 examples/s]43355 examples [00:35, 1264.22 examples/s]43482 examples [00:35, 1253.55 examples/s]43608 examples [00:35, 1231.65 examples/s]43736 examples [00:35, 1242.80 examples/s]43861 examples [00:35, 1237.63 examples/s]43985 examples [00:35, 1201.34 examples/s]44106 examples [00:35, 1199.95 examples/s]44237 examples [00:36, 1230.05 examples/s]44367 examples [00:36, 1250.05 examples/s]44495 examples [00:36, 1258.81 examples/s]44622 examples [00:36, 1221.88 examples/s]44747 examples [00:36, 1227.49 examples/s]44871 examples [00:36, 1226.81 examples/s]44994 examples [00:36, 1192.42 examples/s]45114 examples [00:36, 1169.19 examples/s]45232 examples [00:36, 1147.49 examples/s]45348 examples [00:36, 1101.81 examples/s]45462 examples [00:37, 1112.43 examples/s]45574 examples [00:37, 1111.90 examples/s]45701 examples [00:37, 1154.40 examples/s]45832 examples [00:37, 1195.02 examples/s]45963 examples [00:37, 1226.12 examples/s]46095 examples [00:37, 1251.19 examples/s]46230 examples [00:37, 1277.54 examples/s]46363 examples [00:37, 1292.66 examples/s]46493 examples [00:37, 1272.02 examples/s]46623 examples [00:37, 1278.65 examples/s]46752 examples [00:38, 1275.43 examples/s]46885 examples [00:38, 1291.27 examples/s]47015 examples [00:38, 1259.84 examples/s]47144 examples [00:38, 1267.50 examples/s]47271 examples [00:38, 1265.22 examples/s]47404 examples [00:38, 1282.23 examples/s]47538 examples [00:38, 1296.18 examples/s]47668 examples [00:38, 1289.78 examples/s]47798 examples [00:38, 1267.83 examples/s]47925 examples [00:38, 1255.19 examples/s]48051 examples [00:39, 1242.14 examples/s]48176 examples [00:39, 1226.19 examples/s]48299 examples [00:39, 1216.96 examples/s]48421 examples [00:39, 1178.56 examples/s]48542 examples [00:39, 1186.69 examples/s]48661 examples [00:39, 1167.95 examples/s]48781 examples [00:39, 1175.34 examples/s]48899 examples [00:39, 1172.62 examples/s]49019 examples [00:39, 1178.34 examples/s]49139 examples [00:40, 1184.70 examples/s]49258 examples [00:40, 1179.20 examples/s]49381 examples [00:40, 1191.89 examples/s]49503 examples [00:40, 1198.05 examples/s]49624 examples [00:40, 1199.84 examples/s]49746 examples [00:40, 1203.53 examples/s]49869 examples [00:40, 1209.30 examples/s]49992 examples [00:40, 1213.64 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9544/50000 [00:00<00:00, 95433.72 examples/s] 47%|████▋     | 23266/50000 [00:00<00:00, 105027.26 examples/s] 74%|███████▍  | 37155/50000 [00:00<00:00, 113314.86 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 957.71 examples/s]223 examples [00:00, 1032.58 examples/s]358 examples [00:00, 1109.73 examples/s]492 examples [00:00, 1169.41 examples/s]623 examples [00:00, 1207.50 examples/s]758 examples [00:00, 1244.16 examples/s]886 examples [00:00, 1251.77 examples/s]1016 examples [00:00, 1264.79 examples/s]1156 examples [00:00, 1301.18 examples/s]1290 examples [00:01, 1310.70 examples/s]1420 examples [00:01, 1299.51 examples/s]1557 examples [00:01, 1317.51 examples/s]1688 examples [00:01, 1308.74 examples/s]1820 examples [00:01, 1309.37 examples/s]1951 examples [00:01, 1293.56 examples/s]2087 examples [00:01, 1311.56 examples/s]2219 examples [00:01, 1313.94 examples/s]2351 examples [00:01, 1289.70 examples/s]2481 examples [00:01, 1292.75 examples/s]2613 examples [00:02, 1297.88 examples/s]2743 examples [00:02, 1242.50 examples/s]2868 examples [00:02, 1239.90 examples/s]2995 examples [00:02, 1248.07 examples/s]3134 examples [00:02, 1285.62 examples/s]3268 examples [00:02, 1300.60 examples/s]3399 examples [00:02, 1296.93 examples/s]3534 examples [00:02, 1309.42 examples/s]3668 examples [00:02, 1317.29 examples/s]3803 examples [00:02, 1326.24 examples/s]3936 examples [00:03, 1326.44 examples/s]4069 examples [00:03, 1321.56 examples/s]4202 examples [00:03, 1314.01 examples/s]4334 examples [00:03, 1312.64 examples/s]4469 examples [00:03, 1321.07 examples/s]4602 examples [00:03, 1314.33 examples/s]4734 examples [00:03, 1310.39 examples/s]4871 examples [00:03, 1326.10 examples/s]5004 examples [00:03, 1322.23 examples/s]5137 examples [00:03, 1312.87 examples/s]5273 examples [00:04, 1325.69 examples/s]5406 examples [00:04, 1322.13 examples/s]5542 examples [00:04, 1332.42 examples/s]5676 examples [00:04, 1279.17 examples/s]5810 examples [00:04, 1296.57 examples/s]5942 examples [00:04, 1303.49 examples/s]6076 examples [00:04, 1311.88 examples/s]6208 examples [00:04, 1284.52 examples/s]6337 examples [00:04, 1278.11 examples/s]6466 examples [00:04, 1240.58 examples/s]6591 examples [00:05, 1211.13 examples/s]6713 examples [00:05, 1195.96 examples/s]6833 examples [00:05, 1193.18 examples/s]6963 examples [00:05, 1221.37 examples/s]7100 examples [00:05, 1261.83 examples/s]7236 examples [00:05, 1285.13 examples/s]7366 examples [00:05, 1283.60 examples/s]7497 examples [00:05, 1290.62 examples/s]7628 examples [00:05, 1294.22 examples/s]7764 examples [00:06, 1312.62 examples/s]7897 examples [00:06, 1315.81 examples/s]8029 examples [00:06, 1302.60 examples/s]8160 examples [00:06, 1288.05 examples/s]8291 examples [00:06, 1292.41 examples/s]8424 examples [00:06, 1300.90 examples/s]8562 examples [00:06, 1322.96 examples/s]8700 examples [00:06, 1337.58 examples/s]8834 examples [00:06, 1280.67 examples/s]8963 examples [00:06, 1267.01 examples/s]9101 examples [00:07, 1298.35 examples/s]9232 examples [00:07, 1298.87 examples/s]9363 examples [00:07, 1294.34 examples/s]9496 examples [00:07, 1303.25 examples/s]9627 examples [00:07, 1228.55 examples/s]9751 examples [00:07, 1208.58 examples/s]9873 examples [00:07, 1153.24 examples/s]9992 examples [00:07, 1163.73 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYINAVC/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYINAVC/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ed18c1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ed18c1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ed18c1488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9eb8a6e588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e628e6b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9ed18c1488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9ed18c1488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f9ed18c1488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9e628e6a20>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f9eb89d40f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f9e6005f268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f9e6005f268> 

  function with postional parmater data_info <function split_train_valid at 0x7f9e6005f268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f9e6005f378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f9e6005f378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f9e6005f378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=aa53a72a8a7709f09c26a427e8be82f2cdba07feeb97e268e800e3e45be785e8
  Stored in directory: /tmp/pip-ephem-wheel-cache-m_byvsdl/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:42:15, 12.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:19:30, 18.0kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<9:22:11, 25.6kB/s]  .vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<6:35:22, 36.3kB/s].vector_cache/glove.6B.zip:   0%|          | 1.67M/862M [00:01<4:36:43, 51.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<3:13:30, 74.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.16M/862M [00:01<2:14:49, 106kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<1:34:08, 151kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.9M/862M [00:01<1:05:45, 215kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<45:58, 306kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 22.5M/862M [00:01<32:07, 436kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<22:29, 619kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:01<15:46, 879kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:01<11:04, 1.25MB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.7M/862M [00:02<07:55, 1.74MB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<05:37, 2.43MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<04:02, 3.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.4M/862M [00:02<02:56, 4.62MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<02:20, 5.76MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<01:48, 7.45MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.9M/862M [00:02<01:26, 9.28MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<11:20, 1.19MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<15:33, 864kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<12:26, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<09:41, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.5M/862M [00:05<06:58, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<09:29, 1.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<09:31, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<08:44, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:46, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:06<04:56, 2.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<08:32, 1.56MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<09:07, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<07:04, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:08<05:12, 2.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<07:06, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<08:10, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:30, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.5M/862M [00:10<04:41, 2.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<11:33, 1.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<11:10, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<08:35, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:12<06:11, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<13:15, 987kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<12:31, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<09:36, 1.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<07:31, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<08:30, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:38, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<04:54, 2.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:46, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<08:43, 1.48MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<09:22, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<07:30, 1.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:18<05:35, 2.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:19<04:09, 3.10MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<09:42, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<11:30, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<09:14, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<06:44, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<08:02, 1.59MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<10:11, 1.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<08:08, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:22<04:24, 2.90MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<13:33, 941kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<13:42, 930kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<10:39, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<07:43, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:37, 2.25MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<17:20, 731kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<16:19, 776kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<12:27, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<08:56, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<06:30, 1.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<42:45, 295kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<33:58, 371kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<24:41, 511kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<17:48, 707kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<12:39, 992kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:50, 906kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:36, 922kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:30, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<07:35, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<05:31, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<24:02, 519kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<53:48, 232kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<38:16, 325kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<28:23, 439kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<22:26, 555kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<17:19, 718kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<14:33, 855kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<11:53, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<10:43, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<09:14, 1.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<08:45, 1.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<07:57, 1.56MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<07:41, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<07:11, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<06:59, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<06:52, 1.81MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<07:08, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<07:17, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:42, 1.61MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:14, 1.51MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:40, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:27, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:54, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<07:13, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:52, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:29, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:20, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:09, 2.01MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:04, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:58, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<05:57, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<06:03, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<05:51, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<05:47, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<05:40, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<05:38, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:37, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:36, 2.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:30, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:32, 2.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:30, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:22, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:10, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:22, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:10, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:22, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:21, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:13, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<05:12, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:38<05:09, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:38<05:05, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:01, 2.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<04:58, 2.47MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<04:59, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<04:54, 2.50MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:53, 2.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:46, 2.57MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<04:48, 2.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:42, 2.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:45, 2.57MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:37, 2.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:43, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:55, 2.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:06, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:00, 2.44MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:56, 2.47MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:47, 2.55MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:41, 2.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:34, 2.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<04:33, 2.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<04:28, 2.73MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:20, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:30, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:15, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:23, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:43, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<05:12, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:59, 2.44MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:38, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:28, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<04:12, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<04:05, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:55, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<03:54, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:02, 755kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<14:09, 856kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<11:07, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:44, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:51, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:05, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<04:57, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<04:42, 2.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<03:57, 3.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<03:58, 3.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:02, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:37, 1.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:41, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:50, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:30, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<04:32, 2.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<03:46, 3.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<03:26, 3.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:44, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:45, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:08, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:57, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<05:18, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<04:26, 2.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<03:35, 3.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:47, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<12:19, 967kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:47, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:04, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:57, 2.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:56, 3.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<03:11, 3.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:17, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:02, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:50, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:36, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:56, 2.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:57, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:01, 979kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:19, 955kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<09:23, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:10, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:16, 2.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:06, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:32, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:16, 881kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:54, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:06, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<06:00, 1.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:29, 2.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<11:25, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<10:45, 1.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:15, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<06:05, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:30, 2.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<10:03, 1.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<12:13, 946kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:52, 1.17MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:12, 1.60MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<05:15, 2.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:32, 1.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:03, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:49, 1.30MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:29, 1.77MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:59, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:05, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:20, 1.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:25, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:58, 2.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<43:39, 260kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<31:05, 365kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<22:05, 513kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<15:33, 726kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<28:42, 393kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<23:22, 483kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<17:03, 661kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<12:12, 922kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<08:49, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<11:12, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<14:02, 800kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<11:17, 994kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<08:11, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<06:01, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<07:48, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<11:03, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<09:07, 1.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:44, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:53, 2.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:19, 1.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<12:14, 905kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:49, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:12, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<05:15, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<14:00, 787kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<14:47, 744kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:32, 953kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:21, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<06:09, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<07:48, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<10:25, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:28, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:14, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:33, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<10:24, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<11:46, 924kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:21, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<06:51, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<04:59, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<47:12, 229kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<37:42, 287kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<27:45, 389kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<20:02, 538kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<14:17, 754kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<10:10, 1.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<17:20, 619kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<16:35, 647kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<12:41, 846kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<09:10, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<06:39, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:34, 1.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<10:04, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<07:56, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:52, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:19, 2.46MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:12, 1.47MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<09:07, 1.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<07:23, 1.43MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:25, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:58, 2.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:52, 968kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<11:37, 906kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<09:08, 1.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:35, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:53, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:15, 1.27MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<11:48, 885kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:30, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:44, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:54, 1.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:42, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:19, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:02, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<06:36, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<06:28, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:22, 1.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:14, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<06:22, 1.63MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<06:05, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:04, 1.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:02, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:58, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:42, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<06:09, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:06, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:57, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:06, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:51, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:52, 1.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:41, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:43, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:37, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:31, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<05:26, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:17, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:17, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:10, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:12, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:11, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:05, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:03, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:58, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:00, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:53, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:54, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:49, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:45, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:45, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<04:37, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<04:40, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<04:33, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:36, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:29, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:29, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:27, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<04:26, 2.31MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:20, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:24, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<05:13, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:37, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:57, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:11, 1.98MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:15, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:20, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:18, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:18, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:13, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<05:10, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<05:05, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:01, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:58, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:52, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:49, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:42, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:36, 2.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:34, 2.23MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<04:31, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:30, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:26, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:20, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:18, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:20, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:13, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:22, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:14, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:12, 2.41MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<04:17, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:10, 2.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:06, 2.46MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:08, 2.45MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:16, 2.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:08, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:17, 2.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:09, 2.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:10, 2.42MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:27, 2.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:12, 2.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:19, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:11, 2.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:16, 2.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:10, 2.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:17, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:04, 2.47MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:09, 2.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:00, 2.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:50, 2.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:32, 2.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:18, 2.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<04:01, 2.49MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:04, 2.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:50, 2.61MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:54, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:46, 2.65MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:44, 2.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:42, 2.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<03:30, 2.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<11:26, 873kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<09:42, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<07:54, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:29, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:39, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:06, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:25, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:24, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:04, 2.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:57, 2.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:43, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:39, 2.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:30, 2.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:31, 2.82MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<12:41, 781kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:14, 968kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:11, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:50, 1.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<05:35, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:42, 2.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:23, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<04:03, 2.43MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:39, 2.69MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:34, 2.76MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<03:14, 3.05MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:14, 3.04MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:12, 1.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:40, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:58, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:44, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:45, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:04, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<03:38, 2.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:10, 3.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:00, 3.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<02:45, 3.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<02:36, 3.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<10:24, 940kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<10:31, 929kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:19, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:25, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:00, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:03, 2.40MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:18, 2.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:01, 3.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<09:57, 976kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<09:40, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:30, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<05:58, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:43, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:42, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<03:05, 3.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:03, 3.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:46, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:50, 890kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:10, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:52, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:38, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<04:28, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:40, 2.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:05, 3.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<02:43, 3.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<14:31, 660kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<12:47, 748kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<09:41, 986kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:11, 1.33MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:35, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:20, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<03:34, 2.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<02:54, 3.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<15:19, 620kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<15:32, 611kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:52, 800kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:07, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:44, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:20, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:29, 2.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:26, 1.46MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:09, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<07:35, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:50, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:30, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:28, 2.70MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:01, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:31, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<09:20, 1.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<07:47, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:04, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:40, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:41, 2.53MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:00, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<02:30, 3.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:10, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<09:10, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:31, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:43, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:20, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:32, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:48, 3.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<02:29, 3.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<37:08, 249kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<30:34, 302kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<22:28, 410kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<16:06, 572kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<11:40, 788kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<08:34, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<06:22, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:57, 1.85MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<25:59, 353kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<19:00, 482kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<13:44, 665kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<10:00, 912kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<07:22, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:33, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<35:01, 260kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<28:37, 318kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<21:06, 431kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<15:14, 595kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<11:03, 819kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<08:08, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<06:04, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:39, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<11:00, 819kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<12:12, 739kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:47, 921kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:33, 1.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:44, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:23, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<03:28, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<02:48, 3.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:21, 3.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<1:15:08, 119kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<57:03, 157kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<40:58, 218kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<29:03, 307kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<20:41, 431kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<14:48, 601kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<10:45, 827kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<18:07, 490kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<16:42, 532kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<12:42, 699kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:18, 953kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:54, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:11, 1.71MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<04:00, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:09, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<31:33, 279kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<26:03, 338kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<19:12, 458kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<13:47, 638kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<09:58, 880kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<07:21, 1.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<05:29, 1.59MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<09:28, 924kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<10:13, 856kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<08:03, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:57, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:23, 2.56MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<02:48, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<51:45, 168kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<39:10, 222kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<29:25, 295kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<21:50, 397kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<15:54, 545kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<11:44, 737kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<08:39, 1.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:34, 1.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:04, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<07:31, 1.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:50, 1.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:10, 1.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:54, 2.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:59, 2.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:58, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<10:52, 785kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<09:04, 941kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:45, 1.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:02, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:45, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:53, 2.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<12:32, 676kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<11:23, 743kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<08:36, 982kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<06:17, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:37, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<08:14, 1.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<10:27, 804kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<08:29, 989kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<06:15, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:34, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:22, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:39, 963kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<07:11, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:28, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:08, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:11, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:30, 3.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:56, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<06:06, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:56, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:40, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:52, 2.86MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:20, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:14, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:55, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<04:33, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:27, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<02:44, 2.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:10, 3.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<08:04, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<08:30, 955kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<06:41, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:58, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:40, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:57, 2.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:18, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<07:57, 1.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<06:16, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:39, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:31, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:42, 2.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:14, 1.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:01, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<05:36, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:19, 1.84MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:19, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<02:36, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:04, 3.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:37, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:46, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<06:03, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:30, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:20, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:37, 3.00MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<16:23, 479kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<13:53, 565kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<10:18, 761kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:24, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<05:23, 1.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:59, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<31:03, 250kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<24:00, 324kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<17:22, 447kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<12:22, 626kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<08:48, 877kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<06:28, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<2:12:55, 58.0kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<1:37:47, 78.8kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<1:09:38, 111kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<48:59, 157kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<34:24, 223kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<24:13, 315kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<1:29:28, 85.4kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<1:04:52, 118kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<45:52, 166kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<32:15, 236kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<22:58, 331kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<16:12, 468kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<21:25, 354kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<21:26, 353kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<16:30, 459kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<11:52, 636kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<08:41, 868kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:19, 1.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:44, 1.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<25:56, 289kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<21:00, 357kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<15:27, 485kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<11:05, 675kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:58, 936kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:54, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:33, 984kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<08:08, 914kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:23, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<04:44, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:34, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:41, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<13:25, 548kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<12:12, 603kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<09:15, 795kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<06:43, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:59, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:42, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:52, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<18:28, 395kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<15:29, 471kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<11:30, 634kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<08:18, 876kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:02, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:24, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<28:17, 256kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<22:34, 320kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<16:26, 439kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<11:45, 613kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<08:26, 851kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<06:04, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<23:37, 303kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<19:14, 372kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:00, 510kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<10:23, 687kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:26, 957kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:32, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:18, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:25, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:02:16, 114kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<47:28, 149kB/s]  .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<34:08, 208kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<24:10, 292kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<17:16, 409kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<12:25, 568kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<09:02, 779kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:40, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<08:31, 824kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:27, 742kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:30, 934kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:34, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:10, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:11, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<02:34, 2.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:26, 1.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:38, 911kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:11, 1.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<04:41, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:36, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:55, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:23, 2.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<01:57, 3.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:12, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:46, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:30, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:05, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:17, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:33, 2.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:06, 3.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<01:47, 3.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<11:09, 611kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<11:02, 617kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<08:28, 804kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:20, 1.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:40, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:37, 1.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:46, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:16, 2.98MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<08:07, 830kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<08:45, 770kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:53, 977kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:08, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:55, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:20, 2.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:57, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:23, 2.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:11, 3.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:13, 811kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:46, 761kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:53, 967kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:22, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:09, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:41, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:58, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:22, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:26, 2.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:08, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:04, 3.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<26:56, 245kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<20:14, 326kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<14:45, 447kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<10:47, 611kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<08:04, 815kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<06:06, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<04:47, 1.37MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:47, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:09, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:39, 2.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:22, 2.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<08:39, 755kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<08:33, 765kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<06:42, 975kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:08, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<04:12, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:20, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:55, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:26, 2.67MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:16, 2.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<01:59, 3.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<01:59, 3.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<07:33, 856kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<06:34, 984kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:59, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:08, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:16, 1.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:53, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:29, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:06, 3.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:07, 3.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<01:55, 3.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<01:51, 3.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<1:44:16, 61.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<1:15:25, 84.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<53:27, 120kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<37:43, 169kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<27:02, 236kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<19:16, 331kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<14:06, 451kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<10:13, 621kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<07:47, 816kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<05:55, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:34, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<09:11, 689kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<07:36, 833kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:10, 1.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<04:49, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:51, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:11, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:39, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:21, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:18, 2.72MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:14, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<01:59, 3.14MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<01:50, 3.41MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<06:53, 909kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:58, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<03:40, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:57, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:34, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:10, 2.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:02, 3.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<01:48, 3.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<01:45, 3.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<06:25, 965kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<07:05, 874kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<06:29, 953kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:54, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:25, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<05:00, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:43, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:31, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:20, 1.42MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:12, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:53, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:16, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:54, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:10, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:49, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:08, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<03:49, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:09, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:50, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<04:05, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:46, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:55, 1.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:39, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:52, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:37, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:47, 1.62MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:33, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:23, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:35<03:33, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:23, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:27, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:16, 1.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<03:19, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:15, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:15, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:09, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:08, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:05, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:06, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:07, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:59, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:05, 1.96MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:03, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:58, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:58, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:51, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:56, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:48, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:51, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:44, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<02:49, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:41, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:44, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:38, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:42, 2.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:36, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:40, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:36, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:37, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:34, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:35, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:37, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:33, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:33, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:30, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:27, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<02:26, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:27, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:23, 2.48MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:24, 2.47MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:21, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:19, 2.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:19, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:18, 2.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:17, 2.60MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:15, 2.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:15, 2.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:13, 2.66MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:14, 2.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:08, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:12, 2.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<02:10, 2.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:07, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:03, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:00, 2.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<01:58, 2.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<01:56, 3.03MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<01:51, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:50, 3.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:45, 3.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<01:44, 3.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:27, 1.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<04:55, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:55, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:29, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:11, 1.82MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:58, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:46, 2.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:28, 2.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:24, 2.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:20, 2.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:32, 2.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:15, 2.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:33, 764kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:04, 950kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:14, 1.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:25, 1.30MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:47, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:27, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:05, 1.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:46, 2.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:32, 2.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:22, 2.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:14, 2.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:08, 2.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:05, 2.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:00, 2.84MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:59, 2.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<01:57, 2.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<1:28:37, 64.4kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<1:02:51, 90.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<44:31, 128kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<31:42, 179kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<22:43, 250kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<16:27, 345kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<12:03, 470kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<08:59, 629kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:52, 823kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:29, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:26, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:42, 1.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:10, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<14:32, 388kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<11:00, 512kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<08:15, 681kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:28, 868kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:07, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:10, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:27, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:58, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:37, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:23, 2.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:14, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:08, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:01, 2.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:01, 2.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<07:30, 742kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<06:06, 910kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:46, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:56, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:17, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:46, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<02:30, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:21, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:09, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:03, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:56, 2.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:54, 2.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:05, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<04:55, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:00, 1.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:19, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:49, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:28, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:11, 2.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:04, 2.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:59, 2.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:07, 2.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:01, 2.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:55, 2.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<08:53, 610kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<06:56, 782kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:54, 919kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:28, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:47, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:42, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:23, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:16, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:08, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:03, 1.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<02:56, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:01, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:49, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:55, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:54, 1.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:39, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:55, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:39, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:40, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:42, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:34, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:37, 2.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:32, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:33, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:28, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:32, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:27, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:31, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:24, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:29, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:22, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:30, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:23, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:27, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:57, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:45, 1.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:33, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:30, 2.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:23, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:21, 2.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:17, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:16, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:13, 2.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:13, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:09, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:06, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:09, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:05, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:07, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:04, 2.51MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<10:57, 477kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<08:29, 615kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<06:29, 805kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:26, 957kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:19, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:29, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:12, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:41, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:41, 1.93MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:19, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:24, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:27, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:17, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:30, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:15, 2.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:15, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:49, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:13, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:47, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:29, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:16, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:06, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:59, 2.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<01:54, 2.68MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<01:50, 2.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<01:57, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:56, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:29, 926kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:35, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:50, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:24, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:55, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:33, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:12, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:03, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:52, 2.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:46, 2.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:38, 3.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:36, 3.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:29, 3.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:55, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:28, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:33, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:46, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:16, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:06, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:49, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:41, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:06, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:57, 2.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:46, 2.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<06:31, 759kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<05:41, 869kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:24, 1.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:22, 1.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:55, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:21, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:09, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:52, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:41, 2.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:32, 3.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:26, 3.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:30, 887kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:47, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:45, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:58, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:22, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:57, 2.48MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:45, 2.75MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:30, 3.19MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:29, 3.24MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:01, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<04:39, 1.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:44, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:56, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:22, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:58, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:41, 2.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:24, 3.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:25, 3.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<07:06, 668kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<06:32, 724kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<05:02, 940kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:46, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:56, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:23, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:12, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:49, 2.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:43, 2.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:39, 2.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<1:03:34, 73.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<45:28, 103kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<32:10, 145kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<22:50, 204kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<16:15, 286kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<11:50, 392kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<08:40, 535kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:26, 719kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:48, 961kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:44, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<02:54, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<22:06, 208kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<16:20, 282kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<11:46, 390kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<08:34, 534kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<06:27, 710kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:57, 921kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:54, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:09, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:37, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:15, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:06, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<28:01, 162kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<20:44, 219kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<14:55, 303kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<11:11, 404kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<08:27, 535kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<06:45, 669kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:28, 825kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:30, 1.00MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:37, 1.25MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:13, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:40, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:13, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:58, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:43, 2.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:36, 2.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:18, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:52, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:20, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:50, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:35, 2.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:34, 2.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:24, 3.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:25, 3.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:18, 3.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:21, 3.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:17, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:59, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:12, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:36, 1.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:11, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:53, 2.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:40, 2.61MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:34, 2.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:28, 2.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:22, 3.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:19, 3.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:16, 3.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<06:00, 721kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:10, 836kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:59, 1.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:08, 1.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:30, 1.71MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:09, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:50, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:39, 2.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:31, 2.82MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:31, 2.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:26, 2.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<09:07, 467kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<07:17, 585kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<05:27, 778kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:10, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:14, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:36, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:08, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:49, 2.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:34, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:27, 2.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<09:55, 423kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<07:41, 545kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<05:44, 729kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<04:19, 965kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:17, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:48, 1.48MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:18, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:54, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:39, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:27, 2.85MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<01:19, 3.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:27, 640kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<06:07, 673kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:44, 867kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:46, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:11, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:50, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:30, 2.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:21, 3.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:10, 3.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<16:00, 254kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<12:32, 323kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<09:24, 430kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<07:15, 558kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:43, 706kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:40, 863kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:54, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:16, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:59, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:37, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:26, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:23, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:09, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:14, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:01, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:10, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:00, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:05, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:56, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:03, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:55, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:03, 1.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:53, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:04, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:58, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:04, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:57, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:59, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:55, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:54, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:51, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:53, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:49, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:49, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:45, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:49, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:47, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:46, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:41, 2.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:43, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:39, 2.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:40, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:37, 2.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:37, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:35, 2.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:36, 2.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:36, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:34, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:33, 2.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:32, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:31, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:30, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:31, 2.52MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:28, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:31, 2.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:27, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:30, 2.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:26, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:28, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:24, 2.70MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:24, 2.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<05:00, 760kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:09, 914kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:19, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:43, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:18, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:00, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:36, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:32, 2.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:25, 2.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:24, 2.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:24, 2.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<06:20, 590kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:00, 745kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:53, 956kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:05, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:30, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:06, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:47, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:34, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:23, 2.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:18, 2.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:11, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<05:14, 700kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<04:26, 825kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:27, 1.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:57, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:30, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:56, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:46, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:39, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:34, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:36, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:31, 2.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:29, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:26, 2.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:22, 2.62MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:19, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:54, 922kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<03:15, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:35, 1.39MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:05, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:53, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:36, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:30, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:20, 2.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:19, 2.69MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:12, 2.96MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:14, 2.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:12, 2.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:09, 3.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<07:01, 502kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<05:46, 611kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<04:23, 803kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:22, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:39, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:10, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:49, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:34, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:23, 2.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:15, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:11, 2.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<22:09, 156kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<16:19, 212kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<11:44, 294kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<08:30, 405kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<06:11, 555kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:42, 729kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:31, 970kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:50, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:13, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:58, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:36, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:33, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:12, 806kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:44, 905kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:57, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:22, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:57, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:41, 1.99MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:31, 2.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:23, 2.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:15, 2.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:21, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:14, 2.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:08, 2.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:06, 3.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:09, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:18, 1.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:43, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:16, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:57, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:39, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:24, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:18, 2.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:16, 2.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:11, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:10, 2.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:09, 2.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:27, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:10, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:49, 1.77MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:32, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:29, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:30, 2.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:27, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:26, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:25, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:24, 2.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:22, 2.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:19, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:19, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:17, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:56, 1.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:35, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:10, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:52, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:39, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:31, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:24, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:20, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:13, 2.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:16, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:10, 2.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:14, 2.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:09, 2.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:13, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<16:04, 194kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<11:44, 265kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<08:32, 364kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<06:18, 492kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:44, 652kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:35, 861kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<03:07, 989kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:40, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:21, 1.30MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:07, 1.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:51, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:38, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:29, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:22, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:17, 2.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:13, 2.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<12:19, 247kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<09:05, 335kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<06:40, 456kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<04:58, 609kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:43, 811kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:00, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:21, 1.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:02, 1.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:40, 1.80MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:33, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:19, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:20, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:12, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<03:11, 935kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:39, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:06, 1.41MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:42, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:37, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:22, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:32, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:24, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:19, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:16, 2.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:17, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:13, 2.40MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:13, 2.37MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:10, 2.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<06:19, 461kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:53, 595kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:44, 776kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:55, 990kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:19, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:55, 1.50MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:34, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:33, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:21, 2.11MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:23, 2.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:21, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:15, 2.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:10, 2.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:13, 2.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:13, 882kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:46, 1.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:19, 1.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:56, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:33, 1.81MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:34, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:24, 2.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:24, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:18, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:16, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:13, 2.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:09, 2.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:08, 2.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:09, 2.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:07, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<02:15, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:54, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:39, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:29, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:18, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:14, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:15, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:11, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:11, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:08, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:08, 2.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:07, 2.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:08, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:10, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:08, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:25, 1.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<02:08, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:47, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:33, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:21, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:17, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:10, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:09, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:04, 2.47MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:04, 2.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:02, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:01, 2.57MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:00, 2.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:10, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:49, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:48, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:38, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:30, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:23, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:15, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:10, 2.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:05, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:01, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:03, 2.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:58, 2.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:52, 1.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:35, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:23, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:15, 2.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:06, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:02, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:57, 2.62MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:59, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:55, 2.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:00, 2.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:59, 2.55MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:58, 2.58MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:55, 2.70MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:56, 2.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<09:45, 256kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<07:12, 346kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<05:14, 476kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<04:02, 616kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<03:02, 817kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:25, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:55, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:40, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:23, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:17, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:06, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:59, 2.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:59, 2.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:55, 2.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<07:03, 345kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<05:15, 462kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<03:57, 610kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<03:01, 798kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:22, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:54, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:34, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:21, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:12, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:13, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:13, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:12, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:11, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:11, 1.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:07, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:02, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:58, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:40, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:31, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:20, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:09, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:03, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:07, 2.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<00:58, 2.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:00, 2.33MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:01, 2.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:00, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:01, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:59, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:00, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:57, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:58, 2.36MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:51, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:42, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:27, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:11, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:08, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:13, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:11, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:12, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:12, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:10, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:08, 1.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:09, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:05, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:06, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:03, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:05, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:03, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:03, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:01, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:02, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:00, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:58, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:59, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:59, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:59, 2.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:00, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:58, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:59, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:57, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:57, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:57, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:22, 1.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:14, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:07, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:03, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:58, 2.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:56, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:54, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:51, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:53, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:49, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:56, 2.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:55, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:51, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:52, 2.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:50, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<08:55, 234kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<06:36, 316kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<04:50, 430kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<03:37, 574kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:42, 762kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:11, 942kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:45, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:28, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:16, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:07, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:00, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:51, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:48, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:04, 978kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:45, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:26, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:15, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:10, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:04, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:01, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:58, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.12MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:53, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:52, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:50, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:49, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:48, 2.44MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:30, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:16, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:06, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:59, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:54, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:50, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:48, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:46, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:45, 2.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:43, 2.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:43, 2.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:42, 2.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:41, 2.74MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:40, 2.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<06:59, 270kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<05:10, 363kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<03:48, 492kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:51, 655kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:11, 852kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:42, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:23, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:09, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:59, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:53, 2.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:49, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:33, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:21, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:08, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:59, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:52, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:48, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:44, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:40, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:39, 2.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:38, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:37, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:36, 2.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:30, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:19, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:06, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:52, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:45, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:43, 2.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:41, 2.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:42, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:40, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:37, 2.69MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:36, 2.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:34, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<08:38, 195kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<06:15, 267kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<04:32, 368kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<03:19, 500kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:26, 677kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:53, 869kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:27, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:11, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:58, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:45, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:41, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:38, 2.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<03:09, 511kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<02:35, 622kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:57, 817kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:28, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:12, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:56, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:49, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:40, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:39, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:32, 2.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:33, 2.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:10, 1.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:07, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:54, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:37, 2.45MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:31, 2.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:28, 3.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:25, 3.49MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:23, 3.71MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:27, 1.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:30, 979kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:10, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:54, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:42, 2.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:35, 2.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:29, 2.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:26, 3.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:27, 3.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:26, 971kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:32, 907kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:12, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:56, 1.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:44, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:35, 2.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:30, 2.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:25, 3.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:23, 3.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:52, 464kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:27, 544kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:49, 726kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:21, 973kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:00, 1.29MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:46, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:36, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:30, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:40, 757kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:32, 824kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:10, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:52, 1.43MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:40, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:32, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:25, 2.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:22, 3.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:04, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<01:06, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:50, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:31, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:24, 2.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:20, 3.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:18, 864kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:32, 733kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:12, 931kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:56, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:41, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:34, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:22, 2.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:18, 3.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:52, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:54, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:43, 1.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:34, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:26, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:22, 2.69MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:20, 2.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:19, 3.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:18, 3.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:18, 3.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:18, 3.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:59, 1.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:57, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:43, 1.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:21, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:17, 3.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:15, 3.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:42, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:00, 917kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:49, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:37, 1.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:23, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:18, 2.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:15, 3.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:13, 3.80MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:27, 588kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<01:14, 683kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:55, 905kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:41, 1.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:31, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:19, 2.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:17, 2.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:14, 3.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:57, 823kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:52, 887kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:40, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:29, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:23, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:18, 2.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:14, 2.97MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:12, 3.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:49, 874kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:57, 741kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:45, 947kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:32, 1.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:15, 2.64MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:12, 3.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:28, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:11<00:39, 981kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:32, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:24, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:09, 3.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<01:24, 413kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<01:14, 467kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:55, 617kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:39, 854kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:29, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:15, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:26, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:32, 946kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:25, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:18, 1.57MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:10, 2.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:08, 3.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:24, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:27, 962kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:21, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:15, 1.65MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:08, 2.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:06, 3.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:06, 335kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:54, 409kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:39, 556kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:26, 773kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:18, 1.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:12, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:23, 779kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:21, 825kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:16, 1.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:11, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 3.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:20, 694kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:18, 758kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:13, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:09, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:10, 946kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:14, 685kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:11, 829kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:07, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 736kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:07, 772kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:05, 1.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.39MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 422kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:02, 506kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 685kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<19:38:28,  5.66it/s]  0%|          | 1043/400000 [00:00<13:42:58,  8.08it/s]  1%|          | 2103/400000 [00:00<9:34:44, 11.54it/s]   1%|          | 3192/400000 [00:00<6:41:23, 16.48it/s]  1%|          | 4177/400000 [00:00<4:40:28, 23.52it/s]  1%|▏         | 5182/400000 [00:00<3:16:02, 33.57it/s]  2%|▏         | 6110/400000 [00:00<2:17:06, 47.88it/s]  2%|▏         | 7092/400000 [00:00<1:35:56, 68.26it/s]  2%|▏         | 8181/400000 [00:00<1:07:09, 97.25it/s]  2%|▏         | 9255/400000 [00:01<47:03, 138.39it/s]   3%|▎         | 10322/400000 [00:01<33:02, 196.60it/s]  3%|▎         | 11387/400000 [00:01<23:14, 278.65it/s]  3%|▎         | 12418/400000 [00:01<16:24, 393.50it/s]  3%|▎         | 13459/400000 [00:01<11:38, 553.17it/s]  4%|▎         | 14496/400000 [00:01<08:18, 772.58it/s]  4%|▍         | 15531/400000 [00:01<05:59, 1068.82it/s]  4%|▍         | 16560/400000 [00:01<04:22, 1460.95it/s]  4%|▍         | 17587/400000 [00:01<03:14, 1967.11it/s]  5%|▍         | 18637/400000 [00:01<02:26, 2601.14it/s]  5%|▍         | 19679/400000 [00:02<01:53, 3356.61it/s]  5%|▌         | 20715/400000 [00:02<01:30, 4208.48it/s]  5%|▌         | 21782/400000 [00:02<01:13, 5142.66it/s]  6%|▌         | 22890/400000 [00:02<01:01, 6126.88it/s]  6%|▌         | 23953/400000 [00:02<00:53, 7008.08it/s]  6%|▋         | 25014/400000 [00:02<00:48, 7657.81it/s]  7%|▋         | 26050/400000 [00:02<00:45, 8146.75it/s]  7%|▋         | 27063/400000 [00:02<00:43, 8532.83it/s]  7%|▋         | 28060/400000 [00:02<00:42, 8758.21it/s]  7%|▋         | 29038/400000 [00:03<00:41, 8898.70it/s]  8%|▊         | 30015/400000 [00:03<00:40, 9142.48it/s]  8%|▊         | 31035/400000 [00:03<00:39, 9435.72it/s]  8%|▊         | 32018/400000 [00:03<00:40, 9130.88it/s]  8%|▊         | 33074/400000 [00:03<00:38, 9517.10it/s]  9%|▊         | 34113/400000 [00:03<00:37, 9762.40it/s]  9%|▉         | 35167/400000 [00:03<00:36, 9982.79it/s]  9%|▉         | 36215/400000 [00:03<00:35, 10125.04it/s]  9%|▉         | 37245/400000 [00:03<00:35, 10176.06it/s] 10%|▉         | 38273/400000 [00:03<00:35, 10204.65it/s] 10%|▉         | 39299/400000 [00:04<00:36, 9999.33it/s]  10%|█         | 40304/400000 [00:04<00:35, 10014.05it/s] 10%|█         | 41309/400000 [00:04<00:36, 9803.33it/s]  11%|█         | 42293/400000 [00:04<00:37, 9608.75it/s] 11%|█         | 43258/400000 [00:04<00:37, 9606.87it/s] 11%|█         | 44262/400000 [00:04<00:36, 9731.69it/s] 11%|█▏        | 45238/400000 [00:04<00:36, 9726.34it/s] 12%|█▏        | 46246/400000 [00:04<00:35, 9828.71it/s] 12%|█▏        | 47249/400000 [00:04<00:35, 9886.84it/s] 12%|█▏        | 48239/400000 [00:04<00:35, 9825.42it/s] 12%|█▏        | 49261/400000 [00:05<00:35, 9939.68it/s] 13%|█▎        | 50318/400000 [00:05<00:34, 10120.12it/s] 13%|█▎        | 51337/400000 [00:05<00:34, 10139.02it/s] 13%|█▎        | 52352/400000 [00:05<00:34, 10080.42it/s] 13%|█▎        | 53454/400000 [00:05<00:33, 10344.16it/s] 14%|█▎        | 54542/400000 [00:05<00:32, 10497.19it/s] 14%|█▍        | 55594/400000 [00:05<00:34, 10032.41it/s] 14%|█▍        | 56669/400000 [00:05<00:33, 10236.07it/s] 14%|█▍        | 57719/400000 [00:05<00:33, 10313.60it/s] 15%|█▍        | 58755/400000 [00:05<00:33, 10110.75it/s] 15%|█▍        | 59780/400000 [00:06<00:33, 10150.48it/s] 15%|█▌        | 60798/400000 [00:06<00:33, 10104.38it/s] 15%|█▌        | 61822/400000 [00:06<00:33, 10144.61it/s] 16%|█▌        | 62859/400000 [00:06<00:33, 10208.96it/s] 16%|█▌        | 63881/400000 [00:06<00:32, 10207.70it/s] 16%|█▌        | 64903/400000 [00:06<00:33, 10114.63it/s] 16%|█▋        | 65917/400000 [00:06<00:33, 10121.08it/s] 17%|█▋        | 66952/400000 [00:06<00:32, 10187.15it/s] 17%|█▋        | 67972/400000 [00:06<00:32, 10182.75it/s] 17%|█▋        | 69026/400000 [00:06<00:32, 10287.17it/s] 18%|█▊        | 70071/400000 [00:07<00:31, 10335.33it/s] 18%|█▊        | 71105/400000 [00:07<00:31, 10317.80it/s] 18%|█▊        | 72164/400000 [00:07<00:31, 10396.51it/s] 18%|█▊        | 73204/400000 [00:07<00:31, 10306.97it/s] 19%|█▊        | 74251/400000 [00:07<00:31, 10355.11it/s] 19%|█▉        | 75287/400000 [00:07<00:31, 10240.91it/s] 19%|█▉        | 76312/400000 [00:07<00:32, 10112.48it/s] 19%|█▉        | 77367/400000 [00:07<00:31, 10239.67it/s] 20%|█▉        | 78423/400000 [00:07<00:31, 10333.09it/s] 20%|█▉        | 79482/400000 [00:08<00:30, 10407.09it/s] 20%|██        | 80524/400000 [00:08<00:32, 9980.09it/s]  20%|██        | 81527/400000 [00:08<00:32, 9802.69it/s] 21%|██        | 82582/400000 [00:08<00:31, 10014.06it/s] 21%|██        | 83615/400000 [00:08<00:31, 10104.90it/s] 21%|██        | 84630/400000 [00:08<00:31, 10116.86it/s] 21%|██▏       | 85645/400000 [00:08<00:31, 10126.13it/s] 22%|██▏       | 86694/400000 [00:08<00:30, 10230.20it/s] 22%|██▏       | 87757/400000 [00:08<00:30, 10345.29it/s] 22%|██▏       | 88817/400000 [00:08<00:29, 10419.28it/s] 22%|██▏       | 89860/400000 [00:09<00:30, 10331.71it/s] 23%|██▎       | 90938/400000 [00:09<00:29, 10460.17it/s] 23%|██▎       | 91997/400000 [00:09<00:29, 10498.45it/s] 23%|██▎       | 93048/400000 [00:09<00:29, 10394.60it/s] 24%|██▎       | 94112/400000 [00:09<00:29, 10464.68it/s] 24%|██▍       | 95204/400000 [00:09<00:28, 10594.72it/s] 24%|██▍       | 96265/400000 [00:09<00:29, 10464.45it/s] 24%|██▍       | 97313/400000 [00:09<00:29, 10434.48it/s] 25%|██▍       | 98358/400000 [00:09<00:29, 10284.14it/s] 25%|██▍       | 99388/400000 [00:09<00:29, 10208.74it/s] 25%|██▌       | 100415/400000 [00:10<00:29, 10224.82it/s] 25%|██▌       | 101450/400000 [00:10<00:29, 10260.93it/s] 26%|██▌       | 102477/400000 [00:10<00:29, 10229.70it/s] 26%|██▌       | 103501/400000 [00:10<00:29, 10028.89it/s] 26%|██▌       | 104569/400000 [00:10<00:28, 10213.57it/s] 26%|██▋       | 105636/400000 [00:10<00:28, 10344.98it/s] 27%|██▋       | 106673/400000 [00:10<00:28, 10320.48it/s] 27%|██▋       | 107707/400000 [00:10<00:29, 10053.75it/s] 27%|██▋       | 108741/400000 [00:10<00:28, 10137.48it/s] 27%|██▋       | 109821/400000 [00:10<00:28, 10327.45it/s] 28%|██▊       | 110916/400000 [00:11<00:27, 10504.87it/s] 28%|██▊       | 111979/400000 [00:11<00:27, 10540.68it/s] 28%|██▊       | 113053/400000 [00:11<00:27, 10598.96it/s] 29%|██▊       | 114115/400000 [00:11<00:27, 10501.14it/s] 29%|██▉       | 115167/400000 [00:11<00:27, 10448.47it/s] 29%|██▉       | 116213/400000 [00:11<00:27, 10243.16it/s] 29%|██▉       | 117239/400000 [00:11<00:27, 10213.51it/s] 30%|██▉       | 118306/400000 [00:11<00:27, 10344.43it/s] 30%|██▉       | 119357/400000 [00:11<00:27, 10392.48it/s] 30%|███       | 120428/400000 [00:11<00:26, 10484.42it/s] 30%|███       | 121478/400000 [00:12<00:26, 10488.77it/s] 31%|███       | 122528/400000 [00:12<00:26, 10475.06it/s] 31%|███       | 123576/400000 [00:12<00:27, 10231.27it/s] 31%|███       | 124601/400000 [00:12<00:27, 10184.31it/s] 31%|███▏      | 125673/400000 [00:12<00:26, 10336.94it/s] 32%|███▏      | 126709/400000 [00:12<00:27, 9999.33it/s]  32%|███▏      | 127713/400000 [00:12<00:27, 9799.27it/s] 32%|███▏      | 128747/400000 [00:12<00:27, 9953.82it/s] 32%|███▏      | 129824/400000 [00:12<00:26, 10182.29it/s] 33%|███▎      | 130860/400000 [00:13<00:26, 10233.89it/s] 33%|███▎      | 131886/400000 [00:13<00:26, 10199.18it/s] 33%|███▎      | 132908/400000 [00:13<00:27, 9763.85it/s]  33%|███▎      | 133890/400000 [00:13<00:27, 9773.95it/s] 34%|███▎      | 134917/400000 [00:13<00:26, 9914.52it/s] 34%|███▍      | 135974/400000 [00:13<00:26, 10100.12it/s] 34%|███▍      | 137051/400000 [00:13<00:25, 10289.53it/s] 35%|███▍      | 138096/400000 [00:13<00:25, 10334.08it/s] 35%|███▍      | 139145/400000 [00:13<00:25, 10379.45it/s] 35%|███▌      | 140250/400000 [00:13<00:24, 10571.36it/s] 35%|███▌      | 141310/400000 [00:14<00:24, 10553.41it/s] 36%|███▌      | 142377/400000 [00:14<00:24, 10587.18it/s] 36%|███▌      | 143455/400000 [00:14<00:24, 10644.28it/s] 36%|███▌      | 144521/400000 [00:14<00:24, 10407.58it/s] 36%|███▋      | 145564/400000 [00:14<00:24, 10369.04it/s] 37%|███▋      | 146603/400000 [00:14<00:24, 10163.87it/s] 37%|███▋      | 147622/400000 [00:14<00:25, 10086.93it/s] 37%|███▋      | 148633/400000 [00:14<00:25, 10027.74it/s] 37%|███▋      | 149637/400000 [00:14<00:25, 9975.20it/s]  38%|███▊      | 150636/400000 [00:14<00:25, 9961.18it/s] 38%|███▊      | 151699/400000 [00:15<00:24, 10151.55it/s] 38%|███▊      | 152733/400000 [00:15<00:24, 10204.86it/s] 38%|███▊      | 153806/400000 [00:15<00:23, 10355.83it/s] 39%|███▊      | 154864/400000 [00:15<00:23, 10421.37it/s] 39%|███▉      | 155908/400000 [00:15<00:24, 10152.33it/s] 39%|███▉      | 156930/400000 [00:15<00:23, 10169.95it/s] 39%|███▉      | 157975/400000 [00:15<00:23, 10251.05it/s] 40%|███▉      | 159002/400000 [00:15<00:23, 10105.60it/s] 40%|████      | 160020/400000 [00:15<00:23, 10127.55it/s] 40%|████      | 161087/400000 [00:15<00:23, 10282.68it/s] 41%|████      | 162161/400000 [00:16<00:22, 10415.51it/s] 41%|████      | 163204/400000 [00:16<00:23, 10189.57it/s] 41%|████      | 164247/400000 [00:16<00:22, 10257.54it/s] 41%|████▏     | 165310/400000 [00:16<00:22, 10364.95it/s] 42%|████▏     | 166348/400000 [00:16<00:22, 10204.17it/s] 42%|████▏     | 167383/400000 [00:16<00:22, 10246.18it/s] 42%|████▏     | 168409/400000 [00:16<00:22, 10238.88it/s] 42%|████▏     | 169434/400000 [00:16<00:22, 10210.81it/s] 43%|████▎     | 170470/400000 [00:16<00:22, 10253.99it/s] 43%|████▎     | 171554/400000 [00:16<00:21, 10422.89it/s] 43%|████▎     | 172598/400000 [00:17<00:21, 10384.34it/s] 43%|████▎     | 173638/400000 [00:17<00:21, 10333.93it/s] 44%|████▎     | 174672/400000 [00:17<00:21, 10300.91it/s] 44%|████▍     | 175716/400000 [00:17<00:21, 10340.63it/s] 44%|████▍     | 176751/400000 [00:17<00:21, 10179.07it/s] 44%|████▍     | 177811/400000 [00:17<00:21, 10299.47it/s] 45%|████▍     | 178842/400000 [00:17<00:22, 10011.80it/s] 45%|████▍     | 179846/400000 [00:17<00:22, 9917.84it/s]  45%|████▌     | 180845/400000 [00:17<00:22, 9938.99it/s] 45%|████▌     | 181847/400000 [00:17<00:21, 9959.96it/s] 46%|████▌     | 182844/400000 [00:18<00:22, 9575.44it/s] 46%|████▌     | 183835/400000 [00:18<00:22, 9672.32it/s] 46%|████▌     | 184852/400000 [00:18<00:21, 9815.16it/s] 46%|████▋     | 185874/400000 [00:18<00:21, 9930.46it/s] 47%|████▋     | 186877/400000 [00:18<00:21, 9958.21it/s] 47%|████▋     | 187875/400000 [00:18<00:21, 9940.33it/s] 47%|████▋     | 188928/400000 [00:18<00:20, 10110.04it/s] 47%|████▋     | 189941/400000 [00:18<00:20, 10112.93it/s] 48%|████▊     | 190993/400000 [00:18<00:20, 10228.52it/s] 48%|████▊     | 192017/400000 [00:19<00:20, 10230.12it/s] 48%|████▊     | 193107/400000 [00:19<00:19, 10421.95it/s] 49%|████▊     | 194197/400000 [00:19<00:19, 10559.02it/s] 49%|████▉     | 195255/400000 [00:19<00:19, 10453.96it/s] 49%|████▉     | 196304/400000 [00:19<00:19, 10462.45it/s] 49%|████▉     | 197352/400000 [00:19<00:20, 9789.73it/s]  50%|████▉     | 198341/400000 [00:19<00:20, 9752.38it/s] 50%|████▉     | 199323/400000 [00:19<00:21, 9552.48it/s] 50%|█████     | 200284/400000 [00:19<00:20, 9552.01it/s] 50%|█████     | 201320/400000 [00:19<00:20, 9779.62it/s] 51%|█████     | 202414/400000 [00:20<00:19, 10100.04it/s] 51%|█████     | 203518/400000 [00:20<00:18, 10362.67it/s] 51%|█████     | 204586/400000 [00:20<00:18, 10453.79it/s] 51%|█████▏    | 205636/400000 [00:20<00:18, 10355.52it/s] 52%|█████▏    | 206675/400000 [00:20<00:18, 10270.81it/s] 52%|█████▏    | 207705/400000 [00:20<00:18, 10262.80it/s] 52%|█████▏    | 208733/400000 [00:20<00:18, 10261.74it/s] 52%|█████▏    | 209761/400000 [00:20<00:19, 9965.32it/s]  53%|█████▎    | 210761/400000 [00:20<00:19, 9935.83it/s] 53%|█████▎    | 211769/400000 [00:20<00:18, 9977.54it/s] 53%|█████▎    | 212801/400000 [00:21<00:18, 10077.41it/s] 53%|█████▎    | 213890/400000 [00:21<00:18, 10306.38it/s] 54%|█████▎    | 214960/400000 [00:21<00:17, 10420.07it/s] 54%|█████▍    | 216004/400000 [00:21<00:18, 10197.33it/s] 54%|█████▍    | 217027/400000 [00:21<00:18, 9954.17it/s]  55%|█████▍    | 218036/400000 [00:21<00:18, 9994.13it/s] 55%|█████▍    | 219103/400000 [00:21<00:17, 10185.47it/s] 55%|█████▌    | 220124/400000 [00:21<00:17, 10160.45it/s] 55%|█████▌    | 221190/400000 [00:21<00:17, 10304.29it/s] 56%|█████▌    | 222265/400000 [00:21<00:17, 10433.38it/s] 56%|█████▌    | 223310/400000 [00:22<00:16, 10426.29it/s] 56%|█████▌    | 224379/400000 [00:22<00:16, 10502.46it/s] 56%|█████▋    | 225431/400000 [00:22<00:16, 10476.09it/s] 57%|█████▋    | 226480/400000 [00:22<00:17, 10100.27it/s] 57%|█████▋    | 227494/400000 [00:22<00:17, 9963.39it/s]  57%|█████▋    | 228526/400000 [00:22<00:17, 10066.50it/s] 57%|█████▋    | 229606/400000 [00:22<00:16, 10273.79it/s] 58%|█████▊    | 230644/400000 [00:22<00:16, 10303.25it/s] 58%|█████▊    | 231677/400000 [00:22<00:16, 10300.57it/s] 58%|█████▊    | 232765/400000 [00:23<00:15, 10467.53it/s] 58%|█████▊    | 233831/400000 [00:23<00:15, 10523.79it/s] 59%|█████▊    | 234889/400000 [00:23<00:15, 10537.53it/s] 59%|█████▉    | 235944/400000 [00:23<00:15, 10301.96it/s] 59%|█████▉    | 236994/400000 [00:23<00:15, 10359.00it/s] 60%|█████▉    | 238043/400000 [00:23<00:15, 10396.21it/s] 60%|█████▉    | 239084/400000 [00:23<00:15, 10280.89it/s] 60%|██████    | 240114/400000 [00:23<00:16, 9928.08it/s]  60%|██████    | 241111/400000 [00:23<00:16, 9671.51it/s] 61%|██████    | 242082/400000 [00:23<00:16, 9582.78it/s] 61%|██████    | 243097/400000 [00:24<00:16, 9745.67it/s] 61%|██████    | 244140/400000 [00:24<00:15, 9940.91it/s] 61%|██████▏   | 245219/400000 [00:24<00:15, 10181.13it/s] 62%|██████▏   | 246241/400000 [00:24<00:15, 10073.02it/s] 62%|██████▏   | 247260/400000 [00:24<00:15, 10107.11it/s] 62%|██████▏   | 248324/400000 [00:24<00:14, 10260.16it/s] 62%|██████▏   | 249353/400000 [00:24<00:14, 10266.02it/s] 63%|██████▎   | 250406/400000 [00:24<00:14, 10341.58it/s] 63%|██████▎   | 251460/400000 [00:24<00:14, 10398.05it/s] 63%|██████▎   | 252501/400000 [00:24<00:14, 10289.40it/s] 63%|██████▎   | 253567/400000 [00:25<00:14, 10397.67it/s] 64%|██████▎   | 254641/400000 [00:25<00:13, 10497.74it/s] 64%|██████▍   | 255692/400000 [00:25<00:13, 10491.87it/s] 64%|██████▍   | 256742/400000 [00:25<00:13, 10298.31it/s] 64%|██████▍   | 257787/400000 [00:25<00:13, 10343.29it/s] 65%|██████▍   | 258831/400000 [00:25<00:13, 10371.42it/s] 65%|██████▍   | 259869/400000 [00:25<00:13, 10343.09it/s] 65%|██████▌   | 260904/400000 [00:25<00:13, 10226.10it/s] 65%|██████▌   | 261928/400000 [00:25<00:13, 10176.14it/s] 66%|██████▌   | 262998/400000 [00:25<00:13, 10325.72it/s] 66%|██████▌   | 264045/400000 [00:26<00:13, 10368.48it/s] 66%|██████▋   | 265127/400000 [00:26<00:12, 10498.53it/s] 67%|██████▋   | 266216/400000 [00:26<00:12, 10609.64it/s] 67%|██████▋   | 267292/400000 [00:26<00:12, 10650.90it/s] 67%|██████▋   | 268358/400000 [00:26<00:12, 10489.67it/s] 67%|██████▋   | 269408/400000 [00:26<00:12, 10069.50it/s] 68%|██████▊   | 270420/400000 [00:26<00:13, 9681.63it/s]  68%|██████▊   | 271424/400000 [00:26<00:13, 9785.23it/s] 68%|██████▊   | 272463/400000 [00:26<00:12, 9956.36it/s] 68%|██████▊   | 273480/400000 [00:26<00:12, 10018.69it/s] 69%|██████▊   | 274530/400000 [00:27<00:12, 10157.50it/s] 69%|██████▉   | 275552/400000 [00:27<00:12, 10175.62it/s] 69%|██████▉   | 276586/400000 [00:27<00:12, 10221.33it/s] 69%|██████▉   | 277610/400000 [00:27<00:12, 10171.65it/s] 70%|██████▉   | 278629/400000 [00:27<00:11, 10125.03it/s] 70%|██████▉   | 279643/400000 [00:27<00:11, 10115.94it/s] 70%|███████   | 280656/400000 [00:27<00:11, 10059.40it/s] 70%|███████   | 281693/400000 [00:27<00:11, 10148.27it/s] 71%|███████   | 282723/400000 [00:27<00:11, 10192.72it/s] 71%|███████   | 283768/400000 [00:28<00:11, 10268.39it/s] 71%|███████   | 284825/400000 [00:28<00:11, 10356.95it/s] 71%|███████▏  | 285862/400000 [00:28<00:11, 10132.96it/s] 72%|███████▏  | 286909/400000 [00:28<00:11, 10230.80it/s] 72%|███████▏  | 287969/400000 [00:28<00:10, 10337.70it/s] 72%|███████▏  | 289004/400000 [00:28<00:10, 10188.27it/s] 73%|███████▎  | 290025/400000 [00:28<00:10, 10147.12it/s] 73%|███████▎  | 291093/400000 [00:28<00:10, 10299.50it/s] 73%|███████▎  | 292125/400000 [00:28<00:10, 10287.42it/s] 73%|███████▎  | 293194/400000 [00:28<00:10, 10403.01it/s] 74%|███████▎  | 294236/400000 [00:29<00:10, 10151.67it/s] 74%|███████▍  | 295339/400000 [00:29<00:10, 10399.13it/s] 74%|███████▍  | 296442/400000 [00:29<00:09, 10579.57it/s] 74%|███████▍  | 297503/400000 [00:29<00:09, 10501.25it/s] 75%|███████▍  | 298620/400000 [00:29<00:09, 10691.89it/s] 75%|███████▍  | 299692/400000 [00:29<00:09, 10648.50it/s] 75%|███████▌  | 300759/400000 [00:29<00:09, 10567.83it/s] 75%|███████▌  | 301871/400000 [00:29<00:09, 10727.23it/s] 76%|███████▌  | 302946/400000 [00:29<00:09, 10726.07it/s] 76%|███████▌  | 304020/400000 [00:29<00:09, 10591.06it/s] 76%|███████▋  | 305081/400000 [00:30<00:09, 10456.64it/s] 77%|███████▋  | 306145/400000 [00:30<00:08, 10508.74it/s] 77%|███████▋  | 307197/400000 [00:30<00:09, 10277.32it/s] 77%|███████▋  | 308243/400000 [00:30<00:08, 10330.94it/s] 77%|███████▋  | 309308/400000 [00:30<00:08, 10422.47it/s] 78%|███████▊  | 310352/400000 [00:30<00:08, 10375.08it/s] 78%|███████▊  | 311391/400000 [00:30<00:08, 10311.01it/s] 78%|███████▊  | 312454/400000 [00:30<00:08, 10404.48it/s] 78%|███████▊  | 313523/400000 [00:30<00:08, 10487.48it/s] 79%|███████▊  | 314638/400000 [00:30<00:07, 10676.36it/s] 79%|███████▉  | 315707/400000 [00:31<00:07, 10636.77it/s] 79%|███████▉  | 316772/400000 [00:31<00:07, 10631.78it/s] 79%|███████▉  | 317881/400000 [00:31<00:07, 10763.74it/s] 80%|███████▉  | 318959/400000 [00:31<00:07, 10551.57it/s] 80%|████████  | 320025/400000 [00:31<00:07, 10582.88it/s] 80%|████████  | 321085/400000 [00:31<00:07, 10502.95it/s] 81%|████████  | 322137/400000 [00:31<00:07, 10376.92it/s] 81%|████████  | 323212/400000 [00:31<00:07, 10482.88it/s] 81%|████████  | 324262/400000 [00:31<00:07, 10157.75it/s] 81%|████████▏ | 325281/400000 [00:31<00:07, 9840.20it/s]  82%|████████▏ | 326270/400000 [00:32<00:07, 9790.55it/s] 82%|████████▏ | 327263/400000 [00:32<00:07, 9831.81it/s] 82%|████████▏ | 328319/400000 [00:32<00:07, 10037.80it/s] 82%|████████▏ | 329408/400000 [00:32<00:06, 10276.45it/s] 83%|████████▎ | 330525/400000 [00:32<00:06, 10527.51it/s] 83%|████████▎ | 331582/400000 [00:32<00:06, 10356.01it/s] 83%|████████▎ | 332636/400000 [00:32<00:06, 10407.85it/s] 83%|████████▎ | 333746/400000 [00:32<00:06, 10604.56it/s] 84%|████████▎ | 334831/400000 [00:32<00:06, 10675.84it/s] 84%|████████▍ | 335901/400000 [00:33<00:06, 10353.72it/s] 84%|████████▍ | 336940/400000 [00:33<00:06, 10116.86it/s] 85%|████████▍ | 338067/400000 [00:33<00:05, 10436.80it/s] 85%|████████▍ | 339116/400000 [00:33<00:05, 10186.40it/s] 85%|████████▌ | 340140/400000 [00:33<00:06, 9932.23it/s]  85%|████████▌ | 341140/400000 [00:33<00:05, 9952.20it/s] 86%|████████▌ | 342139/400000 [00:33<00:05, 9929.56it/s] 86%|████████▌ | 343135/400000 [00:33<00:05, 9726.64it/s] 86%|████████▌ | 344141/400000 [00:33<00:05, 9823.97it/s] 86%|████████▋ | 345126/400000 [00:33<00:05, 9752.72it/s] 87%|████████▋ | 346103/400000 [00:34<00:05, 9539.86it/s] 87%|████████▋ | 347060/400000 [00:34<00:05, 9516.37it/s] 87%|████████▋ | 348014/400000 [00:34<00:05, 9361.55it/s] 87%|████████▋ | 348969/400000 [00:34<00:05, 9411.61it/s] 87%|████████▋ | 349954/400000 [00:34<00:05, 9537.78it/s] 88%|████████▊ | 350986/400000 [00:34<00:05, 9758.17it/s] 88%|████████▊ | 352015/400000 [00:34<00:04, 9910.71it/s] 88%|████████▊ | 353046/400000 [00:34<00:04, 10025.21it/s] 89%|████████▊ | 354051/400000 [00:34<00:04, 9972.36it/s]  89%|████████▉ | 355050/400000 [00:34<00:04, 9910.10it/s] 89%|████████▉ | 356043/400000 [00:35<00:04, 9880.89it/s] 89%|████████▉ | 357032/400000 [00:35<00:04, 9752.47it/s] 90%|████████▉ | 358035/400000 [00:35<00:04, 9832.06it/s] 90%|████████▉ | 359019/400000 [00:35<00:04, 9659.98it/s] 90%|█████████ | 360033/400000 [00:35<00:04, 9799.11it/s] 90%|█████████ | 361015/400000 [00:35<00:03, 9805.24it/s] 90%|█████████ | 361997/400000 [00:35<00:03, 9741.28it/s] 91%|█████████ | 362979/400000 [00:35<00:03, 9764.56it/s] 91%|█████████ | 363996/400000 [00:35<00:03, 9880.40it/s] 91%|█████████ | 364985/400000 [00:35<00:03, 9769.90it/s] 91%|█████████▏| 365963/400000 [00:36<00:03, 9503.31it/s] 92%|█████████▏| 366916/400000 [00:36<00:03, 9439.58it/s] 92%|█████████▏| 367862/400000 [00:36<00:03, 9338.59it/s] 92%|█████████▏| 368812/400000 [00:36<00:03, 9384.30it/s] 92%|█████████▏| 369768/400000 [00:36<00:03, 9435.29it/s] 93%|█████████▎| 370713/400000 [00:36<00:03, 9360.67it/s] 93%|█████████▎| 371650/400000 [00:36<00:03, 9236.93it/s] 93%|█████████▎| 372575/400000 [00:36<00:03, 8915.19it/s] 93%|█████████▎| 373557/400000 [00:36<00:02, 9167.47it/s] 94%|█████████▎| 374512/400000 [00:37<00:02, 9277.41it/s] 94%|█████████▍| 375479/400000 [00:37<00:02, 9390.76it/s] 94%|█████████▍| 376427/400000 [00:37<00:02, 9415.49it/s] 94%|█████████▍| 377409/400000 [00:37<00:02, 9532.55it/s] 95%|█████████▍| 378382/400000 [00:37<00:02, 9588.35it/s] 95%|█████████▍| 379343/400000 [00:37<00:02, 9552.26it/s] 95%|█████████▌| 380300/400000 [00:37<00:02, 9377.88it/s] 95%|█████████▌| 381240/400000 [00:37<00:02, 9245.05it/s] 96%|█████████▌| 382239/400000 [00:37<00:01, 9455.11it/s] 96%|█████████▌| 383249/400000 [00:37<00:01, 9637.80it/s] 96%|█████████▌| 384302/400000 [00:38<00:01, 9887.03it/s] 96%|█████████▋| 385297/400000 [00:38<00:01, 9904.19it/s] 97%|█████████▋| 386327/400000 [00:38<00:01, 10017.80it/s] 97%|█████████▋| 387340/400000 [00:38<00:01, 10050.09it/s] 97%|█████████▋| 388347/400000 [00:38<00:01, 9914.68it/s]  97%|█████████▋| 389349/400000 [00:38<00:01, 9945.12it/s] 98%|█████████▊| 390355/400000 [00:38<00:00, 9976.75it/s] 98%|█████████▊| 391354/400000 [00:38<00:00, 9953.28it/s] 98%|█████████▊| 392361/400000 [00:38<00:00, 9986.18it/s] 98%|█████████▊| 393385/400000 [00:38<00:00, 10060.08it/s] 99%|█████████▊| 394392/400000 [00:39<00:00, 10053.34it/s] 99%|█████████▉| 395398/400000 [00:39<00:00, 10045.77it/s] 99%|█████████▉| 396427/400000 [00:39<00:00, 10114.76it/s] 99%|█████████▉| 397439/400000 [00:39<00:00, 9994.68it/s] 100%|█████████▉| 398439/400000 [00:39<00:00, 9981.31it/s]100%|█████████▉| 399438/400000 [00:39<00:00, 9842.58it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10098.60it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f9e600726d8>, <torchtext.data.dataset.TabularDataset object at 0x7f9e60072828>, <torchtext.vocab.Vocab object at 0x7f9e60072748>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.49 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.49 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.49 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]2020-06-21 18:17:17.029919: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-21 18:17:17.033755: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-06-21 18:17:17.033914: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad39726b30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 18:17:17.033926: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3792896/9912422 [00:00<00:00, 37429085.02it/s]9920512it [00:00, 10054694.32it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1240748.51it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:01<?, ?it/s]1654784it [00:01, 1041798.23it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 14892.50it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
