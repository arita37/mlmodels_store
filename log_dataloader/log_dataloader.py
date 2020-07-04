
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7041dde0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7041dde0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f70ad12b1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f70ad12b1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f70cb471e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f70cb471e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f705a459620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f705a459620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f705a459620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3858432/9912422 [00:00<00:00, 38524929.41it/s]9920512it [00:00, 15131857.82it/s]                             
0it [00:00, ?it/s]32768it [00:00, 688354.64it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 3051431.53it/s]          
0it [00:00, ?it/s]8192it [00:00, 242258.31it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f705784b940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f704148dbe0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f705a459268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f705a459268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f705a459268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:47,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 12.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 43.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 66.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 88.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 88.45 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.17s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 88.45 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.32 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.34s/ url]
0 examples [00:00, ? examples/s]2020-07-04 06:11:18.727501: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-04 06:11:18.743083: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-04 06:11:18.743290: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612ff41f010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 06:11:18.743384: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
44 examples [00:00, 437.89 examples/s]153 examples [00:00, 533.41 examples/s]261 examples [00:00, 628.21 examples/s]369 examples [00:00, 718.35 examples/s]479 examples [00:00, 800.99 examples/s]587 examples [00:00, 867.58 examples/s]697 examples [00:00, 924.87 examples/s]799 examples [00:00, 950.09 examples/s]907 examples [00:00, 983.48 examples/s]1009 examples [00:01, 992.51 examples/s]1118 examples [00:01, 1017.16 examples/s]1223 examples [00:01, 1025.30 examples/s]1327 examples [00:01, 1023.51 examples/s]1434 examples [00:01, 1036.10 examples/s]1544 examples [00:01, 1053.26 examples/s]1654 examples [00:01, 1066.22 examples/s]1763 examples [00:01, 1072.20 examples/s]1871 examples [00:01, 1069.69 examples/s]1979 examples [00:01, 1052.16 examples/s]2087 examples [00:02, 1058.76 examples/s]2196 examples [00:02, 1066.22 examples/s]2305 examples [00:02, 1070.92 examples/s]2413 examples [00:02, 1069.60 examples/s]2521 examples [00:02, 1058.32 examples/s]2631 examples [00:02, 1067.94 examples/s]2738 examples [00:02, 1054.71 examples/s]2844 examples [00:02, 1039.14 examples/s]2949 examples [00:02, 1038.69 examples/s]3054 examples [00:02, 1041.11 examples/s]3161 examples [00:03, 1049.52 examples/s]3270 examples [00:03, 1059.19 examples/s]3377 examples [00:03, 1061.46 examples/s]3484 examples [00:03, 1061.90 examples/s]3591 examples [00:03, 1062.87 examples/s]3699 examples [00:03, 1067.18 examples/s]3806 examples [00:03, 1055.80 examples/s]3915 examples [00:03, 1063.45 examples/s]4022 examples [00:03, 1062.09 examples/s]4129 examples [00:03, 1058.31 examples/s]4236 examples [00:04, 1060.79 examples/s]4344 examples [00:04, 1065.51 examples/s]4451 examples [00:04, 1060.62 examples/s]4558 examples [00:04, 1055.96 examples/s]4666 examples [00:04, 1061.91 examples/s]4773 examples [00:04, 1062.09 examples/s]4881 examples [00:04, 1065.29 examples/s]4988 examples [00:04, 1063.35 examples/s]5095 examples [00:04, 1063.96 examples/s]5202 examples [00:04, 1065.29 examples/s]5311 examples [00:05, 1070.39 examples/s]5419 examples [00:05, 1071.57 examples/s]5528 examples [00:05, 1074.56 examples/s]5636 examples [00:05, 1057.75 examples/s]5743 examples [00:05, 1060.45 examples/s]5852 examples [00:05, 1068.63 examples/s]5959 examples [00:05, 1058.53 examples/s]6065 examples [00:05, 1023.24 examples/s]6168 examples [00:05, 1014.89 examples/s]6272 examples [00:05, 1022.22 examples/s]6380 examples [00:06, 1036.50 examples/s]6486 examples [00:06, 1041.11 examples/s]6591 examples [00:06, 1041.91 examples/s]6696 examples [00:06, 1008.28 examples/s]6798 examples [00:06, 1007.28 examples/s]6899 examples [00:06, 987.24 examples/s] 7006 examples [00:06, 1008.74 examples/s]7108 examples [00:06, 1010.22 examples/s]7215 examples [00:06, 1027.13 examples/s]7323 examples [00:07, 1041.49 examples/s]7433 examples [00:07, 1056.45 examples/s]7539 examples [00:07, 1055.21 examples/s]7645 examples [00:07, 1038.29 examples/s]7752 examples [00:07, 1045.15 examples/s]7860 examples [00:07, 1053.33 examples/s]7969 examples [00:07, 1064.03 examples/s]8077 examples [00:07, 1067.67 examples/s]8185 examples [00:07, 1069.38 examples/s]8292 examples [00:07, 1063.90 examples/s]8399 examples [00:08, 1037.96 examples/s]8503 examples [00:08, 1004.42 examples/s]8609 examples [00:08, 1020.33 examples/s]8716 examples [00:08, 1033.91 examples/s]8820 examples [00:08, 962.65 examples/s] 8920 examples [00:08, 970.38 examples/s]9018 examples [00:08, 962.71 examples/s]9115 examples [00:08, 941.82 examples/s]9210 examples [00:08, 919.65 examples/s]9303 examples [00:08, 902.76 examples/s]9404 examples [00:09, 931.19 examples/s]9510 examples [00:09, 964.63 examples/s]9618 examples [00:09, 994.69 examples/s]9726 examples [00:09, 1018.39 examples/s]9835 examples [00:09, 1036.75 examples/s]9943 examples [00:09, 1046.77 examples/s]10049 examples [00:09, 1001.78 examples/s]10150 examples [00:09, 999.72 examples/s] 10251 examples [00:09, 932.46 examples/s]10358 examples [00:10, 967.62 examples/s]10456 examples [00:10, 951.20 examples/s]10561 examples [00:10, 977.78 examples/s]10666 examples [00:10, 997.89 examples/s]10768 examples [00:10, 1003.06 examples/s]10869 examples [00:10, 991.56 examples/s] 10978 examples [00:10, 1016.59 examples/s]11081 examples [00:10, 978.50 examples/s] 11180 examples [00:10, 961.23 examples/s]11277 examples [00:10, 955.07 examples/s]11379 examples [00:11, 971.98 examples/s]11486 examples [00:11, 998.36 examples/s]11596 examples [00:11, 1024.35 examples/s]11699 examples [00:11, 1017.06 examples/s]11804 examples [00:11, 1025.27 examples/s]11909 examples [00:11, 1030.73 examples/s]12016 examples [00:11, 1039.71 examples/s]12121 examples [00:11, 1018.30 examples/s]12228 examples [00:11, 1030.66 examples/s]12336 examples [00:11, 1043.39 examples/s]12441 examples [00:12, 1034.77 examples/s]12545 examples [00:12, 1030.47 examples/s]12652 examples [00:12, 1041.57 examples/s]12761 examples [00:12, 1054.10 examples/s]12870 examples [00:12, 1062.02 examples/s]12977 examples [00:12, 1057.45 examples/s]13084 examples [00:12, 1058.35 examples/s]13190 examples [00:12, 1042.56 examples/s]13299 examples [00:12, 1055.48 examples/s]13408 examples [00:13, 1065.52 examples/s]13516 examples [00:13, 1066.00 examples/s]13624 examples [00:13, 1068.78 examples/s]13731 examples [00:13, 1052.20 examples/s]13837 examples [00:13, 1015.52 examples/s]13943 examples [00:13, 1026.41 examples/s]14047 examples [00:13, 1029.71 examples/s]14157 examples [00:13, 1047.50 examples/s]14266 examples [00:13, 1057.02 examples/s]14374 examples [00:13, 1062.73 examples/s]14482 examples [00:14, 1066.59 examples/s]14591 examples [00:14, 1071.73 examples/s]14699 examples [00:14, 1026.12 examples/s]14808 examples [00:14, 1041.79 examples/s]14915 examples [00:14, 1049.24 examples/s]15024 examples [00:14, 1059.89 examples/s]15132 examples [00:14, 1065.33 examples/s]15239 examples [00:14, 1058.92 examples/s]15346 examples [00:14, 1012.34 examples/s]15448 examples [00:14, 1014.23 examples/s]15550 examples [00:15, 1010.34 examples/s]15654 examples [00:15, 1017.55 examples/s]15766 examples [00:15, 1046.20 examples/s]15877 examples [00:15, 1063.74 examples/s]15986 examples [00:15, 1070.67 examples/s]16094 examples [00:15, 1071.36 examples/s]16202 examples [00:15, 1070.37 examples/s]16310 examples [00:15, 1068.08 examples/s]16417 examples [00:15, 1049.18 examples/s]16525 examples [00:15, 1057.90 examples/s]16631 examples [00:16, 1054.86 examples/s]16740 examples [00:16, 1064.69 examples/s]16849 examples [00:16, 1070.54 examples/s]16958 examples [00:16, 1075.70 examples/s]17066 examples [00:16, 1076.32 examples/s]17174 examples [00:16, 1068.35 examples/s]17281 examples [00:16, 1035.60 examples/s]17388 examples [00:16, 1045.67 examples/s]17499 examples [00:16, 1061.85 examples/s]17608 examples [00:17, 1069.72 examples/s]17716 examples [00:17, 1070.47 examples/s]17824 examples [00:17, 1072.61 examples/s]17932 examples [00:17, 1067.16 examples/s]18039 examples [00:17, 1062.55 examples/s]18146 examples [00:17, 1060.50 examples/s]18253 examples [00:17, 1058.21 examples/s]18361 examples [00:17, 1062.82 examples/s]18470 examples [00:17, 1068.21 examples/s]18577 examples [00:17, 1015.89 examples/s]18686 examples [00:18, 1034.73 examples/s]18794 examples [00:18, 1047.29 examples/s]18904 examples [00:18, 1061.43 examples/s]19013 examples [00:18, 1069.71 examples/s]19121 examples [00:18, 1066.12 examples/s]19228 examples [00:18, 1045.01 examples/s]19333 examples [00:18, 1036.03 examples/s]19437 examples [00:18, 1024.01 examples/s]19546 examples [00:18, 1040.34 examples/s]19655 examples [00:18, 1052.37 examples/s]19763 examples [00:19, 1059.74 examples/s]19870 examples [00:19, 1051.39 examples/s]19977 examples [00:19, 1055.58 examples/s]20083 examples [00:19, 977.81 examples/s] 20193 examples [00:19, 1011.45 examples/s]20302 examples [00:19, 1031.99 examples/s]20407 examples [00:19, 995.92 examples/s] 20515 examples [00:19, 1017.85 examples/s]20622 examples [00:19, 1029.81 examples/s]20730 examples [00:19, 1043.32 examples/s]20837 examples [00:20, 1048.56 examples/s]20943 examples [00:20, 1048.22 examples/s]21049 examples [00:20, 1027.04 examples/s]21159 examples [00:20, 1044.83 examples/s]21269 examples [00:20, 1058.17 examples/s]21376 examples [00:20, 1040.01 examples/s]21481 examples [00:20, 1037.04 examples/s]21585 examples [00:20, 1014.44 examples/s]21689 examples [00:20, 1020.75 examples/s]21792 examples [00:21, 1019.96 examples/s]21899 examples [00:21, 1033.33 examples/s]22006 examples [00:21, 1042.50 examples/s]22111 examples [00:21, 1043.80 examples/s]22216 examples [00:21, 1010.04 examples/s]22323 examples [00:21, 1025.61 examples/s]22426 examples [00:21, 1010.30 examples/s]22528 examples [00:21, 981.71 examples/s] 22627 examples [00:21, 977.26 examples/s]22736 examples [00:21, 1006.53 examples/s]22845 examples [00:22, 1028.86 examples/s]22953 examples [00:22, 1041.56 examples/s]23062 examples [00:22, 1054.06 examples/s]23168 examples [00:22, 1050.58 examples/s]23279 examples [00:22, 1065.54 examples/s]23389 examples [00:22, 1075.48 examples/s]23497 examples [00:22, 1066.62 examples/s]23605 examples [00:22, 1067.94 examples/s]23712 examples [00:22, 1030.38 examples/s]23816 examples [00:22, 999.55 examples/s] 23924 examples [00:23, 1022.32 examples/s]24031 examples [00:23, 1035.54 examples/s]24140 examples [00:23, 1050.17 examples/s]24248 examples [00:23, 1056.80 examples/s]24357 examples [00:23, 1064.24 examples/s]24464 examples [00:23, 1021.30 examples/s]24570 examples [00:23, 1031.47 examples/s]24677 examples [00:23, 1042.02 examples/s]24782 examples [00:23, 1043.38 examples/s]24887 examples [00:24, 948.35 examples/s] 24989 examples [00:24, 966.30 examples/s]25087 examples [00:24, 966.08 examples/s]25194 examples [00:24, 994.31 examples/s]25295 examples [00:24, 974.70 examples/s]25405 examples [00:24, 1006.89 examples/s]25507 examples [00:24, 1001.07 examples/s]25608 examples [00:24, 953.04 examples/s] 25710 examples [00:24, 971.39 examples/s]25811 examples [00:24, 981.05 examples/s]25921 examples [00:25, 1011.70 examples/s]26023 examples [00:25, 1009.75 examples/s]26128 examples [00:25, 1020.55 examples/s]26232 examples [00:25, 1025.43 examples/s]26335 examples [00:25, 982.43 examples/s] 26434 examples [00:25, 909.73 examples/s]26527 examples [00:25, 908.83 examples/s]26624 examples [00:25, 924.44 examples/s]26729 examples [00:25, 957.56 examples/s]26834 examples [00:26, 982.69 examples/s]26934 examples [00:26, 919.68 examples/s]27028 examples [00:26, 921.16 examples/s]27122 examples [00:26, 883.24 examples/s]27212 examples [00:26, 873.06 examples/s]27301 examples [00:26, 860.64 examples/s]27395 examples [00:26, 880.34 examples/s]27497 examples [00:26, 916.99 examples/s]27599 examples [00:26, 944.86 examples/s]27706 examples [00:26, 978.09 examples/s]27805 examples [00:27, 978.17 examples/s]27907 examples [00:27, 988.97 examples/s]28014 examples [00:27, 1011.09 examples/s]28116 examples [00:27, 1013.64 examples/s]28226 examples [00:27, 1036.07 examples/s]28335 examples [00:27, 1049.80 examples/s]28444 examples [00:27, 1059.77 examples/s]28552 examples [00:27, 1064.03 examples/s]28659 examples [00:27, 1041.91 examples/s]28764 examples [00:28, 964.58 examples/s] 28862 examples [00:28, 861.72 examples/s]28960 examples [00:28, 893.76 examples/s]29055 examples [00:28, 907.61 examples/s]29153 examples [00:28, 922.93 examples/s]29247 examples [00:28, 897.27 examples/s]29343 examples [00:28, 914.68 examples/s]29436 examples [00:28, 913.15 examples/s]29528 examples [00:28, 891.02 examples/s]29618 examples [00:29, 863.71 examples/s]29720 examples [00:29, 903.33 examples/s]29812 examples [00:29, 841.71 examples/s]29908 examples [00:29, 856.76 examples/s]29997 examples [00:29, 865.98 examples/s]30085 examples [00:29, 777.28 examples/s]30165 examples [00:29, 772.09 examples/s]30259 examples [00:29, 814.16 examples/s]30369 examples [00:29, 881.07 examples/s]30478 examples [00:29, 933.84 examples/s]30586 examples [00:30, 967.31 examples/s]30686 examples [00:30, 877.65 examples/s]30777 examples [00:30, 876.24 examples/s]30872 examples [00:30, 893.38 examples/s]30965 examples [00:30, 893.93 examples/s]31056 examples [00:30, 894.52 examples/s]31147 examples [00:30, 853.45 examples/s]31247 examples [00:30, 890.92 examples/s]31350 examples [00:30, 926.47 examples/s]31459 examples [00:31, 968.49 examples/s]31558 examples [00:31, 947.33 examples/s]31654 examples [00:31, 919.32 examples/s]31747 examples [00:31, 899.97 examples/s]31853 examples [00:31, 942.41 examples/s]31962 examples [00:31, 981.26 examples/s]32069 examples [00:31, 1004.34 examples/s]32176 examples [00:31, 1021.11 examples/s]32279 examples [00:31, 996.01 examples/s] 32380 examples [00:32, 993.93 examples/s]32485 examples [00:32, 1010.03 examples/s]32592 examples [00:32, 1026.25 examples/s]32695 examples [00:32, 1011.91 examples/s]32801 examples [00:32, 1023.32 examples/s]32905 examples [00:32, 1026.03 examples/s]33008 examples [00:32, 1011.67 examples/s]33115 examples [00:32, 1026.58 examples/s]33221 examples [00:32, 1035.49 examples/s]33326 examples [00:32, 1037.87 examples/s]33430 examples [00:33, 1016.77 examples/s]33535 examples [00:33, 1023.99 examples/s]33638 examples [00:33, 1009.30 examples/s]33740 examples [00:33, 997.91 examples/s] 33847 examples [00:33, 1016.63 examples/s]33952 examples [00:33, 1026.15 examples/s]34058 examples [00:33, 1034.98 examples/s]34162 examples [00:33, 1036.11 examples/s]34268 examples [00:33, 1043.03 examples/s]34376 examples [00:33, 1051.26 examples/s]34483 examples [00:34, 1055.15 examples/s]34590 examples [00:34, 1058.88 examples/s]34697 examples [00:34, 1061.15 examples/s]34804 examples [00:34, 1061.17 examples/s]34911 examples [00:34, 1048.50 examples/s]35016 examples [00:34, 1024.73 examples/s]35124 examples [00:34, 1039.63 examples/s]35231 examples [00:34, 1045.66 examples/s]35336 examples [00:34, 1038.60 examples/s]35442 examples [00:34, 1043.16 examples/s]35549 examples [00:35, 1050.12 examples/s]35657 examples [00:35, 1056.76 examples/s]35764 examples [00:35, 1058.03 examples/s]35870 examples [00:35, 1054.50 examples/s]35976 examples [00:35, 1040.60 examples/s]36081 examples [00:35, 1035.38 examples/s]36189 examples [00:35, 1047.26 examples/s]36294 examples [00:35, 1036.45 examples/s]36398 examples [00:35, 994.43 examples/s] 36498 examples [00:36, 894.17 examples/s]36590 examples [00:36, 843.66 examples/s]36690 examples [00:36, 882.22 examples/s]36781 examples [00:36, 830.44 examples/s]36867 examples [00:36, 817.37 examples/s]36961 examples [00:36, 849.20 examples/s]37061 examples [00:36, 889.12 examples/s]37162 examples [00:36, 922.09 examples/s]37268 examples [00:36, 957.47 examples/s]37374 examples [00:36, 984.36 examples/s]37479 examples [00:37, 1002.86 examples/s]37587 examples [00:37, 1021.88 examples/s]37696 examples [00:37, 1039.96 examples/s]37801 examples [00:37, 1033.76 examples/s]37910 examples [00:37, 1049.69 examples/s]38019 examples [00:37, 1059.45 examples/s]38128 examples [00:37, 1067.10 examples/s]38235 examples [00:37, 1046.89 examples/s]38340 examples [00:37, 1013.09 examples/s]38444 examples [00:38, 1018.37 examples/s]38549 examples [00:38, 1026.50 examples/s]38657 examples [00:38, 1040.77 examples/s]38763 examples [00:38, 1046.25 examples/s]38871 examples [00:38, 1054.14 examples/s]38977 examples [00:38, 1043.47 examples/s]39082 examples [00:38, 1034.19 examples/s]39186 examples [00:38, 998.90 examples/s] 39291 examples [00:38, 1011.35 examples/s]39399 examples [00:38, 1029.18 examples/s]39508 examples [00:39, 1045.35 examples/s]39617 examples [00:39, 1056.23 examples/s]39725 examples [00:39, 1062.16 examples/s]39834 examples [00:39, 1068.16 examples/s]39941 examples [00:39, 1010.50 examples/s]40043 examples [00:39, 963.37 examples/s] 40143 examples [00:39, 973.23 examples/s]40250 examples [00:39, 998.48 examples/s]40358 examples [00:39, 1020.92 examples/s]40461 examples [00:39, 1000.48 examples/s]40568 examples [00:40, 1018.01 examples/s]40675 examples [00:40, 1031.32 examples/s]40783 examples [00:40, 1045.29 examples/s]40889 examples [00:40, 1048.87 examples/s]40997 examples [00:40, 1055.88 examples/s]41105 examples [00:40, 1060.02 examples/s]41213 examples [00:40, 1063.92 examples/s]41323 examples [00:40, 1072.48 examples/s]41432 examples [00:40, 1076.75 examples/s]41540 examples [00:40, 1071.29 examples/s]41648 examples [00:41, 1060.88 examples/s]41758 examples [00:41, 1070.90 examples/s]41869 examples [00:41, 1081.59 examples/s]41978 examples [00:41, 1081.63 examples/s]42087 examples [00:41, 1077.39 examples/s]42196 examples [00:41, 1080.37 examples/s]42305 examples [00:41, 1074.96 examples/s]42415 examples [00:41, 1080.26 examples/s]42524 examples [00:41, 1081.31 examples/s]42633 examples [00:41, 1080.29 examples/s]42742 examples [00:42, 1077.71 examples/s]42850 examples [00:42, 1035.63 examples/s]42959 examples [00:42, 1048.95 examples/s]43068 examples [00:42, 1058.58 examples/s]43175 examples [00:42, 1012.39 examples/s]43284 examples [00:42, 1034.18 examples/s]43391 examples [00:42, 1044.24 examples/s]43496 examples [00:42, 1045.90 examples/s]43606 examples [00:42, 1059.80 examples/s]43713 examples [00:43, 1062.24 examples/s]43823 examples [00:43, 1071.93 examples/s]43931 examples [00:43, 1073.38 examples/s]44039 examples [00:43, 1071.70 examples/s]44148 examples [00:43, 1075.82 examples/s]44256 examples [00:43, 1071.27 examples/s]44364 examples [00:43, 1073.18 examples/s]44472 examples [00:43, 1057.43 examples/s]44578 examples [00:43, 1054.98 examples/s]44684 examples [00:43, 1039.60 examples/s]44792 examples [00:44, 1049.33 examples/s]44900 examples [00:44, 1057.43 examples/s]45006 examples [00:44, 924.36 examples/s] 45102 examples [00:44, 854.42 examples/s]45196 examples [00:44, 871.47 examples/s]45293 examples [00:44, 896.87 examples/s]45385 examples [00:44, 829.04 examples/s]45485 examples [00:44, 870.38 examples/s]45577 examples [00:44, 884.69 examples/s]45676 examples [00:45, 912.87 examples/s]45777 examples [00:45, 934.57 examples/s]45880 examples [00:45, 960.08 examples/s]45984 examples [00:45, 981.22 examples/s]46085 examples [00:45, 987.37 examples/s]46185 examples [00:45, 951.61 examples/s]46288 examples [00:45, 973.71 examples/s]46390 examples [00:45, 986.80 examples/s]46497 examples [00:45, 1010.13 examples/s]46604 examples [00:45, 1026.02 examples/s]46707 examples [00:46, 1024.69 examples/s]46814 examples [00:46, 1037.76 examples/s]46919 examples [00:46, 1028.58 examples/s]47027 examples [00:46, 1041.02 examples/s]47134 examples [00:46, 1048.18 examples/s]47241 examples [00:46, 1052.31 examples/s]47347 examples [00:46, 1054.21 examples/s]47453 examples [00:46, 1048.28 examples/s]47560 examples [00:46, 1053.12 examples/s]47668 examples [00:46, 1058.79 examples/s]47774 examples [00:47, 1058.20 examples/s]47881 examples [00:47, 1059.61 examples/s]47987 examples [00:47, 1020.45 examples/s]48090 examples [00:47, 1011.78 examples/s]48192 examples [00:47, 1008.37 examples/s]48298 examples [00:47, 1022.43 examples/s]48402 examples [00:47, 1025.26 examples/s]48510 examples [00:47, 1038.85 examples/s]48615 examples [00:47, 980.69 examples/s] 48722 examples [00:48, 1004.09 examples/s]48827 examples [00:48, 1016.05 examples/s]48933 examples [00:48, 1028.75 examples/s]49040 examples [00:48, 1038.34 examples/s]49145 examples [00:48, 1039.61 examples/s]49250 examples [00:48, 1017.57 examples/s]49353 examples [00:48, 969.92 examples/s] 49459 examples [00:48, 995.09 examples/s]49566 examples [00:48, 1015.38 examples/s]49669 examples [00:48, 1015.60 examples/s]49774 examples [00:49, 1025.47 examples/s]49877 examples [00:49, 1017.66 examples/s]49983 examples [00:49, 1028.77 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5826/50000 [00:00<00:00, 58259.36 examples/s] 38%|███▊      | 18873/50000 [00:00<00:00, 69858.44 examples/s] 64%|██████▍   | 31983/50000 [00:00<00:00, 81242.59 examples/s] 90%|████████▉ | 44917/50000 [00:00<00:00, 91443.04 examples/s]                                                               0 examples [00:00, ? examples/s]83 examples [00:00, 829.70 examples/s]171 examples [00:00, 837.52 examples/s]266 examples [00:00, 867.45 examples/s]376 examples [00:00, 924.49 examples/s]481 examples [00:00, 957.41 examples/s]588 examples [00:00, 986.84 examples/s]694 examples [00:00, 1006.67 examples/s]793 examples [00:00, 999.87 examples/s] 893 examples [00:00, 998.59 examples/s]997 examples [00:01, 1008.57 examples/s]1096 examples [00:01, 989.33 examples/s]1206 examples [00:01, 1018.43 examples/s]1308 examples [00:01, 1017.09 examples/s]1410 examples [00:01, 1003.50 examples/s]1521 examples [00:01, 1031.34 examples/s]1628 examples [00:01, 1033.10 examples/s]1732 examples [00:01, 973.41 examples/s] 1841 examples [00:01, 1004.69 examples/s]1943 examples [00:01, 985.19 examples/s] 2054 examples [00:02, 1018.92 examples/s]2164 examples [00:02, 1041.47 examples/s]2275 examples [00:02, 1058.89 examples/s]2384 examples [00:02, 1066.95 examples/s]2492 examples [00:02, 1062.66 examples/s]2599 examples [00:02, 1058.53 examples/s]2710 examples [00:02, 1071.78 examples/s]2821 examples [00:02, 1081.60 examples/s]2930 examples [00:02, 1082.72 examples/s]3039 examples [00:02, 1078.14 examples/s]3147 examples [00:03, 1078.21 examples/s]3259 examples [00:03, 1087.66 examples/s]3370 examples [00:03, 1091.88 examples/s]3480 examples [00:03, 1091.65 examples/s]3590 examples [00:03, 1084.69 examples/s]3701 examples [00:03, 1090.63 examples/s]3811 examples [00:03, 1092.45 examples/s]3921 examples [00:03, 1084.95 examples/s]4031 examples [00:03, 1088.14 examples/s]4140 examples [00:03, 1072.43 examples/s]4251 examples [00:04, 1080.93 examples/s]4360 examples [00:04, 1052.78 examples/s]4468 examples [00:04, 1059.99 examples/s]4579 examples [00:04, 1071.87 examples/s]4687 examples [00:04, 1070.46 examples/s]4797 examples [00:04, 1078.05 examples/s]4907 examples [00:04, 1082.98 examples/s]5016 examples [00:04, 1084.40 examples/s]5126 examples [00:04, 1086.61 examples/s]5235 examples [00:05, 1039.83 examples/s]5345 examples [00:05, 1056.45 examples/s]5452 examples [00:05, 1047.42 examples/s]5560 examples [00:05, 1056.74 examples/s]5670 examples [00:05, 1068.93 examples/s]5778 examples [00:05, 996.97 examples/s] 5881 examples [00:05, 1006.51 examples/s]5991 examples [00:05, 1031.11 examples/s]6095 examples [00:05, 1004.88 examples/s]6203 examples [00:05, 1024.56 examples/s]6309 examples [00:06, 1033.50 examples/s]6413 examples [00:06, 1034.61 examples/s]6519 examples [00:06, 1040.17 examples/s]6624 examples [00:06, 1025.92 examples/s]6734 examples [00:06, 1044.50 examples/s]6839 examples [00:06, 998.15 examples/s] 6949 examples [00:06, 1025.78 examples/s]7053 examples [00:06, 1027.30 examples/s]7157 examples [00:06, 1016.72 examples/s]7261 examples [00:06, 1020.55 examples/s]7369 examples [00:07, 1035.99 examples/s]7476 examples [00:07, 1045.53 examples/s]7581 examples [00:07, 1045.12 examples/s]7691 examples [00:07, 1060.98 examples/s]7798 examples [00:07, 1053.94 examples/s]7907 examples [00:07, 1062.76 examples/s]8015 examples [00:07, 1066.54 examples/s]8125 examples [00:07, 1076.36 examples/s]8236 examples [00:07, 1083.89 examples/s]8345 examples [00:07, 1066.09 examples/s]8452 examples [00:08, 1042.11 examples/s]8561 examples [00:08, 1055.40 examples/s]8672 examples [00:08, 1070.99 examples/s]8782 examples [00:08, 1079.32 examples/s]8891 examples [00:08, 1069.81 examples/s]9000 examples [00:08, 1074.53 examples/s]9109 examples [00:08, 1077.04 examples/s]9218 examples [00:08, 1080.02 examples/s]9329 examples [00:08, 1087.79 examples/s]9438 examples [00:09, 1072.30 examples/s]9547 examples [00:09, 1075.27 examples/s]9657 examples [00:09, 1080.67 examples/s]9767 examples [00:09, 1085.16 examples/s]9877 examples [00:09, 1088.88 examples/s]9986 examples [00:09, 1073.57 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRJBJ9J/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRJBJ9J/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f705a459620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f705a459620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f705a459620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6ff4638588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6ff4673978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f705a459620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f705a459620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f705a459620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6ff4661c18>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f704148da90>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f6fe1ad4158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f6fe1ad4158> 

  function with postional parmater data_info <function split_train_valid at 0x7f6fe1ad4158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f6fe1ad4268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f6fe1ad4268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f6fe1ad4268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=93ba9abf8e2c42fd76d08e1486e8229b1736eace8754531137f30c6de9104ee9
  Stored in directory: /tmp/pip-ephem-wheel-cache-tjd4ky7i/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:21:30, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:29:07, 16.5kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:11:08, 23.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:08:18, 33.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.75M/862M [00:01<4:59:24, 47.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.59M/862M [00:01<3:28:45, 68.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.91M/862M [00:01<2:25:41, 97.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:01<1:41:24, 139kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:01<1:10:36, 198kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.0M/862M [00:01<49:09, 283kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<34:27, 403kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.6M/862M [00:01<24:03, 573kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.6M/862M [00:02<16:48, 814kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:02<11:50, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:02<08:18, 1.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<06:44, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<06:37, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:00, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:23, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:05<03:54, 3.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<44:19, 301kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<32:39, 409kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<23:14, 573kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<18:53, 703kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<16:10, 821kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<11:54, 1.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<08:30, 1.56MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<11:07, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<09:07, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<06:43, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<07:46, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<06:47, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:13<05:04, 2.58MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<05:57, 2.19MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<04:29, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<06:13, 2.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 81.9M/862M [00:16<05:27, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<04:06, 3.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:17<03:02, 4.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<54:21, 238kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<40:40, 318kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:19<29:02, 445kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<20:24, 631kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<1:47:07, 120kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<1:16:17, 169kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.7M/862M [00:21<53:37, 239kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<40:26, 317kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<31:01, 413kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<22:21, 572kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:23<15:43, 810kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<57:08, 223kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:24<41:18, 308kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<29:07, 436kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<23:19, 543kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<18:54, 670kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<13:47, 917kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<09:46, 1.29MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<18:49, 669kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<14:28, 869kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:26, 1.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<08:03, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<8:46:27, 23.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<6:08:57, 34.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<4:17:33, 48.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<3:07:50, 66.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:14:07, 92.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:34:27, 132kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<1:06:00, 188kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:16:02, 163kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<54:29, 227kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<38:22, 322kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<29:40, 416kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<22:02, 559kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<15:39, 785kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:49, 887kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:10, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:01, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<06:28, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:14, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:25, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:12, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:10, 1.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:30, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:47, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:10, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<28:18, 426kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<21:02, 573kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<15:00, 801kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:16, 903kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:50, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:49, 1.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<06:17, 1.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<13:39, 872kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:49, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:52, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<08:14, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:00, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:08, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:24, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:54, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:20, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:52, 3.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<10:16, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:22, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:09, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:01, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<07:17, 1.60MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:36, 2.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:04, 2.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<08:39, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:16, 1.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:22, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:26, 1.78MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:44, 2.00MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:18, 2.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:35, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:20, 1.80MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<04:57, 2.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:36, 3.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:59, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:29, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<05:29, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:22, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:51, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:17, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:53, 2.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:31, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:34, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<04:12, 2.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:23, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:08, 1.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:47, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:34, 3.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:40, 1.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:09, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<03:51, 2.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:11, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:48, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:13<03:36, 3.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:00, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:39, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<03:30, 3.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:54, 2.22MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:35, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:29, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:53, 2.21MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:43, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:34, 2.36MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<03:19, 3.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<28:08, 382kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<20:50, 516kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<14:47, 724kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<10:57, 976kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:50:21, 22.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<5:29:28, 32.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<3:49:43, 46.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:51:05, 62.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:02:00, 87.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<1:25:46, 124kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<59:58, 176kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<47:17, 223kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<34:13, 308kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<24:08, 436kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<19:12, 545kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<15:40, 668kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:30, 909kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<08:09, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<36:59, 281kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<26:59, 385kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<19:04, 544kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<15:40, 660kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<13:09, 785kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:41, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:33<06:51, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<18:52, 544kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<14:18, 718kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<10:15, 998kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<09:27, 1.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:44, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:37, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:12, 1.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<06:30, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:05, 1.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:40, 2.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<32:47, 307kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<24:00, 419kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<16:59, 590kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<14:07, 708kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<12:00, 832kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<08:50, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<06:19, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:20, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:53, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:04, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:46, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<04:48, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:28, 2.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<25:49, 379kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<19:06, 512kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<13:33, 719kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<09:35, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<2:03:42, 78.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<1:28:39, 110kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<1:02:24, 155kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<43:48, 221kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<33:06, 292kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<24:11, 399kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<17:06, 562kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<14:05, 680kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<10:53, 879kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:51, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<07:37, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:21, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:36, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:04, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<06:02, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:13, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:51, 2.44MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:48, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:22, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:15, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:04, 3.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<23:41, 393kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<17:33, 529kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<12:28, 744kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:47, 856kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<08:22, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:11, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:25, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<15:05, 607kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<12:30, 732kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<09:09, 998kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<06:31, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<08:37, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:00, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:05, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:35, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<05:50, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:31, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:16, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<24:43, 362kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<18:14, 491kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<12:56, 690kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<11:02, 805kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<09:32, 931kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<07:08, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<05:04, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<1:15:31, 117kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<53:46, 164kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<37:46, 233kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<28:17, 309kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<21:39, 404kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<15:36, 559kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<10:58, 792kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<23:20, 372kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<17:16, 502kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<12:14, 706kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<10:28, 822kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<08:15, 1.04MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<05:57, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:04, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:10, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:47, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:33, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:03, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<03:02, 2.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:05, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:39, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:41, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:41, 3.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<28:18, 295kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<20:43, 402kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<14:41, 566kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<10:42, 772kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<6:01:40, 22.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<4:13:26, 32.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<2:56:31, 46.5kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<2:10:04, 63.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<1:32:08, 89.0kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<1:04:57, 126kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<45:17, 180kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<43:35, 187kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<31:41, 257kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<22:23, 362kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<17:19, 465kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<12:57, 622kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<09:16, 867kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:17, 964kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:39, 1.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:51, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:10, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:19, 1.83MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:14, 2.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:59, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:27, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:32, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:32, 3.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<19:48, 393kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<14:41, 530kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<10:26, 742kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<09:03, 852kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:59, 965kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:56, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<04:14, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:28, 1.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:02, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:25, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:47, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:58, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:49, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:46, 2.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:14, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:28, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:18, 2.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:01, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:24, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:28, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:29, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<37:15, 198kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<26:44, 276kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<18:51, 390kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<14:45, 495kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<11:48, 618kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<08:34, 850kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<06:04, 1.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<07:38, 947kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:07, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:25, 1.63MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:42, 1.52MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:49, 1.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:41, 1.94MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:38, 2.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<12:52, 552kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<09:45, 727kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:57, 1.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:28, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<06:01, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:34, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<03:16, 2.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<23:15, 299kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<17:00, 409kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<12:00, 577kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<09:57, 693kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<08:25, 818kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<04:24, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<06:21, 1.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:10, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:46, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:09, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:20, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:20, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:24, 2.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:04, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<04:57, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:02, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:13, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<03:16, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:21, 2.78MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<26:48, 244kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<19:26, 337kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<13:42, 476kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<11:01, 588kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<09:02, 717kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<06:35, 980kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:40, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:21, 872kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:49, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:22, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:24, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:23, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:27<02:25, 2.59MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<08:48, 713kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<06:49, 919kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<04:53, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:50, 1.28MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:42, 1.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:35, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:47, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<4:13:37, 24.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<2:57:29, 34.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<2:03:35, 49.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<1:28:47, 68.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<1:03:24, 95.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<44:37, 136kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<31:03, 193kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<30:45, 195kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<22:07, 271kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<15:33, 384kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<12:12, 486kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<09:48, 605kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<07:09, 826kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<05:03, 1.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<21:09, 277kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<15:25, 380kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<10:52, 536kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<08:53, 652kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:27, 777kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:30, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<03:54, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<20:02, 286kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<14:31, 394kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<10:17, 554kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<07:13, 784kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<15:43, 360kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<12:12, 463kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<08:49, 639kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<06:11, 903kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<17:42, 316kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<12:58, 430kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<09:11, 605kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<07:38, 723kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<06:30, 847kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:50, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<03:25, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<19:41, 277kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<14:20, 380kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<10:07, 536kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<08:14, 653kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<06:19, 850kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:32, 1.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:24, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:38, 1.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:39, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:02, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:14, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:30, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:49, 2.85MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:42, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:07, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:18, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:47, 1.83MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:02, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:23, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:43, 2.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<16:16, 309kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<11:54, 423kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<08:25, 594kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<07:00, 710kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<05:25, 914kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:53, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:49, 1.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<05:20, 918kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<04:27, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:17, 1.48MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:07, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:29, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:44, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:58, 2.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:14, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:33, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:48, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:01, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:27, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:40, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:49, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:02, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:40, 1.72MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:58, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:28, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:24, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:19, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:15, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:11, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:08, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:58, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:56, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:56, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:55, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:52, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:52, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:50, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:48, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:47, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:33, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:09, 1.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:19, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:26, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:27, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:26, 1.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:22, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<03:16, 1.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:13, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:10, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:06, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<03:01, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<02:59, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:55, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:53, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:21<02:50, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:21<02:48, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:46, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:42, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:42, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<02:42, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:22<02:41, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:38, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:36, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<02:34, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<02:31, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<02:29, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<02:28, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<02:34, 1.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:30, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:27, 1.81MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:23, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:21, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<02:19, 1.91MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<02:18, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<02:15, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<02:13, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:12, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:09, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:08, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<02:06, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<02:06, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<02:04, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:25<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:02, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<01:59, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:57, 2.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:56, 2.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:56, 2.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:56, 2.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:54, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:53, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:54, 2.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:50, 2.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<01:51, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<01:49, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<01:49, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<01:47, 2.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:47, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:45, 2.46MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:43, 2.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:43, 2.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:39, 2.58MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:43, 2.48MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:38, 2.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:42, 2.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<01:37, 2.63MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<01:38, 2.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:28<01:34, 2.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:33, 2.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:30, 2.81MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:27, 2.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:27, 2.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<01:21, 3.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<01:23, 3.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:18, 3.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:20, 3.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:13, 3.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:15, 3.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:12, 3.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:09, 3.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:08, 3.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:03, 3.94MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:05, 3.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<04:53, 853kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:50, 1.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:57, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:19, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:53, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:33, 2.65MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:17, 3.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<04:01, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:20, 768kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<04:20, 944kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:13, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:28, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:53, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:31, 2.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:56, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<04:44, 850kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:47, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:50, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:06, 1.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:37, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<03:07, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<03:40, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:57, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:10, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:38, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:29, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<04:07, 945kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<03:29, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:37, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:56, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:27, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<09:48, 390kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<07:58, 480kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<05:49, 655kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:09, 912kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:58, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<08:29, 443kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<07:46, 483kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<05:53, 636kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:14, 880kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:01, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:14, 869kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<04:28, 825kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<03:27, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:29, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:51, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:33, 1.42MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<03:08, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:30, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:51, 1.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:02, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:40, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:08, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:34, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:56, 1.79MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:25, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:55, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:23, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:02, 3.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:39, 604kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:52<04:52, 699kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<03:38, 932kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:36, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:49, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:53, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:12, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:37, 2.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:06, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:39, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:12, 2.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:56, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:08, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:41, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:12, 2.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:07, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:20, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:51, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:20, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:00, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:14, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:44, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:15, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:05, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:13, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:43, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:14, 2.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:54, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:03, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:34, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:08, 2.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:39, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:51, 1.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:26, 1.97MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:02, 2.70MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:42, 1.64MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:56, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:31, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:06, 2.48MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:43, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:57, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:31, 1.78MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:06, 2.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:43, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:38, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:14, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:18, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:32, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:13, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:52, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<04:31, 555kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:40, 683kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:40, 935kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:52, 1.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:32, 962kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<02:17, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:43, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:12, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<03:35, 663kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<02:50, 836kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:03, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:50, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:42, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:16, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:43, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:24, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:01, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:14, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:16, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:01, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:06, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<00:52, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:56, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:02, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<00:48, 2.51MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:31<00:34, 3.43MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:49, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:37, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:13, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:09, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:09, 1.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:52, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:37, 2.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:27, 1.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:21, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:01, 1.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:59, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:01, 1.71MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:47, 2.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:49, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:53, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:41, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:44, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:49, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:38, 2.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:41, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:46, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:36, 2.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:25, 3.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:32, 582kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:20, 632kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:44, 842kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:14, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:08, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:03, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:48, 1.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:45, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:36, 2.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:37, 2.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:36, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:27, 2.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:32, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:48, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:38, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:27, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:36, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:37, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:28, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:30, 2.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:24, 2.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:17, 3.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:36, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:27, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:19, 2.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:42, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:39, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:29, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:20, 2.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:10, 732kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:58, 879kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:42, 1.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:28, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<09:42, 81.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<06:54, 114kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<04:46, 162kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<03:11, 231kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<02:39, 272kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:59, 363kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<01:23, 509kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:55, 720kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:00, 650kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:48, 799kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:34, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:23, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:52, 669kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:42, 819kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:30, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:20, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:25, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:23, 1.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:17, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:07, 3.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:55, 414kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:42, 534kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:29, 742kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:18, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:24, 769kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:20, 922kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:14, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:08, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:16, 869kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:14, 1.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:05, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:16, 645kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:12, 793kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:04, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 851kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 1.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:01, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:13, 159kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:09, 219kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 309kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 438/400000 [00:00<01:31, 4379.12it/s]  0%|          | 1268/400000 [00:00<01:18, 5101.80it/s]  1%|          | 2071/400000 [00:00<01:09, 5727.67it/s]  1%|          | 2905/400000 [00:00<01:02, 6320.67it/s]  1%|          | 3717/400000 [00:00<00:58, 6769.75it/s]  1%|          | 4560/400000 [00:00<00:54, 7194.41it/s]  1%|▏         | 5385/400000 [00:00<00:52, 7479.25it/s]  2%|▏         | 6181/400000 [00:00<00:51, 7617.11it/s]  2%|▏         | 7026/400000 [00:00<00:50, 7848.19it/s]  2%|▏         | 7812/400000 [00:01<00:50, 7841.49it/s]  2%|▏         | 8622/400000 [00:01<00:49, 7916.77it/s]  2%|▏         | 9466/400000 [00:01<00:48, 8064.07it/s]  3%|▎         | 10301/400000 [00:01<00:47, 8147.72it/s]  3%|▎         | 11154/400000 [00:01<00:47, 8256.94it/s]  3%|▎         | 11982/400000 [00:01<00:46, 8259.02it/s]  3%|▎         | 12809/400000 [00:01<00:47, 8188.67it/s]  3%|▎         | 13638/400000 [00:01<00:47, 8217.37it/s]  4%|▎         | 14467/400000 [00:01<00:46, 8236.44it/s]  4%|▍         | 15343/400000 [00:01<00:45, 8386.40it/s]  4%|▍         | 16200/400000 [00:02<00:45, 8439.34it/s]  4%|▍         | 17045/400000 [00:02<00:45, 8414.62it/s]  4%|▍         | 17887/400000 [00:02<00:46, 8230.60it/s]  5%|▍         | 18712/400000 [00:02<00:46, 8194.89it/s]  5%|▍         | 19537/400000 [00:02<00:46, 8211.12it/s]  5%|▌         | 20368/400000 [00:02<00:46, 8238.75it/s]  5%|▌         | 21194/400000 [00:02<00:45, 8245.11it/s]  6%|▌         | 22019/400000 [00:02<00:46, 8180.00it/s]  6%|▌         | 22861/400000 [00:02<00:45, 8248.24it/s]  6%|▌         | 23693/400000 [00:02<00:45, 8267.90it/s]  6%|▌         | 24521/400000 [00:03<00:45, 8252.50it/s]  6%|▋         | 25347/400000 [00:03<00:45, 8221.77it/s]  7%|▋         | 26198/400000 [00:03<00:45, 8305.90it/s]  7%|▋         | 27053/400000 [00:03<00:44, 8375.04it/s]  7%|▋         | 27896/400000 [00:03<00:44, 8391.18it/s]  7%|▋         | 28736/400000 [00:03<00:44, 8371.30it/s]  7%|▋         | 29574/400000 [00:03<00:45, 8222.17it/s]  8%|▊         | 30418/400000 [00:03<00:44, 8284.77it/s]  8%|▊         | 31259/400000 [00:03<00:44, 8321.69it/s]  8%|▊         | 32092/400000 [00:03<00:44, 8185.41it/s]  8%|▊         | 32943/400000 [00:04<00:44, 8278.57it/s]  8%|▊         | 33788/400000 [00:04<00:43, 8326.69it/s]  9%|▊         | 34636/400000 [00:04<00:43, 8370.40it/s]  9%|▉         | 35489/400000 [00:04<00:43, 8416.86it/s]  9%|▉         | 36332/400000 [00:04<00:45, 8072.33it/s]  9%|▉         | 37174/400000 [00:04<00:44, 8171.63it/s] 10%|▉         | 38027/400000 [00:04<00:43, 8275.59it/s] 10%|▉         | 38857/400000 [00:04<00:43, 8253.22it/s] 10%|▉         | 39721/400000 [00:04<00:43, 8363.69it/s] 10%|█         | 40562/400000 [00:04<00:42, 8374.77it/s] 10%|█         | 41402/400000 [00:05<00:42, 8382.03it/s] 11%|█         | 42258/400000 [00:05<00:42, 8432.66it/s] 11%|█         | 43126/400000 [00:05<00:41, 8503.74it/s] 11%|█         | 43977/400000 [00:05<00:42, 8432.80it/s] 11%|█         | 44821/400000 [00:05<00:42, 8427.63it/s] 11%|█▏        | 45665/400000 [00:05<00:42, 8411.29it/s] 12%|█▏        | 46519/400000 [00:05<00:41, 8440.56it/s] 12%|█▏        | 47369/400000 [00:05<00:41, 8457.10it/s] 12%|█▏        | 48220/400000 [00:05<00:41, 8472.03it/s] 12%|█▏        | 49072/400000 [00:05<00:41, 8483.82it/s] 12%|█▏        | 49921/400000 [00:06<00:41, 8440.61it/s] 13%|█▎        | 50766/400000 [00:06<00:41, 8432.51it/s] 13%|█▎        | 51619/400000 [00:06<00:41, 8460.90it/s] 13%|█▎        | 52466/400000 [00:06<00:41, 8455.92it/s] 13%|█▎        | 53321/400000 [00:06<00:40, 8482.53it/s] 14%|█▎        | 54170/400000 [00:06<00:40, 8447.43it/s] 14%|█▍        | 55042/400000 [00:06<00:40, 8527.09it/s] 14%|█▍        | 55928/400000 [00:06<00:39, 8621.82it/s] 14%|█▍        | 56806/400000 [00:06<00:39, 8666.52it/s] 14%|█▍        | 57694/400000 [00:06<00:39, 8727.40it/s] 15%|█▍        | 58568/400000 [00:07<00:39, 8545.05it/s] 15%|█▍        | 59444/400000 [00:07<00:39, 8607.20it/s] 15%|█▌        | 60311/400000 [00:07<00:39, 8624.41it/s] 15%|█▌        | 61175/400000 [00:07<00:39, 8510.39it/s] 16%|█▌        | 62027/400000 [00:07<00:39, 8482.29it/s] 16%|█▌        | 62876/400000 [00:07<00:39, 8431.25it/s] 16%|█▌        | 63737/400000 [00:07<00:39, 8482.31it/s] 16%|█▌        | 64592/400000 [00:07<00:39, 8501.97it/s] 16%|█▋        | 65443/400000 [00:07<00:39, 8419.59it/s] 17%|█▋        | 66310/400000 [00:07<00:39, 8492.01it/s] 17%|█▋        | 67161/400000 [00:08<00:39, 8495.26it/s] 17%|█▋        | 68033/400000 [00:08<00:38, 8561.38it/s] 17%|█▋        | 68919/400000 [00:08<00:38, 8646.19it/s] 17%|█▋        | 69785/400000 [00:08<00:38, 8600.29it/s] 18%|█▊        | 70672/400000 [00:08<00:37, 8677.23it/s] 18%|█▊        | 71541/400000 [00:08<00:38, 8610.60it/s] 18%|█▊        | 72403/400000 [00:08<00:38, 8608.11it/s] 18%|█▊        | 73286/400000 [00:08<00:37, 8671.61it/s] 19%|█▊        | 74167/400000 [00:08<00:37, 8709.71it/s] 19%|█▉        | 75039/400000 [00:08<00:37, 8694.57it/s] 19%|█▉        | 75909/400000 [00:09<00:37, 8619.72it/s] 19%|█▉        | 76794/400000 [00:09<00:37, 8685.03it/s] 19%|█▉        | 77663/400000 [00:09<00:37, 8679.23it/s] 20%|█▉        | 78532/400000 [00:09<00:37, 8661.20it/s] 20%|█▉        | 79399/400000 [00:09<00:37, 8635.81it/s] 20%|██        | 80263/400000 [00:09<00:37, 8552.70it/s] 20%|██        | 81123/400000 [00:09<00:37, 8566.51it/s] 20%|██        | 81984/400000 [00:09<00:37, 8576.65it/s] 21%|██        | 82846/400000 [00:09<00:36, 8588.24it/s] 21%|██        | 83705/400000 [00:09<00:37, 8477.85it/s] 21%|██        | 84554/400000 [00:10<00:37, 8391.43it/s] 21%|██▏       | 85427/400000 [00:10<00:37, 8490.03it/s] 22%|██▏       | 86277/400000 [00:10<00:36, 8484.74it/s] 22%|██▏       | 87130/400000 [00:10<00:36, 8496.80it/s] 22%|██▏       | 87999/400000 [00:10<00:36, 8551.27it/s] 22%|██▏       | 88855/400000 [00:10<00:36, 8417.15it/s] 22%|██▏       | 89698/400000 [00:10<00:37, 8371.74it/s] 23%|██▎       | 90569/400000 [00:10<00:36, 8468.36it/s] 23%|██▎       | 91431/400000 [00:10<00:36, 8511.60it/s] 23%|██▎       | 92283/400000 [00:11<00:37, 8224.37it/s] 23%|██▎       | 93108/400000 [00:11<00:37, 8224.38it/s] 23%|██▎       | 93947/400000 [00:11<00:36, 8272.27it/s] 24%|██▎       | 94813/400000 [00:11<00:36, 8382.24it/s] 24%|██▍       | 95673/400000 [00:11<00:36, 8444.98it/s] 24%|██▍       | 96530/400000 [00:11<00:35, 8480.13it/s] 24%|██▍       | 97379/400000 [00:11<00:36, 8317.12it/s] 25%|██▍       | 98212/400000 [00:11<00:36, 8226.44it/s] 25%|██▍       | 99036/400000 [00:11<00:37, 8048.47it/s] 25%|██▍       | 99887/400000 [00:11<00:36, 8179.35it/s] 25%|██▌       | 100741/400000 [00:12<00:36, 8283.30it/s] 25%|██▌       | 101582/400000 [00:12<00:35, 8318.44it/s] 26%|██▌       | 102432/400000 [00:12<00:35, 8371.66it/s] 26%|██▌       | 103290/400000 [00:12<00:35, 8431.78it/s] 26%|██▌       | 104148/400000 [00:12<00:34, 8473.67it/s] 26%|██▋       | 105001/400000 [00:12<00:34, 8487.97it/s] 26%|██▋       | 105851/400000 [00:12<00:34, 8407.95it/s] 27%|██▋       | 106706/400000 [00:12<00:34, 8449.78it/s] 27%|██▋       | 107556/400000 [00:12<00:34, 8462.84it/s] 27%|██▋       | 108414/400000 [00:12<00:34, 8495.75it/s] 27%|██▋       | 109275/400000 [00:13<00:34, 8528.31it/s] 28%|██▊       | 110129/400000 [00:13<00:34, 8477.77it/s] 28%|██▊       | 110991/400000 [00:13<00:33, 8519.78it/s] 28%|██▊       | 111851/400000 [00:13<00:33, 8540.90it/s] 28%|██▊       | 112709/400000 [00:13<00:33, 8552.14it/s] 28%|██▊       | 113569/400000 [00:13<00:33, 8565.93it/s] 29%|██▊       | 114426/400000 [00:13<00:33, 8556.31it/s] 29%|██▉       | 115297/400000 [00:13<00:33, 8601.79it/s] 29%|██▉       | 116158/400000 [00:13<00:33, 8592.96it/s] 29%|██▉       | 117018/400000 [00:13<00:32, 8582.98it/s] 29%|██▉       | 117877/400000 [00:14<00:32, 8568.17it/s] 30%|██▉       | 118734/400000 [00:14<00:33, 8462.02it/s] 30%|██▉       | 119589/400000 [00:14<00:33, 8486.94it/s] 30%|███       | 120441/400000 [00:14<00:32, 8494.55it/s] 30%|███       | 121292/400000 [00:14<00:32, 8496.27it/s] 31%|███       | 122142/400000 [00:14<00:32, 8437.68it/s] 31%|███       | 122987/400000 [00:14<00:32, 8440.56it/s] 31%|███       | 123832/400000 [00:14<00:32, 8425.31it/s] 31%|███       | 124700/400000 [00:14<00:32, 8498.87it/s] 31%|███▏      | 125568/400000 [00:14<00:32, 8549.83it/s] 32%|███▏      | 126425/400000 [00:15<00:31, 8552.45it/s] 32%|███▏      | 127281/400000 [00:15<00:32, 8404.17it/s] 32%|███▏      | 128148/400000 [00:15<00:32, 8480.65it/s] 32%|███▏      | 129023/400000 [00:15<00:31, 8557.63it/s] 32%|███▏      | 129880/400000 [00:15<00:31, 8549.32it/s] 33%|███▎      | 130737/400000 [00:15<00:31, 8554.85it/s] 33%|███▎      | 131593/400000 [00:15<00:32, 8366.77it/s] 33%|███▎      | 132446/400000 [00:15<00:31, 8413.99it/s] 33%|███▎      | 133289/400000 [00:15<00:31, 8380.43it/s] 34%|███▎      | 134128/400000 [00:15<00:32, 8262.04it/s] 34%|███▎      | 134956/400000 [00:16<00:32, 8203.79it/s] 34%|███▍      | 135778/400000 [00:16<00:32, 8104.52it/s] 34%|███▍      | 136620/400000 [00:16<00:32, 8194.20it/s] 34%|███▍      | 137470/400000 [00:16<00:31, 8282.34it/s] 35%|███▍      | 138320/400000 [00:16<00:31, 8344.39it/s] 35%|███▍      | 139162/400000 [00:16<00:31, 8365.37it/s] 35%|███▍      | 139999/400000 [00:16<00:31, 8340.69it/s] 35%|███▌      | 140834/400000 [00:16<00:31, 8286.64it/s] 35%|███▌      | 141663/400000 [00:16<00:32, 7990.59it/s] 36%|███▌      | 142471/400000 [00:16<00:32, 8012.41it/s] 36%|███▌      | 143276/400000 [00:17<00:32, 8021.17it/s] 36%|███▌      | 144144/400000 [00:17<00:31, 8207.68it/s] 36%|███▌      | 144967/400000 [00:17<00:31, 7999.15it/s] 36%|███▋      | 145834/400000 [00:17<00:31, 8186.82it/s] 37%|███▋      | 146656/400000 [00:17<00:31, 8068.95it/s] 37%|███▋      | 147488/400000 [00:17<00:31, 8141.49it/s] 37%|███▋      | 148343/400000 [00:17<00:30, 8257.29it/s] 37%|███▋      | 149200/400000 [00:17<00:30, 8347.69it/s] 38%|███▊      | 150078/400000 [00:17<00:29, 8470.11it/s] 38%|███▊      | 150947/400000 [00:18<00:29, 8533.58it/s] 38%|███▊      | 151806/400000 [00:18<00:29, 8550.22it/s] 38%|███▊      | 152672/400000 [00:18<00:28, 8582.55it/s] 38%|███▊      | 153539/400000 [00:18<00:28, 8605.96it/s] 39%|███▊      | 154404/400000 [00:18<00:28, 8618.59it/s] 39%|███▉      | 155267/400000 [00:18<00:28, 8610.22it/s] 39%|███▉      | 156129/400000 [00:18<00:28, 8588.57it/s] 39%|███▉      | 156992/400000 [00:18<00:28, 8599.22it/s] 39%|███▉      | 157853/400000 [00:18<00:28, 8524.46it/s] 40%|███▉      | 158706/400000 [00:18<00:28, 8483.45it/s] 40%|███▉      | 159555/400000 [00:19<00:28, 8443.94it/s] 40%|████      | 160427/400000 [00:19<00:28, 8522.74it/s] 40%|████      | 161287/400000 [00:19<00:27, 8544.34it/s] 41%|████      | 162142/400000 [00:19<00:27, 8504.39it/s] 41%|████      | 162993/400000 [00:19<00:27, 8495.15it/s] 41%|████      | 163843/400000 [00:19<00:27, 8490.19it/s] 41%|████      | 164693/400000 [00:19<00:27, 8444.49it/s] 41%|████▏     | 165552/400000 [00:19<00:27, 8485.23it/s] 42%|████▏     | 166401/400000 [00:19<00:27, 8452.16it/s] 42%|████▏     | 167263/400000 [00:19<00:27, 8499.47it/s] 42%|████▏     | 168124/400000 [00:20<00:27, 8531.94it/s] 42%|████▏     | 168989/400000 [00:20<00:26, 8564.34it/s] 42%|████▏     | 169846/400000 [00:20<00:26, 8557.99it/s] 43%|████▎     | 170702/400000 [00:20<00:27, 8488.41it/s] 43%|████▎     | 171565/400000 [00:20<00:26, 8528.12it/s] 43%|████▎     | 172427/400000 [00:20<00:26, 8553.05it/s] 43%|████▎     | 173286/400000 [00:20<00:26, 8563.91it/s] 44%|████▎     | 174166/400000 [00:20<00:26, 8630.43it/s] 44%|████▍     | 175030/400000 [00:20<00:26, 8558.19it/s] 44%|████▍     | 175899/400000 [00:20<00:26, 8594.45it/s] 44%|████▍     | 176759/400000 [00:21<00:26, 8562.97it/s] 44%|████▍     | 177617/400000 [00:21<00:25, 8566.79it/s] 45%|████▍     | 178474/400000 [00:21<00:25, 8556.00it/s] 45%|████▍     | 179330/400000 [00:21<00:25, 8516.44it/s] 45%|████▌     | 180182/400000 [00:21<00:25, 8474.12it/s] 45%|████▌     | 181042/400000 [00:21<00:25, 8511.08it/s] 45%|████▌     | 181894/400000 [00:21<00:25, 8390.99it/s] 46%|████▌     | 182755/400000 [00:21<00:25, 8454.58it/s] 46%|████▌     | 183601/400000 [00:21<00:25, 8399.76it/s] 46%|████▌     | 184460/400000 [00:21<00:25, 8453.76it/s] 46%|████▋     | 185326/400000 [00:22<00:25, 8514.05it/s] 47%|████▋     | 186182/400000 [00:22<00:25, 8526.18it/s] 47%|████▋     | 187045/400000 [00:22<00:24, 8554.99it/s] 47%|████▋     | 187901/400000 [00:22<00:24, 8500.57it/s] 47%|████▋     | 188752/400000 [00:22<00:24, 8496.59it/s] 47%|████▋     | 189613/400000 [00:22<00:24, 8527.97it/s] 48%|████▊     | 190474/400000 [00:22<00:24, 8552.21it/s] 48%|████▊     | 191338/400000 [00:22<00:24, 8577.18it/s] 48%|████▊     | 192196/400000 [00:22<00:24, 8515.55it/s] 48%|████▊     | 193066/400000 [00:22<00:24, 8568.43it/s] 48%|████▊     | 193930/400000 [00:23<00:23, 8588.46it/s] 49%|████▊     | 194789/400000 [00:23<00:23, 8583.94it/s] 49%|████▉     | 195649/400000 [00:23<00:23, 8588.63it/s] 49%|████▉     | 196508/400000 [00:23<00:23, 8540.59it/s] 49%|████▉     | 197368/400000 [00:23<00:23, 8555.93it/s] 50%|████▉     | 198224/400000 [00:23<00:24, 8394.57it/s] 50%|████▉     | 199085/400000 [00:23<00:23, 8457.17it/s] 50%|████▉     | 199940/400000 [00:23<00:23, 8483.00it/s] 50%|█████     | 200789/400000 [00:23<00:23, 8410.66it/s] 50%|█████     | 201648/400000 [00:23<00:23, 8461.22it/s] 51%|█████     | 202517/400000 [00:24<00:23, 8526.95it/s] 51%|█████     | 203373/400000 [00:24<00:23, 8533.92it/s] 51%|█████     | 204227/400000 [00:24<00:23, 8479.22it/s] 51%|█████▏    | 205076/400000 [00:24<00:23, 8397.47it/s] 51%|█████▏    | 205937/400000 [00:24<00:22, 8457.51it/s] 52%|█████▏    | 206784/400000 [00:24<00:23, 8330.52it/s] 52%|█████▏    | 207633/400000 [00:24<00:22, 8376.38it/s] 52%|█████▏    | 208480/400000 [00:24<00:22, 8401.66it/s] 52%|█████▏    | 209321/400000 [00:24<00:22, 8388.91it/s] 53%|█████▎    | 210184/400000 [00:24<00:22, 8457.29it/s] 53%|█████▎    | 211042/400000 [00:25<00:22, 8490.22it/s] 53%|█████▎    | 211905/400000 [00:25<00:22, 8528.34it/s] 53%|█████▎    | 212759/400000 [00:25<00:21, 8512.42it/s] 53%|█████▎    | 213611/400000 [00:25<00:21, 8508.67it/s] 54%|█████▎    | 214481/400000 [00:25<00:21, 8564.88it/s] 54%|█████▍    | 215351/400000 [00:25<00:21, 8603.59it/s] 54%|█████▍    | 216237/400000 [00:25<00:21, 8678.40it/s] 54%|█████▍    | 217110/400000 [00:25<00:21, 8692.60it/s] 54%|█████▍    | 217980/400000 [00:25<00:20, 8694.57it/s] 55%|█████▍    | 218850/400000 [00:25<00:21, 8471.79it/s] 55%|█████▍    | 219699/400000 [00:26<00:21, 8373.42it/s] 55%|█████▌    | 220582/400000 [00:26<00:21, 8504.35it/s] 55%|█████▌    | 221437/400000 [00:26<00:20, 8514.09it/s] 56%|█████▌    | 222290/400000 [00:26<00:21, 8302.82it/s] 56%|█████▌    | 223157/400000 [00:26<00:21, 8407.99it/s] 56%|█████▌    | 224000/400000 [00:26<00:21, 8310.91it/s] 56%|█████▌    | 224860/400000 [00:26<00:20, 8393.26it/s] 56%|█████▋    | 225701/400000 [00:26<00:20, 8384.93it/s] 57%|█████▋    | 226552/400000 [00:26<00:20, 8421.83it/s] 57%|█████▋    | 227395/400000 [00:27<00:20, 8422.31it/s] 57%|█████▋    | 228288/400000 [00:27<00:20, 8566.37it/s] 57%|█████▋    | 229160/400000 [00:27<00:19, 8610.70it/s] 58%|█████▊    | 230022/400000 [00:27<00:19, 8601.94it/s] 58%|█████▊    | 230883/400000 [00:27<00:19, 8557.42it/s] 58%|█████▊    | 231740/400000 [00:27<00:19, 8478.00it/s] 58%|█████▊    | 232597/400000 [00:27<00:19, 8504.76it/s] 58%|█████▊    | 233458/400000 [00:27<00:19, 8535.62it/s] 59%|█████▊    | 234312/400000 [00:27<00:19, 8501.33it/s] 59%|█████▉    | 235171/400000 [00:27<00:19, 8526.82it/s] 59%|█████▉    | 236056/400000 [00:28<00:19, 8618.77it/s] 59%|█████▉    | 236924/400000 [00:28<00:18, 8636.98it/s] 59%|█████▉    | 237788/400000 [00:28<00:18, 8578.50it/s] 60%|█████▉    | 238651/400000 [00:28<00:18, 8592.00it/s] 60%|█████▉    | 239511/400000 [00:28<00:18, 8562.27it/s] 60%|██████    | 240385/400000 [00:28<00:18, 8612.57it/s] 60%|██████    | 241265/400000 [00:28<00:18, 8667.90it/s] 61%|██████    | 242135/400000 [00:28<00:18, 8676.01it/s] 61%|██████    | 243003/400000 [00:28<00:18, 8564.36it/s] 61%|██████    | 243860/400000 [00:28<00:18, 8557.65it/s] 61%|██████    | 244733/400000 [00:29<00:18, 8606.02it/s] 61%|██████▏   | 245603/400000 [00:29<00:17, 8632.74it/s] 62%|██████▏   | 246469/400000 [00:29<00:17, 8638.52it/s] 62%|██████▏   | 247334/400000 [00:29<00:17, 8620.80it/s] 62%|██████▏   | 248197/400000 [00:29<00:17, 8579.25it/s] 62%|██████▏   | 249060/400000 [00:29<00:17, 8592.48it/s] 62%|██████▏   | 249921/400000 [00:29<00:17, 8592.70it/s] 63%|██████▎   | 250792/400000 [00:29<00:17, 8626.41it/s] 63%|██████▎   | 251655/400000 [00:29<00:17, 8605.59it/s] 63%|██████▎   | 252526/400000 [00:29<00:17, 8634.24it/s] 63%|██████▎   | 253400/400000 [00:30<00:16, 8663.12it/s] 64%|██████▎   | 254287/400000 [00:30<00:16, 8723.95it/s] 64%|██████▍   | 255170/400000 [00:30<00:16, 8753.99it/s] 64%|██████▍   | 256046/400000 [00:30<00:16, 8711.68it/s] 64%|██████▍   | 256918/400000 [00:30<00:16, 8689.73it/s] 64%|██████▍   | 257788/400000 [00:30<00:16, 8632.53it/s] 65%|██████▍   | 258652/400000 [00:30<00:16, 8628.90it/s] 65%|██████▍   | 259516/400000 [00:30<00:16, 8595.39it/s] 65%|██████▌   | 260376/400000 [00:30<00:16, 8568.26it/s] 65%|██████▌   | 261243/400000 [00:30<00:16, 8596.98it/s] 66%|██████▌   | 262103/400000 [00:31<00:16, 8579.66it/s] 66%|██████▌   | 262962/400000 [00:31<00:15, 8579.61it/s] 66%|██████▌   | 263821/400000 [00:31<00:15, 8582.14it/s] 66%|██████▌   | 264680/400000 [00:31<00:15, 8556.37it/s] 66%|██████▋   | 265539/400000 [00:31<00:15, 8565.97it/s] 67%|██████▋   | 266402/400000 [00:31<00:15, 8583.89it/s] 67%|██████▋   | 267261/400000 [00:31<00:15, 8584.98it/s] 67%|██████▋   | 268121/400000 [00:31<00:15, 8588.61it/s] 67%|██████▋   | 268980/400000 [00:31<00:15, 8218.59it/s] 67%|██████▋   | 269825/400000 [00:31<00:15, 8285.70it/s] 68%|██████▊   | 270706/400000 [00:32<00:15, 8435.52it/s] 68%|██████▊   | 271593/400000 [00:32<00:14, 8560.64it/s] 68%|██████▊   | 272488/400000 [00:32<00:14, 8673.75it/s] 68%|██████▊   | 273360/400000 [00:32<00:14, 8687.23it/s] 69%|██████▊   | 274233/400000 [00:32<00:14, 8699.31it/s] 69%|██████▉   | 275104/400000 [00:32<00:14, 8649.06it/s] 69%|██████▉   | 275977/400000 [00:32<00:14, 8672.92it/s] 69%|██████▉   | 276845/400000 [00:32<00:14, 8670.64it/s] 69%|██████▉   | 277715/400000 [00:32<00:14, 8678.56it/s] 70%|██████▉   | 278586/400000 [00:32<00:13, 8686.99it/s] 70%|██████▉   | 279462/400000 [00:33<00:13, 8706.03it/s] 70%|███████   | 280333/400000 [00:33<00:13, 8670.48it/s] 70%|███████   | 281201/400000 [00:33<00:13, 8634.73it/s] 71%|███████   | 282067/400000 [00:33<00:13, 8640.83it/s] 71%|███████   | 282936/400000 [00:33<00:13, 8654.67it/s] 71%|███████   | 283802/400000 [00:33<00:13, 8464.01it/s] 71%|███████   | 284673/400000 [00:33<00:13, 8534.36it/s] 71%|███████▏  | 285528/400000 [00:33<00:13, 8528.34it/s] 72%|███████▏  | 286382/400000 [00:33<00:13, 8359.85it/s] 72%|███████▏  | 287244/400000 [00:33<00:13, 8435.20it/s] 72%|███████▏  | 288108/400000 [00:34<00:13, 8493.09it/s] 72%|███████▏  | 288969/400000 [00:34<00:13, 8526.22it/s] 72%|███████▏  | 289826/400000 [00:34<00:12, 8537.61it/s] 73%|███████▎  | 290688/400000 [00:34<00:12, 8559.92it/s] 73%|███████▎  | 291551/400000 [00:34<00:12, 8580.65it/s] 73%|███████▎  | 292410/400000 [00:34<00:12, 8528.56it/s] 73%|███████▎  | 293271/400000 [00:34<00:12, 8550.12it/s] 74%|███████▎  | 294127/400000 [00:34<00:12, 8532.22it/s] 74%|███████▎  | 294985/400000 [00:34<00:12, 8543.99it/s] 74%|███████▍  | 295847/400000 [00:34<00:12, 8564.08it/s] 74%|███████▍  | 296704/400000 [00:35<00:12, 8541.22it/s] 74%|███████▍  | 297572/400000 [00:35<00:11, 8581.01it/s] 75%|███████▍  | 298431/400000 [00:35<00:11, 8464.26it/s] 75%|███████▍  | 299294/400000 [00:35<00:11, 8512.40it/s] 75%|███████▌  | 300158/400000 [00:35<00:11, 8548.92it/s] 75%|███████▌  | 301029/400000 [00:35<00:11, 8594.24it/s] 75%|███████▌  | 301907/400000 [00:35<00:11, 8646.95it/s] 76%|███████▌  | 302784/400000 [00:35<00:11, 8682.05it/s] 76%|███████▌  | 303662/400000 [00:35<00:11, 8708.77it/s] 76%|███████▌  | 304538/400000 [00:35<00:10, 8722.79it/s] 76%|███████▋  | 305411/400000 [00:36<00:10, 8684.84it/s] 77%|███████▋  | 306280/400000 [00:36<00:10, 8655.00it/s] 77%|███████▋  | 307146/400000 [00:36<00:10, 8597.12it/s] 77%|███████▋  | 308006/400000 [00:36<00:10, 8585.16it/s] 77%|███████▋  | 308873/400000 [00:36<00:10, 8607.59it/s] 77%|███████▋  | 309734/400000 [00:36<00:10, 8579.81it/s] 78%|███████▊  | 310600/400000 [00:36<00:10, 8601.22it/s] 78%|███████▊  | 311461/400000 [00:36<00:10, 8519.09it/s] 78%|███████▊  | 312314/400000 [00:36<00:10, 8487.44it/s] 78%|███████▊  | 313175/400000 [00:36<00:10, 8523.62it/s] 79%|███████▊  | 314048/400000 [00:37<00:10, 8581.59it/s] 79%|███████▊  | 314925/400000 [00:37<00:09, 8635.41it/s] 79%|███████▉  | 315789/400000 [00:37<00:09, 8581.08it/s] 79%|███████▉  | 316649/400000 [00:37<00:09, 8586.39it/s] 79%|███████▉  | 317508/400000 [00:37<00:09, 8585.54it/s] 80%|███████▉  | 318367/400000 [00:37<00:09, 8580.60it/s] 80%|███████▉  | 319226/400000 [00:37<00:09, 8554.10it/s] 80%|████████  | 320082/400000 [00:37<00:09, 8525.76it/s] 80%|████████  | 320939/400000 [00:37<00:09, 8537.86it/s] 80%|████████  | 321793/400000 [00:37<00:09, 8471.82it/s] 81%|████████  | 322646/400000 [00:38<00:09, 8489.06it/s] 81%|████████  | 323501/400000 [00:38<00:08, 8505.49it/s] 81%|████████  | 324352/400000 [00:38<00:08, 8490.06it/s] 81%|████████▏ | 325202/400000 [00:38<00:08, 8481.10it/s] 82%|████████▏ | 326054/400000 [00:38<00:08, 8491.65it/s] 82%|████████▏ | 326906/400000 [00:38<00:08, 8497.44it/s] 82%|████████▏ | 327763/400000 [00:38<00:08, 8516.62it/s] 82%|████████▏ | 328615/400000 [00:38<00:08, 8495.81it/s] 82%|████████▏ | 329479/400000 [00:38<00:08, 8538.12it/s] 83%|████████▎ | 330350/400000 [00:38<00:08, 8587.43it/s] 83%|████████▎ | 331209/400000 [00:39<00:08, 8546.95it/s] 83%|████████▎ | 332064/400000 [00:39<00:07, 8529.25it/s] 83%|████████▎ | 332918/400000 [00:39<00:07, 8487.89it/s] 83%|████████▎ | 333781/400000 [00:39<00:07, 8527.52it/s] 84%|████████▎ | 334644/400000 [00:39<00:07, 8555.18it/s] 84%|████████▍ | 335510/400000 [00:39<00:07, 8583.65it/s] 84%|████████▍ | 336369/400000 [00:39<00:07, 8544.59it/s] 84%|████████▍ | 337224/400000 [00:39<00:07, 8510.58it/s] 85%|████████▍ | 338083/400000 [00:39<00:07, 8532.54it/s] 85%|████████▍ | 338937/400000 [00:40<00:07, 8479.01it/s] 85%|████████▍ | 339797/400000 [00:40<00:07, 8514.55it/s] 85%|████████▌ | 340653/400000 [00:40<00:06, 8527.84it/s] 85%|████████▌ | 341506/400000 [00:40<00:06, 8505.15it/s] 86%|████████▌ | 342357/400000 [00:40<00:06, 8503.21it/s] 86%|████████▌ | 343218/400000 [00:40<00:06, 8534.71it/s] 86%|████████▌ | 344072/400000 [00:40<00:06, 8536.09it/s] 86%|████████▌ | 344927/400000 [00:40<00:06, 8540.22it/s] 86%|████████▋ | 345782/400000 [00:40<00:06, 8453.12it/s] 87%|████████▋ | 346648/400000 [00:40<00:06, 8513.95it/s] 87%|████████▋ | 347505/400000 [00:41<00:06, 8527.71it/s] 87%|████████▋ | 348368/400000 [00:41<00:06, 8555.46it/s] 87%|████████▋ | 349224/400000 [00:41<00:05, 8548.15it/s] 88%|████████▊ | 350083/400000 [00:41<00:05, 8558.83it/s] 88%|████████▊ | 350949/400000 [00:41<00:05, 8587.10it/s] 88%|████████▊ | 351808/400000 [00:41<00:05, 8515.97it/s] 88%|████████▊ | 352660/400000 [00:41<00:05, 8506.22it/s] 88%|████████▊ | 353511/400000 [00:41<00:05, 8491.61it/s] 89%|████████▊ | 354362/400000 [00:41<00:05, 8494.54it/s] 89%|████████▉ | 355217/400000 [00:41<00:05, 8508.96it/s] 89%|████████▉ | 356068/400000 [00:42<00:05, 8376.24it/s] 89%|████████▉ | 356927/400000 [00:42<00:05, 8439.10it/s] 89%|████████▉ | 357772/400000 [00:42<00:05, 8436.36it/s] 90%|████████▉ | 358616/400000 [00:42<00:04, 8435.80it/s] 90%|████████▉ | 359476/400000 [00:42<00:04, 8484.34it/s] 90%|█████████ | 360325/400000 [00:42<00:04, 8263.51it/s] 90%|█████████ | 361157/400000 [00:42<00:04, 8280.09it/s] 91%|█████████ | 362004/400000 [00:42<00:04, 8336.02it/s] 91%|█████████ | 362852/400000 [00:42<00:04, 8378.47it/s] 91%|█████████ | 363691/400000 [00:42<00:04, 8310.91it/s] 91%|█████████ | 364536/400000 [00:43<00:04, 8350.80it/s] 91%|█████████▏| 365396/400000 [00:43<00:04, 8421.63it/s] 92%|█████████▏| 366239/400000 [00:43<00:04, 8417.09it/s] 92%|█████████▏| 367089/400000 [00:43<00:03, 8441.08it/s] 92%|█████████▏| 367939/400000 [00:43<00:03, 8456.29it/s] 92%|█████████▏| 368802/400000 [00:43<00:03, 8507.02it/s] 92%|█████████▏| 369665/400000 [00:43<00:03, 8542.58it/s] 93%|█████████▎| 370520/400000 [00:43<00:03, 8478.62it/s] 93%|█████████▎| 371376/400000 [00:43<00:03, 8502.40it/s] 93%|█████████▎| 372238/400000 [00:43<00:03, 8536.90it/s] 93%|█████████▎| 373105/400000 [00:44<00:03, 8575.37it/s] 93%|█████████▎| 373979/400000 [00:44<00:03, 8623.38it/s] 94%|█████████▎| 374842/400000 [00:44<00:02, 8553.93it/s] 94%|█████████▍| 375698/400000 [00:44<00:02, 8535.21it/s] 94%|█████████▍| 376568/400000 [00:44<00:02, 8583.27it/s] 94%|█████████▍| 377428/400000 [00:44<00:02, 8586.89it/s] 95%|█████████▍| 378296/400000 [00:44<00:02, 8612.78it/s] 95%|█████████▍| 379158/400000 [00:44<00:02, 8489.02it/s] 95%|█████████▌| 380027/400000 [00:44<00:02, 8547.33it/s] 95%|█████████▌| 380892/400000 [00:44<00:02, 8576.67it/s] 95%|█████████▌| 381750/400000 [00:45<00:02, 8541.84it/s] 96%|█████████▌| 382610/400000 [00:45<00:02, 8556.53it/s] 96%|█████████▌| 383466/400000 [00:45<00:01, 8549.38it/s] 96%|█████████▌| 384338/400000 [00:45<00:01, 8599.86it/s] 96%|█████████▋| 385209/400000 [00:45<00:01, 8630.52it/s] 97%|█████████▋| 386077/400000 [00:45<00:01, 8644.07it/s] 97%|█████████▋| 386948/400000 [00:45<00:01, 8663.10it/s] 97%|█████████▋| 387815/400000 [00:45<00:01, 8624.47it/s] 97%|█████████▋| 388694/400000 [00:45<00:01, 8672.92it/s] 97%|█████████▋| 389569/400000 [00:45<00:01, 8693.37it/s] 98%|█████████▊| 390439/400000 [00:46<00:01, 8520.65it/s] 98%|█████████▊| 391302/400000 [00:46<00:01, 8550.85it/s] 98%|█████████▊| 392163/400000 [00:46<00:00, 8566.61it/s] 98%|█████████▊| 393035/400000 [00:46<00:00, 8610.07it/s] 98%|█████████▊| 393897/400000 [00:46<00:00, 8488.90it/s] 99%|█████████▊| 394747/400000 [00:46<00:00, 8470.80it/s] 99%|█████████▉| 395595/400000 [00:46<00:00, 8310.99it/s] 99%|█████████▉| 396428/400000 [00:46<00:00, 8242.95it/s] 99%|█████████▉| 397293/400000 [00:46<00:00, 8360.48it/s]100%|█████████▉| 398162/400000 [00:46<00:00, 8454.24it/s]100%|█████████▉| 399040/400000 [00:47<00:00, 8547.72it/s]100%|█████████▉| 399906/400000 [00:47<00:00, 8580.42it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8477.04it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f6fe1aefb38>, <torchtext.data.dataset.TabularDataset object at 0x7f6ff406a4e0>, <torchtext.vocab.Vocab object at 0x7f6fe1aefb70>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.31 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.13 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.13 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.04 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.82 file/s]2020-07-04 06:20:28.753712: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-04 06:20:28.758330: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-07-04 06:20:28.758634: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557cef1dbaa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 06:20:28.758656: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 90112/9912422 [00:00<00:10, 893731.16it/s] 86%|████████▌ | 8503296/9912422 [00:00<00:01, 1270971.86it/s]9920512it [00:00, 46689403.58it/s]                            
0it [00:00, ?it/s]32768it [00:00, 702484.34it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1032525.20it/s]1654784it [00:00, 12581170.93it/s]                           
0it [00:00, ?it/s]8192it [00:00, 212136.43it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
