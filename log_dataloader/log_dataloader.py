
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f723318c0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f723318c0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f729e4d51e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f729e4d51e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f72bc81de18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f72bc81de18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f724b803620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f724b803620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f724b803620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3457024/9912422 [00:00<00:00, 34555855.13it/s]9920512it [00:00, 32872755.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 909108.04it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 145552.33it/s]1654784it [00:00, 10313961.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185601.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7248a295f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7232837c88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f724b803268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f724b803268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f724b803268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:32,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:31,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:04,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:03,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:02,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:01,  2.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:42,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:42,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:41,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:41,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:40,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:40,  3.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:28,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:27,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:27,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:26,  4.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:18,  6.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:05, 17.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 22.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 29.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 36.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 44.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 53.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 60.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 73.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.50 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.36s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 78.50 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.17 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.48s/ url]
0 examples [00:00, ? examples/s]2020-07-13 18:09:10.062251: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-13 18:09:10.074119: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-13 18:09:10.074278: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fbe92d0540 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-13 18:09:10.074294: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
40 examples [00:00, 399.28 examples/s]149 examples [00:00, 492.58 examples/s]248 examples [00:00, 579.26 examples/s]341 examples [00:00, 652.41 examples/s]442 examples [00:00, 729.38 examples/s]537 examples [00:00, 783.60 examples/s]647 examples [00:00, 856.56 examples/s]748 examples [00:00, 895.96 examples/s]855 examples [00:00, 940.85 examples/s]965 examples [00:01, 982.40 examples/s]1081 examples [00:01, 1029.13 examples/s]1187 examples [00:01, 1026.59 examples/s]1297 examples [00:01, 1045.40 examples/s]1408 examples [00:01, 1061.02 examples/s]1519 examples [00:01, 1073.34 examples/s]1631 examples [00:01, 1085.70 examples/s]1743 examples [00:01, 1095.39 examples/s]1855 examples [00:01, 1101.00 examples/s]1967 examples [00:01, 1105.14 examples/s]2078 examples [00:02, 1098.92 examples/s]2189 examples [00:02, 1101.08 examples/s]2300 examples [00:02, 1095.57 examples/s]2416 examples [00:02, 1112.84 examples/s]2534 examples [00:02, 1130.05 examples/s]2648 examples [00:02, 1110.84 examples/s]2760 examples [00:02, 1077.77 examples/s]2870 examples [00:02, 1082.01 examples/s]2986 examples [00:02, 1102.82 examples/s]3102 examples [00:02, 1118.57 examples/s]3215 examples [00:03, 1100.75 examples/s]3330 examples [00:03, 1115.02 examples/s]3445 examples [00:03, 1124.33 examples/s]3560 examples [00:03, 1131.64 examples/s]3677 examples [00:03, 1141.58 examples/s]3795 examples [00:03, 1151.56 examples/s]3911 examples [00:03, 1152.02 examples/s]4029 examples [00:03, 1159.76 examples/s]4146 examples [00:03, 1158.33 examples/s]4262 examples [00:03, 1134.40 examples/s]4376 examples [00:04, 1132.68 examples/s]4496 examples [00:04, 1151.75 examples/s]4612 examples [00:04, 1110.99 examples/s]4731 examples [00:04, 1131.28 examples/s]4850 examples [00:04, 1147.18 examples/s]4966 examples [00:04, 1138.45 examples/s]5081 examples [00:04, 1133.66 examples/s]5195 examples [00:04, 1133.46 examples/s]5313 examples [00:04, 1144.59 examples/s]5430 examples [00:04, 1151.36 examples/s]5546 examples [00:05, 1143.50 examples/s]5662 examples [00:05, 1145.86 examples/s]5780 examples [00:05, 1155.43 examples/s]5896 examples [00:05, 1141.38 examples/s]6015 examples [00:05, 1155.16 examples/s]6131 examples [00:05, 1150.24 examples/s]6248 examples [00:05, 1155.49 examples/s]6364 examples [00:05, 1155.68 examples/s]6480 examples [00:05, 1151.45 examples/s]6600 examples [00:05, 1164.75 examples/s]6717 examples [00:06, 1154.00 examples/s]6833 examples [00:06, 1155.26 examples/s]6949 examples [00:06, 1154.50 examples/s]7065 examples [00:06, 1145.71 examples/s]7181 examples [00:06, 1149.51 examples/s]7296 examples [00:06, 1145.56 examples/s]7413 examples [00:06, 1152.17 examples/s]7529 examples [00:06, 1140.21 examples/s]7644 examples [00:06, 1124.94 examples/s]7757 examples [00:07, 1117.72 examples/s]7869 examples [00:07, 1105.00 examples/s]7985 examples [00:07, 1119.45 examples/s]8098 examples [00:07, 1089.02 examples/s]8216 examples [00:07, 1114.09 examples/s]8328 examples [00:07, 1115.70 examples/s]8443 examples [00:07, 1122.93 examples/s]8560 examples [00:07, 1134.69 examples/s]8675 examples [00:07, 1138.85 examples/s]8793 examples [00:07, 1148.52 examples/s]8908 examples [00:08, 1147.17 examples/s]9023 examples [00:08, 1147.35 examples/s]9139 examples [00:08, 1149.05 examples/s]9254 examples [00:08, 1146.71 examples/s]9369 examples [00:08, 1139.06 examples/s]9484 examples [00:08, 1139.37 examples/s]9598 examples [00:08, 1100.70 examples/s]9709 examples [00:08, 1098.65 examples/s]9825 examples [00:08, 1115.85 examples/s]9937 examples [00:08, 1113.35 examples/s]10049 examples [00:09, 1070.88 examples/s]10163 examples [00:09, 1089.65 examples/s]10279 examples [00:09, 1109.21 examples/s]10399 examples [00:09, 1132.13 examples/s]10513 examples [00:09, 1123.74 examples/s]10631 examples [00:09, 1139.70 examples/s]10746 examples [00:09, 1139.21 examples/s]10867 examples [00:09, 1156.87 examples/s]10984 examples [00:09, 1159.89 examples/s]11101 examples [00:09, 1157.04 examples/s]11217 examples [00:10, 1142.29 examples/s]11332 examples [00:10, 1129.66 examples/s]11446 examples [00:10, 1125.03 examples/s]11559 examples [00:10, 1086.89 examples/s]11675 examples [00:10, 1107.00 examples/s]11787 examples [00:10, 1090.68 examples/s]11899 examples [00:10, 1096.39 examples/s]12014 examples [00:10, 1110.13 examples/s]12127 examples [00:10, 1115.91 examples/s]12244 examples [00:11, 1131.46 examples/s]12358 examples [00:11, 1131.07 examples/s]12472 examples [00:11, 1127.89 examples/s]12587 examples [00:11, 1132.66 examples/s]12703 examples [00:11, 1139.35 examples/s]12821 examples [00:11, 1149.26 examples/s]12941 examples [00:11, 1163.70 examples/s]13058 examples [00:11, 1160.11 examples/s]13175 examples [00:11, 1152.42 examples/s]13291 examples [00:11, 1152.41 examples/s]13407 examples [00:12, 1154.26 examples/s]13524 examples [00:12, 1155.74 examples/s]13640 examples [00:12, 1152.64 examples/s]13756 examples [00:12, 1149.49 examples/s]13871 examples [00:12, 1147.92 examples/s]13986 examples [00:12, 1143.39 examples/s]14101 examples [00:12, 1129.51 examples/s]14214 examples [00:12, 1087.35 examples/s]14332 examples [00:12, 1111.97 examples/s]14447 examples [00:12, 1122.74 examples/s]14560 examples [00:13, 1118.20 examples/s]14673 examples [00:13, 1121.34 examples/s]14786 examples [00:13, 1123.80 examples/s]14904 examples [00:13, 1138.07 examples/s]15018 examples [00:13, 1113.74 examples/s]15137 examples [00:13, 1133.41 examples/s]15251 examples [00:13, 1135.12 examples/s]15369 examples [00:13, 1145.38 examples/s]15485 examples [00:13, 1147.55 examples/s]15601 examples [00:13, 1149.31 examples/s]15716 examples [00:14, 1146.70 examples/s]15831 examples [00:14, 1139.56 examples/s]15945 examples [00:14, 1136.85 examples/s]16059 examples [00:14, 1130.74 examples/s]16175 examples [00:14, 1138.19 examples/s]16294 examples [00:14, 1151.49 examples/s]16411 examples [00:14, 1154.20 examples/s]16527 examples [00:14, 1149.15 examples/s]16642 examples [00:14, 1149.32 examples/s]16757 examples [00:14, 1144.22 examples/s]16872 examples [00:15, 1139.39 examples/s]16989 examples [00:15, 1148.32 examples/s]17104 examples [00:15, 1143.38 examples/s]17219 examples [00:15, 1142.75 examples/s]17334 examples [00:15, 1143.94 examples/s]17453 examples [00:15, 1156.84 examples/s]17569 examples [00:15, 1126.97 examples/s]17682 examples [00:15, 1124.91 examples/s]17801 examples [00:15, 1143.67 examples/s]17919 examples [00:15, 1153.62 examples/s]18037 examples [00:16, 1159.11 examples/s]18154 examples [00:16, 1152.91 examples/s]18270 examples [00:16, 1143.13 examples/s]18386 examples [00:16, 1147.41 examples/s]18501 examples [00:16, 1116.31 examples/s]18620 examples [00:16, 1135.84 examples/s]18738 examples [00:16, 1148.52 examples/s]18854 examples [00:16, 1147.59 examples/s]18973 examples [00:16, 1157.64 examples/s]19089 examples [00:16, 1153.98 examples/s]19208 examples [00:17, 1161.98 examples/s]19325 examples [00:17, 1164.30 examples/s]19442 examples [00:17, 1153.21 examples/s]19560 examples [00:17, 1160.28 examples/s]19677 examples [00:17, 1153.13 examples/s]19793 examples [00:17, 1143.92 examples/s]19910 examples [00:17, 1149.29 examples/s]20025 examples [00:17, 1092.34 examples/s]20135 examples [00:17, 1087.27 examples/s]20251 examples [00:18, 1106.25 examples/s]20369 examples [00:18, 1125.92 examples/s]20484 examples [00:18, 1130.99 examples/s]20598 examples [00:18, 1111.66 examples/s]20714 examples [00:18, 1123.73 examples/s]20827 examples [00:18, 1116.12 examples/s]20939 examples [00:18, 1114.29 examples/s]21060 examples [00:18, 1139.21 examples/s]21177 examples [00:18, 1146.70 examples/s]21295 examples [00:18, 1155.60 examples/s]21411 examples [00:19, 1151.92 examples/s]21528 examples [00:19, 1155.29 examples/s]21645 examples [00:19, 1158.08 examples/s]21761 examples [00:19, 1145.79 examples/s]21876 examples [00:19, 1143.34 examples/s]21991 examples [00:19, 1077.46 examples/s]22106 examples [00:19, 1097.04 examples/s]22219 examples [00:19, 1106.36 examples/s]22334 examples [00:19, 1116.98 examples/s]22451 examples [00:19, 1130.94 examples/s]22565 examples [00:20, 1131.83 examples/s]22679 examples [00:20, 1133.44 examples/s]22793 examples [00:20, 1129.99 examples/s]22907 examples [00:20, 1118.99 examples/s]23021 examples [00:20, 1123.24 examples/s]23134 examples [00:20, 1120.17 examples/s]23247 examples [00:20, 1121.36 examples/s]23360 examples [00:20, 1104.09 examples/s]23471 examples [00:20, 1097.89 examples/s]23585 examples [00:20, 1109.09 examples/s]23701 examples [00:21, 1121.59 examples/s]23814 examples [00:21, 1120.20 examples/s]23928 examples [00:21, 1125.42 examples/s]24041 examples [00:21, 1115.18 examples/s]24154 examples [00:21, 1119.24 examples/s]24270 examples [00:21, 1129.37 examples/s]24385 examples [00:21, 1135.36 examples/s]24502 examples [00:21, 1144.64 examples/s]24617 examples [00:21, 1126.13 examples/s]24731 examples [00:22, 1130.25 examples/s]24845 examples [00:22, 1095.90 examples/s]24965 examples [00:22, 1123.28 examples/s]25078 examples [00:22, 1121.49 examples/s]25191 examples [00:22, 1121.71 examples/s]25308 examples [00:22, 1133.14 examples/s]25422 examples [00:22, 1072.64 examples/s]25536 examples [00:22, 1089.91 examples/s]25652 examples [00:22, 1108.72 examples/s]25768 examples [00:22, 1121.08 examples/s]25883 examples [00:23, 1129.02 examples/s]25998 examples [00:23, 1135.16 examples/s]26114 examples [00:23, 1141.28 examples/s]26229 examples [00:23, 1142.02 examples/s]26344 examples [00:23, 1141.30 examples/s]26460 examples [00:23, 1145.52 examples/s]26575 examples [00:23, 1137.19 examples/s]26689 examples [00:23, 1135.64 examples/s]26804 examples [00:23, 1137.13 examples/s]26918 examples [00:23, 1131.19 examples/s]27032 examples [00:24, 1117.02 examples/s]27146 examples [00:24, 1123.77 examples/s]27260 examples [00:24, 1127.13 examples/s]27373 examples [00:24, 1125.00 examples/s]27489 examples [00:24, 1133.55 examples/s]27604 examples [00:24, 1137.55 examples/s]27718 examples [00:24, 1116.78 examples/s]27834 examples [00:24, 1127.47 examples/s]27952 examples [00:24, 1141.39 examples/s]28067 examples [00:24, 1135.05 examples/s]28181 examples [00:25, 1134.14 examples/s]28295 examples [00:25, 1134.61 examples/s]28411 examples [00:25, 1140.04 examples/s]28527 examples [00:25, 1143.71 examples/s]28642 examples [00:25, 1131.30 examples/s]28756 examples [00:25, 1125.19 examples/s]28869 examples [00:25, 1100.40 examples/s]28984 examples [00:25, 1112.74 examples/s]29101 examples [00:25, 1128.72 examples/s]29216 examples [00:25, 1133.59 examples/s]29330 examples [00:26, 1135.46 examples/s]29444 examples [00:26, 1135.65 examples/s]29558 examples [00:26, 1134.57 examples/s]29672 examples [00:26, 1125.16 examples/s]29785 examples [00:26, 1113.88 examples/s]29901 examples [00:26, 1125.03 examples/s]30014 examples [00:26, 1055.21 examples/s]30128 examples [00:26, 1076.80 examples/s]30242 examples [00:26, 1092.32 examples/s]30353 examples [00:27, 1094.87 examples/s]30464 examples [00:27, 1099.11 examples/s]30580 examples [00:27, 1114.70 examples/s]30692 examples [00:27, 1115.96 examples/s]30805 examples [00:27, 1119.39 examples/s]30918 examples [00:27, 1114.51 examples/s]31030 examples [00:27, 1108.48 examples/s]31141 examples [00:27, 1081.03 examples/s]31255 examples [00:27, 1096.59 examples/s]31365 examples [00:27, 1078.15 examples/s]31475 examples [00:28, 1082.09 examples/s]31586 examples [00:28, 1087.86 examples/s]31696 examples [00:28, 1089.85 examples/s]31807 examples [00:28, 1093.65 examples/s]31919 examples [00:28, 1098.82 examples/s]32029 examples [00:28, 1088.79 examples/s]32141 examples [00:28, 1093.40 examples/s]32251 examples [00:28, 1069.29 examples/s]32359 examples [00:28, 1054.46 examples/s]32465 examples [00:28, 1046.46 examples/s]32578 examples [00:29, 1069.29 examples/s]32691 examples [00:29, 1086.63 examples/s]32802 examples [00:29, 1091.07 examples/s]32918 examples [00:29, 1108.74 examples/s]33031 examples [00:29, 1115.01 examples/s]33144 examples [00:29, 1117.85 examples/s]33256 examples [00:29, 1113.56 examples/s]33368 examples [00:29, 1107.17 examples/s]33479 examples [00:29, 1105.44 examples/s]33590 examples [00:29, 1104.40 examples/s]33701 examples [00:30, 1100.59 examples/s]33815 examples [00:30, 1111.12 examples/s]33927 examples [00:30, 1103.06 examples/s]34038 examples [00:30, 1094.98 examples/s]34150 examples [00:30, 1101.34 examples/s]34261 examples [00:30, 1098.84 examples/s]34372 examples [00:30, 1099.21 examples/s]34482 examples [00:30, 1098.51 examples/s]34599 examples [00:30, 1118.96 examples/s]34711 examples [00:30, 1096.24 examples/s]34823 examples [00:31, 1101.45 examples/s]34934 examples [00:31, 1099.79 examples/s]35045 examples [00:31, 1093.54 examples/s]35161 examples [00:31, 1110.21 examples/s]35273 examples [00:31, 1102.70 examples/s]35384 examples [00:31, 1099.39 examples/s]35498 examples [00:31, 1111.19 examples/s]35610 examples [00:31, 1112.34 examples/s]35722 examples [00:31, 1106.81 examples/s]35833 examples [00:32, 1080.82 examples/s]35944 examples [00:32, 1086.61 examples/s]36057 examples [00:32, 1098.25 examples/s]36171 examples [00:32, 1108.41 examples/s]36288 examples [00:32, 1123.40 examples/s]36401 examples [00:32, 1119.23 examples/s]36514 examples [00:32, 1103.27 examples/s]36625 examples [00:32, 1075.01 examples/s]36739 examples [00:32, 1092.27 examples/s]36853 examples [00:32, 1104.61 examples/s]36964 examples [00:33, 1094.44 examples/s]37074 examples [00:33, 1080.15 examples/s]37186 examples [00:33, 1090.78 examples/s]37296 examples [00:33, 1083.96 examples/s]37408 examples [00:33, 1093.46 examples/s]37521 examples [00:33, 1102.63 examples/s]37632 examples [00:33, 1101.13 examples/s]37743 examples [00:33, 1090.39 examples/s]37853 examples [00:33, 998.36 examples/s] 37967 examples [00:33, 1034.88 examples/s]38079 examples [00:34, 1056.48 examples/s]38193 examples [00:34, 1078.89 examples/s]38308 examples [00:34, 1098.95 examples/s]38422 examples [00:34, 1108.32 examples/s]38534 examples [00:34, 1108.69 examples/s]38646 examples [00:34, 1103.91 examples/s]38759 examples [00:34, 1111.51 examples/s]38873 examples [00:34, 1118.49 examples/s]38989 examples [00:34, 1128.59 examples/s]39102 examples [00:34, 1098.46 examples/s]39213 examples [00:35, 1099.94 examples/s]39328 examples [00:35, 1112.57 examples/s]39441 examples [00:35, 1115.54 examples/s]39556 examples [00:35, 1125.49 examples/s]39669 examples [00:35, 1116.39 examples/s]39781 examples [00:35, 1114.13 examples/s]39893 examples [00:35, 1115.01 examples/s]40005 examples [00:35, 1061.05 examples/s]40121 examples [00:35, 1087.39 examples/s]40237 examples [00:36, 1108.19 examples/s]40350 examples [00:36, 1112.61 examples/s]40466 examples [00:36, 1123.77 examples/s]40579 examples [00:36, 1118.19 examples/s]40692 examples [00:36, 1118.77 examples/s]40805 examples [00:36, 1121.52 examples/s]40918 examples [00:36, 1122.77 examples/s]41031 examples [00:36, 1082.94 examples/s]41142 examples [00:36, 1088.19 examples/s]41252 examples [00:36, 1079.18 examples/s]41370 examples [00:37, 1106.47 examples/s]41481 examples [00:37, 1104.39 examples/s]41592 examples [00:37, 1105.62 examples/s]41706 examples [00:37, 1114.25 examples/s]41820 examples [00:37, 1119.33 examples/s]41933 examples [00:37, 1120.90 examples/s]42046 examples [00:37, 1116.65 examples/s]42162 examples [00:37, 1127.31 examples/s]42276 examples [00:37, 1129.16 examples/s]42390 examples [00:37, 1132.25 examples/s]42504 examples [00:38, 1073.68 examples/s]42616 examples [00:38, 1085.65 examples/s]42726 examples [00:38, 1060.32 examples/s]42835 examples [00:38, 1068.84 examples/s]42946 examples [00:38, 1079.25 examples/s]43058 examples [00:38, 1089.53 examples/s]43168 examples [00:38, 1072.33 examples/s]43281 examples [00:38, 1087.65 examples/s]43395 examples [00:38, 1100.94 examples/s]43506 examples [00:38, 1101.37 examples/s]43617 examples [00:39, 1100.58 examples/s]43728 examples [00:39, 1082.60 examples/s]43839 examples [00:39, 1089.94 examples/s]43952 examples [00:39, 1101.59 examples/s]44064 examples [00:39, 1105.97 examples/s]44176 examples [00:39, 1107.42 examples/s]44287 examples [00:39, 1095.02 examples/s]44401 examples [00:39, 1106.38 examples/s]44516 examples [00:39, 1116.80 examples/s]44628 examples [00:40, 1113.56 examples/s]44740 examples [00:40, 1107.08 examples/s]44851 examples [00:40, 1105.62 examples/s]44964 examples [00:40, 1109.92 examples/s]45080 examples [00:40, 1123.63 examples/s]45193 examples [00:40, 1119.99 examples/s]45306 examples [00:40, 1078.67 examples/s]45415 examples [00:40, 1056.91 examples/s]45528 examples [00:40, 1075.58 examples/s]45646 examples [00:40, 1103.30 examples/s]45762 examples [00:41, 1117.39 examples/s]45875 examples [00:41, 1071.91 examples/s]45989 examples [00:41, 1090.44 examples/s]46104 examples [00:41, 1107.45 examples/s]46220 examples [00:41, 1120.85 examples/s]46335 examples [00:41, 1128.75 examples/s]46449 examples [00:41, 1131.86 examples/s]46565 examples [00:41, 1140.03 examples/s]46680 examples [00:41, 1137.34 examples/s]46796 examples [00:41, 1142.43 examples/s]46911 examples [00:42, 1139.45 examples/s]47025 examples [00:42, 1134.47 examples/s]47139 examples [00:42, 1132.78 examples/s]47253 examples [00:42, 1107.48 examples/s]47364 examples [00:42, 1107.47 examples/s]47476 examples [00:42, 1110.74 examples/s]47588 examples [00:42, 1105.04 examples/s]47699 examples [00:42, 1091.98 examples/s]47813 examples [00:42, 1103.91 examples/s]47928 examples [00:42, 1115.87 examples/s]48040 examples [00:43, 1115.94 examples/s]48153 examples [00:43, 1118.62 examples/s]48265 examples [00:43, 1116.36 examples/s]48380 examples [00:43, 1125.91 examples/s]48493 examples [00:43, 1120.34 examples/s]48606 examples [00:43, 1107.19 examples/s]48717 examples [00:43, 1102.34 examples/s]48828 examples [00:43, 1061.67 examples/s]48942 examples [00:43, 1083.07 examples/s]49057 examples [00:43, 1101.58 examples/s]49168 examples [00:44, 1102.97 examples/s]49279 examples [00:44, 1064.20 examples/s]49386 examples [00:44, 1047.25 examples/s]49504 examples [00:44, 1083.00 examples/s]49622 examples [00:44, 1108.90 examples/s]49738 examples [00:44, 1121.75 examples/s]49851 examples [00:44, 1114.99 examples/s]49964 examples [00:44, 1119.24 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7873/50000 [00:00<00:00, 78728.01 examples/s] 43%|████▎     | 21610/50000 [00:00<00:00, 90290.72 examples/s] 72%|███████▏  | 35804/50000 [00:00<00:00, 101354.84 examples/s] 99%|█████████▉| 49716/50000 [00:00<00:00, 110339.70 examples/s]                                                                0 examples [00:00, ? examples/s]98 examples [00:00, 971.00 examples/s]216 examples [00:00, 1024.12 examples/s]334 examples [00:00, 1065.14 examples/s]450 examples [00:00, 1090.21 examples/s]566 examples [00:00, 1107.55 examples/s]685 examples [00:00, 1130.23 examples/s]803 examples [00:00, 1143.91 examples/s]920 examples [00:00, 1149.98 examples/s]1038 examples [00:00, 1156.91 examples/s]1156 examples [00:01, 1160.93 examples/s]1272 examples [00:01, 1158.05 examples/s]1390 examples [00:01, 1163.08 examples/s]1507 examples [00:01, 1164.87 examples/s]1626 examples [00:01, 1169.56 examples/s]1743 examples [00:01, 1164.20 examples/s]1859 examples [00:01, 1149.53 examples/s]1974 examples [00:01, 1132.04 examples/s]2093 examples [00:01, 1147.51 examples/s]2208 examples [00:01, 1140.67 examples/s]2323 examples [00:02, 1107.65 examples/s]2438 examples [00:02, 1117.78 examples/s]2554 examples [00:02, 1128.90 examples/s]2668 examples [00:02, 1125.53 examples/s]2781 examples [00:02, 1123.97 examples/s]2895 examples [00:02, 1126.21 examples/s]3012 examples [00:02, 1138.82 examples/s]3126 examples [00:02, 1122.61 examples/s]3239 examples [00:02, 1122.19 examples/s]3355 examples [00:02, 1132.00 examples/s]3469 examples [00:03, 1111.34 examples/s]3585 examples [00:03, 1123.83 examples/s]3702 examples [00:03, 1135.20 examples/s]3820 examples [00:03, 1147.59 examples/s]3935 examples [00:03, 1131.70 examples/s]4049 examples [00:03, 1133.76 examples/s]4167 examples [00:03, 1147.05 examples/s]4282 examples [00:03, 1115.46 examples/s]4400 examples [00:03, 1133.31 examples/s]4521 examples [00:03, 1153.19 examples/s]4639 examples [00:04, 1159.82 examples/s]4759 examples [00:04, 1170.04 examples/s]4879 examples [00:04, 1178.12 examples/s]4997 examples [00:04, 1176.92 examples/s]5115 examples [00:04, 1170.09 examples/s]5233 examples [00:04, 1169.98 examples/s]5351 examples [00:04, 1165.78 examples/s]5469 examples [00:04, 1166.91 examples/s]5586 examples [00:04, 1164.72 examples/s]5703 examples [00:04, 1160.50 examples/s]5820 examples [00:05, 1086.43 examples/s]5933 examples [00:05, 1097.01 examples/s]6047 examples [00:05, 1108.53 examples/s]6159 examples [00:05, 1110.60 examples/s]6271 examples [00:05, 1109.53 examples/s]6383 examples [00:05, 1112.16 examples/s]6495 examples [00:05, 1107.59 examples/s]6609 examples [00:05, 1114.71 examples/s]6726 examples [00:05, 1128.29 examples/s]6841 examples [00:06, 1133.23 examples/s]6955 examples [00:06, 1134.41 examples/s]7071 examples [00:06, 1141.90 examples/s]7186 examples [00:06, 1138.22 examples/s]7302 examples [00:06, 1143.86 examples/s]7417 examples [00:06, 1143.76 examples/s]7532 examples [00:06, 1131.71 examples/s]7649 examples [00:06, 1140.70 examples/s]7764 examples [00:06, 1110.07 examples/s]7880 examples [00:06, 1121.78 examples/s]7994 examples [00:07, 1124.61 examples/s]8107 examples [00:07, 1098.20 examples/s]8223 examples [00:07, 1115.22 examples/s]8339 examples [00:07, 1127.44 examples/s]8452 examples [00:07, 1121.78 examples/s]8566 examples [00:07, 1125.65 examples/s]8682 examples [00:07, 1133.93 examples/s]8796 examples [00:07, 1133.39 examples/s]8910 examples [00:07, 1128.27 examples/s]9028 examples [00:07, 1142.77 examples/s]9143 examples [00:08, 1122.98 examples/s]9256 examples [00:08, 1109.68 examples/s]9370 examples [00:08, 1117.26 examples/s]9482 examples [00:08, 1115.75 examples/s]9597 examples [00:08, 1125.14 examples/s]9710 examples [00:08, 1105.74 examples/s]9823 examples [00:08, 1110.87 examples/s]9935 examples [00:08, 1100.57 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteH2WDCY/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteH2WDCY/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f724b803620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f724b803620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f724b803620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f71d4c792b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f71d4cafb70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f724b803620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f724b803620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f724b803620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f72328374a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7232837550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f71c3282268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f71c3282268> 

  function with postional parmater data_info <function split_train_valid at 0x7f71c3282268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f71c3282378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f71c3282378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f71c3282378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=d399766cf2e889b10954b7abf28675dde060454adb18c36cc3f421e1e31dd2da
  Stored in directory: /tmp/pip-ephem-wheel-cache-e7ze4l4j/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:00:35, 9.97kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:02:26, 14.1kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:58:53, 20.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:23:39, 28.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:51:39, 40.7kB/s].vector_cache/glove.6B.zip:   1%|          | 9.14M/862M [00:01<4:04:39, 58.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:01<2:50:48, 82.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:58:54, 118kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:01<1:23:02, 169kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:01<57:50, 241kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<40:27, 343kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.6M/862M [00:02<28:12, 489kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<19:48, 694kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.3M/862M [00:02<13:50, 986kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<09:47, 1.39MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<06:52, 1.96MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<09:47, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<08:44, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<09:44, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:46, 1.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<05:38, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<09:10, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<07:56, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<05:52, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:08<06:58, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<07:33, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<05:57, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<06:18, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<05:46, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<04:19, 3.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<06:07, 2.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:56, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<05:26, 2.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<04:01, 3.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:21, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:31, 2.01MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<04:54, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:26, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<05:49, 2.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<04:20, 2.99MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:05, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:52, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<05:23, 2.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<03:55, 3.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<09:55, 1.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<08:16, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:14, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<07:40, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<05:54, 2.16MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:22<04:18, 2.96MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:46, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<07:25, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<05:31, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:50, 1.85MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:21, 1.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:44, 2.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:09, 3.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<45:46, 275kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<33:22, 378kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<23:35, 533kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<19:23, 646kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<16:06, 778kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<11:54, 1.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<10:19, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:30, 1.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:16, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:17, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:36, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:57, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:09, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:33, 2.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:09, 2.96MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:47, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:34, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:06, 2.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:45, 3.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:17, 1.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:21, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:45, 2.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:08, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:51, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:25, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:55, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<46:24, 260kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<33:43, 357kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<23:51, 504kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<19:26, 617kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<14:48, 809kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<10:38, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:15, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:35, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:18, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<05:13, 2.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:29:21, 133kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:03:43, 186kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<44:48, 264kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<34:02, 346kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<26:14, 449kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<18:52, 623kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<13:18, 881kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<18:03, 648kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:50, 846kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:58, 1.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:41, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<09:08, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:58, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:59, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<11:15:08, 17.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<7:53:32, 24.4kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<5:30:58, 34.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<3:53:40, 49.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<2:44:38, 69.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<1:55:13, 99.6kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<1:23:08, 138kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<59:18, 193kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<41:42, 274kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<31:52, 357kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<23:25, 485kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<16:36, 683kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<14:15, 792kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<11:08, 1.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<08:04, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:16, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:55, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:07, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:12, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:30, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:03, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:39, 3.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<12:47, 868kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:06, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<07:17, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:41, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:36, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:47, 1.90MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:12, 2.61MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<07:40, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:28, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:48, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:54, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:20, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:58, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:13, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:46, 2.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:36, 2.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:02, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:29, 3.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:33, 2.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:27, 3.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:56, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:30, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:19, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:07, 3.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<13:19, 791kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<10:25, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:32, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<07:43, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:31, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:43, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<04:06, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<11:26, 910kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:54, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:55, 1.49MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:43, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:13, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:33<03:06, 3.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<1:11:32, 144kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<50:16, 204kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<36:48, 277kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<26:48, 380kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<18:56, 537kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<15:33, 651kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<12:56, 783kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:33, 1.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<08:17, 1.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:50, 1.47MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:02, 1.99MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:50, 1.71MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:12, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:51, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:29, 2.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<38:21, 259kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<27:51, 356kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<19:39, 503kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<16:02, 615kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:14, 744kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:40, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<06:53, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:16, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:29, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:28, 1.78MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:06, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:14, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:53, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:00, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:28, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:22, 2.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:37, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:10, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:02, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<02:55, 3.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<10:50, 878kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<08:32, 1.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<06:10, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:31, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:34, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:00, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:36, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:43, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<06:22, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:40, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:27, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:42, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:23, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<03:10, 2.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:55, 932kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<07:52, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<05:43, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<06:09, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:09, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:42, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<03:23, 2.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<11:30, 792kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<08:58, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<06:29, 1.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:38, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<06:33, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:59, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:10<03:34, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<13:06, 685kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<10:05, 888kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<07:16, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<07:08, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:52, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:12, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<03:43, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<19:59, 442kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<14:53, 593kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<10:35, 831kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<09:25, 930kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:28, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:26, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:51, 1.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:59, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:41, 2.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:37, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:59, 2.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:06, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:35, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:36, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:35, 3.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<19:47, 429kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<14:42, 577kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<10:27, 810kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:16, 908kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<08:11, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:05, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<04:20, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:21, 999kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:41, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:51, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:19, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:24, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:08, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:59, 2.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:26, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:21, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:56, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:39, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:55, 1.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:51, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:59, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:36, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:42, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:46, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:16, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:19, 2.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:26, 3.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<05:12, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:27, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:16, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:08, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:33, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:35, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:35, 3.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<57:42, 135kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<41:09, 189kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<28:55, 269kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<21:58, 352kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<16:56, 456kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<12:13, 631kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<08:35, 893kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<1:01:22, 125kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<43:42, 175kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<30:41, 249kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<23:09, 328kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<16:57, 448kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<12:01, 630kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<10:09, 741kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<07:45, 971kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:35, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:40, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:28, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:12, 1.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<04:07, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:37, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<02:41, 2.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:36, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:00, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:07, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:16, 3.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<06:30, 1.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:18, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:51, 1.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:22, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:31, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:31, 2.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<02:30, 2.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<3:29:46, 33.9kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<2:27:25, 48.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<1:42:58, 68.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<1:13:23, 96.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<52:42, 134kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<37:11, 189kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<26:58, 259kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<19:35, 356kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<13:50, 502kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<11:14, 615kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<09:16, 745kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<06:47, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<04:48, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<07:34, 904kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:00, 1.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:21, 1.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:37, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:36, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:33, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:11, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:23, 2.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:13, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:36, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:49, 2.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<02:02, 3.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:02, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:53, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:34, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:00, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:28, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:35, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:19, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:14, 2.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:03, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:25, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:40, 2.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<01:56, 3.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<07:10, 878kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<05:39, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:06, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:18, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:39, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:42, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:20, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:35, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:46, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:00, 3.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:42, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:55, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:53, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:25, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:00, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:14, 2.66MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:58, 2.00MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:40, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:59, 2.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:47, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:09, 1.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:29, 2.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:40, 2.17MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:51, 3.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:38, 2.17MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:00, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:21, 2.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<01:41, 3.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<07:46, 731kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<06:01, 942kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<04:20, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:20, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:12, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:13, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:08, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:39, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:59, 2.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<01:28, 3.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<33:40, 163kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<24:05, 227kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<16:55, 322kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<13:03, 414kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<10:14, 528kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<07:23, 730kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:54<05:11, 1.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<08:20, 640kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:23, 835kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<04:33, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:24, 1.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:09, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:09, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:02, 1.71MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:39, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:57, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<01:26, 3.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<31:13, 164kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<22:20, 229kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<15:40, 326kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<12:06, 419kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<08:53, 569kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<06:18, 798kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<04:27, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<40:24, 124kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<29:17, 170kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<20:42, 240kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<15:08, 325kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<11:05, 443kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<07:49, 625kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<06:35, 738kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<05:05, 953kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:40, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:41, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:03, 1.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:41, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:21, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:45, 2.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:19, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:06, 2.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:34, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:10, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:27, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:54, 2.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<01:22, 3.28MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:54, 1.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:11, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:19, 1.93MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:46, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:07, 2.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:32, 2.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:30, 1.25MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:54, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:08, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:29, 1.73MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:36, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:00, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:27, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<03:07, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:36, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:55, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:22, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:31, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:58, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:25, 2.89MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<30:12, 136kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:31, 190kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<15:04, 270kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<11:24, 354kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:47, 458kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<06:18, 637kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<04:24, 900kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<06:48, 583kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<05:10, 766kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<03:41, 1.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:28, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:50, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:03, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:24, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:52, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:54, 1.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:42, 2.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:16, 2.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:45, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:11, 3.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:41, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:54, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:29, 2.42MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:04, 3.32MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:42, 957kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:57, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<02:07, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:17, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:18, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:47, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:47, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:35, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:11, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:36, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:48, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:25, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:31, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:23, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:03, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:41, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:20, 2.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:19, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<00:59, 3.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:24, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:17, 2.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<00:58, 3.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:23, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:34, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:14, 2.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:20, 2.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:13, 2.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:54, 3.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<00:40, 4.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<21:44, 132kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<15:46, 182kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<11:06, 257kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<07:43, 365kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<06:47, 412kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<05:01, 556kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<03:32, 781kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:05, 883kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:26, 1.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:45, 1.53MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:50, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:33, 1.70MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:08, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:24, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:31, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:10, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:51, 2.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:13, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<00:54, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:39, 3.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<06:39, 369kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<04:53, 501kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<03:27, 702kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:56, 810kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:33, 935kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:53, 1.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:40, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:24, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:02, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:14, 1.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:19, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:01, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:43, 3.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:49, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:29, 1.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:05, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:14, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:05, 1.94MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:48, 2.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:02, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:55, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:41, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:56, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:51, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:38, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:53, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:48, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:34, 3.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:38, 1.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:20, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:58, 1.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:41, 2.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:23, 739kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:51, 953kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:19, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:18, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:56, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:40, 2.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:19, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:05, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:47, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:55, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:58, 1.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:45, 2.06MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:45, 1.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:40, 2.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:30, 2.91MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:46, 1.86MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:35, 2.38MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:25, 3.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:39, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:29, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:38, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:42, 1.83MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:32, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:23, 3.22MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:54, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:45, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:32, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:38, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:31, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:28, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:21, 2.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:28, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:17, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:24, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:21, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:22, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:20, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:14, 3.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:13, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:15, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:11, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:14, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:16, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:09, 2.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:07, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 3.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:23<00:04, 2.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.17MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<34:20:26,  3.24it/s]  0%|          | 902/400000 [00:00<23:59:17,  4.62it/s]  0%|          | 1829/400000 [00:00<16:45:22,  6.60it/s]  1%|          | 2740/400000 [00:00<11:42:22,  9.43it/s]  1%|          | 3684/400000 [00:00<8:10:42, 13.46it/s]   1%|          | 4642/400000 [00:00<5:42:52, 19.22it/s]  1%|▏         | 5557/400000 [00:00<3:59:40, 27.43it/s]  2%|▏         | 6519/400000 [00:01<2:47:33, 39.14it/s]  2%|▏         | 7515/400000 [00:01<1:57:11, 55.82it/s]  2%|▏         | 8457/400000 [00:01<1:22:02, 79.54it/s]  2%|▏         | 9418/400000 [00:01<57:29, 113.22it/s]   3%|▎         | 10381/400000 [00:01<40:20, 160.93it/s]  3%|▎         | 11320/400000 [00:01<28:23, 228.20it/s]  3%|▎         | 12277/400000 [00:01<20:01, 322.70it/s]  3%|▎         | 13243/400000 [00:01<14:10, 454.49it/s]  4%|▎         | 14192/400000 [00:01<10:06, 635.93it/s]  4%|▍         | 15135/400000 [00:01<07:16, 881.24it/s]  4%|▍         | 16097/400000 [00:02<05:16, 1211.32it/s]  4%|▍         | 17033/400000 [00:02<03:54, 1634.33it/s]  4%|▍         | 17953/400000 [00:02<02:56, 2168.56it/s]  5%|▍         | 18871/400000 [00:02<02:15, 2810.54it/s]  5%|▍         | 19793/400000 [00:02<01:47, 3550.87it/s]  5%|▌         | 20767/400000 [00:02<01:26, 4386.80it/s]  5%|▌         | 21701/400000 [00:02<01:13, 5152.70it/s]  6%|▌         | 22664/400000 [00:02<01:03, 5987.87it/s]  6%|▌         | 23604/400000 [00:02<00:56, 6718.75it/s]  6%|▌         | 24537/400000 [00:02<00:51, 7250.76it/s]  6%|▋         | 25487/400000 [00:03<00:47, 7804.46it/s]  7%|▋         | 26448/400000 [00:03<00:45, 8270.57it/s]  7%|▋         | 27393/400000 [00:03<00:43, 8591.02it/s]  7%|▋         | 28353/400000 [00:03<00:41, 8869.33it/s]  7%|▋         | 29299/400000 [00:03<00:41, 8896.27it/s]  8%|▊         | 30230/400000 [00:03<00:41, 8931.67it/s]  8%|▊         | 31179/400000 [00:03<00:40, 9090.70it/s]  8%|▊         | 32110/400000 [00:03<00:40, 9067.60it/s]  8%|▊         | 33064/400000 [00:03<00:39, 9203.42it/s]  8%|▊         | 33996/400000 [00:03<00:39, 9172.38it/s]  9%|▊         | 34976/400000 [00:04<00:39, 9351.42it/s]  9%|▉         | 35949/400000 [00:04<00:38, 9459.83it/s]  9%|▉         | 36900/400000 [00:04<00:38, 9472.99it/s]  9%|▉         | 37861/400000 [00:04<00:38, 9512.86it/s] 10%|▉         | 38815/400000 [00:04<00:38, 9424.09it/s] 10%|▉         | 39800/400000 [00:04<00:37, 9546.45it/s] 10%|█         | 40785/400000 [00:04<00:37, 9634.07it/s] 10%|█         | 41752/400000 [00:04<00:37, 9643.41it/s] 11%|█         | 42718/400000 [00:04<00:37, 9555.31it/s] 11%|█         | 43675/400000 [00:04<00:37, 9500.42it/s] 11%|█         | 44659/400000 [00:05<00:37, 9599.04it/s] 11%|█▏        | 45620/400000 [00:05<00:37, 9494.19it/s] 12%|█▏        | 46589/400000 [00:05<00:37, 9550.11it/s] 12%|█▏        | 47586/400000 [00:05<00:36, 9670.93it/s] 12%|█▏        | 48554/400000 [00:05<00:36, 9500.97it/s] 12%|█▏        | 49506/400000 [00:05<00:36, 9504.89it/s] 13%|█▎        | 50458/400000 [00:05<00:37, 9416.48it/s] 13%|█▎        | 51417/400000 [00:05<00:36, 9464.66it/s] 13%|█▎        | 52386/400000 [00:05<00:36, 9530.10it/s] 13%|█▎        | 53340/400000 [00:05<00:36, 9488.64it/s] 14%|█▎        | 54295/400000 [00:06<00:36, 9504.73it/s] 14%|█▍        | 55246/400000 [00:06<00:36, 9317.79it/s] 14%|█▍        | 56179/400000 [00:06<00:37, 9109.25it/s] 14%|█▍        | 57092/400000 [00:06<00:38, 9004.47it/s] 14%|█▍        | 57994/400000 [00:06<00:38, 8893.23it/s] 15%|█▍        | 58885/400000 [00:06<00:38, 8860.20it/s] 15%|█▍        | 59804/400000 [00:06<00:37, 8955.54it/s] 15%|█▌        | 60718/400000 [00:06<00:37, 9008.55it/s] 15%|█▌        | 61650/400000 [00:06<00:37, 9098.41it/s] 16%|█▌        | 62580/400000 [00:07<00:36, 9156.10it/s] 16%|█▌        | 63512/400000 [00:07<00:36, 9203.19it/s] 16%|█▌        | 64468/400000 [00:07<00:36, 9306.03it/s] 16%|█▋        | 65448/400000 [00:07<00:35, 9446.92it/s] 17%|█▋        | 66435/400000 [00:07<00:34, 9568.14it/s] 17%|█▋        | 67393/400000 [00:07<00:35, 9493.26it/s] 17%|█▋        | 68361/400000 [00:07<00:34, 9546.47it/s] 17%|█▋        | 69343/400000 [00:07<00:34, 9626.27it/s] 18%|█▊        | 70319/400000 [00:07<00:34, 9665.95it/s] 18%|█▊        | 71287/400000 [00:07<00:34, 9621.11it/s] 18%|█▊        | 72250/400000 [00:08<00:34, 9479.69it/s] 18%|█▊        | 73199/400000 [00:08<00:34, 9457.58it/s] 19%|█▊        | 74157/400000 [00:08<00:34, 9491.93it/s] 19%|█▉        | 75157/400000 [00:08<00:33, 9638.31it/s] 19%|█▉        | 76122/400000 [00:08<00:33, 9608.65it/s] 19%|█▉        | 77084/400000 [00:08<00:34, 9381.70it/s] 20%|█▉        | 78024/400000 [00:08<00:35, 9188.50it/s] 20%|█▉        | 78956/400000 [00:08<00:34, 9226.37it/s] 20%|█▉        | 79918/400000 [00:08<00:34, 9340.94it/s] 20%|██        | 80894/400000 [00:08<00:33, 9462.70it/s] 20%|██        | 81842/400000 [00:09<00:34, 9345.94it/s] 21%|██        | 82778/400000 [00:09<00:34, 9323.77it/s] 21%|██        | 83743/400000 [00:09<00:33, 9416.99it/s] 21%|██        | 84693/400000 [00:09<00:33, 9441.49it/s] 21%|██▏       | 85650/400000 [00:09<00:33, 9477.71it/s] 22%|██▏       | 86599/400000 [00:09<00:33, 9413.40it/s] 22%|██▏       | 87577/400000 [00:09<00:32, 9518.32it/s] 22%|██▏       | 88556/400000 [00:09<00:32, 9597.27it/s] 22%|██▏       | 89517/400000 [00:09<00:32, 9526.45it/s] 23%|██▎       | 90471/400000 [00:09<00:32, 9455.64it/s] 23%|██▎       | 91418/400000 [00:10<00:33, 9262.66it/s] 23%|██▎       | 92346/400000 [00:10<00:33, 9259.33it/s] 23%|██▎       | 93273/400000 [00:10<00:33, 9199.82it/s] 24%|██▎       | 94194/400000 [00:10<00:34, 8963.87it/s] 24%|██▍       | 95093/400000 [00:10<00:34, 8853.97it/s] 24%|██▍       | 95980/400000 [00:10<00:34, 8848.60it/s] 24%|██▍       | 96877/400000 [00:10<00:34, 8884.50it/s] 24%|██▍       | 97866/400000 [00:10<00:32, 9163.98it/s] 25%|██▍       | 98786/400000 [00:10<00:33, 9047.44it/s] 25%|██▍       | 99698/400000 [00:10<00:33, 9066.65it/s] 25%|██▌       | 100622/400000 [00:11<00:32, 9117.87it/s] 25%|██▌       | 101582/400000 [00:11<00:32, 9256.47it/s] 26%|██▌       | 102534/400000 [00:11<00:31, 9333.03it/s] 26%|██▌       | 103508/400000 [00:11<00:31, 9449.88it/s] 26%|██▌       | 104455/400000 [00:11<00:31, 9405.78it/s] 26%|██▋       | 105397/400000 [00:11<00:31, 9304.53it/s] 27%|██▋       | 106329/400000 [00:11<00:31, 9253.02it/s] 27%|██▋       | 107305/400000 [00:11<00:31, 9399.33it/s] 27%|██▋       | 108246/400000 [00:11<00:31, 9339.99it/s] 27%|██▋       | 109181/400000 [00:11<00:31, 9321.51it/s] 28%|██▊       | 110171/400000 [00:12<00:30, 9485.10it/s] 28%|██▊       | 111121/400000 [00:12<00:30, 9399.86it/s] 28%|██▊       | 112080/400000 [00:12<00:30, 9454.15it/s] 28%|██▊       | 113027/400000 [00:12<00:30, 9458.06it/s] 28%|██▊       | 113974/400000 [00:12<00:30, 9432.36it/s] 29%|██▊       | 114949/400000 [00:12<00:29, 9523.80it/s] 29%|██▉       | 115902/400000 [00:12<00:30, 9432.65it/s] 29%|██▉       | 116846/400000 [00:12<00:30, 9381.69it/s] 29%|██▉       | 117785/400000 [00:12<00:30, 9294.93it/s] 30%|██▉       | 118715/400000 [00:12<00:30, 9286.57it/s] 30%|██▉       | 119649/400000 [00:13<00:30, 9301.25it/s] 30%|███       | 120580/400000 [00:13<00:30, 9028.42it/s] 30%|███       | 121567/400000 [00:13<00:30, 9264.92it/s] 31%|███       | 122497/400000 [00:13<00:29, 9273.55it/s] 31%|███       | 123427/400000 [00:13<00:29, 9230.38it/s] 31%|███       | 124388/400000 [00:13<00:29, 9339.50it/s] 31%|███▏      | 125324/400000 [00:13<00:29, 9302.49it/s] 32%|███▏      | 126278/400000 [00:13<00:29, 9370.54it/s] 32%|███▏      | 127216/400000 [00:13<00:29, 9309.21it/s] 32%|███▏      | 128189/400000 [00:14<00:28, 9429.98it/s] 32%|███▏      | 129134/400000 [00:14<00:28, 9434.18it/s] 33%|███▎      | 130088/400000 [00:14<00:28, 9465.58it/s] 33%|███▎      | 131044/400000 [00:14<00:28, 9490.94it/s] 33%|███▎      | 132025/400000 [00:14<00:27, 9582.11it/s] 33%|███▎      | 132984/400000 [00:14<00:28, 9494.10it/s] 33%|███▎      | 133966/400000 [00:14<00:27, 9589.56it/s] 34%|███▎      | 134926/400000 [00:14<00:28, 9411.24it/s] 34%|███▍      | 135869/400000 [00:14<00:28, 9221.32it/s] 34%|███▍      | 136820/400000 [00:14<00:28, 9305.50it/s] 34%|███▍      | 137770/400000 [00:15<00:28, 9361.14it/s] 35%|███▍      | 138731/400000 [00:15<00:27, 9434.09it/s] 35%|███▍      | 139676/400000 [00:15<00:27, 9335.25it/s] 35%|███▌      | 140611/400000 [00:15<00:27, 9315.73it/s] 35%|███▌      | 141544/400000 [00:15<00:27, 9266.94it/s] 36%|███▌      | 142472/400000 [00:15<00:27, 9270.51it/s] 36%|███▌      | 143400/400000 [00:15<00:27, 9237.97it/s] 36%|███▌      | 144325/400000 [00:15<00:28, 9009.59it/s] 36%|███▋      | 145228/400000 [00:15<00:29, 8745.75it/s] 37%|███▋      | 146125/400000 [00:15<00:28, 8810.15it/s] 37%|███▋      | 147009/400000 [00:16<00:28, 8817.58it/s] 37%|███▋      | 147917/400000 [00:16<00:28, 8894.13it/s] 37%|███▋      | 148808/400000 [00:16<00:28, 8759.97it/s] 37%|███▋      | 149711/400000 [00:16<00:28, 8838.86it/s] 38%|███▊      | 150605/400000 [00:16<00:28, 8866.32it/s] 38%|███▊      | 151493/400000 [00:16<00:28, 8826.38it/s] 38%|███▊      | 152381/400000 [00:16<00:28, 8841.19it/s] 38%|███▊      | 153266/400000 [00:16<00:27, 8817.24it/s] 39%|███▊      | 154166/400000 [00:16<00:27, 8869.70it/s] 39%|███▉      | 155060/400000 [00:16<00:27, 8888.02it/s] 39%|███▉      | 155963/400000 [00:17<00:27, 8929.00it/s] 39%|███▉      | 156876/400000 [00:17<00:27, 8985.92it/s] 39%|███▉      | 157775/400000 [00:17<00:27, 8866.18it/s] 40%|███▉      | 158687/400000 [00:17<00:26, 8938.21it/s] 40%|███▉      | 159582/400000 [00:17<00:27, 8887.44it/s] 40%|████      | 160505/400000 [00:17<00:26, 8985.79it/s] 40%|████      | 161418/400000 [00:17<00:26, 9025.91it/s] 41%|████      | 162322/400000 [00:17<00:26, 8958.49it/s] 41%|████      | 163223/400000 [00:17<00:26, 8972.57it/s] 41%|████      | 164132/400000 [00:17<00:26, 9005.66it/s] 41%|████▏     | 165033/400000 [00:18<00:26, 8999.90it/s] 41%|████▏     | 165942/400000 [00:18<00:25, 9024.17it/s] 42%|████▏     | 166845/400000 [00:18<00:26, 8898.64it/s] 42%|████▏     | 167736/400000 [00:18<00:26, 8852.52it/s] 42%|████▏     | 168622/400000 [00:18<00:26, 8818.69it/s] 42%|████▏     | 169524/400000 [00:18<00:25, 8876.74it/s] 43%|████▎     | 170420/400000 [00:18<00:25, 8900.77it/s] 43%|████▎     | 171311/400000 [00:18<00:26, 8760.78it/s] 43%|████▎     | 172201/400000 [00:18<00:25, 8800.99it/s] 43%|████▎     | 173109/400000 [00:18<00:25, 8882.75it/s] 44%|████▎     | 174018/400000 [00:19<00:25, 8941.11it/s] 44%|████▎     | 174913/400000 [00:19<00:25, 8833.04it/s] 44%|████▍     | 175797/400000 [00:19<00:25, 8728.24it/s] 44%|████▍     | 176675/400000 [00:19<00:25, 8741.79it/s] 44%|████▍     | 177556/400000 [00:19<00:25, 8760.96it/s] 45%|████▍     | 178459/400000 [00:19<00:25, 8836.85it/s] 45%|████▍     | 179383/400000 [00:19<00:24, 8953.38it/s] 45%|████▌     | 180279/400000 [00:19<00:24, 8896.36it/s] 45%|████▌     | 181172/400000 [00:19<00:24, 8903.96it/s] 46%|████▌     | 182076/400000 [00:20<00:24, 8941.64it/s] 46%|████▌     | 182972/400000 [00:20<00:24, 8947.13it/s] 46%|████▌     | 183879/400000 [00:20<00:24, 8981.69it/s] 46%|████▌     | 184778/400000 [00:20<00:24, 8857.77it/s] 46%|████▋     | 185677/400000 [00:20<00:24, 8896.65it/s] 47%|████▋     | 186591/400000 [00:20<00:23, 8967.99it/s] 47%|████▋     | 187489/400000 [00:20<00:23, 8963.49it/s] 47%|████▋     | 188386/400000 [00:20<00:23, 8916.96it/s] 47%|████▋     | 189278/400000 [00:20<00:23, 8866.70it/s] 48%|████▊     | 190165/400000 [00:20<00:23, 8806.95it/s] 48%|████▊     | 191069/400000 [00:21<00:23, 8874.15it/s] 48%|████▊     | 191957/400000 [00:21<00:23, 8841.05it/s] 48%|████▊     | 192855/400000 [00:21<00:23, 8880.54it/s] 48%|████▊     | 193744/400000 [00:21<00:23, 8787.19it/s] 49%|████▊     | 194658/400000 [00:21<00:23, 8887.50it/s] 49%|████▉     | 195562/400000 [00:21<00:22, 8931.27it/s] 49%|████▉     | 196458/400000 [00:21<00:22, 8936.92it/s] 49%|████▉     | 197369/400000 [00:21<00:22, 8985.98it/s] 50%|████▉     | 198268/400000 [00:21<00:22, 8908.13it/s] 50%|████▉     | 199177/400000 [00:21<00:22, 8959.29it/s] 50%|█████     | 200081/400000 [00:22<00:22, 8980.67it/s] 50%|█████     | 200999/400000 [00:22<00:22, 9039.25it/s] 50%|█████     | 201926/400000 [00:22<00:21, 9105.17it/s] 51%|█████     | 202837/400000 [00:22<00:22, 8864.49it/s] 51%|█████     | 203726/400000 [00:22<00:22, 8856.18it/s] 51%|█████     | 204622/400000 [00:22<00:21, 8885.97it/s] 51%|█████▏    | 205529/400000 [00:22<00:21, 8937.67it/s] 52%|█████▏    | 206447/400000 [00:22<00:21, 9005.93it/s] 52%|█████▏    | 207349/400000 [00:22<00:21, 8988.17it/s] 52%|█████▏    | 208249/400000 [00:22<00:21, 8984.80it/s] 52%|█████▏    | 209167/400000 [00:23<00:21, 9040.28it/s] 53%|█████▎    | 210085/400000 [00:23<00:20, 9081.49it/s] 53%|█████▎    | 210994/400000 [00:23<00:20, 9066.93it/s] 53%|█████▎    | 211901/400000 [00:23<00:21, 8922.43it/s] 53%|█████▎    | 212794/400000 [00:23<00:21, 8884.18it/s] 53%|█████▎    | 213709/400000 [00:23<00:20, 8959.99it/s] 54%|█████▎    | 214635/400000 [00:23<00:20, 9046.87it/s] 54%|█████▍    | 215541/400000 [00:23<00:20, 8969.33it/s] 54%|█████▍    | 216439/400000 [00:23<00:20, 8741.65it/s] 54%|█████▍    | 217351/400000 [00:23<00:20, 8851.04it/s] 55%|█████▍    | 218247/400000 [00:24<00:20, 8883.42it/s] 55%|█████▍    | 219159/400000 [00:24<00:20, 8949.98it/s] 55%|█████▌    | 220084/400000 [00:24<00:19, 9035.65it/s] 55%|█████▌    | 220989/400000 [00:24<00:19, 9011.64it/s] 55%|█████▌    | 221891/400000 [00:24<00:19, 8949.62it/s] 56%|█████▌    | 222799/400000 [00:24<00:19, 8986.09it/s] 56%|█████▌    | 223717/400000 [00:24<00:19, 9042.79it/s] 56%|█████▌    | 224622/400000 [00:24<00:19, 8823.92it/s] 56%|█████▋    | 225506/400000 [00:24<00:20, 8695.85it/s] 57%|█████▋    | 226391/400000 [00:24<00:19, 8740.01it/s] 57%|█████▋    | 227278/400000 [00:25<00:19, 8777.39it/s] 57%|█████▋    | 228157/400000 [00:25<00:19, 8724.03it/s] 57%|█████▋    | 229031/400000 [00:25<00:19, 8619.58it/s] 57%|█████▋    | 229943/400000 [00:25<00:19, 8762.12it/s] 58%|█████▊    | 230859/400000 [00:25<00:19, 8877.61it/s] 58%|█████▊    | 231781/400000 [00:25<00:18, 8976.56it/s] 58%|█████▊    | 232692/400000 [00:25<00:18, 9014.19it/s] 58%|█████▊    | 233620/400000 [00:25<00:18, 9091.85it/s] 59%|█████▊    | 234530/400000 [00:25<00:18, 8974.81it/s] 59%|█████▉    | 235429/400000 [00:25<00:18, 8890.19it/s] 59%|█████▉    | 236332/400000 [00:26<00:18, 8929.99it/s] 59%|█████▉    | 237270/400000 [00:26<00:17, 9059.40it/s] 60%|█████▉    | 238177/400000 [00:26<00:17, 9051.52it/s] 60%|█████▉    | 239083/400000 [00:26<00:18, 8934.21it/s] 60%|█████▉    | 239984/400000 [00:26<00:17, 8954.30it/s] 60%|██████    | 240880/400000 [00:26<00:17, 8894.78it/s] 60%|██████    | 241782/400000 [00:26<00:17, 8931.77it/s] 61%|██████    | 242686/400000 [00:26<00:17, 8961.65it/s] 61%|██████    | 243583/400000 [00:26<00:17, 8927.31it/s] 61%|██████    | 244480/400000 [00:26<00:17, 8938.30it/s] 61%|██████▏   | 245374/400000 [00:27<00:17, 8887.72it/s] 62%|██████▏   | 246315/400000 [00:27<00:17, 9036.72it/s] 62%|██████▏   | 247247/400000 [00:27<00:16, 9119.40it/s] 62%|██████▏   | 248160/400000 [00:27<00:16, 9042.81it/s] 62%|██████▏   | 249065/400000 [00:27<00:16, 9023.92it/s] 62%|██████▏   | 249970/400000 [00:27<00:16, 9030.76it/s] 63%|██████▎   | 250874/400000 [00:27<00:16, 9003.44it/s] 63%|██████▎   | 251783/400000 [00:27<00:16, 9026.86it/s] 63%|██████▎   | 252698/400000 [00:27<00:16, 9061.03it/s] 63%|██████▎   | 253612/400000 [00:27<00:16, 9084.15it/s] 64%|██████▎   | 254521/400000 [00:28<00:16, 9060.88it/s] 64%|██████▍   | 255434/400000 [00:28<00:15, 9078.93it/s] 64%|██████▍   | 256344/400000 [00:28<00:15, 9082.66it/s] 64%|██████▍   | 257253/400000 [00:28<00:15, 9040.98it/s] 65%|██████▍   | 258158/400000 [00:28<00:15, 9035.48it/s] 65%|██████▍   | 259074/400000 [00:28<00:15, 9070.88it/s] 65%|██████▌   | 260016/400000 [00:28<00:15, 9170.38it/s] 65%|██████▌   | 260934/400000 [00:28<00:15, 9135.68it/s] 65%|██████▌   | 261856/400000 [00:28<00:15, 9158.07it/s] 66%|██████▌   | 262772/400000 [00:29<00:15, 8983.28it/s] 66%|██████▌   | 263713/400000 [00:29<00:14, 9105.06it/s] 66%|██████▌   | 264650/400000 [00:29<00:14, 9180.91it/s] 66%|██████▋   | 265598/400000 [00:29<00:14, 9267.57it/s] 67%|██████▋   | 266526/400000 [00:29<00:14, 9231.91it/s] 67%|██████▋   | 267450/400000 [00:29<00:14, 9146.46it/s] 67%|██████▋   | 268407/400000 [00:29<00:14, 9267.68it/s] 67%|██████▋   | 269335/400000 [00:29<00:14, 9241.79it/s] 68%|██████▊   | 270260/400000 [00:29<00:14, 9169.55it/s] 68%|██████▊   | 271178/400000 [00:29<00:14, 9097.89it/s] 68%|██████▊   | 272089/400000 [00:30<00:14, 9072.30it/s] 68%|██████▊   | 273017/400000 [00:30<00:13, 9132.13it/s] 68%|██████▊   | 273931/400000 [00:30<00:13, 9088.64it/s] 69%|██████▊   | 274842/400000 [00:30<00:13, 9092.49it/s] 69%|██████▉   | 275763/400000 [00:30<00:13, 9125.01it/s] 69%|██████▉   | 276676/400000 [00:30<00:13, 9037.68it/s] 69%|██████▉   | 277581/400000 [00:30<00:13, 8965.92it/s] 70%|██████▉   | 278488/400000 [00:30<00:13, 8994.08it/s] 70%|██████▉   | 279395/400000 [00:30<00:13, 9015.79it/s] 70%|███████   | 280324/400000 [00:30<00:13, 9095.74it/s] 70%|███████   | 281234/400000 [00:31<00:13, 9075.16it/s] 71%|███████   | 282142/400000 [00:31<00:12, 9066.58it/s] 71%|███████   | 283065/400000 [00:31<00:12, 9114.01it/s] 71%|███████   | 283977/400000 [00:31<00:12, 9095.95it/s] 71%|███████   | 284890/400000 [00:31<00:12, 9104.70it/s] 71%|███████▏  | 285801/400000 [00:31<00:12, 9059.46it/s] 72%|███████▏  | 286716/400000 [00:31<00:12, 9086.41it/s] 72%|███████▏  | 287625/400000 [00:31<00:12, 9038.41it/s] 72%|███████▏  | 288529/400000 [00:31<00:12, 8887.36it/s] 72%|███████▏  | 289444/400000 [00:31<00:12, 8963.25it/s] 73%|███████▎  | 290347/400000 [00:32<00:12, 8982.85it/s] 73%|███████▎  | 291248/400000 [00:32<00:12, 8988.78it/s] 73%|███████▎  | 292165/400000 [00:32<00:11, 9040.62it/s] 73%|███████▎  | 293098/400000 [00:32<00:11, 9123.78it/s] 74%|███████▎  | 294021/400000 [00:32<00:11, 9152.83it/s] 74%|███████▎  | 294937/400000 [00:32<00:11, 9064.67it/s] 74%|███████▍  | 295844/400000 [00:32<00:11, 9034.29it/s] 74%|███████▍  | 296756/400000 [00:32<00:11, 9057.28it/s] 74%|███████▍  | 297662/400000 [00:32<00:11, 9018.70it/s] 75%|███████▍  | 298570/400000 [00:32<00:11, 9036.50it/s] 75%|███████▍  | 299474/400000 [00:33<00:11, 9004.90it/s] 75%|███████▌  | 300375/400000 [00:33<00:11, 8767.48it/s] 75%|███████▌  | 301254/400000 [00:33<00:11, 8704.53it/s] 76%|███████▌  | 302126/400000 [00:33<00:11, 8639.84it/s] 76%|███████▌  | 303017/400000 [00:33<00:11, 8716.54it/s] 76%|███████▌  | 303890/400000 [00:33<00:11, 8679.79it/s] 76%|███████▌  | 304759/400000 [00:33<00:11, 8600.39it/s] 76%|███████▋  | 305620/400000 [00:33<00:10, 8589.16it/s] 77%|███████▋  | 306504/400000 [00:33<00:10, 8660.93it/s] 77%|███████▋  | 307447/400000 [00:33<00:10, 8877.67it/s] 77%|███████▋  | 308361/400000 [00:34<00:10, 8953.24it/s] 77%|███████▋  | 309278/400000 [00:34<00:10, 9016.09it/s] 78%|███████▊  | 310217/400000 [00:34<00:09, 9123.38it/s] 78%|███████▊  | 311135/400000 [00:34<00:09, 9139.30it/s] 78%|███████▊  | 312077/400000 [00:34<00:09, 9221.39it/s] 78%|███████▊  | 313000/400000 [00:34<00:09, 9175.72it/s] 78%|███████▊  | 313925/400000 [00:34<00:09, 9195.46it/s] 79%|███████▊  | 314845/400000 [00:34<00:09, 9170.64it/s] 79%|███████▉  | 315763/400000 [00:34<00:09, 9149.10it/s] 79%|███████▉  | 316679/400000 [00:34<00:09, 9121.25it/s] 79%|███████▉  | 317592/400000 [00:35<00:09, 9027.89it/s] 80%|███████▉  | 318522/400000 [00:35<00:08, 9105.22it/s] 80%|███████▉  | 319458/400000 [00:35<00:08, 9178.26it/s] 80%|████████  | 320380/400000 [00:35<00:08, 9189.16it/s] 80%|████████  | 321300/400000 [00:35<00:08, 9131.92it/s] 81%|████████  | 322241/400000 [00:35<00:08, 9213.19it/s] 81%|████████  | 323216/400000 [00:35<00:08, 9367.90it/s] 81%|████████  | 324171/400000 [00:35<00:08, 9420.54it/s] 81%|████████▏ | 325152/400000 [00:35<00:07, 9533.77it/s] 82%|████████▏ | 326121/400000 [00:35<00:07, 9580.04it/s] 82%|████████▏ | 327080/400000 [00:36<00:07, 9430.17it/s] 82%|████████▏ | 328024/400000 [00:36<00:07, 9336.18it/s] 82%|████████▏ | 328959/400000 [00:36<00:07, 9306.11it/s] 82%|████████▏ | 329891/400000 [00:36<00:07, 9207.61it/s] 83%|████████▎ | 330819/400000 [00:36<00:07, 9227.50it/s] 83%|████████▎ | 331743/400000 [00:36<00:07, 9227.29it/s] 83%|████████▎ | 332693/400000 [00:36<00:07, 9305.66it/s] 83%|████████▎ | 333624/400000 [00:36<00:07, 9295.47it/s] 84%|████████▎ | 334569/400000 [00:36<00:07, 9340.40it/s] 84%|████████▍ | 335504/400000 [00:36<00:06, 9267.57it/s] 84%|████████▍ | 336432/400000 [00:37<00:06, 9125.40it/s] 84%|████████▍ | 337389/400000 [00:37<00:06, 9253.23it/s] 85%|████████▍ | 338373/400000 [00:37<00:06, 9419.00it/s] 85%|████████▍ | 339317/400000 [00:37<00:06, 9339.62it/s] 85%|████████▌ | 340297/400000 [00:37<00:06, 9471.44it/s] 85%|████████▌ | 341246/400000 [00:37<00:06, 9311.79it/s] 86%|████████▌ | 342179/400000 [00:37<00:06, 9300.48it/s] 86%|████████▌ | 343111/400000 [00:37<00:06, 9234.03it/s] 86%|████████▌ | 344036/400000 [00:37<00:06, 9174.63it/s] 86%|████████▌ | 344979/400000 [00:38<00:05, 9247.48it/s] 86%|████████▋ | 345905/400000 [00:38<00:05, 9186.88it/s] 87%|████████▋ | 346841/400000 [00:38<00:05, 9236.85it/s] 87%|████████▋ | 347811/400000 [00:38<00:05, 9370.55it/s] 87%|████████▋ | 348749/400000 [00:38<00:05, 9342.64it/s] 87%|████████▋ | 349684/400000 [00:38<00:05, 9312.14it/s] 88%|████████▊ | 350621/400000 [00:38<00:05, 9328.62it/s] 88%|████████▊ | 351570/400000 [00:38<00:05, 9375.40it/s] 88%|████████▊ | 352519/400000 [00:38<00:05, 9409.48it/s] 88%|████████▊ | 353474/400000 [00:38<00:04, 9449.89it/s] 89%|████████▊ | 354420/400000 [00:39<00:04, 9418.35it/s] 89%|████████▉ | 355362/400000 [00:39<00:04, 9278.75it/s] 89%|████████▉ | 356291/400000 [00:39<00:04, 9237.50it/s] 89%|████████▉ | 357259/400000 [00:39<00:04, 9364.42it/s] 90%|████████▉ | 358197/400000 [00:39<00:04, 9266.48it/s] 90%|████████▉ | 359125/400000 [00:39<00:04, 9251.46it/s] 90%|█████████ | 360076/400000 [00:39<00:04, 9327.33it/s] 90%|█████████ | 361013/400000 [00:39<00:04, 9337.15it/s] 90%|█████████ | 361948/400000 [00:39<00:04, 9255.68it/s] 91%|█████████ | 362876/400000 [00:39<00:04, 9260.73it/s] 91%|█████████ | 363826/400000 [00:40<00:03, 9329.50it/s] 91%|█████████ | 364760/400000 [00:40<00:03, 9265.62it/s] 91%|█████████▏| 365691/400000 [00:40<00:03, 9276.25it/s] 92%|█████████▏| 366650/400000 [00:40<00:03, 9366.83it/s] 92%|█████████▏| 367596/400000 [00:40<00:03, 9393.54it/s] 92%|█████████▏| 368541/400000 [00:40<00:03, 9409.78it/s] 92%|█████████▏| 369483/400000 [00:40<00:03, 9306.03it/s] 93%|█████████▎| 370449/400000 [00:40<00:03, 9407.42it/s] 93%|█████████▎| 371391/400000 [00:40<00:03, 9393.56it/s] 93%|█████████▎| 372331/400000 [00:40<00:02, 9368.34it/s] 93%|█████████▎| 373269/400000 [00:41<00:02, 9311.93it/s] 94%|█████████▎| 374201/400000 [00:41<00:02, 9254.40it/s] 94%|█████████▍| 375142/400000 [00:41<00:02, 9297.71it/s] 94%|█████████▍| 376106/400000 [00:41<00:02, 9396.87it/s] 94%|█████████▍| 377058/400000 [00:41<00:02, 9431.00it/s] 95%|█████████▍| 378002/400000 [00:41<00:02, 9315.69it/s] 95%|█████████▍| 378935/400000 [00:41<00:02, 9254.87it/s] 95%|█████████▍| 379879/400000 [00:41<00:02, 9309.05it/s] 95%|█████████▌| 380811/400000 [00:41<00:02, 9288.73it/s] 95%|█████████▌| 381752/400000 [00:41<00:01, 9322.67it/s] 96%|█████████▌| 382685/400000 [00:42<00:01, 9276.85it/s] 96%|█████████▌| 383613/400000 [00:42<00:01, 9256.05it/s] 96%|█████████▌| 384539/400000 [00:42<00:01, 9240.73it/s] 96%|█████████▋| 385464/400000 [00:42<00:01, 9028.11it/s] 97%|█████████▋| 386408/400000 [00:42<00:01, 9145.05it/s] 97%|█████████▋| 387373/400000 [00:42<00:01, 9290.22it/s] 97%|█████████▋| 388323/400000 [00:42<00:01, 9350.17it/s] 97%|█████████▋| 389260/400000 [00:42<00:01, 9247.92it/s] 98%|█████████▊| 390186/400000 [00:42<00:01, 9188.41it/s] 98%|█████████▊| 391106/400000 [00:42<00:00, 9176.11it/s] 98%|█████████▊| 392025/400000 [00:43<00:00, 9178.46it/s] 98%|█████████▊| 392944/400000 [00:43<00:00, 9046.16it/s] 98%|█████████▊| 393855/400000 [00:43<00:00, 9062.93it/s] 99%|█████████▊| 394789/400000 [00:43<00:00, 9141.54it/s] 99%|█████████▉| 395721/400000 [00:43<00:00, 9191.82it/s] 99%|█████████▉| 396647/400000 [00:43<00:00, 9210.72it/s] 99%|█████████▉| 397569/400000 [00:43<00:00, 9179.76it/s]100%|█████████▉| 398508/400000 [00:43<00:00, 9238.98it/s]100%|█████████▉| 399433/400000 [00:43<00:00, 9106.92it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9101.11it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f71c329ffd0>, <torchtext.data.dataset.TabularDataset object at 0x7f71c329f208>, <torchtext.vocab.Vocab object at 0x7f71c329f128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.61 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.53 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.70 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.21 file/s]2020-07-13 18:18:08.431944: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-13 18:18:08.436183: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-13 18:18:08.436362: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c7100d0e40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-13 18:18:08.436376: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 10%|█         | 1040384/9912422 [00:00<00:00, 10111753.49it/s] 31%|███▏      | 3112960/9912422 [00:00<00:00, 11890888.80it/s] 51%|█████▏    | 5095424/9912422 [00:00<00:00, 13417615.26it/s] 72%|███████▏  | 7118848/9912422 [00:00<00:00, 14805766.09it/s] 87%|████████▋ | 8617984/9912422 [00:00<00:00, 14587443.55it/s]9920512it [00:00, 13614313.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 514696.73it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 149742.39it/s]1654784it [00:00, 10825712.10it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180721.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
