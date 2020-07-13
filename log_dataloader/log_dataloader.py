
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fcec7a680d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fcec7a680d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fcf32db21e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fcf32db21e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fcf510f9e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fcf510f9e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fcee00df620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fcee00df620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fcee00df620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 152349.73it/s] 36%|███▌      | 3538944/9912422 [00:00<00:29, 217222.63it/s]9920512it [00:00, 33048548.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 498894.52it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162049.40it/s]1654784it [00:00, 11084052.08it/s]                         
0it [00:00, ?it/s]8192it [00:00, 208121.01it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcec711f978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcec7113c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fcee00df268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fcee00df268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fcee00df268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:16,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:15,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:14,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:13,  2.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:51,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:50,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:49,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:49,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:33,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 50.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 65.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.23s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.42 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.22s/ url]
0 examples [00:00, ? examples/s]2020-07-13 00:09:06.438200: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-13 00:09:06.456326: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-07-13 00:09:06.457114: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55885f4dc850 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-13 00:09:06.457136: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.17 examples/s]108 examples [00:00,  3.10 examples/s]219 examples [00:00,  4.42 examples/s]335 examples [00:00,  6.30 examples/s]450 examples [00:00,  8.98 examples/s]573 examples [00:00, 12.79 examples/s]685 examples [00:01, 18.19 examples/s]799 examples [00:01, 25.81 examples/s]911 examples [00:01, 36.50 examples/s]1025 examples [00:01, 51.44 examples/s]1142 examples [00:01, 72.11 examples/s]1254 examples [00:01, 100.21 examples/s]1378 examples [00:01, 138.32 examples/s]1493 examples [00:01, 187.62 examples/s]1608 examples [00:01, 250.47 examples/s]1726 examples [00:01, 327.95 examples/s]1844 examples [00:02, 418.30 examples/s]1960 examples [00:02, 511.34 examples/s]2073 examples [00:02, 604.40 examples/s]2188 examples [00:02, 704.22 examples/s]2305 examples [00:02, 798.96 examples/s]2419 examples [00:02, 870.42 examples/s]2532 examples [00:02, 916.61 examples/s]2645 examples [00:02, 969.45 examples/s]2761 examples [00:02, 1018.38 examples/s]2874 examples [00:03, 1010.65 examples/s]2983 examples [00:03, 1009.29 examples/s]3092 examples [00:03, 1030.51 examples/s]3199 examples [00:03, 1034.47 examples/s]3307 examples [00:03, 1046.47 examples/s]3414 examples [00:03, 1047.49 examples/s]3521 examples [00:03, 1040.96 examples/s]3631 examples [00:03, 1057.82 examples/s]3741 examples [00:03, 1068.30 examples/s]3849 examples [00:03, 1044.02 examples/s]3961 examples [00:04, 1064.62 examples/s]4069 examples [00:04, 1069.06 examples/s]4181 examples [00:04, 1081.82 examples/s]4290 examples [00:04, 1072.85 examples/s]4399 examples [00:04, 1075.63 examples/s]4507 examples [00:04, 1071.13 examples/s]4616 examples [00:04, 1076.02 examples/s]4724 examples [00:04, 1042.95 examples/s]4838 examples [00:04, 1068.81 examples/s]4957 examples [00:04, 1102.36 examples/s]5071 examples [00:05, 1111.60 examples/s]5189 examples [00:05, 1128.95 examples/s]5307 examples [00:05, 1143.26 examples/s]5422 examples [00:05, 1122.38 examples/s]5535 examples [00:05, 1112.78 examples/s]5647 examples [00:05, 1111.05 examples/s]5759 examples [00:05, 1110.32 examples/s]5873 examples [00:05, 1118.26 examples/s]5985 examples [00:05, 1101.13 examples/s]6096 examples [00:05, 1103.07 examples/s]6215 examples [00:06, 1126.67 examples/s]6330 examples [00:06, 1133.11 examples/s]6445 examples [00:06, 1135.82 examples/s]6564 examples [00:06, 1148.91 examples/s]6679 examples [00:06, 1137.43 examples/s]6793 examples [00:06, 1114.36 examples/s]6905 examples [00:06, 1061.32 examples/s]7012 examples [00:06, 1044.64 examples/s]7117 examples [00:06, 1029.90 examples/s]7221 examples [00:07, 1026.00 examples/s]7333 examples [00:07, 1051.21 examples/s]7447 examples [00:07, 1074.82 examples/s]7559 examples [00:07, 1087.69 examples/s]7669 examples [00:07, 1081.19 examples/s]7778 examples [00:07, 1078.23 examples/s]7886 examples [00:07, 1072.28 examples/s]7994 examples [00:07, 1043.25 examples/s]8099 examples [00:07, 1033.65 examples/s]8207 examples [00:07, 1045.48 examples/s]8321 examples [00:08, 1071.48 examples/s]8431 examples [00:08, 1079.69 examples/s]8547 examples [00:08, 1099.92 examples/s]8659 examples [00:08, 1104.39 examples/s]8770 examples [00:08, 1097.91 examples/s]8884 examples [00:08, 1107.68 examples/s]8995 examples [00:08, 1106.83 examples/s]9109 examples [00:08, 1113.36 examples/s]9221 examples [00:08, 1104.65 examples/s]9332 examples [00:08, 1079.86 examples/s]9442 examples [00:09, 1083.46 examples/s]9551 examples [00:09, 1084.58 examples/s]9663 examples [00:09, 1094.53 examples/s]9773 examples [00:09, 1072.32 examples/s]9881 examples [00:09, 1070.20 examples/s]9989 examples [00:09, 1060.41 examples/s]10096 examples [00:09, 979.70 examples/s]10204 examples [00:09, 1007.75 examples/s]10312 examples [00:09, 1028.23 examples/s]10428 examples [00:10, 1064.50 examples/s]10536 examples [00:10, 1060.32 examples/s]10649 examples [00:10, 1080.21 examples/s]10764 examples [00:10, 1098.20 examples/s]10876 examples [00:10, 1102.48 examples/s]10988 examples [00:10, 1106.28 examples/s]11099 examples [00:10, 1103.97 examples/s]11217 examples [00:10, 1124.47 examples/s]11330 examples [00:10, 1110.66 examples/s]11443 examples [00:10, 1113.78 examples/s]11560 examples [00:11, 1129.61 examples/s]11678 examples [00:11, 1142.14 examples/s]11796 examples [00:11, 1152.25 examples/s]11914 examples [00:11, 1159.45 examples/s]12031 examples [00:11, 1153.28 examples/s]12147 examples [00:11, 1126.22 examples/s]12260 examples [00:11, 1121.62 examples/s]12373 examples [00:11, 1113.84 examples/s]12485 examples [00:11, 1100.06 examples/s]12599 examples [00:11, 1110.23 examples/s]12712 examples [00:12, 1113.37 examples/s]12830 examples [00:12, 1130.63 examples/s]12944 examples [00:12, 1116.28 examples/s]13056 examples [00:12, 1116.20 examples/s]13168 examples [00:12, 1109.61 examples/s]13280 examples [00:12, 1109.16 examples/s]13395 examples [00:12, 1119.44 examples/s]13508 examples [00:12, 1085.03 examples/s]13626 examples [00:12, 1111.45 examples/s]13738 examples [00:12, 1108.16 examples/s]13850 examples [00:13, 1111.00 examples/s]13962 examples [00:13, 1108.33 examples/s]14080 examples [00:13, 1126.98 examples/s]14195 examples [00:13, 1133.52 examples/s]14310 examples [00:13, 1136.42 examples/s]14429 examples [00:13, 1150.91 examples/s]14545 examples [00:13, 1144.78 examples/s]14664 examples [00:13, 1155.67 examples/s]14780 examples [00:13, 1141.23 examples/s]14895 examples [00:13, 1141.52 examples/s]15015 examples [00:14, 1156.19 examples/s]15132 examples [00:14, 1159.03 examples/s]15251 examples [00:14, 1165.67 examples/s]15368 examples [00:14, 1162.99 examples/s]15486 examples [00:14, 1166.80 examples/s]15603 examples [00:14, 1167.09 examples/s]15720 examples [00:14, 1155.52 examples/s]15836 examples [00:14, 1147.37 examples/s]15951 examples [00:14, 1132.23 examples/s]16067 examples [00:14, 1139.42 examples/s]16183 examples [00:15, 1144.39 examples/s]16298 examples [00:15, 1139.65 examples/s]16413 examples [00:15, 1121.78 examples/s]16526 examples [00:15, 1123.41 examples/s]16639 examples [00:15, 1119.68 examples/s]16753 examples [00:15, 1125.05 examples/s]16867 examples [00:15, 1127.62 examples/s]16980 examples [00:15, 1067.59 examples/s]17088 examples [00:15, 1045.02 examples/s]17194 examples [00:16, 1027.45 examples/s]17301 examples [00:16, 1036.67 examples/s]17415 examples [00:16, 1064.31 examples/s]17529 examples [00:16, 1084.04 examples/s]17641 examples [00:16, 1093.11 examples/s]17751 examples [00:16, 1082.69 examples/s]17866 examples [00:16, 1101.63 examples/s]17981 examples [00:16, 1114.72 examples/s]18093 examples [00:16, 1103.27 examples/s]18208 examples [00:16, 1115.11 examples/s]18320 examples [00:17, 1106.73 examples/s]18439 examples [00:17, 1129.32 examples/s]18560 examples [00:17, 1149.67 examples/s]18676 examples [00:17, 1135.56 examples/s]18790 examples [00:17, 1136.57 examples/s]18907 examples [00:17, 1146.25 examples/s]19026 examples [00:17, 1156.38 examples/s]19142 examples [00:17, 1150.89 examples/s]19258 examples [00:17, 1120.77 examples/s]19371 examples [00:17, 1108.81 examples/s]19483 examples [00:18, 1101.44 examples/s]19599 examples [00:18, 1116.89 examples/s]19717 examples [00:18, 1134.03 examples/s]19836 examples [00:18, 1148.92 examples/s]19960 examples [00:18, 1173.25 examples/s]20078 examples [00:18, 1114.17 examples/s]20194 examples [00:18, 1125.79 examples/s]20308 examples [00:18, 1092.42 examples/s]20425 examples [00:18, 1111.96 examples/s]20537 examples [00:19, 1107.28 examples/s]20651 examples [00:19, 1115.14 examples/s]20763 examples [00:19, 1113.36 examples/s]20875 examples [00:19, 1114.44 examples/s]20995 examples [00:19, 1135.80 examples/s]21112 examples [00:19, 1143.35 examples/s]21233 examples [00:19, 1162.49 examples/s]21350 examples [00:19, 1142.59 examples/s]21465 examples [00:19, 1144.05 examples/s]21580 examples [00:19, 1139.93 examples/s]21695 examples [00:20, 1137.56 examples/s]21814 examples [00:20, 1151.88 examples/s]21933 examples [00:20, 1159.93 examples/s]22050 examples [00:20, 1157.84 examples/s]22166 examples [00:20, 1151.33 examples/s]22286 examples [00:20, 1165.27 examples/s]22407 examples [00:20, 1178.19 examples/s]22529 examples [00:20, 1189.29 examples/s]22650 examples [00:20, 1193.10 examples/s]22770 examples [00:20, 1179.17 examples/s]22888 examples [00:21, 1162.35 examples/s]23006 examples [00:21, 1165.06 examples/s]23123 examples [00:21, 1148.08 examples/s]23239 examples [00:21, 1149.14 examples/s]23356 examples [00:21, 1155.31 examples/s]23472 examples [00:21, 1150.12 examples/s]23588 examples [00:21, 1152.69 examples/s]23709 examples [00:21, 1168.25 examples/s]23826 examples [00:21, 1155.85 examples/s]23942 examples [00:21, 1108.00 examples/s]24058 examples [00:22, 1121.34 examples/s]24176 examples [00:22, 1136.15 examples/s]24295 examples [00:22, 1149.71 examples/s]24411 examples [00:22, 1146.22 examples/s]24526 examples [00:22, 1128.54 examples/s]24640 examples [00:22, 1129.89 examples/s]24760 examples [00:22, 1149.14 examples/s]24880 examples [00:22, 1161.66 examples/s]24997 examples [00:22, 1155.32 examples/s]25113 examples [00:22, 1144.91 examples/s]25228 examples [00:23, 1146.08 examples/s]25350 examples [00:23, 1166.98 examples/s]25473 examples [00:23, 1182.39 examples/s]25593 examples [00:23, 1187.61 examples/s]25712 examples [00:23, 1153.87 examples/s]25828 examples [00:23, 1151.96 examples/s]25944 examples [00:23, 1150.16 examples/s]26060 examples [00:23, 1122.94 examples/s]26177 examples [00:23, 1134.69 examples/s]26291 examples [00:24, 1118.12 examples/s]26404 examples [00:24, 1117.09 examples/s]26521 examples [00:24, 1132.11 examples/s]26635 examples [00:24, 1121.85 examples/s]26748 examples [00:24, 1117.52 examples/s]26860 examples [00:24, 1116.40 examples/s]26980 examples [00:24, 1138.59 examples/s]27096 examples [00:24, 1143.88 examples/s]27211 examples [00:24, 1102.91 examples/s]27322 examples [00:24, 1104.44 examples/s]27437 examples [00:25, 1115.71 examples/s]27557 examples [00:25, 1138.09 examples/s]27678 examples [00:25, 1156.39 examples/s]27797 examples [00:25, 1164.73 examples/s]27914 examples [00:25, 1163.22 examples/s]28035 examples [00:25, 1175.57 examples/s]28153 examples [00:25, 1152.59 examples/s]28275 examples [00:25, 1170.43 examples/s]28393 examples [00:25, 1167.24 examples/s]28510 examples [00:25, 1154.94 examples/s]28626 examples [00:26, 1142.89 examples/s]28741 examples [00:26, 1128.69 examples/s]28857 examples [00:26, 1135.42 examples/s]28971 examples [00:26, 1135.71 examples/s]29094 examples [00:26, 1160.39 examples/s]29211 examples [00:26, 1152.92 examples/s]29327 examples [00:26, 1146.75 examples/s]29443 examples [00:26, 1150.34 examples/s]29559 examples [00:26, 1142.38 examples/s]29677 examples [00:26, 1150.84 examples/s]29793 examples [00:27, 1088.61 examples/s]29903 examples [00:27, 1086.00 examples/s]30013 examples [00:27, 1021.84 examples/s]30117 examples [00:27, 1022.19 examples/s]30228 examples [00:27, 1046.98 examples/s]30334 examples [00:27, 1036.09 examples/s]30439 examples [00:27, 1000.16 examples/s]30549 examples [00:27, 1026.80 examples/s]30665 examples [00:27, 1062.20 examples/s]30772 examples [00:28, 1046.37 examples/s]30884 examples [00:28, 1066.88 examples/s]30998 examples [00:28, 1085.53 examples/s]31107 examples [00:28, 1076.96 examples/s]31216 examples [00:28, 1045.29 examples/s]31331 examples [00:28, 1074.25 examples/s]31442 examples [00:28, 1083.39 examples/s]31556 examples [00:28, 1097.71 examples/s]31667 examples [00:28, 1096.18 examples/s]31780 examples [00:28, 1105.68 examples/s]31892 examples [00:29, 1108.31 examples/s]32003 examples [00:29, 1106.95 examples/s]32116 examples [00:29, 1111.70 examples/s]32231 examples [00:29, 1121.66 examples/s]32344 examples [00:29, 1123.75 examples/s]32460 examples [00:29, 1132.56 examples/s]32574 examples [00:29, 1128.44 examples/s]32687 examples [00:29, 1127.98 examples/s]32807 examples [00:29, 1146.57 examples/s]32922 examples [00:29, 1124.65 examples/s]33035 examples [00:30, 1125.63 examples/s]33148 examples [00:30, 1109.88 examples/s]33260 examples [00:30, 1102.52 examples/s]33376 examples [00:30, 1117.76 examples/s]33490 examples [00:30, 1123.73 examples/s]33606 examples [00:30, 1132.04 examples/s]33720 examples [00:30, 1130.59 examples/s]33834 examples [00:30, 1117.61 examples/s]33946 examples [00:30, 1112.52 examples/s]34058 examples [00:30, 1113.79 examples/s]34170 examples [00:31, 1044.65 examples/s]34281 examples [00:31, 1060.84 examples/s]34390 examples [00:31, 1067.28 examples/s]34502 examples [00:31, 1082.47 examples/s]34611 examples [00:31, 1084.40 examples/s]34720 examples [00:31, 1074.67 examples/s]34830 examples [00:31, 1080.64 examples/s]34943 examples [00:31, 1093.33 examples/s]35053 examples [00:31, 1086.33 examples/s]35168 examples [00:32, 1100.10 examples/s]35279 examples [00:32, 1102.92 examples/s]35390 examples [00:32, 1103.31 examples/s]35501 examples [00:32, 1093.55 examples/s]35611 examples [00:32, 1089.95 examples/s]35724 examples [00:32, 1099.88 examples/s]35840 examples [00:32, 1116.20 examples/s]35953 examples [00:32, 1119.59 examples/s]36066 examples [00:32, 1122.28 examples/s]36179 examples [00:32, 1111.88 examples/s]36291 examples [00:33, 1108.42 examples/s]36406 examples [00:33, 1119.89 examples/s]36519 examples [00:33, 1110.64 examples/s]36633 examples [00:33, 1118.07 examples/s]36745 examples [00:33, 1115.11 examples/s]36858 examples [00:33, 1117.33 examples/s]36970 examples [00:33, 1105.81 examples/s]37081 examples [00:33, 1091.50 examples/s]37191 examples [00:33, 1073.00 examples/s]37299 examples [00:33, 1071.28 examples/s]37407 examples [00:34, 1048.70 examples/s]37515 examples [00:34, 1057.52 examples/s]37621 examples [00:34, 1054.75 examples/s]37738 examples [00:34, 1085.65 examples/s]37849 examples [00:34, 1089.53 examples/s]37959 examples [00:34, 1067.40 examples/s]38069 examples [00:34, 1074.54 examples/s]38185 examples [00:34, 1096.14 examples/s]38299 examples [00:34, 1108.26 examples/s]38411 examples [00:34, 1104.19 examples/s]38525 examples [00:35, 1114.25 examples/s]38637 examples [00:35, 1115.45 examples/s]38749 examples [00:35, 1065.04 examples/s]38865 examples [00:35, 1091.63 examples/s]38975 examples [00:35, 1064.61 examples/s]39082 examples [00:35, 1063.42 examples/s]39194 examples [00:35, 1077.52 examples/s]39303 examples [00:35, 1080.60 examples/s]39412 examples [00:35, 1060.40 examples/s]39519 examples [00:36, 1058.53 examples/s]39626 examples [00:36, 1042.22 examples/s]39738 examples [00:36, 1061.78 examples/s]39845 examples [00:36, 1024.85 examples/s]39948 examples [00:36, 1024.02 examples/s]40051 examples [00:36, 970.63 examples/s] 40152 examples [00:36, 981.79 examples/s]40258 examples [00:36, 1003.03 examples/s]40359 examples [00:36, 1000.59 examples/s]40464 examples [00:36, 1013.34 examples/s]40571 examples [00:37, 1028.97 examples/s]40675 examples [00:37, 1021.52 examples/s]40786 examples [00:37, 1046.42 examples/s]40899 examples [00:37, 1069.06 examples/s]41012 examples [00:37, 1084.74 examples/s]41126 examples [00:37, 1098.91 examples/s]41237 examples [00:37, 1079.19 examples/s]41346 examples [00:37, 1035.30 examples/s]41457 examples [00:37, 1054.94 examples/s]41564 examples [00:37, 1057.55 examples/s]41676 examples [00:38, 1073.62 examples/s]41791 examples [00:38, 1093.05 examples/s]41907 examples [00:38, 1110.67 examples/s]42019 examples [00:38, 1099.02 examples/s]42134 examples [00:38, 1112.77 examples/s]42250 examples [00:38, 1125.15 examples/s]42366 examples [00:38, 1133.32 examples/s]42480 examples [00:38, 1129.61 examples/s]42594 examples [00:38, 1130.16 examples/s]42708 examples [00:39, 1113.78 examples/s]42821 examples [00:39, 1118.45 examples/s]42935 examples [00:39, 1124.23 examples/s]43048 examples [00:39, 1123.96 examples/s]43163 examples [00:39, 1131.31 examples/s]43277 examples [00:39, 1044.95 examples/s]43387 examples [00:39, 1059.37 examples/s]43502 examples [00:39, 1083.56 examples/s]43612 examples [00:39, 1031.52 examples/s]43717 examples [00:39, 1011.62 examples/s]43820 examples [00:40, 993.82 examples/s] 43921 examples [00:40, 967.75 examples/s]44030 examples [00:40, 999.28 examples/s]44133 examples [00:40, 1008.20 examples/s]44244 examples [00:40, 1034.50 examples/s]44352 examples [00:40, 1046.81 examples/s]44458 examples [00:40, 1034.05 examples/s]44568 examples [00:40, 1050.90 examples/s]44681 examples [00:40, 1072.23 examples/s]44792 examples [00:40, 1082.49 examples/s]44901 examples [00:41, 1081.92 examples/s]45010 examples [00:41, 1081.28 examples/s]45120 examples [00:41, 1085.55 examples/s]45231 examples [00:41, 1090.44 examples/s]45341 examples [00:41, 1082.15 examples/s]45450 examples [00:41, 1067.12 examples/s]45557 examples [00:41, 1039.11 examples/s]45662 examples [00:41, 1033.49 examples/s]45766 examples [00:41, 1003.66 examples/s]45867 examples [00:42, 992.13 examples/s] 45970 examples [00:42, 1003.14 examples/s]46071 examples [00:42, 962.15 examples/s] 46179 examples [00:42, 992.77 examples/s]46279 examples [00:42, 972.42 examples/s]46390 examples [00:42, 1007.52 examples/s]46500 examples [00:42, 1031.11 examples/s]46604 examples [00:42, 1031.57 examples/s]46708 examples [00:42, 994.30 examples/s] 46809 examples [00:42, 977.36 examples/s]46908 examples [00:43, 976.73 examples/s]47017 examples [00:43, 1007.00 examples/s]47119 examples [00:43, 986.40 examples/s] 47226 examples [00:43, 1008.69 examples/s]47333 examples [00:43, 1024.53 examples/s]47436 examples [00:43, 992.83 examples/s] 47536 examples [00:43, 939.73 examples/s]47632 examples [00:43, 943.54 examples/s]47727 examples [00:43, 938.47 examples/s]47829 examples [00:44, 959.59 examples/s]47929 examples [00:44, 970.65 examples/s]48027 examples [00:44, 882.01 examples/s]48131 examples [00:44, 923.94 examples/s]48242 examples [00:44, 970.87 examples/s]48353 examples [00:44, 1006.54 examples/s]48459 examples [00:44, 1019.35 examples/s]48572 examples [00:44, 1049.31 examples/s]48688 examples [00:44, 1078.31 examples/s]48797 examples [00:44, 1048.94 examples/s]48913 examples [00:45, 1078.24 examples/s]49033 examples [00:45, 1111.39 examples/s]49150 examples [00:45, 1127.15 examples/s]49267 examples [00:45, 1139.33 examples/s]49382 examples [00:45, 1139.25 examples/s]49500 examples [00:45, 1149.46 examples/s]49616 examples [00:45, 1151.23 examples/s]49737 examples [00:45, 1159.98 examples/s]49854 examples [00:45, 1142.72 examples/s]49969 examples [00:45, 1137.76 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7424/50000 [00:00<00:00, 74239.89 examples/s] 40%|███▉      | 19921/50000 [00:00<00:00, 84534.46 examples/s] 65%|██████▌   | 32585/50000 [00:00<00:00, 93865.36 examples/s] 92%|█████████▏| 46024/50000 [00:00<00:00, 103199.86 examples/s]                                                                0 examples [00:00, ? examples/s]98 examples [00:00, 978.04 examples/s]213 examples [00:00, 1022.92 examples/s]336 examples [00:00, 1075.80 examples/s]456 examples [00:00, 1107.66 examples/s]568 examples [00:00, 1111.29 examples/s]688 examples [00:00, 1135.26 examples/s]807 examples [00:00, 1149.42 examples/s]923 examples [00:00, 1150.86 examples/s]1043 examples [00:00, 1164.38 examples/s]1159 examples [00:01, 1160.49 examples/s]1274 examples [00:01, 1155.04 examples/s]1388 examples [00:01, 1144.79 examples/s]1502 examples [00:01, 1111.59 examples/s]1618 examples [00:01, 1124.35 examples/s]1731 examples [00:01, 1116.50 examples/s]1844 examples [00:01, 1117.86 examples/s]1958 examples [00:01, 1121.59 examples/s]2071 examples [00:01, 1120.09 examples/s]2186 examples [00:01, 1126.96 examples/s]2299 examples [00:02, 1124.09 examples/s]2414 examples [00:02, 1130.62 examples/s]2530 examples [00:02, 1138.85 examples/s]2644 examples [00:02, 1126.21 examples/s]2757 examples [00:02, 1080.84 examples/s]2871 examples [00:02, 1096.31 examples/s]2981 examples [00:02, 1040.89 examples/s]3091 examples [00:02, 1055.67 examples/s]3198 examples [00:02, 1018.66 examples/s]3307 examples [00:02, 1038.41 examples/s]3415 examples [00:03, 1048.01 examples/s]3526 examples [00:03, 1064.78 examples/s]3636 examples [00:03, 1073.88 examples/s]3744 examples [00:03, 1051.17 examples/s]3850 examples [00:03, 1030.96 examples/s]3956 examples [00:03, 1038.80 examples/s]4068 examples [00:03, 1059.28 examples/s]4183 examples [00:03, 1082.82 examples/s]4292 examples [00:03, 1080.53 examples/s]4406 examples [00:03, 1096.14 examples/s]4518 examples [00:04, 1102.54 examples/s]4629 examples [00:04, 1104.41 examples/s]4740 examples [00:04, 1088.78 examples/s]4851 examples [00:04, 1093.73 examples/s]4961 examples [00:04, 1071.81 examples/s]5069 examples [00:04, 1059.97 examples/s]5182 examples [00:04, 1078.19 examples/s]5297 examples [00:04, 1096.31 examples/s]5411 examples [00:04, 1108.30 examples/s]5524 examples [00:05, 1113.25 examples/s]5636 examples [00:05, 1106.17 examples/s]5747 examples [00:05, 1081.51 examples/s]5861 examples [00:05, 1096.79 examples/s]5979 examples [00:05, 1119.92 examples/s]6092 examples [00:05, 1122.45 examples/s]6207 examples [00:05, 1130.12 examples/s]6322 examples [00:05, 1134.89 examples/s]6436 examples [00:05, 1121.99 examples/s]6549 examples [00:05, 1067.83 examples/s]6666 examples [00:06, 1094.37 examples/s]6777 examples [00:06, 1069.65 examples/s]6898 examples [00:06, 1106.90 examples/s]7010 examples [00:06, 1110.05 examples/s]7126 examples [00:06, 1124.05 examples/s]7239 examples [00:06, 1094.44 examples/s]7349 examples [00:06, 1092.84 examples/s]7464 examples [00:06, 1106.66 examples/s]7579 examples [00:06, 1117.27 examples/s]7691 examples [00:06, 1098.61 examples/s]7811 examples [00:07, 1125.23 examples/s]7926 examples [00:07, 1131.17 examples/s]8043 examples [00:07, 1141.28 examples/s]8158 examples [00:07, 1116.51 examples/s]8273 examples [00:07, 1126.21 examples/s]8387 examples [00:07, 1127.40 examples/s]8500 examples [00:07, 1115.78 examples/s]8616 examples [00:07, 1127.10 examples/s]8729 examples [00:07, 1117.28 examples/s]8841 examples [00:08, 1099.74 examples/s]8952 examples [00:08, 1084.05 examples/s]9062 examples [00:08, 1087.43 examples/s]9179 examples [00:08, 1108.11 examples/s]9290 examples [00:08, 1100.14 examples/s]9401 examples [00:08, 1091.85 examples/s]9515 examples [00:08, 1103.78 examples/s]9626 examples [00:08, 1099.64 examples/s]9737 examples [00:08, 1087.90 examples/s]9846 examples [00:08, 1081.99 examples/s]9955 examples [00:09, 1056.77 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2D5U1X/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2D5U1X/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fcee00df620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fcee00df620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fcee00df620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce6559e4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fce655da908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fcee00df620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fcee00df620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fcee00df620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcec71135f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fcec7113390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fce57b9d510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fce57b9d510> 

  function with postional parmater data_info <function split_train_valid at 0x7fce57b9d510> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fce57b9d620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fce57b9d620> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fce57b9d620> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=f83e33fb791316500cfcf9a4cd846e7d44e30c7262e0013377151b6ac2642c98
  Stored in directory: /tmp/pip-ephem-wheel-cache-b2ny692h/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:38:55, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:41:05, 16.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:19:57, 23.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:14:28, 33.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:03:20, 47.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.49M/862M [00:01<3:31:13, 67.4kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:27:20, 96.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:42:33, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<1:11:24, 196kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<49:44, 279kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:01<34:40, 398kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.7M/862M [00:02<24:12, 566kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<16:53, 805kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<11:55, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<09:13, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:04<08:20, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<08:00, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<06:03, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<04:22, 3.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<39:12, 341kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<30:15, 441kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<21:45, 613kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:07<15:22, 865kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<16:55, 785kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<13:14, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<09:32, 1.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<09:41, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<07:53, 1.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<05:50, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<07:11, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<07:41, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<05:58, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:13<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<39:45, 329kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<29:08, 449kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<20:39, 631kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<17:28, 745kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<14:51, 875kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<11:03, 1.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<09:48, 1.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<08:10, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:15, 1.78MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<06:24, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<04:45, 2.70MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:18, 2.03MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<07:00, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<05:27, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:22<03:56, 3.23MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<16:59, 750kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<13:10, 966kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<09:28, 1.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:36, 1.32MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:17, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:08, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:01, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:11, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:36, 2.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:09, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:50, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:19, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<03:52, 3.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<11:11, 1.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<09:07, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<06:39, 1.87MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:33, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:47, 1.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:04, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:14, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:38, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:15, 2.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:50, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:35, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:08, 2.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:48, 3.21MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:47, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:59, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:29, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:58, 2.03MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:25, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:06, 2.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:41, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:12, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:56, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:31, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:11, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:55, 2.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:23, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:00, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:45, 3.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:22, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:09, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:54, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:19, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:07, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:47, 2.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:29, 3.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:07, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:34, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<05:35, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:37, 1.76MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:06, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:01, 2.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:17:02, 150kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<55:05, 210kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<38:46, 298kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<29:45, 387kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<22:00, 522kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<15:37, 734kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<13:36, 841kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<10:40, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<07:44, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:06, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:52, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:05, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:04, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:24, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:01, 2.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:26, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:04, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:49, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:10, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:37, 3.07MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:06, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:49, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:37, 2.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:37, 3.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:10:58, 155kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<50:46, 217kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<35:44, 307kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<27:32, 398kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<21:28, 510kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<15:29, 706kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<10:55, 997kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<16:21, 666kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<12:33, 866kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<09:00, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:50, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:22, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:24, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:12, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:01, 2.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:19, 2.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:53, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:39, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:58, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:33, 2.32MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:24, 3.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:51, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:32, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:24, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:32, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:20, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:08, 3.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:38, 978kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:31, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<06:12, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:45, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:36, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:07, 2.50MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<03:03, 3.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:29:39, 115kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:04:49, 158kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<45:50, 224kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<32:03, 318kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<1:45:49, 96.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<1:15:03, 136kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<52:37, 193kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<39:05, 259kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<28:21, 357kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<20:02, 504kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<16:22, 615kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<13:30, 745kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:57, 1.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:33, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:59, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<05:05, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:54, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:08, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<04:47, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:56, 1.99MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:27, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:20, 2.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:39, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:17, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:12, 3.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:30, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:09, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:08, 3.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:27, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:05, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:02, 2.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:22, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:01, 2.38MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:03, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:22, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:01, 2.36MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:03, 3.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:21, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:00, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:00, 3.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:55, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:54, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:54, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<02:57, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:13, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:50, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:47, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<02:44, 3.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<11:15, 814kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:50, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<06:21, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:34, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:31, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<04:02, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:58, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:18, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:09, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<02:59, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<8:39:56, 17.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<6:04:37, 24.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<4:14:41, 35.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<2:59:39, 49.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<2:07:31, 69.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<1:29:35, 99.1kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<1:03:46, 138kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<45:30, 194kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<31:59, 275kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<24:21, 360kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<17:55, 488kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<12:41, 687kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<10:55, 795kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<09:25, 922kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<06:57, 1.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<04:58, 1.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:19, 1.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:43, 1.28MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:54, 1.75MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:26, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:36, 1.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<04:21, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<03:08, 2.70MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<1:02:58, 135kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<44:54, 189kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<31:33, 268kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<23:58, 351kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<18:28, 455kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<13:16, 633kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<09:23, 892kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<09:52, 845kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<07:45, 1.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:39, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<05:51, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:21, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:00, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:17, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:15, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:13, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:34<02:20, 3.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<16:22, 497kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<12:42, 641kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<09:07, 890kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<06:26, 1.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<19:48, 407kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<15:02, 536kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<10:45, 748kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<07:36, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<13:01, 614kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<11:23, 702kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<08:34, 932kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<06:06, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:01, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<06:07, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:32, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:35, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:38, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:32, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<03:18, 2.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:00, 1.55MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:39, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<03:32, 2.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:52, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:54, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:59, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:55, 2.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:53, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:34, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:29, 2.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:48, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:48, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:55, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:24, 2.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:39, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:43, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<02:43, 2.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:09, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:00, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:01, 2.45MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:12, 3.36MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<11:33, 638kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<09:11, 803kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:40, 1.10MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:58, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<06:24, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:56, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:33, 2.04MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:52, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:28, 1.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:23, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:41, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:45, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:51, 1.86MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<02:49, 2.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:26, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<04:09, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:10, 2.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:29, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:28, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:38, 2.65MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<01:55, 3.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<08:54, 782kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<08:16, 842kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<06:18, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:30, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:32, 1.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:55, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<03:41, 1.86MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<03:48, 1.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:42, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:48, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:11, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<04:18, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:30, 1.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:33, 2.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<04:07, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:51, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:56, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:15, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:15, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:29, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:55, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:03, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:20, 2.79MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<01:42, 3.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<17:58, 361kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<13:33, 478kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<09:41, 667kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<06:49, 940kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<09:31, 673kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<07:38, 839kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<05:32, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:00, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:25, 1.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:16, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<03:05, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:20, 1.45MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:57, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:12, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<04:08, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:19, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:25, 2.55MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:27, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:21, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:33, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:52, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:56, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:14, 2.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:39, 2.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:42, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:57, 2.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:09, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:17, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<03:11, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:24, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<01:44, 3.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<23:03, 254kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<16:57, 345kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<12:03, 484kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<09:26, 613kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:22, 692kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<06:16, 920kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:43<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:06, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:25, 1.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:17, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:19, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:00, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:16, 2.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:42, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:14, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:36, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<01:52, 2.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:47, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:09, 1.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:05, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:08, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:45, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:02, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:12, 2.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:24, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:10, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:24, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:39, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:39, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:02, 2.59MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:16, 1.60MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:40, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:57, 2.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:06, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:57, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:15, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:31, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:55, 2.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:23, 3.60MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<08:17, 606kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<06:34, 765kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:45, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:12, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<03:41, 1.34MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:45, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:48, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:27, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:43, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:02, 2.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:23, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:22, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:48, 2.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:20, 3.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:04, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:35, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:04, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:04, 1.52MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:51, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:20, 1.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:02, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:28, 1.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:48, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:50, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:39, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:59, 2.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:26, 3.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<10:32, 425kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<08:43, 513kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<06:24, 698kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<04:31, 978kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<04:38, 949kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:55, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:53, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:47, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:17, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:37, 1.65MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:53, 2.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:48, 1.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:36, 1.63MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:57, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:45, 1.52MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:35, 2.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:09, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:38, 2.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:11, 3.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<05:25, 748kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:57, 819kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:43, 1.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:39, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:02, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:42, 1.47MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<02:01, 1.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:07, 1.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:40, 1.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:07, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:33, 2.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:01, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:59, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:31, 2.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:44, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:47, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:22, 2.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:37, 2.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:15, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:51, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:21, 2.72MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:05, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:35, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:45, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:47, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:22, 2.58MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:35, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:37, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:14, 2.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:29, 2.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:33, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:11, 2.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:26, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:30, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:09, 2.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<00:50, 3.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<22:26, 147kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<16:42, 198kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<11:55, 276kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<08:18, 392kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<06:53, 469kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<05:13, 619kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<03:43, 861kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:13, 982kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<03:13, 980kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:29, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:46, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<02:19, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:05, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:33, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:06, 2.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<06:58, 434kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<05:20, 566kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:49, 785kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:10, 931kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:07, 945kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:24, 1.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:43, 1.69MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:11, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:57, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:28, 1.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:31, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:29, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:07, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<00:48, 3.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<08:45, 314kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<06:58, 395kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<05:01, 545kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<03:33, 764kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:03, 877kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:32, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:51, 1.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:19, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:24, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:18, 1.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:45, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:14, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:21, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:01, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:29, 1.69MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:44, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:22, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:59, 2.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:25, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:21, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:02, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:08, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:21, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:04, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:45, 2.98MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:41, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:30, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:07, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:50, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:15, 1.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:23, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:05, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:46, 2.75MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:19, 914kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:07, 1.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:35, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:26, 845kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:10, 943kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:37, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:08, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:33, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:30, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:08, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:48, 2.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:39, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:36, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:12, 1.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:51, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:15, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:18, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:00, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:36<00:42, 2.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:27, 1.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:25, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:45, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:32, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:27, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:05, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:46, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:04, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:04, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:49, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:34, 2.73MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:13, 301kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<04:00, 392kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:51, 544kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:58, 770kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:18, 649kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:54, 789kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:22, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:57, 1.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:07, 674kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:45, 817kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:16, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:53, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<07:08, 191kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<05:14, 260kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<03:41, 365kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<02:30, 518kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<07:56, 163kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<05:50, 221kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<04:07, 311kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<02:49, 442kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<02:30, 489kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:58, 621kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:24, 858kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:58, 1.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:22, 843kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:12, 961kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:53, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:37, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<01:40, 652kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:21, 798kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:59, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:49, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:48, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:37, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:05, 873kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:57, 988kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:42, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:28, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:59, 888kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:53, 992kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:39, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:28, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:21, 2.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:55, 808kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:00, 741kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:46, 942kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:32, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:29, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:21, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:19, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:13, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:46, 691kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:37, 841kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:26, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:17, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:46, 599kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:38, 715kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:28, 963kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:18, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:32, 740kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:27, 862kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:19, 1.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:12, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:38, 516kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 630kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:21, 858kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:13, 1.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:25, 602kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:21, 728kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:14, 986kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:08, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:22, 504kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:17, 638kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:11, 876kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 1.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.50MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:01, 2.09MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 454kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 582kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 805kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<66:13:37,  1.68it/s]  0%|          | 1045/400000 [00:00<46:14:27,  2.40it/s]  1%|          | 2140/400000 [00:00<32:16:58,  3.42it/s]  1%|          | 3176/400000 [00:00<22:32:32,  4.89it/s]  1%|          | 4238/400000 [00:00<15:44:26,  6.98it/s]  1%|▏         | 5239/400000 [00:01<10:59:37,  9.97it/s]  2%|▏         | 6342/400000 [00:01<7:40:37, 14.24it/s]   2%|▏         | 7469/400000 [00:01<5:21:41, 20.34it/s]  2%|▏         | 8595/400000 [00:01<3:44:42, 29.03it/s]  2%|▏         | 9730/400000 [00:01<2:37:00, 41.43it/s]  3%|▎         | 10804/400000 [00:01<1:49:47, 59.08it/s]  3%|▎         | 11919/400000 [00:01<1:16:48, 84.21it/s]  3%|▎         | 13000/400000 [00:01<53:48, 119.88it/s]   4%|▎         | 14064/400000 [00:01<37:45, 170.35it/s]  4%|▍         | 15100/400000 [00:02<26:33, 241.60it/s]  4%|▍         | 16125/400000 [00:02<18:43, 341.69it/s]  4%|▍         | 17150/400000 [00:02<13:15, 480.98it/s]  5%|▍         | 18249/400000 [00:02<09:26, 674.45it/s]  5%|▍         | 19288/400000 [00:02<06:47, 935.24it/s]  5%|▌         | 20302/400000 [00:02<04:55, 1284.66it/s]  5%|▌         | 21315/400000 [00:02<03:37, 1740.55it/s]  6%|▌         | 22353/400000 [00:02<02:42, 2319.76it/s]  6%|▌         | 23440/400000 [00:02<02:04, 3036.20it/s]  6%|▌         | 24494/400000 [00:02<01:37, 3860.47it/s]  6%|▋         | 25589/400000 [00:03<01:18, 4790.70it/s]  7%|▋         | 26648/400000 [00:03<01:05, 5706.56it/s]  7%|▋         | 27699/400000 [00:03<00:56, 6606.25it/s]  7%|▋         | 28748/400000 [00:03<00:50, 7415.61it/s]  7%|▋         | 29794/400000 [00:03<00:46, 8026.37it/s]  8%|▊         | 30840/400000 [00:03<00:42, 8628.52it/s]  8%|▊         | 31875/400000 [00:03<00:40, 8990.48it/s]  8%|▊         | 32899/400000 [00:03<00:39, 9203.92it/s]  9%|▊         | 34034/400000 [00:03<00:37, 9757.18it/s]  9%|▉         | 35143/400000 [00:03<00:36, 10119.57it/s]  9%|▉         | 36208/400000 [00:04<00:35, 10267.05it/s]  9%|▉         | 37273/400000 [00:04<00:35, 10160.40it/s] 10%|▉         | 38378/400000 [00:04<00:34, 10410.50it/s] 10%|▉         | 39447/400000 [00:04<00:34, 10490.87it/s] 10%|█         | 40553/400000 [00:04<00:33, 10655.35it/s] 10%|█         | 41729/400000 [00:04<00:32, 10963.12it/s] 11%|█         | 42835/400000 [00:04<00:32, 10860.67it/s] 11%|█         | 43960/400000 [00:04<00:32, 10973.33it/s] 11%|█▏        | 45077/400000 [00:04<00:32, 11030.84it/s] 12%|█▏        | 46184/400000 [00:04<00:32, 10851.16it/s] 12%|█▏        | 47273/400000 [00:05<00:33, 10618.98it/s] 12%|█▏        | 48339/400000 [00:05<00:33, 10390.91it/s] 12%|█▏        | 49382/400000 [00:05<00:34, 10244.34it/s] 13%|█▎        | 50410/400000 [00:05<00:34, 10139.11it/s] 13%|█▎        | 51427/400000 [00:05<00:34, 10103.71it/s] 13%|█▎        | 52509/400000 [00:05<00:33, 10307.51it/s] 13%|█▎        | 53542/400000 [00:05<00:34, 10115.88it/s] 14%|█▎        | 54584/400000 [00:05<00:33, 10205.07it/s] 14%|█▍        | 55684/400000 [00:05<00:33, 10431.17it/s] 14%|█▍        | 56730/400000 [00:05<00:32, 10435.24it/s] 14%|█▍        | 57842/400000 [00:06<00:32, 10630.76it/s] 15%|█▍        | 58918/400000 [00:06<00:31, 10668.04it/s] 15%|█▍        | 59987/400000 [00:06<00:31, 10637.87it/s] 15%|█▌        | 61058/400000 [00:06<00:31, 10657.19it/s] 16%|█▌        | 62157/400000 [00:06<00:31, 10753.87it/s] 16%|█▌        | 63234/400000 [00:06<00:31, 10757.69it/s] 16%|█▌        | 64323/400000 [00:06<00:31, 10794.63it/s] 16%|█▋        | 65417/400000 [00:06<00:30, 10835.94it/s] 17%|█▋        | 66511/400000 [00:06<00:30, 10866.56it/s] 17%|█▋        | 67598/400000 [00:07<00:31, 10614.73it/s] 17%|█▋        | 68661/400000 [00:07<00:31, 10466.76it/s] 17%|█▋        | 69710/400000 [00:07<00:31, 10372.04it/s] 18%|█▊        | 70837/400000 [00:07<00:30, 10624.81it/s] 18%|█▊        | 71909/400000 [00:07<00:30, 10652.77it/s] 18%|█▊        | 73015/400000 [00:07<00:30, 10771.08it/s] 19%|█▊        | 74094/400000 [00:07<00:30, 10724.24it/s] 19%|█▉        | 75168/400000 [00:07<00:30, 10610.58it/s] 19%|█▉        | 76312/400000 [00:07<00:29, 10845.28it/s] 19%|█▉        | 77432/400000 [00:07<00:29, 10948.79it/s] 20%|█▉        | 78529/400000 [00:08<00:29, 10886.21it/s] 20%|█▉        | 79619/400000 [00:08<00:30, 10668.04it/s] 20%|██        | 80688/400000 [00:08<00:31, 10241.88it/s] 20%|██        | 81718/400000 [00:08<00:31, 10079.77it/s] 21%|██        | 82803/400000 [00:08<00:30, 10297.74it/s] 21%|██        | 83941/400000 [00:08<00:29, 10598.40it/s] 21%|██▏       | 85046/400000 [00:08<00:29, 10729.92it/s] 22%|██▏       | 86123/400000 [00:08<00:29, 10555.04it/s] 22%|██▏       | 87209/400000 [00:08<00:29, 10643.33it/s] 22%|██▏       | 88276/400000 [00:08<00:29, 10435.57it/s] 22%|██▏       | 89323/400000 [00:09<00:29, 10437.25it/s] 23%|██▎       | 90465/400000 [00:09<00:28, 10711.49it/s] 23%|██▎       | 91540/400000 [00:09<00:29, 10580.63it/s] 23%|██▎       | 92664/400000 [00:09<00:28, 10769.84it/s] 23%|██▎       | 93744/400000 [00:09<00:28, 10684.35it/s] 24%|██▎       | 94877/400000 [00:09<00:28, 10869.56it/s] 24%|██▍       | 95967/400000 [00:09<00:28, 10670.81it/s] 24%|██▍       | 97037/400000 [00:09<00:28, 10555.32it/s] 25%|██▍       | 98103/400000 [00:09<00:28, 10585.41it/s] 25%|██▍       | 99241/400000 [00:09<00:27, 10809.61it/s] 25%|██▌       | 100360/400000 [00:10<00:27, 10920.08it/s] 25%|██▌       | 101497/400000 [00:10<00:27, 11049.13it/s] 26%|██▌       | 102604/400000 [00:10<00:27, 10834.49it/s] 26%|██▌       | 103690/400000 [00:10<00:27, 10793.36it/s] 26%|██▌       | 104797/400000 [00:10<00:27, 10874.60it/s] 26%|██▋       | 105930/400000 [00:10<00:26, 11005.78it/s] 27%|██▋       | 107032/400000 [00:10<00:26, 10852.54it/s] 27%|██▋       | 108119/400000 [00:10<00:27, 10594.89it/s] 27%|██▋       | 109181/400000 [00:10<00:28, 10187.33it/s] 28%|██▊       | 110208/400000 [00:11<00:28, 10211.59it/s] 28%|██▊       | 111239/400000 [00:11<00:28, 10239.38it/s] 28%|██▊       | 112266/400000 [00:11<00:28, 10203.82it/s] 28%|██▊       | 113300/400000 [00:11<00:27, 10242.35it/s] 29%|██▊       | 114326/400000 [00:11<00:27, 10222.13it/s] 29%|██▉       | 115353/400000 [00:11<00:27, 10235.39it/s] 29%|██▉       | 116426/400000 [00:11<00:27, 10377.04it/s] 29%|██▉       | 117480/400000 [00:11<00:27, 10423.47it/s] 30%|██▉       | 118524/400000 [00:11<00:27, 10394.94it/s] 30%|██▉       | 119667/400000 [00:11<00:26, 10684.60it/s] 30%|███       | 120738/400000 [00:12<00:26, 10605.28it/s] 30%|███       | 121823/400000 [00:12<00:26, 10677.10it/s] 31%|███       | 122916/400000 [00:12<00:25, 10749.21it/s] 31%|███       | 123992/400000 [00:12<00:25, 10736.24it/s] 31%|███▏      | 125067/400000 [00:12<00:25, 10703.72it/s] 32%|███▏      | 126215/400000 [00:12<00:25, 10925.25it/s] 32%|███▏      | 127357/400000 [00:12<00:24, 11068.61it/s] 32%|███▏      | 128466/400000 [00:12<00:25, 10836.65it/s] 32%|███▏      | 129552/400000 [00:12<00:25, 10657.80it/s] 33%|███▎      | 130638/400000 [00:12<00:25, 10640.11it/s] 33%|███▎      | 131704/400000 [00:13<00:25, 10425.33it/s] 33%|███▎      | 132777/400000 [00:13<00:25, 10512.09it/s] 33%|███▎      | 133918/400000 [00:13<00:24, 10766.06it/s] 34%|███▎      | 134998/400000 [00:13<00:24, 10747.45it/s] 34%|███▍      | 136081/400000 [00:13<00:24, 10769.69it/s] 34%|███▍      | 137160/400000 [00:13<00:24, 10645.13it/s] 35%|███▍      | 138275/400000 [00:13<00:24, 10789.72it/s] 35%|███▍      | 139356/400000 [00:13<00:25, 10405.62it/s] 35%|███▌      | 140409/400000 [00:13<00:24, 10440.02it/s] 35%|███▌      | 141504/400000 [00:13<00:24, 10586.10it/s] 36%|███▌      | 142566/400000 [00:14<00:24, 10340.64it/s] 36%|███▌      | 143604/400000 [00:14<00:24, 10338.17it/s] 36%|███▌      | 144640/400000 [00:14<00:24, 10287.47it/s] 36%|███▋      | 145706/400000 [00:14<00:24, 10394.37it/s] 37%|███▋      | 146804/400000 [00:14<00:23, 10560.72it/s] 37%|███▋      | 147889/400000 [00:14<00:23, 10643.21it/s] 37%|███▋      | 148991/400000 [00:14<00:23, 10752.56it/s] 38%|███▊      | 150120/400000 [00:14<00:22, 10906.72it/s] 38%|███▊      | 151213/400000 [00:14<00:23, 10585.38it/s] 38%|███▊      | 152306/400000 [00:14<00:23, 10685.73it/s] 38%|███▊      | 153377/400000 [00:15<00:23, 10646.50it/s] 39%|███▊      | 154449/400000 [00:15<00:23, 10666.64it/s] 39%|███▉      | 155517/400000 [00:15<00:23, 10626.71it/s] 39%|███▉      | 156581/400000 [00:15<00:23, 10314.01it/s] 39%|███▉      | 157658/400000 [00:15<00:23, 10446.02it/s] 40%|███▉      | 158789/400000 [00:15<00:22, 10690.72it/s] 40%|███▉      | 159884/400000 [00:15<00:22, 10765.12it/s] 40%|████      | 160963/400000 [00:15<00:22, 10766.10it/s] 41%|████      | 162042/400000 [00:15<00:22, 10656.24it/s] 41%|████      | 163125/400000 [00:16<00:22, 10706.13it/s] 41%|████      | 164197/400000 [00:16<00:22, 10287.71it/s] 41%|████▏     | 165342/400000 [00:16<00:22, 10610.55it/s] 42%|████▏     | 166425/400000 [00:16<00:21, 10673.17it/s] 42%|████▏     | 167497/400000 [00:16<00:22, 10415.57it/s] 42%|████▏     | 168592/400000 [00:16<00:21, 10569.47it/s] 42%|████▏     | 169653/400000 [00:16<00:22, 10460.68it/s] 43%|████▎     | 170727/400000 [00:16<00:21, 10541.72it/s] 43%|████▎     | 171878/400000 [00:16<00:21, 10812.23it/s] 43%|████▎     | 172963/400000 [00:16<00:21, 10764.19it/s] 44%|████▎     | 174042/400000 [00:17<00:21, 10544.20it/s] 44%|████▍     | 175099/400000 [00:17<00:22, 10114.59it/s] 44%|████▍     | 176116/400000 [00:17<00:22, 10110.46it/s] 44%|████▍     | 177131/400000 [00:17<00:22, 9892.59it/s]  45%|████▍     | 178202/400000 [00:17<00:21, 10123.58it/s] 45%|████▍     | 179255/400000 [00:17<00:21, 10241.76it/s] 45%|████▌     | 180346/400000 [00:17<00:21, 10432.58it/s] 45%|████▌     | 181451/400000 [00:17<00:20, 10608.17it/s] 46%|████▌     | 182533/400000 [00:17<00:20, 10668.95it/s] 46%|████▌     | 183602/400000 [00:17<00:20, 10606.89it/s] 46%|████▌     | 184665/400000 [00:18<00:20, 10332.29it/s] 46%|████▋     | 185718/400000 [00:18<00:20, 10389.03it/s] 47%|████▋     | 186842/400000 [00:18<00:20, 10629.23it/s] 47%|████▋     | 187917/400000 [00:18<00:19, 10664.89it/s] 47%|████▋     | 189024/400000 [00:18<00:19, 10781.03it/s] 48%|████▊     | 190110/400000 [00:18<00:19, 10803.59it/s] 48%|████▊     | 191192/400000 [00:18<00:19, 10690.30it/s] 48%|████▊     | 192263/400000 [00:18<00:19, 10608.05it/s] 48%|████▊     | 193325/400000 [00:18<00:19, 10440.92it/s] 49%|████▊     | 194448/400000 [00:18<00:19, 10664.25it/s] 49%|████▉     | 195517/400000 [00:19<00:19, 10423.29it/s] 49%|████▉     | 196562/400000 [00:19<00:19, 10337.35it/s] 49%|████▉     | 197638/400000 [00:19<00:19, 10459.52it/s] 50%|████▉     | 198686/400000 [00:19<00:19, 10431.73it/s] 50%|████▉     | 199774/400000 [00:19<00:18, 10560.14it/s] 50%|█████     | 200927/400000 [00:19<00:18, 10833.20it/s] 51%|█████     | 202022/400000 [00:19<00:18, 10865.62it/s] 51%|█████     | 203111/400000 [00:19<00:18, 10830.60it/s] 51%|█████     | 204196/400000 [00:19<00:18, 10599.24it/s] 51%|█████▏    | 205312/400000 [00:20<00:18, 10759.86it/s] 52%|█████▏    | 206390/400000 [00:20<00:18, 10513.50it/s] 52%|█████▏    | 207474/400000 [00:20<00:18, 10607.66it/s] 52%|█████▏    | 208537/400000 [00:20<00:18, 10421.28it/s] 52%|█████▏    | 209582/400000 [00:20<00:18, 10253.17it/s] 53%|█████▎    | 210732/400000 [00:20<00:17, 10597.11it/s] 53%|█████▎    | 211797/400000 [00:20<00:17, 10596.95it/s] 53%|█████▎    | 212907/400000 [00:20<00:17, 10741.03it/s] 54%|█████▎    | 214031/400000 [00:20<00:17, 10884.60it/s] 54%|█████▍    | 215122/400000 [00:20<00:17, 10765.96it/s] 54%|█████▍    | 216201/400000 [00:21<00:17, 10678.34it/s] 54%|█████▍    | 217316/400000 [00:21<00:16, 10813.04it/s] 55%|█████▍    | 218442/400000 [00:21<00:16, 10941.90it/s] 55%|█████▍    | 219594/400000 [00:21<00:16, 11107.36it/s] 55%|█████▌    | 220707/400000 [00:21<00:16, 10931.34it/s] 55%|█████▌    | 221866/400000 [00:21<00:16, 11119.14it/s] 56%|█████▌    | 222980/400000 [00:21<00:16, 11043.35it/s] 56%|█████▌    | 224086/400000 [00:21<00:16, 10816.05it/s] 56%|█████▋    | 225170/400000 [00:21<00:16, 10719.09it/s] 57%|█████▋    | 226244/400000 [00:21<00:16, 10706.76it/s] 57%|█████▋    | 227316/400000 [00:22<00:16, 10522.29it/s] 57%|█████▋    | 228370/400000 [00:22<00:16, 10484.48it/s] 57%|█████▋    | 229491/400000 [00:22<00:15, 10689.48it/s] 58%|█████▊    | 230625/400000 [00:22<00:15, 10876.50it/s] 58%|█████▊    | 231715/400000 [00:22<00:15, 10616.67it/s] 58%|█████▊    | 232883/400000 [00:22<00:15, 10913.88it/s] 58%|█████▊    | 233979/400000 [00:22<00:15, 10847.22it/s] 59%|█████▉    | 235089/400000 [00:22<00:15, 10920.40it/s] 59%|█████▉    | 236229/400000 [00:22<00:14, 11058.18it/s] 59%|█████▉    | 237337/400000 [00:22<00:15, 10570.61it/s] 60%|█████▉    | 238400/400000 [00:23<00:15, 10145.60it/s] 60%|█████▉    | 239423/400000 [00:23<00:16, 9642.82it/s]  60%|██████    | 240514/400000 [00:23<00:15, 9990.08it/s] 60%|██████    | 241524/400000 [00:23<00:16, 9669.47it/s] 61%|██████    | 242501/400000 [00:23<00:16, 9383.45it/s] 61%|██████    | 243524/400000 [00:23<00:16, 9620.73it/s] 61%|██████    | 244563/400000 [00:23<00:15, 9837.79it/s] 61%|██████▏   | 245554/400000 [00:23<00:15, 9768.04it/s] 62%|██████▏   | 246585/400000 [00:23<00:15, 9924.02it/s] 62%|██████▏   | 247582/400000 [00:24<00:16, 9486.74it/s] 62%|██████▏   | 248538/400000 [00:24<00:16, 9418.33it/s] 62%|██████▏   | 249538/400000 [00:24<00:15, 9585.56it/s] 63%|██████▎   | 250587/400000 [00:24<00:15, 9838.67it/s] 63%|██████▎   | 251652/400000 [00:24<00:14, 10030.39it/s] 63%|██████▎   | 252700/400000 [00:24<00:14, 10160.99it/s] 63%|██████▎   | 253720/400000 [00:24<00:14, 9937.15it/s]  64%|██████▎   | 254787/400000 [00:24<00:14, 10144.12it/s] 64%|██████▍   | 255935/400000 [00:24<00:13, 10511.00it/s] 64%|██████▍   | 257075/400000 [00:24<00:13, 10762.55it/s] 65%|██████▍   | 258157/400000 [00:25<00:13, 10230.37it/s] 65%|██████▍   | 259192/400000 [00:25<00:13, 10262.92it/s] 65%|██████▌   | 260225/400000 [00:25<00:13, 10271.71it/s] 65%|██████▌   | 261309/400000 [00:25<00:13, 10432.84it/s] 66%|██████▌   | 262382/400000 [00:25<00:13, 10519.19it/s] 66%|██████▌   | 263522/400000 [00:25<00:12, 10767.59it/s] 66%|██████▌   | 264666/400000 [00:25<00:12, 10957.31it/s] 66%|██████▋   | 265765/400000 [00:25<00:12, 10881.67it/s] 67%|██████▋   | 266892/400000 [00:25<00:12, 10993.72it/s] 67%|██████▋   | 267994/400000 [00:26<00:12, 10791.93it/s] 67%|██████▋   | 269076/400000 [00:26<00:12, 10508.92it/s] 68%|██████▊   | 270190/400000 [00:26<00:12, 10688.33it/s] 68%|██████▊   | 271262/400000 [00:26<00:12, 10696.74it/s] 68%|██████▊   | 272350/400000 [00:26<00:11, 10750.12it/s] 68%|██████▊   | 273427/400000 [00:26<00:12, 10438.19it/s] 69%|██████▊   | 274515/400000 [00:26<00:11, 10564.62it/s] 69%|██████▉   | 275611/400000 [00:26<00:11, 10678.73it/s] 69%|██████▉   | 276702/400000 [00:26<00:11, 10746.37it/s] 69%|██████▉   | 277849/400000 [00:26<00:11, 10951.25it/s] 70%|██████▉   | 278947/400000 [00:27<00:11, 10686.77it/s] 70%|███████   | 280019/400000 [00:27<00:12, 9887.54it/s]  70%|███████   | 281022/400000 [00:27<00:12, 9612.68it/s] 71%|███████   | 282056/400000 [00:27<00:12, 9816.64it/s] 71%|███████   | 283117/400000 [00:27<00:11, 10041.57it/s] 71%|███████   | 284168/400000 [00:27<00:11, 10176.70it/s] 71%|███████▏  | 285318/400000 [00:27<00:10, 10539.82it/s] 72%|███████▏  | 286380/400000 [00:27<00:10, 10516.41it/s] 72%|███████▏  | 287524/400000 [00:27<00:10, 10776.67it/s] 72%|███████▏  | 288613/400000 [00:27<00:10, 10808.59it/s] 72%|███████▏  | 289698/400000 [00:28<00:10, 10573.41it/s] 73%|███████▎  | 290760/400000 [00:28<00:10, 10569.07it/s] 73%|███████▎  | 291820/400000 [00:28<00:10, 9966.13it/s]  73%|███████▎  | 292826/400000 [00:28<00:11, 9651.33it/s] 73%|███████▎  | 293800/400000 [00:28<00:11, 9550.33it/s] 74%|███████▎  | 294846/400000 [00:28<00:10, 9804.92it/s] 74%|███████▍  | 295914/400000 [00:28<00:10, 10049.10it/s] 74%|███████▍  | 296925/400000 [00:28<00:10, 9953.76it/s]  74%|███████▍  | 297949/400000 [00:28<00:10, 10034.37it/s] 75%|███████▍  | 298976/400000 [00:29<00:09, 10102.47it/s] 75%|███████▌  | 300006/400000 [00:29<00:09, 10160.03it/s] 75%|███████▌  | 301074/400000 [00:29<00:09, 10309.17it/s] 76%|███████▌  | 302155/400000 [00:29<00:09, 10453.06it/s] 76%|███████▌  | 303202/400000 [00:29<00:09, 10422.61it/s] 76%|███████▌  | 304246/400000 [00:29<00:09, 10156.78it/s] 76%|███████▋  | 305265/400000 [00:29<00:09, 9900.31it/s]  77%|███████▋  | 306259/400000 [00:29<00:09, 9676.17it/s] 77%|███████▋  | 307237/400000 [00:29<00:09, 9705.66it/s] 77%|███████▋  | 308273/400000 [00:29<00:09, 9890.89it/s] 77%|███████▋  | 309265/400000 [00:30<00:09, 9879.29it/s] 78%|███████▊  | 310255/400000 [00:30<00:09, 9749.83it/s] 78%|███████▊  | 311232/400000 [00:30<00:09, 9520.20it/s] 78%|███████▊  | 312187/400000 [00:30<00:09, 9330.34it/s] 78%|███████▊  | 313123/400000 [00:30<00:09, 9333.59it/s] 79%|███████▊  | 314059/400000 [00:30<00:09, 9322.08it/s] 79%|███████▉  | 315038/400000 [00:30<00:08, 9454.91it/s] 79%|███████▉  | 315997/400000 [00:30<00:08, 9494.81it/s] 79%|███████▉  | 316948/400000 [00:30<00:08, 9352.63it/s] 79%|███████▉  | 317885/400000 [00:30<00:08, 9127.77it/s] 80%|███████▉  | 318800/400000 [00:31<00:09, 9014.58it/s] 80%|███████▉  | 319783/400000 [00:31<00:08, 9244.33it/s] 80%|████████  | 320811/400000 [00:31<00:08, 9531.43it/s] 80%|████████  | 321775/400000 [00:31<00:08, 9561.25it/s] 81%|████████  | 322735/400000 [00:31<00:08, 9559.38it/s] 81%|████████  | 323693/400000 [00:31<00:08, 9427.46it/s] 81%|████████  | 324638/400000 [00:31<00:08, 9115.63it/s] 81%|████████▏ | 325587/400000 [00:31<00:08, 9223.96it/s] 82%|████████▏ | 326513/400000 [00:31<00:07, 9190.74it/s] 82%|████████▏ | 327445/400000 [00:32<00:07, 9228.19it/s] 82%|████████▏ | 328385/400000 [00:32<00:07, 9275.12it/s] 82%|████████▏ | 329388/400000 [00:32<00:07, 9486.93it/s] 83%|████████▎ | 330398/400000 [00:32<00:07, 9662.61it/s] 83%|████████▎ | 331367/400000 [00:32<00:07, 9623.41it/s] 83%|████████▎ | 332335/400000 [00:32<00:07, 9638.14it/s] 83%|████████▎ | 333311/400000 [00:32<00:06, 9673.16it/s] 84%|████████▎ | 334280/400000 [00:32<00:06, 9630.88it/s] 84%|████████▍ | 335284/400000 [00:32<00:06, 9748.03it/s] 84%|████████▍ | 336282/400000 [00:32<00:06, 9815.37it/s] 84%|████████▍ | 337292/400000 [00:33<00:06, 9896.60it/s] 85%|████████▍ | 338283/400000 [00:33<00:06, 9824.84it/s] 85%|████████▍ | 339272/400000 [00:33<00:06, 9842.64it/s] 85%|████████▌ | 340257/400000 [00:33<00:06, 9819.85it/s] 85%|████████▌ | 341267/400000 [00:33<00:05, 9901.59it/s] 86%|████████▌ | 342319/400000 [00:33<00:05, 10076.63it/s] 86%|████████▌ | 343372/400000 [00:33<00:05, 10206.24it/s] 86%|████████▌ | 344410/400000 [00:33<00:05, 10255.60it/s] 86%|████████▋ | 345494/400000 [00:33<00:05, 10422.47it/s] 87%|████████▋ | 346597/400000 [00:33<00:05, 10597.52it/s] 87%|████████▋ | 347678/400000 [00:34<00:04, 10659.65it/s] 87%|████████▋ | 348746/400000 [00:34<00:04, 10482.00it/s] 87%|████████▋ | 349833/400000 [00:34<00:04, 10594.23it/s] 88%|████████▊ | 350946/400000 [00:34<00:04, 10747.66it/s] 88%|████████▊ | 352023/400000 [00:34<00:04, 10543.39it/s] 88%|████████▊ | 353080/400000 [00:34<00:04, 10337.12it/s] 89%|████████▊ | 354116/400000 [00:34<00:04, 10129.02it/s] 89%|████████▉ | 355132/400000 [00:34<00:04, 9569.43it/s]  89%|████████▉ | 356098/400000 [00:34<00:04, 9012.08it/s] 89%|████████▉ | 357228/400000 [00:34<00:04, 9593.36it/s] 90%|████████▉ | 358317/400000 [00:35<00:04, 9946.91it/s] 90%|████████▉ | 359352/400000 [00:35<00:04, 10062.78it/s] 90%|█████████ | 360483/400000 [00:35<00:03, 10406.68it/s] 90%|█████████ | 361598/400000 [00:35<00:03, 10618.82it/s] 91%|█████████ | 362736/400000 [00:35<00:03, 10835.74it/s] 91%|█████████ | 363835/400000 [00:35<00:03, 10881.01it/s] 91%|█████████ | 364929/400000 [00:35<00:03, 10526.61it/s] 91%|█████████▏| 365988/400000 [00:35<00:03, 10326.94it/s] 92%|█████████▏| 367113/400000 [00:35<00:03, 10586.72it/s] 92%|█████████▏| 368182/400000 [00:36<00:02, 10614.86it/s] 92%|█████████▏| 369248/400000 [00:36<00:03, 10107.83it/s] 93%|█████████▎| 370267/400000 [00:36<00:02, 9952.94it/s]  93%|█████████▎| 371269/400000 [00:36<00:02, 9820.99it/s] 93%|█████████▎| 372256/400000 [00:36<00:02, 9815.28it/s] 93%|█████████▎| 373241/400000 [00:36<00:02, 9798.04it/s] 94%|█████████▎| 374242/400000 [00:36<00:02, 9856.21it/s] 94%|█████████▍| 375230/400000 [00:36<00:02, 9801.22it/s] 94%|█████████▍| 376250/400000 [00:36<00:02, 9916.99it/s] 94%|█████████▍| 377243/400000 [00:36<00:02, 9900.27it/s] 95%|█████████▍| 378234/400000 [00:37<00:02, 9858.45it/s] 95%|█████████▍| 379221/400000 [00:37<00:02, 9651.06it/s] 95%|█████████▌| 380188/400000 [00:37<00:02, 9538.42it/s] 95%|█████████▌| 381170/400000 [00:37<00:01, 9619.74it/s] 96%|█████████▌| 382134/400000 [00:37<00:01, 9548.16it/s] 96%|█████████▌| 383090/400000 [00:37<00:01, 9535.84it/s] 96%|█████████▌| 384045/400000 [00:37<00:01, 9400.15it/s] 96%|█████████▌| 384986/400000 [00:37<00:01, 9375.87it/s] 96%|█████████▋| 385942/400000 [00:37<00:01, 9425.05it/s] 97%|█████████▋| 386885/400000 [00:37<00:01, 9420.72it/s] 97%|█████████▋| 387828/400000 [00:38<00:01, 9237.93it/s] 97%|█████████▋| 388753/400000 [00:38<00:01, 9154.47it/s] 97%|█████████▋| 389704/400000 [00:38<00:01, 9256.00it/s] 98%|█████████▊| 390667/400000 [00:38<00:00, 9363.43it/s] 98%|█████████▊| 391652/400000 [00:38<00:00, 9502.23it/s] 98%|█████████▊| 392604/400000 [00:38<00:00, 9498.50it/s] 98%|█████████▊| 393555/400000 [00:38<00:00, 9330.35it/s] 99%|█████████▊| 394490/400000 [00:38<00:00, 9247.37it/s] 99%|█████████▉| 395432/400000 [00:38<00:00, 9295.05it/s] 99%|█████████▉| 396363/400000 [00:38<00:00, 9242.73it/s] 99%|█████████▉| 397290/400000 [00:39<00:00, 9248.31it/s]100%|█████████▉| 398218/400000 [00:39<00:00, 9257.58it/s]100%|█████████▉| 399160/400000 [00:39<00:00, 9305.64it/s]100%|█████████▉| 399999/400000 [00:39<00:00, 10160.23it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fce57bb8da0>, <torchtext.data.dataset.TabularDataset object at 0x7fce57bb8ef0>, <torchtext.vocab.Vocab object at 0x7fce57bb8e10>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.51 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  5.51 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  5.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  5.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.72 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.04 file/s]2020-07-13 00:18:01.196756: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-13 00:18:01.201167: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-07-13 00:18:01.201303: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56185b7f95c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-13 00:18:01.201315: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 146915.60it/s] 85%|████████▌ | 8437760/9912422 [00:00<00:07, 209714.18it/s]9920512it [00:00, 35883219.57it/s]                           
0it [00:00, ?it/s]32768it [00:00, 572428.56it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 471021.17it/s]1654784it [00:00, 11599716.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 184182.18it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
