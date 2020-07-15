
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fad2c07a0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fad2c07a0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fad973c31e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fad973c31e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fadb570be18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fadb570be18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad446f1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad446f1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fad446f1620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 151092.25it/s] 69%|██████▉   | 6856704/9912422 [00:00<00:14, 215641.07it/s]9920512it [00:00, 42155771.18it/s]                           
0it [00:00, ?it/s]32768it [00:00, 589113.29it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 135666.87it/s]1654784it [00:00, 10161212.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168355.82it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fad41ae7908>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fad2b725b70>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fad446f1268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fad446f1268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fad446f1268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:37,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:36,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:06, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:06, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 20.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 20.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:05, 20.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:05, 20.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 66.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 77.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 79.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.30 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.68s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.11s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.30 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.68s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.69s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.58 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.69s/ url]
0 examples [00:00, ? examples/s]2020-07-15 00:09:25.838202: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-15 00:09:25.892750: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-15 00:09:25.893519: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ea96f2be60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-15 00:09:25.893542: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.99 examples/s]97 examples [00:00,  5.68 examples/s]190 examples [00:00,  8.10 examples/s]294 examples [00:00, 11.53 examples/s]398 examples [00:00, 16.40 examples/s]503 examples [00:00, 23.27 examples/s]606 examples [00:00, 32.92 examples/s]710 examples [00:00, 46.39 examples/s]814 examples [00:01, 65.03 examples/s]916 examples [00:01, 90.42 examples/s]1022 examples [00:01, 124.57 examples/s]1127 examples [00:01, 169.30 examples/s]1232 examples [00:01, 226.14 examples/s]1335 examples [00:01, 295.03 examples/s]1439 examples [00:01, 375.55 examples/s]1545 examples [00:01, 465.42 examples/s]1652 examples [00:01, 560.14 examples/s]1758 examples [00:01, 651.55 examples/s]1864 examples [00:02, 736.03 examples/s]1970 examples [00:02, 809.46 examples/s]2076 examples [00:02, 868.24 examples/s]2182 examples [00:02, 916.93 examples/s]2288 examples [00:02, 945.56 examples/s]2393 examples [00:02, 972.82 examples/s]2499 examples [00:02, 995.36 examples/s]2605 examples [00:02, 1013.32 examples/s]2710 examples [00:02, 1010.62 examples/s]2815 examples [00:02, 1020.38 examples/s]2919 examples [00:03, 1026.13 examples/s]3025 examples [00:03, 1033.91 examples/s]3132 examples [00:03, 1041.66 examples/s]3237 examples [00:03, 1039.47 examples/s]3342 examples [00:03, 1030.88 examples/s]3446 examples [00:03, 1030.62 examples/s]3550 examples [00:03, 1022.13 examples/s]3653 examples [00:03, 1012.43 examples/s]3755 examples [00:03, 1000.90 examples/s]3856 examples [00:03, 1002.64 examples/s]3959 examples [00:04, 1008.92 examples/s]4060 examples [00:04, 989.16 examples/s] 4163 examples [00:04, 1000.79 examples/s]4264 examples [00:04, 998.95 examples/s] 4367 examples [00:04, 1005.66 examples/s]4468 examples [00:04, 991.33 examples/s] 4573 examples [00:04, 1006.36 examples/s]4676 examples [00:04, 1013.09 examples/s]4782 examples [00:04, 1024.64 examples/s]4885 examples [00:05, 1024.63 examples/s]4990 examples [00:05, 1030.51 examples/s]5096 examples [00:05, 1038.03 examples/s]5200 examples [00:05, 1024.18 examples/s]5303 examples [00:05, 1021.69 examples/s]5408 examples [00:05, 1028.47 examples/s]5511 examples [00:05, 1028.77 examples/s]5614 examples [00:05, 1021.06 examples/s]5717 examples [00:05, 990.35 examples/s] 5822 examples [00:05, 1004.72 examples/s]5925 examples [00:06, 1011.69 examples/s]6029 examples [00:06, 1018.04 examples/s]6133 examples [00:06, 1022.41 examples/s]6236 examples [00:06, 1001.33 examples/s]6339 examples [00:06, 1008.60 examples/s]6441 examples [00:06, 1010.30 examples/s]6543 examples [00:06, 1013.14 examples/s]6645 examples [00:06, 1011.42 examples/s]6747 examples [00:06, 1013.58 examples/s]6849 examples [00:06, 1015.18 examples/s]6951 examples [00:07, 1013.52 examples/s]7053 examples [00:07, 1004.70 examples/s]7154 examples [00:07, 996.34 examples/s] 7258 examples [00:07, 1007.03 examples/s]7363 examples [00:07, 1019.02 examples/s]7468 examples [00:07, 1026.86 examples/s]7571 examples [00:07, 1013.47 examples/s]7673 examples [00:07, 1013.59 examples/s]7777 examples [00:07, 1020.68 examples/s]7884 examples [00:07, 1032.00 examples/s]7988 examples [00:08, 1027.25 examples/s]8091 examples [00:08, 1022.91 examples/s]8194 examples [00:08, 1010.65 examples/s]8296 examples [00:08, 1003.89 examples/s]8401 examples [00:08, 1014.79 examples/s]8505 examples [00:08, 1022.18 examples/s]8609 examples [00:08, 1026.15 examples/s]8713 examples [00:08, 1029.60 examples/s]8817 examples [00:08, 1032.40 examples/s]8923 examples [00:08, 1038.61 examples/s]9027 examples [00:09, 1024.52 examples/s]9130 examples [00:09, 1013.15 examples/s]9233 examples [00:09, 1016.96 examples/s]9335 examples [00:09, 1010.92 examples/s]9437 examples [00:09, 1011.18 examples/s]9540 examples [00:09, 1015.30 examples/s]9642 examples [00:09, 1008.66 examples/s]9743 examples [00:09, 970.43 examples/s] 9848 examples [00:09, 991.94 examples/s]9953 examples [00:10, 1006.16 examples/s]10054 examples [00:10, 957.59 examples/s]10154 examples [00:10, 967.79 examples/s]10257 examples [00:10, 984.41 examples/s]10361 examples [00:10, 999.29 examples/s]10462 examples [00:10, 1001.10 examples/s]10567 examples [00:10, 1012.79 examples/s]10673 examples [00:10, 1023.74 examples/s]10776 examples [00:10, 1023.94 examples/s]10880 examples [00:10, 1027.05 examples/s]10985 examples [00:11, 1032.70 examples/s]11089 examples [00:11, 1034.47 examples/s]11193 examples [00:11, 1030.47 examples/s]11297 examples [00:11, 1027.72 examples/s]11400 examples [00:11, 1021.98 examples/s]11505 examples [00:11, 1029.52 examples/s]11608 examples [00:11, 1022.97 examples/s]11711 examples [00:11, 1017.86 examples/s]11814 examples [00:11, 1018.69 examples/s]11918 examples [00:11, 1022.82 examples/s]12024 examples [00:12, 1031.05 examples/s]12128 examples [00:12, 1031.60 examples/s]12232 examples [00:12, 1019.84 examples/s]12335 examples [00:12, 1015.28 examples/s]12439 examples [00:12, 1022.24 examples/s]12544 examples [00:12, 1029.35 examples/s]12647 examples [00:12, 1025.85 examples/s]12750 examples [00:12, 1015.10 examples/s]12852 examples [00:12, 1001.35 examples/s]12958 examples [00:12, 1017.25 examples/s]13063 examples [00:13, 1025.23 examples/s]13166 examples [00:13, 1024.83 examples/s]13269 examples [00:13, 996.30 examples/s] 13374 examples [00:13, 1010.15 examples/s]13480 examples [00:13, 1023.28 examples/s]13585 examples [00:13, 1030.40 examples/s]13691 examples [00:13, 1038.14 examples/s]13796 examples [00:13, 1039.71 examples/s]13901 examples [00:13, 1038.74 examples/s]14007 examples [00:13, 1043.38 examples/s]14112 examples [00:14, 1039.99 examples/s]14218 examples [00:14, 1044.19 examples/s]14323 examples [00:14, 1042.42 examples/s]14428 examples [00:14, 1037.54 examples/s]14532 examples [00:14, 1033.41 examples/s]14637 examples [00:14, 1036.69 examples/s]14742 examples [00:14, 1040.44 examples/s]14848 examples [00:14, 1042.58 examples/s]14954 examples [00:14, 1046.37 examples/s]15059 examples [00:14, 1046.10 examples/s]15164 examples [00:15, 1038.38 examples/s]15270 examples [00:15, 1043.16 examples/s]15376 examples [00:15, 1045.33 examples/s]15481 examples [00:15, 1035.89 examples/s]15585 examples [00:15, 1033.69 examples/s]15690 examples [00:15, 1035.72 examples/s]15796 examples [00:15, 1039.99 examples/s]15901 examples [00:15, 1023.41 examples/s]16004 examples [00:15, 1011.92 examples/s]16106 examples [00:16, 1008.33 examples/s]16211 examples [00:16, 1018.35 examples/s]16315 examples [00:16, 1022.20 examples/s]16418 examples [00:16, 1016.01 examples/s]16521 examples [00:16, 1018.92 examples/s]16625 examples [00:16, 1024.86 examples/s]16728 examples [00:16, 1021.04 examples/s]16831 examples [00:16, 1004.81 examples/s]16934 examples [00:16, 1010.95 examples/s]17038 examples [00:16, 1017.98 examples/s]17142 examples [00:17, 1023.91 examples/s]17247 examples [00:17, 1029.09 examples/s]17350 examples [00:17, 1024.82 examples/s]17453 examples [00:17, 1012.83 examples/s]17558 examples [00:17, 1021.25 examples/s]17663 examples [00:17, 1029.63 examples/s]17768 examples [00:17, 1034.18 examples/s]17872 examples [00:17, 1035.57 examples/s]17976 examples [00:17, 1031.60 examples/s]18082 examples [00:17, 1037.63 examples/s]18187 examples [00:18, 1039.06 examples/s]18291 examples [00:18, 1038.13 examples/s]18398 examples [00:18, 1045.07 examples/s]18504 examples [00:18, 1045.62 examples/s]18609 examples [00:18, 1046.11 examples/s]18714 examples [00:18, 1042.00 examples/s]18819 examples [00:18, 1024.19 examples/s]18924 examples [00:18, 1029.88 examples/s]19028 examples [00:18, 1030.41 examples/s]19132 examples [00:18, 1032.54 examples/s]19239 examples [00:19, 1041.36 examples/s]19344 examples [00:19, 1039.13 examples/s]19448 examples [00:19, 1032.18 examples/s]19552 examples [00:19, 1031.49 examples/s]19657 examples [00:19, 1035.88 examples/s]19762 examples [00:19, 1039.63 examples/s]19868 examples [00:19, 1043.24 examples/s]19974 examples [00:19, 1048.20 examples/s]20079 examples [00:19, 994.59 examples/s] 20184 examples [00:19, 1009.40 examples/s]20286 examples [00:20, 1001.94 examples/s]20391 examples [00:20, 1014.34 examples/s]20494 examples [00:20, 1017.89 examples/s]20598 examples [00:20, 1023.63 examples/s]20702 examples [00:20, 1027.28 examples/s]20807 examples [00:20, 1031.64 examples/s]20912 examples [00:20, 1034.69 examples/s]21016 examples [00:20, 1033.12 examples/s]21120 examples [00:20, 1033.72 examples/s]21224 examples [00:20, 1025.06 examples/s]21327 examples [00:21, 1025.66 examples/s]21431 examples [00:21, 1027.49 examples/s]21536 examples [00:21, 1031.64 examples/s]21640 examples [00:21, 1032.61 examples/s]21744 examples [00:21, 1030.95 examples/s]21848 examples [00:21, 1031.34 examples/s]21952 examples [00:21, 1029.82 examples/s]22055 examples [00:21, 1029.07 examples/s]22158 examples [00:21, 1015.71 examples/s]22263 examples [00:21, 1023.29 examples/s]22368 examples [00:22, 1028.93 examples/s]22473 examples [00:22, 1032.44 examples/s]22577 examples [00:22, 1019.26 examples/s]22679 examples [00:22, 1002.44 examples/s]22782 examples [00:22, 1009.86 examples/s]22885 examples [00:22, 1015.66 examples/s]22987 examples [00:22, 1010.84 examples/s]23089 examples [00:22, 1009.06 examples/s]23193 examples [00:22, 1016.07 examples/s]23298 examples [00:23, 1025.44 examples/s]23401 examples [00:23, 1023.81 examples/s]23506 examples [00:23, 1031.34 examples/s]23611 examples [00:23, 1035.49 examples/s]23715 examples [00:23, 1036.18 examples/s]23819 examples [00:23, 1034.23 examples/s]23923 examples [00:23, 1034.72 examples/s]24028 examples [00:23, 1037.68 examples/s]24133 examples [00:23, 1039.01 examples/s]24238 examples [00:23, 1040.44 examples/s]24343 examples [00:24, 1038.88 examples/s]24447 examples [00:24, 1034.34 examples/s]24551 examples [00:24, 1031.49 examples/s]24655 examples [00:24, 1025.16 examples/s]24758 examples [00:24, 998.19 examples/s] 24859 examples [00:24, 1000.42 examples/s]24964 examples [00:24, 1002.22 examples/s]25067 examples [00:24, 1009.46 examples/s]25172 examples [00:24, 1020.45 examples/s]25277 examples [00:24, 1026.52 examples/s]25380 examples [00:25, 1024.97 examples/s]25486 examples [00:25, 1032.43 examples/s]25590 examples [00:25, 1018.10 examples/s]25695 examples [00:25, 1026.43 examples/s]25798 examples [00:25, 1025.55 examples/s]25901 examples [00:25, 1022.99 examples/s]26005 examples [00:25, 1027.49 examples/s]26109 examples [00:25, 1028.30 examples/s]26212 examples [00:25, 1021.98 examples/s]26316 examples [00:25, 1024.91 examples/s]26420 examples [00:26, 1027.57 examples/s]26523 examples [00:26, 1026.49 examples/s]26626 examples [00:26, 1023.17 examples/s]26730 examples [00:26, 1027.21 examples/s]26834 examples [00:26, 1029.75 examples/s]26937 examples [00:26, 1009.75 examples/s]27041 examples [00:26, 1017.58 examples/s]27145 examples [00:26, 1023.25 examples/s]27249 examples [00:26, 1025.52 examples/s]27353 examples [00:26, 1027.81 examples/s]27456 examples [00:27, 1022.99 examples/s]27562 examples [00:27, 1031.95 examples/s]27667 examples [00:27, 1036.46 examples/s]27771 examples [00:27, 1035.55 examples/s]27875 examples [00:27, 1036.09 examples/s]27979 examples [00:27, 1032.58 examples/s]28084 examples [00:27, 1035.03 examples/s]28190 examples [00:27, 1039.48 examples/s]28294 examples [00:27, 1022.22 examples/s]28398 examples [00:27, 1025.80 examples/s]28501 examples [00:28, 1014.24 examples/s]28603 examples [00:28, 1014.15 examples/s]28706 examples [00:28, 1017.41 examples/s]28811 examples [00:28, 1025.85 examples/s]28915 examples [00:28, 1029.77 examples/s]29019 examples [00:28, 1021.26 examples/s]29122 examples [00:28, 1018.87 examples/s]29227 examples [00:28, 1025.86 examples/s]29330 examples [00:28, 1018.33 examples/s]29432 examples [00:28, 1018.40 examples/s]29534 examples [00:29, 1012.24 examples/s]29638 examples [00:29, 1020.19 examples/s]29741 examples [00:29, 1021.88 examples/s]29845 examples [00:29, 1025.43 examples/s]29948 examples [00:29, 1026.43 examples/s]30051 examples [00:29, 970.09 examples/s] 30155 examples [00:29, 987.94 examples/s]30260 examples [00:29, 1003.10 examples/s]30364 examples [00:29, 1011.20 examples/s]30467 examples [00:30, 1016.47 examples/s]30569 examples [00:30, 1016.45 examples/s]30674 examples [00:30, 1023.97 examples/s]30778 examples [00:30, 1028.47 examples/s]30881 examples [00:30, 1025.80 examples/s]30987 examples [00:30, 1035.59 examples/s]31091 examples [00:30, 1030.93 examples/s]31196 examples [00:30, 1034.60 examples/s]31300 examples [00:30, 1034.16 examples/s]31404 examples [00:30, 1035.77 examples/s]31509 examples [00:31, 1038.31 examples/s]31613 examples [00:31, 1034.70 examples/s]31717 examples [00:31, 1031.51 examples/s]31821 examples [00:31, 1027.68 examples/s]31925 examples [00:31, 1030.79 examples/s]32029 examples [00:31, 1021.09 examples/s]32132 examples [00:31, 1011.99 examples/s]32238 examples [00:31, 1023.21 examples/s]32341 examples [00:31, 1002.59 examples/s]32445 examples [00:31, 1011.31 examples/s]32549 examples [00:32, 1019.24 examples/s]32653 examples [00:32, 1022.45 examples/s]32757 examples [00:32, 1027.37 examples/s]32860 examples [00:32, 1025.69 examples/s]32964 examples [00:32, 1028.06 examples/s]33068 examples [00:32, 1029.03 examples/s]33171 examples [00:32, 1025.89 examples/s]33276 examples [00:32, 1032.47 examples/s]33382 examples [00:32, 1037.81 examples/s]33487 examples [00:32, 1041.42 examples/s]33593 examples [00:33, 1044.35 examples/s]33698 examples [00:33, 1037.65 examples/s]33804 examples [00:33, 1042.94 examples/s]33910 examples [00:33, 1045.20 examples/s]34015 examples [00:33, 1046.57 examples/s]34120 examples [00:33, 1047.51 examples/s]34225 examples [00:33, 1044.79 examples/s]34330 examples [00:33, 1031.23 examples/s]34434 examples [00:33, 1019.22 examples/s]34536 examples [00:33, 1012.71 examples/s]34641 examples [00:34, 1021.44 examples/s]34744 examples [00:34, 1020.38 examples/s]34847 examples [00:34, 1007.36 examples/s]34953 examples [00:34, 1020.11 examples/s]35056 examples [00:34, 1001.81 examples/s]35157 examples [00:34, 979.73 examples/s] 35256 examples [00:34, 970.24 examples/s]35356 examples [00:34, 978.34 examples/s]35460 examples [00:34, 995.27 examples/s]35566 examples [00:35, 1012.82 examples/s]35671 examples [00:35, 1022.42 examples/s]35775 examples [00:35, 1026.01 examples/s]35880 examples [00:35, 1031.76 examples/s]35984 examples [00:35, 1033.13 examples/s]36088 examples [00:35, 1026.12 examples/s]36191 examples [00:35, 1022.58 examples/s]36295 examples [00:35, 1025.11 examples/s]36400 examples [00:35, 1030.42 examples/s]36504 examples [00:35, 1031.16 examples/s]36608 examples [00:36, 1032.71 examples/s]36715 examples [00:36, 1040.86 examples/s]36820 examples [00:36, 1032.46 examples/s]36924 examples [00:36, 965.25 examples/s] 37029 examples [00:36, 987.54 examples/s]37129 examples [00:36, 976.41 examples/s]37228 examples [00:36, 949.89 examples/s]37324 examples [00:36, 952.33 examples/s]37429 examples [00:36, 977.45 examples/s]37529 examples [00:36, 981.56 examples/s]37630 examples [00:37, 988.57 examples/s]37733 examples [00:37, 998.54 examples/s]37836 examples [00:37, 1007.47 examples/s]37938 examples [00:37, 1009.62 examples/s]38043 examples [00:37, 1020.79 examples/s]38147 examples [00:37, 1026.18 examples/s]38252 examples [00:37, 1032.27 examples/s]38356 examples [00:37, 1030.09 examples/s]38460 examples [00:37, 1027.24 examples/s]38565 examples [00:37, 1031.20 examples/s]38670 examples [00:38, 1035.97 examples/s]38775 examples [00:38, 1038.47 examples/s]38879 examples [00:38, 1036.05 examples/s]38983 examples [00:38, 1035.18 examples/s]39087 examples [00:38, 1036.00 examples/s]39192 examples [00:38, 1039.15 examples/s]39296 examples [00:38, 1038.60 examples/s]39400 examples [00:38, 1030.35 examples/s]39505 examples [00:38, 1033.61 examples/s]39610 examples [00:38, 1038.29 examples/s]39714 examples [00:39, 1033.86 examples/s]39818 examples [00:39, 1022.69 examples/s]39921 examples [00:39, 1021.40 examples/s]40024 examples [00:39, 966.59 examples/s] 40128 examples [00:39, 985.80 examples/s]40231 examples [00:39, 998.06 examples/s]40332 examples [00:39, 993.00 examples/s]40435 examples [00:39, 1001.22 examples/s]40536 examples [00:39, 999.15 examples/s] 40637 examples [00:40, 999.83 examples/s]40741 examples [00:40, 1008.49 examples/s]40845 examples [00:40, 1015.81 examples/s]40947 examples [00:40, 1007.11 examples/s]41048 examples [00:40, 998.32 examples/s] 41153 examples [00:40, 1010.77 examples/s]41256 examples [00:40, 1015.60 examples/s]41360 examples [00:40, 1021.85 examples/s]41463 examples [00:40, 1019.78 examples/s]41566 examples [00:40, 1015.13 examples/s]41670 examples [00:41, 1022.35 examples/s]41774 examples [00:41, 1025.82 examples/s]41877 examples [00:41, 1020.75 examples/s]41982 examples [00:41, 1027.13 examples/s]42086 examples [00:41, 1029.23 examples/s]42191 examples [00:41, 1034.32 examples/s]42295 examples [00:41, 1030.69 examples/s]42401 examples [00:41, 1036.84 examples/s]42505 examples [00:41, 1028.41 examples/s]42608 examples [00:41, 1021.00 examples/s]42712 examples [00:42, 1024.82 examples/s]42817 examples [00:42, 1029.81 examples/s]42922 examples [00:42, 1035.16 examples/s]43026 examples [00:42, 1028.04 examples/s]43129 examples [00:42, 1017.65 examples/s]43234 examples [00:42, 1025.85 examples/s]43339 examples [00:42, 1031.45 examples/s]43443 examples [00:42, 1006.64 examples/s]43544 examples [00:42, 996.11 examples/s] 43649 examples [00:42, 1010.93 examples/s]43751 examples [00:43, 1012.08 examples/s]43855 examples [00:43, 1017.81 examples/s]43961 examples [00:43, 1027.99 examples/s]44065 examples [00:43, 1028.90 examples/s]44169 examples [00:43, 1030.76 examples/s]44275 examples [00:43, 1038.48 examples/s]44379 examples [00:43, 1037.45 examples/s]44483 examples [00:43, 1036.54 examples/s]44587 examples [00:43, 1033.86 examples/s]44691 examples [00:43, 1023.95 examples/s]44796 examples [00:44, 1029.06 examples/s]44900 examples [00:44, 1031.62 examples/s]45005 examples [00:44, 1034.55 examples/s]45109 examples [00:44, 1026.96 examples/s]45212 examples [00:44, 1015.43 examples/s]45316 examples [00:44, 1021.27 examples/s]45419 examples [00:44, 1010.95 examples/s]45522 examples [00:44, 1014.96 examples/s]45627 examples [00:44, 1023.27 examples/s]45730 examples [00:44, 1013.39 examples/s]45835 examples [00:45, 1022.77 examples/s]45938 examples [00:45, 1011.94 examples/s]46041 examples [00:45, 1016.55 examples/s]46144 examples [00:45, 1019.52 examples/s]46246 examples [00:45, 1013.37 examples/s]46353 examples [00:45, 1028.33 examples/s]46459 examples [00:45, 1036.11 examples/s]46563 examples [00:45, 1028.52 examples/s]46666 examples [00:45, 1015.76 examples/s]46768 examples [00:46, 993.08 examples/s] 46868 examples [00:46, 990.28 examples/s]46973 examples [00:46, 1005.60 examples/s]47074 examples [00:46, 1006.52 examples/s]47178 examples [00:46, 1014.14 examples/s]47282 examples [00:46, 1020.88 examples/s]47385 examples [00:46, 1023.41 examples/s]47490 examples [00:46, 1029.40 examples/s]47593 examples [00:46, 1009.93 examples/s]47698 examples [00:46, 1021.31 examples/s]47801 examples [00:47, 1019.28 examples/s]47904 examples [00:47, 1021.34 examples/s]48007 examples [00:47, 1020.40 examples/s]48110 examples [00:47, 1015.29 examples/s]48212 examples [00:47, 1014.12 examples/s]48317 examples [00:47, 1024.53 examples/s]48422 examples [00:47, 1031.96 examples/s]48526 examples [00:47, 1031.89 examples/s]48631 examples [00:47, 1034.47 examples/s]48735 examples [00:47, 1030.59 examples/s]48841 examples [00:48, 1036.96 examples/s]48947 examples [00:48, 1041.15 examples/s]49053 examples [00:48, 1044.12 examples/s]49158 examples [00:48, 1043.59 examples/s]49263 examples [00:48, 1041.86 examples/s]49368 examples [00:48, 1043.03 examples/s]49473 examples [00:48, 1043.22 examples/s]49579 examples [00:48, 1046.09 examples/s]49684 examples [00:48, 1020.22 examples/s]49790 examples [00:48, 1031.61 examples/s]49894 examples [00:49, 1032.83 examples/s]49998 examples [00:49, 1008.26 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6312/50000 [00:00<00:00, 63115.55 examples/s] 38%|███▊      | 18829/50000 [00:00<00:00, 74142.62 examples/s] 63%|██████▎   | 31572/50000 [00:00<00:00, 84777.85 examples/s] 87%|████████▋ | 43738/50000 [00:00<00:00, 93258.55 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 855.37 examples/s]190 examples [00:00, 903.28 examples/s]288 examples [00:00, 924.06 examples/s]393 examples [00:00, 956.57 examples/s]497 examples [00:00, 978.83 examples/s]602 examples [00:00, 997.86 examples/s]709 examples [00:00, 1016.33 examples/s]812 examples [00:00, 1018.18 examples/s]917 examples [00:00, 1026.35 examples/s]1024 examples [00:01, 1037.68 examples/s]1132 examples [00:01, 1048.55 examples/s]1238 examples [00:01, 1050.91 examples/s]1343 examples [00:01, 1049.77 examples/s]1448 examples [00:01, 1047.33 examples/s]1554 examples [00:01, 1049.53 examples/s]1660 examples [00:01, 1051.88 examples/s]1766 examples [00:01, 1051.61 examples/s]1872 examples [00:01, 1051.79 examples/s]1978 examples [00:01, 1048.81 examples/s]2085 examples [00:02, 1053.10 examples/s]2192 examples [00:02, 1055.02 examples/s]2299 examples [00:02, 1059.19 examples/s]2405 examples [00:02, 1057.29 examples/s]2511 examples [00:02, 1031.40 examples/s]2618 examples [00:02, 1042.32 examples/s]2725 examples [00:02, 1049.37 examples/s]2831 examples [00:02, 1044.35 examples/s]2936 examples [00:02, 1043.09 examples/s]3041 examples [00:02, 1039.08 examples/s]3145 examples [00:03, 1039.14 examples/s]3249 examples [00:03, 1037.62 examples/s]3355 examples [00:03, 1043.09 examples/s]3460 examples [00:03, 1031.37 examples/s]3566 examples [00:03, 1038.45 examples/s]3670 examples [00:03, 1031.27 examples/s]3778 examples [00:03, 1043.20 examples/s]3885 examples [00:03, 1049.21 examples/s]3992 examples [00:03, 1054.21 examples/s]4098 examples [00:03, 1028.16 examples/s]4201 examples [00:04, 1028.00 examples/s]4309 examples [00:04, 1040.52 examples/s]4415 examples [00:04, 1046.12 examples/s]4520 examples [00:04, 1025.15 examples/s]4623 examples [00:04, 1020.28 examples/s]4728 examples [00:04, 1027.65 examples/s]4834 examples [00:04, 1037.12 examples/s]4940 examples [00:04, 1042.40 examples/s]5045 examples [00:04, 1041.90 examples/s]5150 examples [00:04, 1025.63 examples/s]5257 examples [00:05, 1035.54 examples/s]5364 examples [00:05, 1042.89 examples/s]5470 examples [00:05, 1045.49 examples/s]5576 examples [00:05, 1049.22 examples/s]5681 examples [00:05, 1037.80 examples/s]5785 examples [00:05, 1031.87 examples/s]5889 examples [00:05, 1027.14 examples/s]5992 examples [00:05, 1023.31 examples/s]6095 examples [00:05, 1020.36 examples/s]6200 examples [00:05, 1028.97 examples/s]6306 examples [00:06, 1036.30 examples/s]6410 examples [00:06, 1036.08 examples/s]6514 examples [00:06, 1017.14 examples/s]6617 examples [00:06, 1020.35 examples/s]6720 examples [00:06, 1021.20 examples/s]6825 examples [00:06, 1027.01 examples/s]6932 examples [00:06, 1038.25 examples/s]7036 examples [00:06, 1012.32 examples/s]7138 examples [00:06, 1002.31 examples/s]7241 examples [00:07, 1009.16 examples/s]7347 examples [00:07, 1021.54 examples/s]7453 examples [00:07, 1031.78 examples/s]7557 examples [00:07, 1030.85 examples/s]7662 examples [00:07, 1036.51 examples/s]7767 examples [00:07, 1037.20 examples/s]7871 examples [00:07, 1037.70 examples/s]7975 examples [00:07, 1036.99 examples/s]8079 examples [00:07, 1035.54 examples/s]8183 examples [00:07, 1036.55 examples/s]8287 examples [00:08, 1035.45 examples/s]8391 examples [00:08, 1033.97 examples/s]8498 examples [00:08, 1042.58 examples/s]8605 examples [00:08, 1049.12 examples/s]8712 examples [00:08, 1054.11 examples/s]8818 examples [00:08, 1004.03 examples/s]8923 examples [00:08, 1016.01 examples/s]9027 examples [00:08, 1022.47 examples/s]9130 examples [00:08, 1011.51 examples/s]9236 examples [00:08, 1024.20 examples/s]9341 examples [00:09, 1030.44 examples/s]9445 examples [00:09, 1032.81 examples/s]9551 examples [00:09, 1040.10 examples/s]9656 examples [00:09, 1042.62 examples/s]9763 examples [00:09, 1047.97 examples/s]9868 examples [00:09, 1032.37 examples/s]9972 examples [00:09, 1028.63 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKHECRT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKHECRT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad446f1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad446f1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fad446f1620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7facc9b3b080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7facc9b72a90>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad446f1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad446f1620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fad446f1620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fad2b725550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fad2b7252e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7facc8069268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7facc8069268> 

  function with postional parmater data_info <function split_train_valid at 0x7facc8069268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7facc8069378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7facc8069378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7facc8069378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=2bb454e99dbf967daa391689f8aa76e1acd836db0b21236522bf3302a69a02e2
  Stored in directory: /tmp/pip-ephem-wheel-cache-vha4i0jr/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:05:02, 11.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:17:30, 16.8kB/s].vector_cache/glove.6B.zip:   0%|          | 205k/862M [00:00<10:03:48, 23.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 836k/862M [00:01<7:03:15, 33.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.38M/862M [00:01<4:55:37, 48.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.29M/862M [00:01<3:25:35, 69.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:23:33, 98.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:01<1:39:58, 141kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:09:49, 201kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:01<48:37, 286kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<34:03, 408kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<23:45, 580kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<16:42, 822kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:02<11:44, 1.16MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<08:16, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<05:50, 2.31MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:02<05:34, 2.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<05:48, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<07:54, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<06:20, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<04:39, 2.87MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<08:54, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<07:51, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:06<05:50, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<06:49, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<06:08, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:08<04:38, 2.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<06:19, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<07:07, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:10<05:34, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<04:03, 3.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<11:00, 1.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<09:02, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<06:39, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<07:42, 1.70MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<08:02, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<06:11, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:14<04:29, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<11:30, 1.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<09:23, 1.39MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<06:53, 1.88MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<07:51, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<08:08, 1.59MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<06:17, 2.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<04:31, 2.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<19:42, 654kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<15:06, 852kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<10:53, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<10:35, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<08:40, 1.48MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:22<06:23, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<07:28, 1.70MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<07:51, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<06:05, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<04:22, 2.90MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<1:10:44, 179kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<50:35, 250kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<35:37, 355kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<25:03, 503kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:33:08, 135kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<1:07:46, 186kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<48:02, 262kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<33:39, 373kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<2:06:04, 99.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<1:29:29, 140kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<1:02:50, 199kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<46:46, 267kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<35:16, 353kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<25:18, 492kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<19:39, 631kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<15:02, 824kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<10:49, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:25, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:51, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:07, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:50, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:01, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:30, 2.69MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:01, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:39, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:11, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:46, 3.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:23, 1.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<08:31, 1.41MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:15, 1.92MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<07:09, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:04, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:54, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:22, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:50, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:19, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:02, 2.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:33, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:01, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:39, 3.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<1:26:56, 136kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<1:02:01, 190kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<43:37, 269kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<33:11, 353kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<25:38, 457kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<18:31, 632kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<13:03, 892kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<1:32:33, 126kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<1:05:58, 176kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<46:19, 251kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<35:03, 330kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<25:42, 450kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<18:15, 633kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<15:27, 745kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<11:58, 960kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<08:36, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:44, 1.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:25, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:36, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<11:02, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:54, 1.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<06:30, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:12, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:19, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:42, 1.98MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:47, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:12, 2.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:53, 2.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<02:51, 3.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:37:00, 115kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:10:08, 159kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<49:31, 225kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<34:41, 320kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<30:41, 362kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<22:37, 490kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<16:05, 688kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<13:49, 798kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<12:00, 919kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:52, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<06:24, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:59, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:43, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:56, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:00, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:24, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:02, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<03:37, 2.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:20:54, 134kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<57:43, 187kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<40:35, 266kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<30:50, 349kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<22:39, 474kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<16:06, 666kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<13:44, 778kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<11:48, 905kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:43, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:12, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<12:23, 857kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<09:46, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:03, 1.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:24, 1.43MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:20, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:40, 1.86MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<04:04, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<1:18:28, 134kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<55:57, 187kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<39:20, 266kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<29:53, 348kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<21:58, 474kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<15:34, 666kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<13:17, 778kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<11:25, 905kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:30, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<07:35, 1.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:22, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:40, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:39, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:03, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:57, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:30, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:24, 2.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:43, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:20, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:16, 3.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:16, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:12, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:02, 3.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<9:32:14, 17.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<6:41:10, 24.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<4:40:37, 35.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<3:15:45, 50.4kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<2:41:29, 61.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<1:53:57, 86.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<1:19:48, 123kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<57:59, 169kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<42:33, 230kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<30:14, 323kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<22:38, 430kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<16:48, 578kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<11:57, 811kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<10:36, 910kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<09:23, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<06:59, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:02, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:14, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:02, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<04:28, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:22, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:43, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:32, 2.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:43, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:15, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:12, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:27, 2.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:45, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:48, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<02:46, 3.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<07:44, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:21, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:39, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:24, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:43, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:31, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:38, 1.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:05, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:01, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:16, 2.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:54, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:54, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:08<02:49, 3.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<16:15, 556kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<6:59:01, 21.6kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<4:53:21, 30.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<3:24:17, 43.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<2:43:34, 54.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<1:56:16, 77.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<1:21:43, 110kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<58:19, 153kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<42:42, 208kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<30:15, 294kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<21:11, 418kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<19:59, 442kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<14:53, 593kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<10:35, 831kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<09:27, 928kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<08:24, 1.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:18, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:48, 1.50MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:56, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:40, 2.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:57, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<03:50, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:48, 3.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<05:35, 1.53MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:47, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:33, 2.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:27, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:46, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:45, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:59, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:39, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:45, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:32, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:41, 3.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:50, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:22, 1.90MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:24, 2.42MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:30, 3.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:19, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:33, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:21, 2.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:37, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:37, 2.25MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<02:36, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<18:49, 430kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<13:59, 577kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<09:58, 807kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<08:49, 908kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<07:47, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:49, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<04:07, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<1:01:25, 129kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<43:46, 181kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<30:44, 257kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<23:16, 338kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<17:52, 441kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<12:50, 612kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<09:00, 866kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<26:21, 296kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<19:13, 406kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<13:35, 572kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<11:17, 686kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<09:27, 818kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<06:59, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<04:57, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<58:54, 130kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<41:59, 183kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<29:29, 259kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<22:19, 341kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<17:09, 443kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<12:18, 616kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<08:41, 869kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<09:50, 765kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<07:38, 984kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<05:31, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:36, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:26, 1.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:10, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<02:59, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<1:03:17, 117kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<45:00, 164kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<31:35, 233kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<23:43, 309kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<18:06, 405kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<12:57, 564kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<09:08, 796kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<07:06, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<5:37:16, 21.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<3:56:03, 30.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<2:44:08, 43.8kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<2:11:21, 54.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<1:33:21, 77.0kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<1:05:35, 109kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<46:44, 152kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<34:12, 208kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<24:13, 293kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<16:57, 417kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<15:28, 456kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<11:33, 610kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<08:12, 855kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<07:21, 949kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<06:33, 1.06MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<04:53, 1.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<03:29, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:39, 1.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<05:22, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:55, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:19, 1.58MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:24, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:25, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:28, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:07, 2.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:20, 2.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:12, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:56, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<02:11, 3.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:05, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:30, 1.89MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:00, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:40, 2.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<02:02, 3.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:56, 2.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:42, 2.40MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:02, 3.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:57, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:22, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:37, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:55, 3.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<04:20, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:41, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:42, 2.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:22, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:59, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:14, 2.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:02, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:23, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:37, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<01:54, 3.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:50, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:42, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:24, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:48, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:53, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:01, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:10, 2.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<5:50:07, 17.2kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<4:05:26, 24.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<2:51:14, 35.0kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<2:00:31, 49.4kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<1:25:31, 69.6kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<59:59, 99.0kB/s]  .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<41:48, 141kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<32:28, 181kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<23:18, 252kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<16:23, 357kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<12:46, 456kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<10:06, 575kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<07:21, 789kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<06:02, 952kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<04:43, 1.22MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<03:26, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:42, 1.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:45, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:52, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<02:03, 2.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<13:15, 423kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<09:51, 569kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<07:00, 797kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<06:10, 898kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<05:29, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:04, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<02:54, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:46, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<4:10:08, 21.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<2:54:55, 31.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<2:01:47, 44.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:26:59, 62.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:01:57, 87.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<43:31, 124kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<31:03, 172kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<22:16, 239kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<15:39, 339kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<12:05, 436kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<09:31, 553kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<06:55, 759kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:38, 922kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:29, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:14, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:27, 1.49MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:56, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:10, 2.35MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:55, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:17, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<01:38, 3.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<4:50:39, 17.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<3:23:43, 24.5kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<2:21:58, 35.0kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<1:39:48, 49.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<1:10:49, 69.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<49:39, 98.9kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<34:30, 141kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<29:14, 166kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<20:56, 232kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<14:42, 328kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<11:20, 423kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<08:54, 538kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<06:27, 740kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<04:32, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<07:03, 669kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:24, 872kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:53, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:47, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:37, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:46, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:58, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<34:10, 134kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<24:21, 188kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<17:03, 267kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<12:54, 350kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<09:28, 476kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<06:42, 669kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<05:42, 779kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:53, 908kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:36, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<02:33, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<04:09, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:20, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:26, 1.78MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:42, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:43, 1.58MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:07, 2.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:09, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:24, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:53, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:21, 3.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<27:41, 151kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<19:46, 211kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<13:52, 299kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<10:34, 388kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<08:13, 499kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<05:56, 688kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:46, 847kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:11, 962kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:07, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<02:12, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<29:55, 133kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<21:18, 186kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<14:55, 264kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<11:16, 346kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<08:16, 471kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<05:50, 661kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:57, 773kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:14, 903kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:47, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:20, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:42, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:03, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:41, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:12, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<08:29, 427kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<06:17, 575kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<04:27, 806kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:55, 905kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:27, 1.03MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:34, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:49, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:20, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:41, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:57, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:09, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:14, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:44, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:14, 2.70MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<24:49, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<17:41, 189kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<12:22, 268kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<09:20, 352kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<07:11, 456kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<05:09, 633kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<03:36, 894kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:15, 754kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:18, 970kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:22, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:23, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:58, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:27, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:50, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:00, 2.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<18:27, 163kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<13:12, 227kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<09:14, 322kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<07:04, 415kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<05:32, 530kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<04:00, 729kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<02:47, 1.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<25:41, 112kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<18:13, 157kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<12:42, 223kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<09:26, 297kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<07:08, 392kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<05:05, 547kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<03:33, 773kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:03, 673kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:07, 874kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:13, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:10, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:17, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:34, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:13, 2.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:15, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:07, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:50, 2.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:09, 2.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:17, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:00, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:43, 3.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:50, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:31, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:07, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:19, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:25, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:06, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:22<00:47, 2.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<10:27, 216kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<07:31, 299kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<05:16, 422kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<04:08, 529kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:20, 655kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<02:25, 893kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:00, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:36, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<01:10, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:18, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:00, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:42, 2.80MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:21, 842kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:50, 1.07MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:19, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:21, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:06, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:48, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<00:35, 3.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<12:39, 146kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<09:13, 200kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<06:29, 281kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<04:42, 377kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<03:27, 511kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<02:25, 716kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:04, 825kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:37, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:09, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:10, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:10, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:53, 1.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:37, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<11:38, 135kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<08:15, 189kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<05:45, 268kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<03:56, 381kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<06:19, 237kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<04:43, 317kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:20, 444kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<02:17, 629kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:43, 526kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<02:02, 698kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:26, 972kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:18, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:11, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:52, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:36, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:18, 984kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:44, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:48, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:36, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:31, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:21, 3.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<07:13, 151kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<05:08, 211kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<03:32, 298kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<02:37, 388kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:56, 524kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:20, 735kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<01:07, 840kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:58, 974kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:43, 1.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:29, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<51:25, 17.2kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<35:49, 24.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<24:21, 34.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<16:30, 49.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<11:40, 69.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<08:04, 98.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<05:24, 138kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<03:49, 193kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:07<02:36, 274kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<01:53, 359kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<01:27, 464kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<01:01, 642kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:45, 798kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:35, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:24, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:23, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:19, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:13, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:12, 2.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:08, 3.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:16, 1.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.31MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:05, 3.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.73MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:02, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.81MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.81MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<102:07:30,  1.09it/s]  0%|          | 792/400000 [00:01<71:21:01,  1.55it/s]  0%|          | 1596/400000 [00:01<49:50:55,  2.22it/s]  1%|          | 2405/400000 [00:01<34:49:38,  3.17it/s]  1%|          | 3219/400000 [00:01<24:20:00,  4.53it/s]  1%|          | 4016/400000 [00:01<17:00:11,  6.47it/s]  1%|          | 4821/400000 [00:01<11:52:55,  9.24it/s]  1%|▏         | 5635/400000 [00:01<8:18:15, 13.19it/s]   2%|▏         | 6430/400000 [00:01<5:48:19, 18.83it/s]  2%|▏         | 7222/400000 [00:01<4:03:35, 26.87it/s]  2%|▏         | 8024/400000 [00:01<2:50:24, 38.34it/s]  2%|▏         | 8824/400000 [00:02<1:59:17, 54.65it/s]  2%|▏         | 9634/400000 [00:02<1:23:34, 77.85it/s]  3%|▎         | 10426/400000 [00:02<58:38, 110.73it/s]  3%|▎         | 11239/400000 [00:02<41:11, 157.27it/s]  3%|▎         | 12053/400000 [00:02<29:00, 222.83it/s]  3%|▎         | 12852/400000 [00:02<20:30, 314.54it/s]  3%|▎         | 13667/400000 [00:02<14:34, 442.03it/s]  4%|▎         | 14486/400000 [00:02<10:24, 617.18it/s]  4%|▍         | 15304/400000 [00:02<07:30, 854.04it/s]  4%|▍         | 16115/400000 [00:02<05:28, 1167.16it/s]  4%|▍         | 16925/400000 [00:03<04:05, 1560.53it/s]  4%|▍         | 17711/400000 [00:03<03:06, 2052.88it/s]  5%|▍         | 18519/400000 [00:03<02:24, 2644.64it/s]  5%|▍         | 19333/400000 [00:03<01:54, 3316.27it/s]  5%|▌         | 20147/400000 [00:03<01:34, 4033.20it/s]  5%|▌         | 20958/400000 [00:03<01:19, 4748.67it/s]  5%|▌         | 21763/400000 [00:03<01:10, 5377.24it/s]  6%|▌         | 22580/400000 [00:03<01:03, 5990.68it/s]  6%|▌         | 23395/400000 [00:03<00:57, 6506.40it/s]  6%|▌         | 24210/400000 [00:03<00:54, 6924.79it/s]  6%|▋         | 25019/400000 [00:04<00:51, 7221.54it/s]  6%|▋         | 25826/400000 [00:04<00:50, 7376.95it/s]  7%|▋         | 26627/400000 [00:04<00:49, 7556.08it/s]  7%|▋         | 27438/400000 [00:04<00:48, 7712.09it/s]  7%|▋         | 28244/400000 [00:04<00:47, 7812.54it/s]  7%|▋         | 29053/400000 [00:04<00:47, 7891.72it/s]  7%|▋         | 29865/400000 [00:04<00:46, 7956.28it/s]  8%|▊         | 30672/400000 [00:04<00:46, 7956.92it/s]  8%|▊         | 31483/400000 [00:04<00:46, 8000.55it/s]  8%|▊         | 32293/400000 [00:04<00:45, 8027.34it/s]  8%|▊         | 33106/400000 [00:05<00:45, 8057.05it/s]  8%|▊         | 33915/400000 [00:05<00:45, 8026.79it/s]  9%|▊         | 34720/400000 [00:05<00:45, 8014.03it/s]  9%|▉         | 35532/400000 [00:05<00:45, 8043.14it/s]  9%|▉         | 36338/400000 [00:05<00:45, 8016.65it/s]  9%|▉         | 37141/400000 [00:05<00:45, 8007.11it/s]  9%|▉         | 37948/400000 [00:05<00:45, 8025.17it/s] 10%|▉         | 38751/400000 [00:05<00:45, 7946.69it/s] 10%|▉         | 39547/400000 [00:05<00:45, 7930.91it/s] 10%|█         | 40356/400000 [00:05<00:45, 7977.94it/s] 10%|█         | 41155/400000 [00:06<00:44, 7978.75it/s] 10%|█         | 41956/400000 [00:06<00:44, 7986.44it/s] 11%|█         | 42766/400000 [00:06<00:44, 8018.14it/s] 11%|█         | 43568/400000 [00:06<00:44, 8002.59it/s] 11%|█         | 44381/400000 [00:06<00:44, 8040.19it/s] 11%|█▏        | 45195/400000 [00:06<00:43, 8068.43it/s] 12%|█▏        | 46008/400000 [00:06<00:43, 8084.18it/s] 12%|█▏        | 46821/400000 [00:06<00:43, 8095.76it/s] 12%|█▏        | 47631/400000 [00:06<00:43, 8076.58it/s] 12%|█▏        | 48443/400000 [00:06<00:43, 8089.31it/s] 12%|█▏        | 49255/400000 [00:07<00:43, 8097.57it/s] 13%|█▎        | 50065/400000 [00:07<00:43, 8097.02it/s] 13%|█▎        | 50875/400000 [00:07<00:43, 8035.34it/s] 13%|█▎        | 51679/400000 [00:07<00:43, 7999.42it/s] 13%|█▎        | 52481/400000 [00:07<00:43, 8001.73it/s] 13%|█▎        | 53296/400000 [00:07<00:43, 8043.04it/s] 14%|█▎        | 54106/400000 [00:07<00:42, 8058.61it/s] 14%|█▎        | 54912/400000 [00:07<00:42, 8059.01it/s] 14%|█▍        | 55728/400000 [00:07<00:42, 8087.09it/s] 14%|█▍        | 56537/400000 [00:07<00:42, 8076.73it/s] 14%|█▍        | 57352/400000 [00:08<00:42, 8095.95it/s] 15%|█▍        | 58162/400000 [00:08<00:42, 8083.45it/s] 15%|█▍        | 58971/400000 [00:08<00:42, 8067.06it/s] 15%|█▍        | 59787/400000 [00:08<00:42, 8093.90it/s] 15%|█▌        | 60597/400000 [00:08<00:42, 8055.83it/s] 15%|█▌        | 61403/400000 [00:08<00:42, 8028.55it/s] 16%|█▌        | 62208/400000 [00:08<00:42, 8033.01it/s] 16%|█▌        | 63025/400000 [00:08<00:41, 8072.42it/s] 16%|█▌        | 63842/400000 [00:08<00:41, 8098.90it/s] 16%|█▌        | 64652/400000 [00:08<00:41, 8073.33it/s] 16%|█▋        | 65468/400000 [00:09<00:41, 8096.38it/s] 17%|█▋        | 66278/400000 [00:09<00:41, 7970.87it/s] 17%|█▋        | 67090/400000 [00:09<00:41, 8010.90it/s] 17%|█▋        | 67892/400000 [00:09<00:42, 7796.70it/s] 17%|█▋        | 68706/400000 [00:09<00:42, 7883.00it/s] 17%|█▋        | 69507/400000 [00:09<00:41, 7919.59it/s] 18%|█▊        | 70300/400000 [00:09<00:41, 7903.88it/s] 18%|█▊        | 71115/400000 [00:09<00:41, 7973.78it/s] 18%|█▊        | 71923/400000 [00:09<00:40, 8003.07it/s] 18%|█▊        | 72724/400000 [00:09<00:40, 7989.11it/s] 18%|█▊        | 73526/400000 [00:10<00:40, 7997.77it/s] 19%|█▊        | 74327/400000 [00:10<00:40, 7999.28it/s] 19%|█▉        | 75138/400000 [00:10<00:40, 8030.71it/s] 19%|█▉        | 75942/400000 [00:10<00:40, 8027.88it/s] 19%|█▉        | 76755/400000 [00:10<00:40, 8056.43it/s] 19%|█▉        | 77568/400000 [00:10<00:39, 8075.51it/s] 20%|█▉        | 78376/400000 [00:10<00:39, 8065.35it/s] 20%|█▉        | 79183/400000 [00:10<00:39, 8050.56it/s] 20%|█▉        | 79993/400000 [00:10<00:39, 8063.37it/s] 20%|██        | 80800/400000 [00:10<00:39, 8053.52it/s] 20%|██        | 81616/400000 [00:11<00:39, 8084.88it/s] 21%|██        | 82425/400000 [00:11<00:39, 7994.72it/s] 21%|██        | 83232/400000 [00:11<00:39, 8015.71it/s] 21%|██        | 84034/400000 [00:11<00:39, 7956.40it/s] 21%|██        | 84843/400000 [00:11<00:39, 7995.13it/s] 21%|██▏       | 85644/400000 [00:11<00:39, 7999.59it/s] 22%|██▏       | 86445/400000 [00:11<00:39, 7932.75it/s] 22%|██▏       | 87257/400000 [00:11<00:39, 7986.24it/s] 22%|██▏       | 88065/400000 [00:11<00:38, 8011.95it/s] 22%|██▏       | 88867/400000 [00:11<00:38, 8014.05it/s] 22%|██▏       | 89680/400000 [00:12<00:38, 8045.57it/s] 23%|██▎       | 90485/400000 [00:12<00:38, 8032.54it/s] 23%|██▎       | 91289/400000 [00:12<00:38, 8024.17it/s] 23%|██▎       | 92096/400000 [00:12<00:38, 8035.24it/s] 23%|██▎       | 92900/400000 [00:12<00:38, 8011.34it/s] 23%|██▎       | 93716/400000 [00:12<00:38, 8053.96it/s] 24%|██▎       | 94528/400000 [00:12<00:37, 8072.29it/s] 24%|██▍       | 95336/400000 [00:12<00:37, 8062.96it/s] 24%|██▍       | 96150/400000 [00:12<00:37, 8085.77it/s] 24%|██▍       | 96959/400000 [00:13<00:37, 8082.73it/s] 24%|██▍       | 97775/400000 [00:13<00:37, 8105.23it/s] 25%|██▍       | 98586/400000 [00:13<00:37, 8037.49it/s] 25%|██▍       | 99390/400000 [00:13<00:37, 8006.71it/s] 25%|██▌       | 100195/400000 [00:13<00:37, 8017.74it/s] 25%|██▌       | 101010/400000 [00:13<00:37, 8054.05it/s] 25%|██▌       | 101816/400000 [00:13<00:37, 8016.46it/s] 26%|██▌       | 102632/400000 [00:13<00:36, 8058.79it/s] 26%|██▌       | 103444/400000 [00:13<00:36, 8074.70it/s] 26%|██▌       | 104252/400000 [00:13<00:36, 8066.93it/s] 26%|██▋       | 105067/400000 [00:14<00:36, 8090.04it/s] 26%|██▋       | 105881/400000 [00:14<00:36, 8103.44it/s] 27%|██▋       | 106697/400000 [00:14<00:36, 8120.28it/s] 27%|██▋       | 107510/400000 [00:14<00:36, 8059.73it/s] 27%|██▋       | 108317/400000 [00:14<00:36, 8028.45it/s] 27%|██▋       | 109130/400000 [00:14<00:36, 8058.58it/s] 27%|██▋       | 109946/400000 [00:14<00:35, 8087.09it/s] 28%|██▊       | 110755/400000 [00:14<00:35, 8070.35it/s] 28%|██▊       | 111563/400000 [00:14<00:35, 8039.61it/s] 28%|██▊       | 112373/400000 [00:14<00:35, 8055.38it/s] 28%|██▊       | 113179/400000 [00:15<00:35, 8014.56it/s] 28%|██▊       | 113993/400000 [00:15<00:35, 8051.61it/s] 29%|██▊       | 114800/400000 [00:15<00:35, 8055.23it/s] 29%|██▉       | 115612/400000 [00:15<00:35, 8074.39it/s] 29%|██▉       | 116425/400000 [00:15<00:35, 8089.74it/s] 29%|██▉       | 117235/400000 [00:15<00:35, 8013.75it/s] 30%|██▉       | 118037/400000 [00:15<00:35, 7968.73it/s] 30%|██▉       | 118848/400000 [00:15<00:35, 8009.86it/s] 30%|██▉       | 119659/400000 [00:15<00:34, 8038.64it/s] 30%|███       | 120464/400000 [00:15<00:34, 8035.45it/s] 30%|███       | 121276/400000 [00:16<00:34, 8058.41it/s] 31%|███       | 122090/400000 [00:16<00:34, 8081.82it/s] 31%|███       | 122903/400000 [00:16<00:34, 8095.59it/s] 31%|███       | 123713/400000 [00:16<00:34, 8091.80it/s] 31%|███       | 124525/400000 [00:16<00:34, 8098.39it/s] 31%|███▏      | 125337/400000 [00:16<00:33, 8102.45it/s] 32%|███▏      | 126150/400000 [00:16<00:33, 8108.78it/s] 32%|███▏      | 126967/400000 [00:16<00:33, 8126.17it/s] 32%|███▏      | 127780/400000 [00:16<00:33, 8117.61it/s] 32%|███▏      | 128592/400000 [00:16<00:33, 8117.62it/s] 32%|███▏      | 129405/400000 [00:17<00:33, 8120.15it/s] 33%|███▎      | 130218/400000 [00:17<00:33, 8108.90it/s] 33%|███▎      | 131029/400000 [00:17<00:33, 8051.45it/s] 33%|███▎      | 131837/400000 [00:17<00:33, 8057.53it/s] 33%|███▎      | 132649/400000 [00:17<00:33, 8074.67it/s] 33%|███▎      | 133457/400000 [00:17<00:33, 8048.54it/s] 34%|███▎      | 134262/400000 [00:17<00:33, 8044.09it/s] 34%|███▍      | 135074/400000 [00:17<00:32, 8066.70it/s] 34%|███▍      | 135885/400000 [00:17<00:32, 8077.07it/s] 34%|███▍      | 136693/400000 [00:17<00:32, 8051.95it/s] 34%|███▍      | 137499/400000 [00:18<00:32, 8008.28it/s] 35%|███▍      | 138312/400000 [00:18<00:32, 8041.90it/s] 35%|███▍      | 139123/400000 [00:18<00:32, 8061.50it/s] 35%|███▍      | 139936/400000 [00:18<00:32, 8080.16it/s] 35%|███▌      | 140749/400000 [00:18<00:32, 8093.85it/s] 35%|███▌      | 141559/400000 [00:18<00:32, 8069.04it/s] 36%|███▌      | 142366/400000 [00:18<00:32, 8041.63it/s] 36%|███▌      | 143181/400000 [00:18<00:31, 8072.30it/s] 36%|███▌      | 143998/400000 [00:18<00:31, 8100.81it/s] 36%|███▌      | 144809/400000 [00:18<00:31, 8089.26it/s] 36%|███▋      | 145618/400000 [00:19<00:31, 8053.09it/s] 37%|███▋      | 146424/400000 [00:19<00:31, 7974.79it/s] 37%|███▋      | 147222/400000 [00:19<00:31, 7966.18it/s] 37%|███▋      | 148027/400000 [00:19<00:31, 7991.02it/s] 37%|███▋      | 148827/400000 [00:19<00:31, 7986.61it/s] 37%|███▋      | 149626/400000 [00:19<00:31, 7958.24it/s] 38%|███▊      | 150431/400000 [00:19<00:31, 7985.12it/s] 38%|███▊      | 151230/400000 [00:19<00:31, 7979.55it/s] 38%|███▊      | 152043/400000 [00:19<00:30, 8023.66it/s] 38%|███▊      | 152846/400000 [00:19<00:30, 7989.63it/s] 38%|███▊      | 153661/400000 [00:20<00:30, 8035.12it/s] 39%|███▊      | 154476/400000 [00:20<00:30, 8068.94it/s] 39%|███▉      | 155288/400000 [00:20<00:30, 8082.58it/s] 39%|███▉      | 156097/400000 [00:20<00:30, 8042.11it/s] 39%|███▉      | 156906/400000 [00:20<00:30, 8053.92it/s] 39%|███▉      | 157721/400000 [00:20<00:29, 8080.05it/s] 40%|███▉      | 158537/400000 [00:20<00:29, 8101.58it/s] 40%|███▉      | 159351/400000 [00:20<00:29, 8110.49it/s] 40%|████      | 160163/400000 [00:20<00:29, 8082.93it/s] 40%|████      | 160975/400000 [00:20<00:29, 8093.38it/s] 40%|████      | 161785/400000 [00:21<00:29, 8084.64it/s] 41%|████      | 162594/400000 [00:21<00:29, 8044.42it/s] 41%|████      | 163402/400000 [00:21<00:29, 8054.18it/s] 41%|████      | 164216/400000 [00:21<00:29, 8077.43it/s] 41%|████▏     | 165024/400000 [00:21<00:29, 8063.96it/s] 41%|████▏     | 165837/400000 [00:21<00:28, 8081.47it/s] 42%|████▏     | 166649/400000 [00:21<00:28, 8091.57it/s] 42%|████▏     | 167459/400000 [00:21<00:29, 8007.82it/s] 42%|████▏     | 168269/400000 [00:21<00:28, 8034.54it/s] 42%|████▏     | 169073/400000 [00:21<00:28, 8027.47it/s] 42%|████▏     | 169885/400000 [00:22<00:28, 8052.82it/s] 43%|████▎     | 170691/400000 [00:22<00:28, 8001.65it/s] 43%|████▎     | 171505/400000 [00:22<00:28, 8040.53it/s] 43%|████▎     | 172320/400000 [00:22<00:28, 8072.50it/s] 43%|████▎     | 173128/400000 [00:22<00:28, 7986.55it/s] 43%|████▎     | 173942/400000 [00:22<00:28, 8030.78it/s] 44%|████▎     | 174746/400000 [00:22<00:28, 7835.81it/s] 44%|████▍     | 175554/400000 [00:22<00:28, 7906.59it/s] 44%|████▍     | 176362/400000 [00:22<00:28, 7957.58it/s] 44%|████▍     | 177171/400000 [00:22<00:27, 7994.12it/s] 44%|████▍     | 177983/400000 [00:23<00:27, 8030.13it/s] 45%|████▍     | 178799/400000 [00:23<00:27, 8067.31it/s] 45%|████▍     | 179614/400000 [00:23<00:27, 8090.81it/s] 45%|████▌     | 180424/400000 [00:23<00:27, 8093.38it/s] 45%|████▌     | 181234/400000 [00:23<00:27, 7967.15it/s] 46%|████▌     | 182032/400000 [00:23<00:27, 7874.88it/s] 46%|████▌     | 182846/400000 [00:23<00:27, 7951.89it/s] 46%|████▌     | 183661/400000 [00:23<00:27, 8008.80it/s] 46%|████▌     | 184473/400000 [00:23<00:26, 8040.16it/s] 46%|████▋     | 185278/400000 [00:23<00:26, 8013.67it/s] 47%|████▋     | 186086/400000 [00:24<00:26, 8031.58it/s] 47%|████▋     | 186890/400000 [00:24<00:26, 7912.64it/s] 47%|████▋     | 187706/400000 [00:24<00:26, 7985.25it/s] 47%|████▋     | 188520/400000 [00:24<00:26, 8030.75it/s] 47%|████▋     | 189324/400000 [00:24<00:26, 8007.46it/s] 48%|████▊     | 190130/400000 [00:24<00:26, 8020.63it/s] 48%|████▊     | 190933/400000 [00:24<00:26, 8006.19it/s] 48%|████▊     | 191748/400000 [00:24<00:25, 8047.88it/s] 48%|████▊     | 192553/400000 [00:24<00:25, 8038.08it/s] 48%|████▊     | 193368/400000 [00:24<00:25, 8070.65it/s] 49%|████▊     | 194176/400000 [00:25<00:25, 8027.25it/s] 49%|████▊     | 194987/400000 [00:25<00:25, 8051.35it/s] 49%|████▉     | 195801/400000 [00:25<00:25, 8074.91it/s] 49%|████▉     | 196616/400000 [00:25<00:25, 8094.52it/s] 49%|████▉     | 197426/400000 [00:25<00:25, 8052.44it/s] 50%|████▉     | 198232/400000 [00:25<00:25, 8031.32it/s] 50%|████▉     | 199042/400000 [00:25<00:24, 8050.94it/s] 50%|████▉     | 199856/400000 [00:25<00:24, 8076.93it/s] 50%|█████     | 200665/400000 [00:25<00:24, 8079.24it/s] 50%|█████     | 201473/400000 [00:25<00:24, 8039.50it/s] 51%|█████     | 202287/400000 [00:26<00:24, 8067.30it/s] 51%|█████     | 203094/400000 [00:26<00:24, 8017.82it/s] 51%|█████     | 203896/400000 [00:26<00:24, 8011.02it/s] 51%|█████     | 204713/400000 [00:26<00:24, 8056.77it/s] 51%|█████▏    | 205520/400000 [00:26<00:24, 8060.22it/s] 52%|█████▏    | 206336/400000 [00:26<00:23, 8088.11it/s] 52%|█████▏    | 207145/400000 [00:26<00:23, 8076.60it/s] 52%|█████▏    | 207957/400000 [00:26<00:23, 8088.03it/s] 52%|█████▏    | 208772/400000 [00:26<00:23, 8105.12it/s] 52%|█████▏    | 209588/400000 [00:26<00:23, 8118.83it/s] 53%|█████▎    | 210400/400000 [00:27<00:23, 8099.92it/s] 53%|█████▎    | 211211/400000 [00:27<00:23, 8090.09it/s] 53%|█████▎    | 212021/400000 [00:27<00:23, 8078.04it/s] 53%|█████▎    | 212837/400000 [00:27<00:23, 8100.98it/s] 53%|█████▎    | 213648/400000 [00:27<00:23, 8092.49it/s] 54%|█████▎    | 214458/400000 [00:27<00:22, 8076.83it/s] 54%|█████▍    | 215266/400000 [00:27<00:22, 8053.75it/s] 54%|█████▍    | 216078/400000 [00:27<00:22, 8071.93it/s] 54%|█████▍    | 216893/400000 [00:27<00:22, 8093.60it/s] 54%|█████▍    | 217703/400000 [00:28<00:22, 8079.55it/s] 55%|█████▍    | 218519/400000 [00:28<00:22, 8100.74it/s] 55%|█████▍    | 219331/400000 [00:28<00:22, 8105.65it/s] 55%|█████▌    | 220142/400000 [00:28<00:22, 8095.44it/s] 55%|█████▌    | 220952/400000 [00:28<00:22, 8085.97it/s] 55%|█████▌    | 221768/400000 [00:28<00:21, 8107.60it/s] 56%|█████▌    | 222585/400000 [00:28<00:21, 8123.61it/s] 56%|█████▌    | 223398/400000 [00:28<00:21, 8122.90it/s] 56%|█████▌    | 224211/400000 [00:28<00:21, 8049.07it/s] 56%|█████▋    | 225017/400000 [00:28<00:21, 8034.51it/s] 56%|█████▋    | 225821/400000 [00:29<00:21, 7947.53it/s] 57%|█████▋    | 226627/400000 [00:29<00:21, 7979.20it/s] 57%|█████▋    | 227426/400000 [00:29<00:21, 7902.92it/s] 57%|█████▋    | 228225/400000 [00:29<00:21, 7926.86it/s] 57%|█████▋    | 229040/400000 [00:29<00:21, 7991.19it/s] 57%|█████▋    | 229847/400000 [00:29<00:21, 8013.74it/s] 58%|█████▊    | 230663/400000 [00:29<00:21, 8056.65it/s] 58%|█████▊    | 231477/400000 [00:29<00:20, 8078.58it/s] 58%|█████▊    | 232287/400000 [00:29<00:20, 8083.81it/s] 58%|█████▊    | 233096/400000 [00:29<00:20, 8074.12it/s] 58%|█████▊    | 233904/400000 [00:30<00:20, 8056.50it/s] 59%|█████▊    | 234720/400000 [00:30<00:20, 8085.81it/s] 59%|█████▉    | 235534/400000 [00:30<00:20, 8101.31it/s] 59%|█████▉    | 236350/400000 [00:30<00:20, 8117.01it/s] 59%|█████▉    | 237162/400000 [00:30<00:20, 8116.30it/s] 59%|█████▉    | 237974/400000 [00:30<00:20, 8069.44it/s] 60%|█████▉    | 238792/400000 [00:30<00:19, 8101.59it/s] 60%|█████▉    | 239604/400000 [00:30<00:19, 8104.74it/s] 60%|██████    | 240423/400000 [00:30<00:19, 8129.25it/s] 60%|██████    | 241236/400000 [00:30<00:19, 8058.23it/s] 61%|██████    | 242043/400000 [00:31<00:19, 8057.77it/s] 61%|██████    | 242853/400000 [00:31<00:19, 8067.65it/s] 61%|██████    | 243668/400000 [00:31<00:19, 8091.62it/s] 61%|██████    | 244483/400000 [00:31<00:19, 8107.39it/s] 61%|██████▏   | 245294/400000 [00:31<00:19, 8079.83it/s] 62%|██████▏   | 246103/400000 [00:31<00:19, 7970.00it/s] 62%|██████▏   | 246913/400000 [00:31<00:19, 8005.80it/s] 62%|██████▏   | 247728/400000 [00:31<00:18, 8046.91it/s] 62%|██████▏   | 248543/400000 [00:31<00:18, 8075.83it/s] 62%|██████▏   | 249351/400000 [00:31<00:18, 8057.26it/s] 63%|██████▎   | 250157/400000 [00:32<00:18, 7986.60it/s] 63%|██████▎   | 250962/400000 [00:32<00:18, 8003.54it/s] 63%|██████▎   | 251770/400000 [00:32<00:18, 8024.68it/s] 63%|██████▎   | 252575/400000 [00:32<00:18, 8030.19it/s] 63%|██████▎   | 253388/400000 [00:32<00:18, 8058.62it/s] 64%|██████▎   | 254194/400000 [00:32<00:18, 8022.45it/s] 64%|██████▍   | 255009/400000 [00:32<00:17, 8058.09it/s] 64%|██████▍   | 255823/400000 [00:32<00:17, 8081.56it/s] 64%|██████▍   | 256635/400000 [00:32<00:17, 8092.32it/s] 64%|██████▍   | 257447/400000 [00:32<00:17, 8098.74it/s] 65%|██████▍   | 258257/400000 [00:33<00:17, 7905.29it/s] 65%|██████▍   | 259068/400000 [00:33<00:17, 7964.25it/s] 65%|██████▍   | 259879/400000 [00:33<00:17, 8005.40it/s] 65%|██████▌   | 260689/400000 [00:33<00:17, 8033.20it/s] 65%|██████▌   | 261499/400000 [00:33<00:17, 8051.24it/s] 66%|██████▌   | 262305/400000 [00:33<00:17, 8045.21it/s] 66%|██████▌   | 263117/400000 [00:33<00:16, 8065.12it/s] 66%|██████▌   | 263930/400000 [00:33<00:16, 8083.82it/s] 66%|██████▌   | 264746/400000 [00:33<00:16, 8105.60it/s] 66%|██████▋   | 265560/400000 [00:33<00:16, 8114.30it/s] 67%|██████▋   | 266372/400000 [00:34<00:16, 8052.01it/s] 67%|██████▋   | 267178/400000 [00:34<00:16, 8010.43it/s] 67%|██████▋   | 267980/400000 [00:34<00:16, 7912.92it/s] 67%|██████▋   | 268772/400000 [00:34<00:16, 7823.00it/s] 67%|██████▋   | 269564/400000 [00:34<00:16, 7850.20it/s] 68%|██████▊   | 270367/400000 [00:34<00:16, 7900.48it/s] 68%|██████▊   | 271171/400000 [00:34<00:16, 7940.32it/s] 68%|██████▊   | 271968/400000 [00:34<00:16, 7948.73it/s] 68%|██████▊   | 272764/400000 [00:34<00:16, 7948.89it/s] 68%|██████▊   | 273574/400000 [00:34<00:15, 7992.85it/s] 69%|██████▊   | 274384/400000 [00:35<00:15, 8022.75it/s] 69%|██████▉   | 275187/400000 [00:35<00:15, 7980.14it/s] 69%|██████▉   | 276001/400000 [00:35<00:15, 8027.02it/s] 69%|██████▉   | 276817/400000 [00:35<00:15, 8065.11it/s] 69%|██████▉   | 277633/400000 [00:35<00:15, 8090.71it/s] 70%|██████▉   | 278448/400000 [00:35<00:14, 8106.48it/s] 70%|██████▉   | 279261/400000 [00:35<00:14, 8110.71it/s] 70%|███████   | 280074/400000 [00:35<00:14, 8114.84it/s] 70%|███████   | 280888/400000 [00:35<00:14, 8121.41it/s] 70%|███████   | 281702/400000 [00:35<00:14, 8125.78it/s] 71%|███████   | 282515/400000 [00:36<00:14, 8093.74it/s] 71%|███████   | 283325/400000 [00:36<00:14, 8045.04it/s] 71%|███████   | 284130/400000 [00:36<00:14, 8034.14it/s] 71%|███████   | 284935/400000 [00:36<00:14, 8038.06it/s] 71%|███████▏  | 285747/400000 [00:36<00:14, 8060.82it/s] 72%|███████▏  | 286558/400000 [00:36<00:14, 8073.48it/s] 72%|███████▏  | 287368/400000 [00:36<00:13, 8079.36it/s] 72%|███████▏  | 288176/400000 [00:36<00:13, 8050.65it/s] 72%|███████▏  | 288989/400000 [00:36<00:13, 8073.20it/s] 72%|███████▏  | 289803/400000 [00:36<00:13, 8092.78it/s] 73%|███████▎  | 290613/400000 [00:37<00:13, 8094.65it/s] 73%|███████▎  | 291423/400000 [00:37<00:13, 8088.84it/s] 73%|███████▎  | 292232/400000 [00:37<00:13, 8063.49it/s] 73%|███████▎  | 293044/400000 [00:37<00:13, 8077.55it/s] 73%|███████▎  | 293852/400000 [00:37<00:13, 8014.08it/s] 74%|███████▎  | 294666/400000 [00:37<00:13, 8048.92it/s] 74%|███████▍  | 295472/400000 [00:37<00:13, 8033.23it/s] 74%|███████▍  | 296284/400000 [00:37<00:12, 8057.51it/s] 74%|███████▍  | 297090/400000 [00:37<00:12, 8017.31it/s] 74%|███████▍  | 297898/400000 [00:37<00:12, 8033.80it/s] 75%|███████▍  | 298707/400000 [00:38<00:12, 8048.82it/s] 75%|███████▍  | 299519/400000 [00:38<00:12, 8068.94it/s] 75%|███████▌  | 300336/400000 [00:38<00:12, 8097.61it/s] 75%|███████▌  | 301146/400000 [00:38<00:12, 8023.92it/s] 75%|███████▌  | 301960/400000 [00:38<00:12, 8056.39it/s] 76%|███████▌  | 302766/400000 [00:38<00:12, 8046.49it/s] 76%|███████▌  | 303571/400000 [00:38<00:12, 8032.37it/s] 76%|███████▌  | 304388/400000 [00:38<00:11, 8072.53it/s] 76%|███████▋  | 305196/400000 [00:38<00:11, 8069.52it/s] 77%|███████▋  | 306006/400000 [00:38<00:11, 8076.14it/s] 77%|███████▋  | 306814/400000 [00:39<00:11, 8051.71it/s] 77%|███████▋  | 307620/400000 [00:39<00:11, 8022.58it/s] 77%|███████▋  | 308435/400000 [00:39<00:11, 8058.97it/s] 77%|███████▋  | 309241/400000 [00:39<00:11, 8032.20it/s] 78%|███████▊  | 310053/400000 [00:39<00:11, 8057.73it/s] 78%|███████▊  | 310859/400000 [00:39<00:11, 7912.92it/s] 78%|███████▊  | 311663/400000 [00:39<00:11, 7949.73it/s] 78%|███████▊  | 312459/400000 [00:39<00:11, 7941.25it/s] 78%|███████▊  | 313274/400000 [00:39<00:10, 8000.64it/s] 79%|███████▊  | 314079/400000 [00:39<00:10, 8012.94it/s] 79%|███████▊  | 314892/400000 [00:40<00:10, 8046.65it/s] 79%|███████▉  | 315697/400000 [00:40<00:10, 8041.26it/s] 79%|███████▉  | 316508/400000 [00:40<00:10, 8059.71it/s] 79%|███████▉  | 317319/400000 [00:40<00:10, 8073.74it/s] 80%|███████▉  | 318127/400000 [00:40<00:10, 8070.49it/s] 80%|███████▉  | 318935/400000 [00:40<00:10, 8055.87it/s] 80%|███████▉  | 319741/400000 [00:40<00:09, 8042.42it/s] 80%|████████  | 320550/400000 [00:40<00:09, 8054.17it/s] 80%|████████  | 321366/400000 [00:40<00:09, 8085.16it/s] 81%|████████  | 322175/400000 [00:40<00:09, 8053.24it/s] 81%|████████  | 322988/400000 [00:41<00:09, 8075.10it/s] 81%|████████  | 323798/400000 [00:41<00:09, 8081.97it/s] 81%|████████  | 324610/400000 [00:41<00:09, 8091.96it/s] 81%|████████▏ | 325423/400000 [00:41<00:09, 8102.15it/s] 82%|████████▏ | 326234/400000 [00:41<00:09, 8060.41it/s] 82%|████████▏ | 327041/400000 [00:41<00:09, 8027.49it/s] 82%|████████▏ | 327844/400000 [00:41<00:09, 8015.37it/s] 82%|████████▏ | 328659/400000 [00:41<00:08, 8055.13it/s] 82%|████████▏ | 329475/400000 [00:41<00:08, 8083.88it/s] 83%|████████▎ | 330291/400000 [00:41<00:08, 8104.25it/s] 83%|████████▎ | 331102/400000 [00:42<00:08, 8071.76it/s] 83%|████████▎ | 331910/400000 [00:42<00:08, 8062.72it/s] 83%|████████▎ | 332717/400000 [00:42<00:08, 7880.70it/s] 83%|████████▎ | 333530/400000 [00:42<00:08, 7952.66it/s] 84%|████████▎ | 334346/400000 [00:42<00:08, 8013.56it/s] 84%|████████▍ | 335157/400000 [00:42<00:08, 8041.73it/s] 84%|████████▍ | 335962/400000 [00:42<00:07, 8044.04it/s] 84%|████████▍ | 336767/400000 [00:42<00:07, 7946.37it/s] 84%|████████▍ | 337572/400000 [00:42<00:07, 7975.28it/s] 85%|████████▍ | 338370/400000 [00:43<00:07, 7971.75it/s] 85%|████████▍ | 339169/400000 [00:43<00:07, 7975.69it/s] 85%|████████▍ | 339981/400000 [00:43<00:07, 8015.60it/s] 85%|████████▌ | 340794/400000 [00:43<00:07, 8047.51it/s] 85%|████████▌ | 341600/400000 [00:43<00:07, 8048.71it/s] 86%|████████▌ | 342416/400000 [00:43<00:07, 8079.02it/s] 86%|████████▌ | 343225/400000 [00:43<00:07, 8056.60it/s] 86%|████████▌ | 344031/400000 [00:43<00:06, 8048.51it/s] 86%|████████▌ | 344848/400000 [00:43<00:06, 8081.96it/s] 86%|████████▋ | 345657/400000 [00:43<00:06, 8055.36it/s] 87%|████████▋ | 346468/400000 [00:44<00:06, 8071.22it/s] 87%|████████▋ | 347279/400000 [00:44<00:06, 8081.86it/s] 87%|████████▋ | 348088/400000 [00:44<00:06, 7941.67it/s] 87%|████████▋ | 348905/400000 [00:44<00:06, 8006.59it/s] 87%|████████▋ | 349722/400000 [00:44<00:06, 8054.03it/s] 88%|████████▊ | 350537/400000 [00:44<00:06, 8081.58it/s] 88%|████████▊ | 351356/400000 [00:44<00:05, 8110.98it/s] 88%|████████▊ | 352168/400000 [00:44<00:05, 8069.24it/s] 88%|████████▊ | 352976/400000 [00:44<00:05, 8063.54it/s] 88%|████████▊ | 353783/400000 [00:44<00:05, 8055.37it/s] 89%|████████▊ | 354589/400000 [00:45<00:05, 8046.99it/s] 89%|████████▉ | 355394/400000 [00:45<00:05, 8030.45it/s] 89%|████████▉ | 356202/400000 [00:45<00:05, 8045.14it/s] 89%|████████▉ | 357018/400000 [00:45<00:05, 8078.11it/s] 89%|████████▉ | 357835/400000 [00:45<00:05, 8102.75it/s] 90%|████████▉ | 358648/400000 [00:45<00:05, 8110.09it/s] 90%|████████▉ | 359460/400000 [00:45<00:05, 7960.19it/s] 90%|█████████ | 360257/400000 [00:45<00:04, 7953.14it/s] 90%|█████████ | 361067/400000 [00:45<00:04, 7994.21it/s] 90%|█████████ | 361880/400000 [00:45<00:04, 8032.45it/s] 91%|█████████ | 362691/400000 [00:46<00:04, 8055.25it/s] 91%|█████████ | 363497/400000 [00:46<00:04, 8048.37it/s] 91%|█████████ | 364309/400000 [00:46<00:04, 8067.75it/s] 91%|█████████▏| 365116/400000 [00:46<00:04, 8039.78it/s] 91%|█████████▏| 365930/400000 [00:46<00:04, 8069.27it/s] 92%|█████████▏| 366738/400000 [00:46<00:04, 8002.42it/s] 92%|█████████▏| 367553/400000 [00:46<00:04, 8045.05it/s] 92%|█████████▏| 368358/400000 [00:46<00:03, 8034.29it/s] 92%|█████████▏| 369162/400000 [00:46<00:03, 8028.59it/s] 92%|█████████▏| 369975/400000 [00:46<00:03, 8058.55it/s] 93%|█████████▎| 370787/400000 [00:47<00:03, 8074.18it/s] 93%|█████████▎| 371597/400000 [00:47<00:03, 8081.88it/s] 93%|█████████▎| 372406/400000 [00:47<00:03, 8080.77it/s] 93%|█████████▎| 373215/400000 [00:47<00:03, 8002.60it/s] 94%|█████████▎| 374026/400000 [00:47<00:03, 8033.53it/s] 94%|█████████▎| 374830/400000 [00:47<00:03, 8023.81it/s] 94%|█████████▍| 375643/400000 [00:47<00:03, 8054.89it/s] 94%|█████████▍| 376449/400000 [00:47<00:02, 8053.68it/s] 94%|█████████▍| 377255/400000 [00:47<00:02, 8026.07it/s] 95%|█████████▍| 378061/400000 [00:47<00:02, 8035.23it/s] 95%|█████████▍| 378874/400000 [00:48<00:02, 8060.88it/s] 95%|█████████▍| 379683/400000 [00:48<00:02, 8068.94it/s] 95%|█████████▌| 380490/400000 [00:48<00:02, 8061.78it/s] 95%|█████████▌| 381301/400000 [00:48<00:02, 8074.21it/s] 96%|█████████▌| 382109/400000 [00:48<00:02, 8064.13it/s] 96%|█████████▌| 382924/400000 [00:48<00:02, 8089.10it/s] 96%|█████████▌| 383733/400000 [00:48<00:02, 8076.04it/s] 96%|█████████▌| 384541/400000 [00:48<00:01, 8071.63it/s] 96%|█████████▋| 385349/400000 [00:48<00:01, 8053.78it/s] 97%|█████████▋| 386155/400000 [00:48<00:01, 8036.39it/s] 97%|█████████▋| 386970/400000 [00:49<00:01, 8069.36it/s] 97%|█████████▋| 387777/400000 [00:49<00:01, 8061.99it/s] 97%|█████████▋| 388592/400000 [00:49<00:01, 8086.58it/s] 97%|█████████▋| 389401/400000 [00:49<00:01, 8079.51it/s] 98%|█████████▊| 390209/400000 [00:49<00:01, 8057.41it/s] 98%|█████████▊| 391022/400000 [00:49<00:01, 8077.57it/s] 98%|█████████▊| 391830/400000 [00:49<00:01, 7990.72it/s] 98%|█████████▊| 392630/400000 [00:49<00:00, 7982.52it/s] 98%|█████████▊| 393439/400000 [00:49<00:00, 8013.59it/s] 99%|█████████▊| 394241/400000 [00:49<00:00, 7996.03it/s] 99%|█████████▉| 395052/400000 [00:50<00:00, 8029.56it/s] 99%|█████████▉| 395856/400000 [00:50<00:00, 7998.57it/s] 99%|█████████▉| 396663/400000 [00:50<00:00, 8018.52it/s] 99%|█████████▉| 397466/400000 [00:50<00:00, 8019.72it/s]100%|█████████▉| 398274/400000 [00:50<00:00, 8035.61it/s]100%|█████████▉| 399078/400000 [00:50<00:00, 8026.64it/s]100%|█████████▉| 399881/400000 [00:50<00:00, 8027.16it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7894.73it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7facc8081ef0>, <torchtext.data.dataset.TabularDataset object at 0x7facc80810b8>, <torchtext.vocab.Vocab object at 0x7facc8081f60>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.43 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.43 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.43 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.92 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.92 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.64 file/s]2020-07-15 00:18:42.231208: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-15 00:18:42.236264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-15 00:18:42.236464: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562637e602a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-15 00:18:42.236479: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134320.04it/s] 58%|█████▊    | 5791744/9912422 [00:00<00:21, 191684.45it/s]9920512it [00:00, 37774956.25it/s]                           
0it [00:00, ?it/s]32768it [00:00, 558338.59it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161250.49it/s]1654784it [00:00, 11405797.25it/s]                         
0it [00:00, ?it/s]8192it [00:00, 182559.67it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
