
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f42ffa8b6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f42ffa8b6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f436b0530d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f436b0530d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4384d7fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4384d7fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f43188ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f43188ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f43188ce950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 28%|██▊       | 2744320/9912422 [00:00<00:00, 27432826.77it/s]9920512it [00:00, 35878918.75it/s]                             
0it [00:00, ?it/s]32768it [00:00, 606901.68it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 454061.45it/s]1654784it [00:00, 11890690.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 195045.15it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4315ee6a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4315edad30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f43188ce598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f43188ce598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f43188ce598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:57,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:40,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:27,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:27,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:09, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 18.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 29.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 42.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.44 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.63s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 70.44 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.51 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.98s/ url]
0 examples [00:00, ? examples/s]2020-06-13 00:09:43.666488: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 00:09:43.679837: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-13 00:09:43.680039: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559c8dd6a3d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 00:09:43.680061: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
49 examples [00:00, 488.85 examples/s]151 examples [00:00, 578.58 examples/s]251 examples [00:00, 661.19 examples/s]351 examples [00:00, 735.17 examples/s]448 examples [00:00, 792.12 examples/s]550 examples [00:00, 848.58 examples/s]647 examples [00:00, 879.70 examples/s]739 examples [00:00, 889.33 examples/s]832 examples [00:00, 900.79 examples/s]928 examples [00:01, 916.09 examples/s]1023 examples [00:01, 925.88 examples/s]1116 examples [00:01, 911.55 examples/s]1211 examples [00:01, 920.93 examples/s]1308 examples [00:01, 935.09 examples/s]1408 examples [00:01, 952.91 examples/s]1505 examples [00:01, 955.43 examples/s]1604 examples [00:01, 963.01 examples/s]1704 examples [00:01, 971.77 examples/s]1802 examples [00:01, 973.29 examples/s]1903 examples [00:02, 983.82 examples/s]2003 examples [00:02, 985.91 examples/s]2104 examples [00:02, 990.60 examples/s]2204 examples [00:02, 989.47 examples/s]2303 examples [00:02, 988.81 examples/s]2405 examples [00:02, 995.31 examples/s]2505 examples [00:02, 994.48 examples/s]2605 examples [00:02, 991.77 examples/s]2705 examples [00:02, 991.45 examples/s]2805 examples [00:02, 991.48 examples/s]2905 examples [00:03, 985.68 examples/s]3005 examples [00:03, 984.50 examples/s]3104 examples [00:03, 953.28 examples/s]3201 examples [00:03, 955.60 examples/s]3297 examples [00:03, 929.65 examples/s]3391 examples [00:03, 928.69 examples/s]3485 examples [00:03, 925.54 examples/s]3586 examples [00:03, 946.91 examples/s]3681 examples [00:03, 932.62 examples/s]3778 examples [00:03, 942.27 examples/s]3880 examples [00:04, 962.13 examples/s]3982 examples [00:04, 976.31 examples/s]4080 examples [00:04, 954.04 examples/s]4177 examples [00:04, 957.97 examples/s]4273 examples [00:04, 952.86 examples/s]4370 examples [00:04, 956.01 examples/s]4472 examples [00:04, 971.76 examples/s]4571 examples [00:04, 975.51 examples/s]4672 examples [00:04, 983.09 examples/s]4771 examples [00:04, 981.97 examples/s]4872 examples [00:05, 988.07 examples/s]4973 examples [00:05, 994.06 examples/s]5074 examples [00:05, 997.59 examples/s]5174 examples [00:05, 996.66 examples/s]5274 examples [00:05, 992.37 examples/s]5374 examples [00:05, 993.95 examples/s]5474 examples [00:05, 990.16 examples/s]5574 examples [00:05, 992.09 examples/s]5674 examples [00:05, 992.37 examples/s]5774 examples [00:05, 974.24 examples/s]5872 examples [00:06, 945.47 examples/s]5973 examples [00:06, 963.33 examples/s]6073 examples [00:06, 973.86 examples/s]6171 examples [00:06, 971.47 examples/s]6272 examples [00:06, 980.06 examples/s]6373 examples [00:06, 988.65 examples/s]6474 examples [00:06, 994.57 examples/s]6575 examples [00:06, 997.57 examples/s]6675 examples [00:06, 991.12 examples/s]6775 examples [00:07, 976.23 examples/s]6876 examples [00:07, 984.43 examples/s]6975 examples [00:07, 974.98 examples/s]7073 examples [00:07, 954.94 examples/s]7169 examples [00:07, 955.48 examples/s]7270 examples [00:07, 969.72 examples/s]7368 examples [00:07, 947.96 examples/s]7467 examples [00:07, 959.74 examples/s]7567 examples [00:07, 970.14 examples/s]7665 examples [00:07, 960.26 examples/s]7762 examples [00:08, 947.15 examples/s]7857 examples [00:08, 947.76 examples/s]7954 examples [00:08, 954.11 examples/s]8053 examples [00:08, 964.24 examples/s]8150 examples [00:08, 965.05 examples/s]8251 examples [00:08, 975.66 examples/s]8351 examples [00:08, 980.61 examples/s]8450 examples [00:08, 982.90 examples/s]8550 examples [00:08, 986.36 examples/s]8649 examples [00:08, 965.12 examples/s]8748 examples [00:09, 969.86 examples/s]8846 examples [00:09, 969.22 examples/s]8946 examples [00:09, 976.91 examples/s]9046 examples [00:09, 981.28 examples/s]9145 examples [00:09, 975.95 examples/s]9245 examples [00:09, 981.62 examples/s]9345 examples [00:09, 986.36 examples/s]9445 examples [00:09, 989.65 examples/s]9545 examples [00:09, 989.72 examples/s]9644 examples [00:09, 980.71 examples/s]9744 examples [00:10, 985.37 examples/s]9844 examples [00:10, 986.84 examples/s]9944 examples [00:10, 988.99 examples/s]10043 examples [00:10, 921.93 examples/s]10139 examples [00:10, 931.84 examples/s]10240 examples [00:10, 952.31 examples/s]10340 examples [00:10, 965.55 examples/s]10439 examples [00:10, 970.54 examples/s]10537 examples [00:10, 958.86 examples/s]10634 examples [00:11, 949.84 examples/s]10730 examples [00:11, 936.48 examples/s]10827 examples [00:11, 944.08 examples/s]10927 examples [00:11, 959.41 examples/s]11024 examples [00:11, 958.32 examples/s]11121 examples [00:11, 961.27 examples/s]11223 examples [00:11, 975.45 examples/s]11324 examples [00:11, 983.61 examples/s]11425 examples [00:11, 990.81 examples/s]11525 examples [00:11, 976.75 examples/s]11626 examples [00:12, 984.69 examples/s]11727 examples [00:12, 990.72 examples/s]11829 examples [00:12, 997.19 examples/s]11930 examples [00:12, 998.96 examples/s]12030 examples [00:12, 983.67 examples/s]12129 examples [00:12, 982.76 examples/s]12228 examples [00:12, 963.20 examples/s]12325 examples [00:12, 945.45 examples/s]12423 examples [00:12, 954.07 examples/s]12521 examples [00:12, 961.35 examples/s]12621 examples [00:13, 971.48 examples/s]12722 examples [00:13, 980.86 examples/s]12821 examples [00:13, 968.59 examples/s]12922 examples [00:13, 978.83 examples/s]13020 examples [00:13, 939.28 examples/s]13121 examples [00:13, 956.51 examples/s]13221 examples [00:13, 966.92 examples/s]13322 examples [00:13, 977.92 examples/s]13422 examples [00:13, 981.94 examples/s]13521 examples [00:13, 983.32 examples/s]13622 examples [00:14, 990.79 examples/s]13722 examples [00:14, 980.34 examples/s]13821 examples [00:14, 946.18 examples/s]13918 examples [00:14, 952.22 examples/s]14016 examples [00:14, 960.29 examples/s]14113 examples [00:14, 957.20 examples/s]14214 examples [00:14, 969.63 examples/s]14315 examples [00:14, 980.93 examples/s]14415 examples [00:14, 985.60 examples/s]14514 examples [00:15, 963.01 examples/s]14615 examples [00:15, 974.37 examples/s]14714 examples [00:15, 978.02 examples/s]14815 examples [00:15, 984.60 examples/s]14916 examples [00:15, 990.36 examples/s]15016 examples [00:15, 978.24 examples/s]15117 examples [00:15, 985.20 examples/s]15218 examples [00:15, 991.99 examples/s]15319 examples [00:15, 996.85 examples/s]15419 examples [00:15, 997.32 examples/s]15519 examples [00:16, 978.32 examples/s]15617 examples [00:16, 964.09 examples/s]15716 examples [00:16, 971.22 examples/s]15815 examples [00:16, 976.36 examples/s]15913 examples [00:16, 972.09 examples/s]16011 examples [00:16, 928.23 examples/s]16111 examples [00:16, 946.04 examples/s]16212 examples [00:16, 962.40 examples/s]16313 examples [00:16, 974.00 examples/s]16414 examples [00:16, 982.52 examples/s]16513 examples [00:17, 962.59 examples/s]16613 examples [00:17, 973.50 examples/s]16711 examples [00:17, 969.14 examples/s]16809 examples [00:17, 971.40 examples/s]16909 examples [00:17, 977.63 examples/s]17007 examples [00:17, 963.95 examples/s]17104 examples [00:17, 959.34 examples/s]17201 examples [00:17, 941.52 examples/s]17300 examples [00:17, 955.13 examples/s]17400 examples [00:17, 967.29 examples/s]17497 examples [00:18, 949.27 examples/s]17597 examples [00:18, 961.21 examples/s]17694 examples [00:18, 962.85 examples/s]17795 examples [00:18, 973.66 examples/s]17893 examples [00:18, 961.89 examples/s]17990 examples [00:18, 963.47 examples/s]18091 examples [00:18, 974.84 examples/s]18190 examples [00:18, 977.75 examples/s]18290 examples [00:18, 984.27 examples/s]18389 examples [00:18, 948.20 examples/s]18486 examples [00:19, 954.26 examples/s]18583 examples [00:19, 957.01 examples/s]18679 examples [00:19, 945.81 examples/s]18774 examples [00:19, 919.50 examples/s]18867 examples [00:19, 900.06 examples/s]18958 examples [00:19, 902.76 examples/s]19058 examples [00:19, 928.62 examples/s]19158 examples [00:19, 948.05 examples/s]19254 examples [00:19, 948.52 examples/s]19353 examples [00:20, 959.55 examples/s]19450 examples [00:20, 951.07 examples/s]19551 examples [00:20, 965.63 examples/s]19651 examples [00:20, 975.27 examples/s]19749 examples [00:20, 976.25 examples/s]19849 examples [00:20, 980.47 examples/s]19948 examples [00:20, 981.23 examples/s]20047 examples [00:20, 943.59 examples/s]20144 examples [00:20, 950.43 examples/s]20240 examples [00:20, 920.51 examples/s]20334 examples [00:21, 924.89 examples/s]20427 examples [00:21, 891.05 examples/s]20527 examples [00:21, 918.66 examples/s]20627 examples [00:21, 941.57 examples/s]20722 examples [00:21, 941.99 examples/s]20818 examples [00:21, 944.42 examples/s]20919 examples [00:21, 960.66 examples/s]21016 examples [00:21, 963.43 examples/s]21116 examples [00:21, 974.04 examples/s]21216 examples [00:21, 979.00 examples/s]21315 examples [00:22, 976.78 examples/s]21416 examples [00:22, 983.85 examples/s]21516 examples [00:22, 987.96 examples/s]21615 examples [00:22, 982.13 examples/s]21716 examples [00:22, 988.07 examples/s]21815 examples [00:22, 956.94 examples/s]21911 examples [00:22, 912.95 examples/s]22003 examples [00:22, 899.45 examples/s]22098 examples [00:22, 911.99 examples/s]22198 examples [00:23, 935.02 examples/s]22297 examples [00:23, 949.34 examples/s]22394 examples [00:23, 952.98 examples/s]22494 examples [00:23, 962.59 examples/s]22591 examples [00:23, 959.73 examples/s]22692 examples [00:23, 971.64 examples/s]22790 examples [00:23, 967.98 examples/s]22890 examples [00:23, 975.27 examples/s]22991 examples [00:23, 983.15 examples/s]23091 examples [00:23, 985.73 examples/s]23190 examples [00:24, 985.82 examples/s]23289 examples [00:24, 977.05 examples/s]23388 examples [00:24, 979.01 examples/s]23488 examples [00:24, 983.78 examples/s]23587 examples [00:24, 974.23 examples/s]23685 examples [00:24, 973.39 examples/s]23783 examples [00:24, 960.05 examples/s]23880 examples [00:24, 941.40 examples/s]23979 examples [00:24, 955.12 examples/s]24078 examples [00:24, 963.76 examples/s]24175 examples [00:25, 960.50 examples/s]24272 examples [00:25, 962.97 examples/s]24369 examples [00:25, 960.57 examples/s]24467 examples [00:25, 965.65 examples/s]24567 examples [00:25, 972.97 examples/s]24666 examples [00:25, 975.85 examples/s]24764 examples [00:25, 941.51 examples/s]24863 examples [00:25, 953.51 examples/s]24960 examples [00:25, 955.51 examples/s]25056 examples [00:25, 953.18 examples/s]25152 examples [00:26, 949.49 examples/s]25248 examples [00:26, 952.23 examples/s]25348 examples [00:26, 964.72 examples/s]25447 examples [00:26, 969.56 examples/s]25547 examples [00:26, 975.68 examples/s]25646 examples [00:26, 977.53 examples/s]25744 examples [00:26, 970.48 examples/s]25844 examples [00:26, 977.66 examples/s]25943 examples [00:26, 979.87 examples/s]26042 examples [00:26, 982.50 examples/s]26143 examples [00:27, 987.93 examples/s]26243 examples [00:27, 990.14 examples/s]26343 examples [00:27, 979.43 examples/s]26444 examples [00:27, 987.84 examples/s]26547 examples [00:27, 997.86 examples/s]26647 examples [00:27, 995.98 examples/s]26747 examples [00:27, 994.98 examples/s]26847 examples [00:27, 983.44 examples/s]26946 examples [00:27, 983.74 examples/s]27046 examples [00:27, 986.65 examples/s]27147 examples [00:28, 993.01 examples/s]27247 examples [00:28, 990.19 examples/s]27347 examples [00:28, 992.06 examples/s]27448 examples [00:28, 995.17 examples/s]27550 examples [00:28, 999.84 examples/s]27651 examples [00:28, 1001.32 examples/s]27752 examples [00:28, 967.15 examples/s] 27849 examples [00:28, 964.88 examples/s]27951 examples [00:28, 979.31 examples/s]28053 examples [00:29, 990.21 examples/s]28155 examples [00:29, 996.53 examples/s]28255 examples [00:29, 985.11 examples/s]28355 examples [00:29, 988.24 examples/s]28454 examples [00:29, 970.20 examples/s]28552 examples [00:29, 971.25 examples/s]28653 examples [00:29, 980.70 examples/s]28752 examples [00:29, 947.25 examples/s]28848 examples [00:29, 936.47 examples/s]28946 examples [00:29, 948.06 examples/s]29046 examples [00:30, 961.03 examples/s]29143 examples [00:30, 957.85 examples/s]29239 examples [00:30, 945.18 examples/s]29334 examples [00:30, 946.39 examples/s]29429 examples [00:30, 944.66 examples/s]29527 examples [00:30, 953.37 examples/s]29626 examples [00:30, 963.15 examples/s]29723 examples [00:30, 952.00 examples/s]29821 examples [00:30, 957.86 examples/s]29917 examples [00:30, 957.41 examples/s]30013 examples [00:31, 908.63 examples/s]30105 examples [00:31, 903.54 examples/s]30200 examples [00:31, 916.35 examples/s]30299 examples [00:31, 932.52 examples/s]30395 examples [00:31, 938.31 examples/s]30490 examples [00:31, 936.66 examples/s]30590 examples [00:31, 952.84 examples/s]30686 examples [00:31, 939.68 examples/s]30781 examples [00:31, 930.78 examples/s]30880 examples [00:31, 946.61 examples/s]30979 examples [00:32, 956.51 examples/s]31075 examples [00:32, 956.43 examples/s]31172 examples [00:32, 959.10 examples/s]31268 examples [00:32, 951.26 examples/s]31368 examples [00:32, 963.05 examples/s]31465 examples [00:32, 964.38 examples/s]31563 examples [00:32, 967.83 examples/s]31660 examples [00:32, 948.08 examples/s]31755 examples [00:32, 947.71 examples/s]31850 examples [00:33, 934.52 examples/s]31948 examples [00:33, 947.07 examples/s]32047 examples [00:33, 959.31 examples/s]32144 examples [00:33, 931.89 examples/s]32241 examples [00:33, 940.41 examples/s]32340 examples [00:33, 953.39 examples/s]32440 examples [00:33, 964.81 examples/s]32540 examples [00:33, 972.60 examples/s]32638 examples [00:33, 968.18 examples/s]32738 examples [00:33, 975.50 examples/s]32838 examples [00:34, 980.56 examples/s]32937 examples [00:34, 977.97 examples/s]33036 examples [00:34, 978.57 examples/s]33134 examples [00:34, 965.10 examples/s]33231 examples [00:34, 949.93 examples/s]33330 examples [00:34, 958.92 examples/s]33430 examples [00:34, 968.20 examples/s]33529 examples [00:34, 973.83 examples/s]33627 examples [00:34, 972.32 examples/s]33725 examples [00:34, 928.58 examples/s]33819 examples [00:35, 907.69 examples/s]33920 examples [00:35, 935.13 examples/s]34017 examples [00:35, 945.23 examples/s]34116 examples [00:35, 956.99 examples/s]34215 examples [00:35, 965.55 examples/s]34316 examples [00:35, 977.14 examples/s]34417 examples [00:35, 986.52 examples/s]34516 examples [00:35, 984.89 examples/s]34616 examples [00:35, 986.69 examples/s]34715 examples [00:35, 974.87 examples/s]34813 examples [00:36, 967.12 examples/s]34912 examples [00:36, 971.15 examples/s]35012 examples [00:36, 977.35 examples/s]35113 examples [00:36, 986.23 examples/s]35212 examples [00:36, 973.53 examples/s]35313 examples [00:36, 981.79 examples/s]35412 examples [00:36, 972.94 examples/s]35512 examples [00:36, 980.09 examples/s]35612 examples [00:36, 983.44 examples/s]35712 examples [00:37, 988.04 examples/s]35811 examples [00:37, 969.11 examples/s]35911 examples [00:37, 977.26 examples/s]36010 examples [00:37, 981.04 examples/s]36110 examples [00:37, 984.78 examples/s]36209 examples [00:37, 981.79 examples/s]36308 examples [00:37, 963.04 examples/s]36408 examples [00:37, 973.49 examples/s]36506 examples [00:37, 960.57 examples/s]36604 examples [00:37, 964.63 examples/s]36701 examples [00:38, 947.68 examples/s]36801 examples [00:38, 956.84 examples/s]36897 examples [00:38, 956.67 examples/s]36993 examples [00:38, 952.24 examples/s]37089 examples [00:38, 948.15 examples/s]37189 examples [00:38, 961.13 examples/s]37290 examples [00:38, 974.42 examples/s]37390 examples [00:38, 981.71 examples/s]37490 examples [00:38, 985.43 examples/s]37590 examples [00:38, 987.81 examples/s]37689 examples [00:39, 985.78 examples/s]37791 examples [00:39, 994.22 examples/s]37891 examples [00:39, 991.49 examples/s]37991 examples [00:39, 982.84 examples/s]38090 examples [00:39, 937.52 examples/s]38189 examples [00:39, 950.90 examples/s]38290 examples [00:39, 964.90 examples/s]38391 examples [00:39, 976.45 examples/s]38489 examples [00:39, 971.75 examples/s]38591 examples [00:39, 984.07 examples/s]38693 examples [00:40, 994.10 examples/s]38793 examples [00:40, 992.60 examples/s]38896 examples [00:40, 1001.66 examples/s]38997 examples [00:40, 1000.22 examples/s]39098 examples [00:40, 973.88 examples/s] 39199 examples [00:40, 982.59 examples/s]39298 examples [00:40, 983.79 examples/s]39400 examples [00:40, 991.59 examples/s]39500 examples [00:40, 987.78 examples/s]39599 examples [00:40, 983.30 examples/s]39698 examples [00:41, 940.64 examples/s]39793 examples [00:41, 915.10 examples/s]39885 examples [00:41, 898.51 examples/s]39976 examples [00:41, 892.54 examples/s]40066 examples [00:41, 864.74 examples/s]40167 examples [00:41, 902.71 examples/s]40269 examples [00:41, 932.73 examples/s]40369 examples [00:41, 950.76 examples/s]40466 examples [00:41, 954.48 examples/s]40566 examples [00:42, 966.96 examples/s]40667 examples [00:42, 976.79 examples/s]40767 examples [00:42, 981.66 examples/s]40866 examples [00:42, 983.75 examples/s]40965 examples [00:42, 971.05 examples/s]41063 examples [00:42, 966.54 examples/s]41164 examples [00:42, 976.92 examples/s]41263 examples [00:42, 978.40 examples/s]41361 examples [00:42, 969.56 examples/s]41459 examples [00:42, 970.92 examples/s]41559 examples [00:43, 977.83 examples/s]41659 examples [00:43, 983.77 examples/s]41760 examples [00:43, 989.10 examples/s]41859 examples [00:43, 969.20 examples/s]41957 examples [00:43, 961.80 examples/s]42054 examples [00:43, 961.62 examples/s]42154 examples [00:43, 970.90 examples/s]42254 examples [00:43, 977.22 examples/s]42353 examples [00:43, 978.34 examples/s]42452 examples [00:43, 980.02 examples/s]42551 examples [00:44, 982.89 examples/s]42650 examples [00:44, 944.53 examples/s]42750 examples [00:44, 960.23 examples/s]42847 examples [00:44, 949.68 examples/s]42943 examples [00:44, 919.37 examples/s]43036 examples [00:44, 920.24 examples/s]43137 examples [00:44, 944.92 examples/s]43238 examples [00:44, 960.32 examples/s]43339 examples [00:44, 974.44 examples/s]43438 examples [00:45, 979.00 examples/s]43539 examples [00:45, 987.70 examples/s]43642 examples [00:45, 997.58 examples/s]43742 examples [00:45, 993.38 examples/s]43842 examples [00:45, 942.66 examples/s]43938 examples [00:45, 946.39 examples/s]44038 examples [00:45, 960.74 examples/s]44139 examples [00:45, 974.34 examples/s]44240 examples [00:45, 982.64 examples/s]44341 examples [00:45, 988.18 examples/s]44440 examples [00:46, 981.59 examples/s]44539 examples [00:46, 958.36 examples/s]44636 examples [00:46, 912.10 examples/s]44737 examples [00:46, 937.93 examples/s]44832 examples [00:46, 905.37 examples/s]44926 examples [00:46, 913.91 examples/s]45027 examples [00:46, 938.69 examples/s]45128 examples [00:46, 957.98 examples/s]45227 examples [00:46, 966.26 examples/s]45326 examples [00:46, 970.53 examples/s]45425 examples [00:47, 975.82 examples/s]45523 examples [00:47, 936.55 examples/s]45624 examples [00:47, 956.66 examples/s]45725 examples [00:47, 969.50 examples/s]45823 examples [00:47, 955.14 examples/s]45921 examples [00:47, 962.19 examples/s]46022 examples [00:47, 975.17 examples/s]46120 examples [00:47, 969.78 examples/s]46218 examples [00:47, 959.96 examples/s]46315 examples [00:48, 954.81 examples/s]46413 examples [00:48, 960.57 examples/s]46510 examples [00:48, 959.12 examples/s]46606 examples [00:48, 953.15 examples/s]46705 examples [00:48, 961.87 examples/s]46802 examples [00:48, 943.07 examples/s]46897 examples [00:48, 931.31 examples/s]46994 examples [00:48, 941.03 examples/s]47089 examples [00:48, 937.46 examples/s]47189 examples [00:48, 952.77 examples/s]47287 examples [00:49, 959.60 examples/s]47386 examples [00:49, 966.35 examples/s]47487 examples [00:49, 976.89 examples/s]47588 examples [00:49, 985.82 examples/s]47687 examples [00:49, 970.46 examples/s]47785 examples [00:49, 966.22 examples/s]47884 examples [00:49, 972.37 examples/s]47984 examples [00:49, 979.70 examples/s]48084 examples [00:49, 984.60 examples/s]48183 examples [00:49, 973.48 examples/s]48283 examples [00:50, 978.47 examples/s]48383 examples [00:50, 983.28 examples/s]48482 examples [00:50, 948.87 examples/s]48582 examples [00:50, 962.31 examples/s]48679 examples [00:50, 960.56 examples/s]48776 examples [00:50, 948.44 examples/s]48874 examples [00:50, 956.92 examples/s]48975 examples [00:50, 971.81 examples/s]49077 examples [00:50, 983.60 examples/s]49176 examples [00:50, 983.20 examples/s]49275 examples [00:51, 966.92 examples/s]49375 examples [00:51, 974.44 examples/s]49473 examples [00:51, 956.99 examples/s]49570 examples [00:51, 960.03 examples/s]49670 examples [00:51, 970.76 examples/s]49768 examples [00:51, 970.15 examples/s]49868 examples [00:51, 976.43 examples/s]49968 examples [00:51, 981.79 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7039/50000 [00:00<00:00, 70386.88 examples/s] 38%|███▊      | 18986/50000 [00:00<00:00, 80281.68 examples/s] 62%|██████▏   | 31010/50000 [00:00<00:00, 89170.20 examples/s] 87%|████████▋ | 43267/50000 [00:00<00:00, 97107.69 examples/s]                                                               0 examples [00:00, ? examples/s]85 examples [00:00, 843.43 examples/s]186 examples [00:00, 886.85 examples/s]286 examples [00:00, 915.94 examples/s]387 examples [00:00, 940.17 examples/s]489 examples [00:00, 961.32 examples/s]587 examples [00:00, 964.14 examples/s]681 examples [00:00, 956.64 examples/s]780 examples [00:00, 965.49 examples/s]881 examples [00:00, 977.70 examples/s]976 examples [00:01, 955.22 examples/s]1077 examples [00:01, 969.48 examples/s]1178 examples [00:01, 978.55 examples/s]1275 examples [00:01, 974.60 examples/s]1372 examples [00:01, 759.54 examples/s]1473 examples [00:01, 819.62 examples/s]1575 examples [00:01, 870.01 examples/s]1673 examples [00:01, 899.64 examples/s]1774 examples [00:01, 927.67 examples/s]1876 examples [00:02, 952.36 examples/s]1977 examples [00:02, 968.21 examples/s]2076 examples [00:02, 965.91 examples/s]2174 examples [00:02, 940.40 examples/s]2273 examples [00:02, 951.97 examples/s]2374 examples [00:02, 967.48 examples/s]2475 examples [00:02, 978.77 examples/s]2577 examples [00:02, 989.60 examples/s]2677 examples [00:02, 991.54 examples/s]2779 examples [00:02, 997.37 examples/s]2881 examples [00:03, 1002.54 examples/s]2982 examples [00:03, 991.65 examples/s] 3084 examples [00:03, 998.17 examples/s]3184 examples [00:03, 979.32 examples/s]3284 examples [00:03, 985.02 examples/s]3384 examples [00:03, 987.44 examples/s]3485 examples [00:03, 992.89 examples/s]3587 examples [00:03, 999.43 examples/s]3687 examples [00:03, 989.08 examples/s]3787 examples [00:03, 990.43 examples/s]3887 examples [00:04, 955.72 examples/s]3986 examples [00:04, 965.49 examples/s]4087 examples [00:04, 977.93 examples/s]4185 examples [00:04, 978.09 examples/s]4283 examples [00:04, 977.15 examples/s]4381 examples [00:04, 975.53 examples/s]4480 examples [00:04, 977.60 examples/s]4578 examples [00:04, 972.59 examples/s]4678 examples [00:04, 979.30 examples/s]4780 examples [00:04, 988.97 examples/s]4881 examples [00:05, 993.92 examples/s]4981 examples [00:05, 982.86 examples/s]5080 examples [00:05, 984.27 examples/s]5179 examples [00:05, 984.45 examples/s]5280 examples [00:05, 990.54 examples/s]5380 examples [00:05, 990.15 examples/s]5480 examples [00:05, 954.24 examples/s]5576 examples [00:05, 931.00 examples/s]5675 examples [00:05, 945.27 examples/s]5775 examples [00:05, 959.78 examples/s]5872 examples [00:06, 960.71 examples/s]5972 examples [00:06, 969.79 examples/s]6072 examples [00:06, 978.53 examples/s]6173 examples [00:06, 985.35 examples/s]6274 examples [00:06, 990.69 examples/s]6374 examples [00:06, 972.89 examples/s]6476 examples [00:06, 984.07 examples/s]6577 examples [00:06, 990.95 examples/s]6677 examples [00:06, 992.77 examples/s]6777 examples [00:07, 960.13 examples/s]6874 examples [00:07, 901.21 examples/s]6966 examples [00:07, 875.21 examples/s]7055 examples [00:07, 869.11 examples/s]7151 examples [00:07, 893.89 examples/s]7251 examples [00:07, 922.85 examples/s]7352 examples [00:07, 944.96 examples/s]7448 examples [00:07, 945.24 examples/s]7548 examples [00:07, 958.34 examples/s]7649 examples [00:07, 971.68 examples/s]7747 examples [00:08, 960.18 examples/s]7844 examples [00:08, 958.75 examples/s]7945 examples [00:08, 971.96 examples/s]8046 examples [00:08, 979.18 examples/s]8146 examples [00:08, 983.59 examples/s]8246 examples [00:08, 987.23 examples/s]8346 examples [00:08, 988.64 examples/s]8447 examples [00:08, 994.29 examples/s]8547 examples [00:08, 955.96 examples/s]8643 examples [00:09, 908.73 examples/s]8739 examples [00:09, 922.31 examples/s]8837 examples [00:09, 937.60 examples/s]8936 examples [00:09, 950.84 examples/s]9036 examples [00:09, 964.87 examples/s]9138 examples [00:09, 978.15 examples/s]9239 examples [00:09, 987.40 examples/s]9339 examples [00:09, 990.26 examples/s]9441 examples [00:09, 998.11 examples/s]9543 examples [00:09, 1001.64 examples/s]9644 examples [00:10, 1000.31 examples/s]9745 examples [00:10, 988.69 examples/s] 9844 examples [00:10, 939.50 examples/s]9944 examples [00:10, 956.59 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteT6ZOD1/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteT6ZOD1/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f43188ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f43188ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f43188ce950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42a2e52cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f43017d2860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f43188ce950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f43188ce950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f43188ce950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42a2e35b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f42fec8d080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f42978942f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f42978942f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f42978942f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f4297894400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f4297894400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f4297894400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=09881847da71aa24cacdf290235538b4a31a5d3d424cbddb623e4f06755d24f5
  Stored in directory: /tmp/pip-ephem-wheel-cache-87296syy/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:46:11, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:46:18, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:23:38, 23.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:17:03, 32.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:05:10, 46.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.76M/862M [00:01<3:32:09, 67.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.7M/862M [00:01<2:28:10, 95.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:43:07, 136kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<1:11:46, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.9M/862M [00:01<50:00, 277kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<34:51, 395kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<24:19, 562kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:02<17:00, 799kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:02<11:53, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:03<49:07, 275kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<36:08, 371kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<27:09, 494kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<19:34, 685kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:05<13:49, 967kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<22:43, 588kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<18:36, 717kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:07<13:36, 980kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:07<09:40, 1.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<13:30, 983kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:09<10:48, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:09<07:50, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<08:35, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<08:50, 1.49MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<06:46, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:11<04:54, 2.68MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<10:22, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<08:37, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.3M/862M [00:13<06:21, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:30, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:15<04:53, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:29, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<07:11, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<05:41, 2.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<05:34, 2.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<04:13, 3.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<05:58, 2.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<06:49, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<05:18, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:21<03:53, 3.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<08:28, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<07:15, 1.76MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:22<05:23, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:41, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<07:16, 1.75MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<05:37, 2.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:05, 3.10MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<10:00, 1.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:19, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<06:05, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:11, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:35, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:57, 2.11MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:45, 2.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<7:54:02, 26.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<5:31:40, 37.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<3:51:29, 53.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<3:04:48, 67.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:12:34, 94.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:33:30, 133kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<1:05:24, 190kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<52:22, 237kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<38:02, 326kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<26:52, 460kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<21:23, 576kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<17:28, 705kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<12:45, 965kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<09:02, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<15:23, 797kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:03, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<08:43, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:57, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:44, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:40, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:48, 2.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:53, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:51, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:21, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:31, 1.60MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:52, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:13, 2.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:34:32, 17.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:07:10, 24.6kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<5:40:36, 35.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<4:00:27, 49.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:50:48, 69.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<2:00:03, 99.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<1:23:49, 141kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<2:23:25, 82.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<1:41:34, 117kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<1:11:15, 166kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<52:29, 224kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<39:08, 301kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<27:58, 421kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<19:37, 597kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<1:38:33, 119kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:10:08, 167kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<49:17, 237kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<37:07, 314kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<28:21, 411kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<20:25, 569kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<14:23, 805kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:44:36, 111kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:14:21, 156kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<52:13, 221kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<39:08, 294kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<29:43, 387kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<21:22, 538kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<16:45, 683kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<12:54, 885kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<09:18, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:10, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:34, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:34, 2.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:34, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:54, 1.64MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:19, 2.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:50, 2.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<15:40, 716kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<12:08, 925kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<08:42, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<08:42, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:13, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<05:20, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:26, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<7:29:46, 24.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<5:15:16, 35.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<3:40:03, 50.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<2:38:59, 69.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<1:53:49, 96.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<1:20:07, 138kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<56:02, 196kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<44:26, 247kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<32:14, 340kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<22:45, 480kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<18:25, 591kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<15:06, 721kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:03, 984kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<07:48, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<58:40, 185kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<42:08, 257kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<29:42, 363kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<23:15, 463kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<17:10, 626kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<12:13, 878kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<08:40, 1.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<49:10, 217kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<36:35, 292kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<26:02, 410kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<18:18, 581kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<18:43, 567kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<14:12, 747kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<10:11, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:35, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:51, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:39, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:45, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<13:00, 806kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<10:11, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:23, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<07:34, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<06:21, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:42, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:44, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:04, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:48, 2.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:05, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<05:39, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:23, 2.33MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:11, 3.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<09:52, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:56, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:48, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<06:26, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<06:34, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:06, 1.98MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:11, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:28, 2.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:47, 2.09MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:23, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:12, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:02, 3.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<17:59, 552kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<13:37, 729kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<09:43, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<09:06, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:23, 1.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<05:24, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:06, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:16, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:53, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:00, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:29, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:15, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<03:05, 3.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<05:05, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<6:16:59, 25.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<4:23:38, 36.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<3:05:27, 51.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<2:12:07, 72.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<1:32:57, 103kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<1:04:54, 147kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<52:18, 182kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<37:26, 254kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<26:20, 361kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<18:30, 511kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<26:30, 357kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<20:32, 460kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<14:47, 638kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<10:25, 902kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<14:18, 657kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:59, 854kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<07:55, 1.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<07:41, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:19, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:39, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:26, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:41, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:26, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<03:11, 2.88MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<33:47, 272kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<24:36, 373kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<17:23, 527kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<14:11, 643kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<10:52, 838kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<07:50, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:33, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<07:12, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:30, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<03:57, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<29:45, 302kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<21:46, 412kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<15:26, 580kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<12:46, 698kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:49, 823kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:02, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<05:41, 1.55MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<25:20, 349kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<18:40, 473kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<13:14, 666kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<11:12, 782kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<09:42, 904kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<07:11, 1.22MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<05:07, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<08:59, 969kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<07:12, 1.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:13, 1.66MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:38, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:46, 1.49MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:29, 1.92MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:14, 2.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<29:26, 291kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<21:30, 398kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<15:14, 560kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<12:32, 678kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<10:34, 803kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<07:50, 1.08MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<05:33, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<23:20, 361kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<17:12, 489kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<12:12, 688kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<10:26, 801kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<09:00, 928kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<06:40, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<07:09, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<05:52, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:19, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:54, 1.68MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:10, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:00, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:54, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:37, 1.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:47, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<03:33, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:20, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:53, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:55, 2.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:32, 3.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<5:27:34, 24.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<3:49:25, 34.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<2:39:59, 49.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<1:54:45, 69.4kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<1:22:04, 96.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<57:49, 137kB/s]   .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:41<40:21, 196kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<36:07, 218kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<26:06, 302kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<18:25, 427kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<14:35, 536kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<11:52, 659kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<08:39, 902kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<06:07, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<08:22, 926kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:40, 1.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<04:50, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:08, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:24, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:15, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:01, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:26, 1.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:30, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:32, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<25:42, 293kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<18:46, 402kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<13:17, 565kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<10:59, 681kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<09:16, 806kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:50, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<04:51, 1.53MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<19:49, 374kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<14:38, 506kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<10:24, 709kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<08:56, 821kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:48, 939kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:51, 1.25MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<04:09, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<19:43, 369kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<14:35, 498kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<10:21, 699kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<08:52, 812kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<07:44, 930kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<05:47, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<04:06, 1.74MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<19:22, 368kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<14:18, 498kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<10:09, 699kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<08:42, 812kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<06:50, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:57, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:01, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<05:00, 1.40MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:51, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:46, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<18:26, 376kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<13:37, 508kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<09:40, 713kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<08:18, 826kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<07:16, 944kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:26, 1.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:51, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<18:25, 369kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<13:35, 499kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<09:37, 702kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<08:15, 813kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<07:09, 940kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<05:20, 1.25MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<03:47, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<56:59, 117kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<40:32, 164kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<28:26, 233kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<21:19, 309kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<16:15, 405kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<11:38, 564kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<08:12, 797kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:47, 741kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:49, 954kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<04:55, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:49, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<4:31:28, 23.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<3:10:05, 33.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<2:12:14, 48.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<1:36:16, 66.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<1:08:42, 92.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<48:18, 132kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<33:37, 188kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<32:23, 195kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<23:19, 271kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<16:24, 383kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<12:50, 487kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<10:18, 606kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:39, 815kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:50, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:29, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:35, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:57, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:30, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:11, 2.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<17:25, 355kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<14:30, 426kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<10:41, 578kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:56, 775kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:53, 1.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<04:30, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:32, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:50, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:47, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<06:02, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<04:42, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:40, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:59, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:26, 2.48MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:03, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:47, 3.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<09:26, 640kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<08:26, 715kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<06:25, 938kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:47, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:36, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:06, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:30, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:05, 2.86MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:41, 894kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:30, 919kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:01, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:51, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:26, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:01, 2.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:25, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<06:42, 881kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:37, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:14, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:47, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:12, 2.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:48, 3.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<01:41, 3.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:58, 836kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<06:39, 875kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:09, 1.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:52, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:01, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:26, 2.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:03, 2.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<04:36, 1.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:45, 853kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<05:38, 1.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:20, 1.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:19, 1.73MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:41, 2.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:11, 2.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:47, 3.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:45, 988kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<05:41, 1.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:26, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:22, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:35, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:19, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:55, 2.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:36, 3.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<09:22, 600kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<08:18, 677kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<06:15, 898kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:40, 1.20MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:34, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:47, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:12, 2.53MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<05:29, 1.01MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<07:13, 769kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:52, 945kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<04:24, 1.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:24, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:39, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<02:05, 2.63MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<01:44, 3.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<1:47:57, 50.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<1:17:01, 71.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<54:10, 101kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<38:06, 143kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<26:54, 203kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<19:00, 286kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<13:31, 401kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<25:14, 215kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<20:35, 263kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<15:06, 359kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<10:50, 498kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<07:46, 693kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<05:36, 957kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<04:18, 1.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<05:36, 954kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<06:32, 818kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<05:12, 1.03MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:50, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:53, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:13, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:03, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<04:59, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:03, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:01, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:19, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:50, 2.84MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:26, 3.63MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<49:32, 105kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<36:38, 142kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<26:01, 200kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<18:22, 283kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<12:59, 398kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<09:10, 561kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<15:16, 337kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<12:28, 412kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<09:04, 566kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<06:31, 784kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<04:43, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:13, 972kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<06:47, 747kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:33, 914kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:04, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:58, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:32, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<05:11, 963kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:19, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:13, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:22, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:13, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:27, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:43, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:01, 2.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:41, 1.81MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<04:03, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<03:23, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:31, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:50, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:17, 909kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<05:39, 849kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:26, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:14, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<02:19, 2.03MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<09:46, 484kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<08:36, 549kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<06:27, 732kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:37, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:13, 1.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:36, 1.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:37, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:38, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:55, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:27, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:46, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:00, 2.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:47, 2.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<2:46:18, 27.3kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<1:56:33, 38.8kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<1:21:08, 55.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<57:42, 77.4kB/s]  .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<43:07, 104kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<30:45, 145kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<21:33, 206kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<15:02, 293kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<14:44, 298kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<11:30, 382kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<08:19, 527kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<05:51, 743kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:38, 767kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:00, 865kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:42, 1.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:38, 1.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:27, 1.23MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:38, 1.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:50, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<02:03, 2.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:55, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:02, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:22, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:43, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:01, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:07, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:24, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:43, 2.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:00, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:08, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:25, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:45, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:25, 1.64MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:40, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:06, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:31, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:44, 1.43MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:03, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:23, 1.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:44, 2.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:36, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:52, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:13, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:35, 2.37MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:21, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:15, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:42, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<11:19, 327kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<08:42, 425kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<06:16, 588kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<04:22, 830kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<14:02, 259kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:20, 351kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<07:18, 494kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<05:05, 700kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<13:14, 269kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<09:43, 366kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<06:53, 514kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<05:26, 643kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<04:27, 783kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<03:15, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<02:16, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<06:55, 494kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:29, 623kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:59, 855kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:18, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:55, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:10, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:33, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:48, 1.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:33, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:55, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:21, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<07:06, 453kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:33, 579kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<04:01, 796kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:18, 955kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<03:29, 901kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:44, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:57, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:07, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:02, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:33, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:34, 1.91MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<02:14, 1.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:20, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:27, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:33, 1.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:13, 2.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:19, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:01, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:41, 1.70MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:13, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:32, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:35, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:12, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:52, 3.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<04:17, 638kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<03:30, 781kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:33, 1.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:12, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:32, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:59, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:27, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:29, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:10, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:13, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:01, 2.44MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:44, 3.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:00, 821kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:32, 968kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:52, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:41, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:36, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:13, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:57, 2.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<1:36:34, 24.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<1:07:36, 34.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<46:46, 49.1kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<32:53, 68.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<23:38, 95.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<16:42, 135kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<11:35, 192kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<08:32, 257kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<06:21, 344kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<04:31, 481kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:27, 616kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:47, 759kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<02:01, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:25, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:15, 910kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:57, 1.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:27, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:19, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:58, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:59, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:02, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:48, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:51, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:56, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:44, 2.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:47, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:52, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:45, 2.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:07, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:56, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:41, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:52, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:55, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:44, 2.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:06, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:54, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:39, 2.35MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:46, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:48, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:43, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:33, 2.54MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:23, 3.48MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:22, 997kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:12, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:53, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:03, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:50, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:36, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:43, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:43, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:33, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:34, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:29, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:25, 2.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:30, 2.03MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:23, 2.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:16, 3.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:27, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:21, 2.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:24, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:19, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:22, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:17, 2.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:15, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:20, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:14, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:18, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:13, 2.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:10, 2.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<19:56, 24.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<13:47, 34.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<09:03, 48.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<05:59, 68.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<04:19, 94.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<03:00, 133kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<01:58, 189kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:20, 254kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:59, 339kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:40, 476kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:24, 674kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:40, 403kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:30, 519kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:21, 717kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:13, 874kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:08, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:04, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 825kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:09, 815kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 1.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.89MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<143:51:19,  1.29s/it]  0%|          | 727/400000 [00:01<100:31:14,  1.10it/s]  0%|          | 1478/400000 [00:01<70:14:11,  1.58it/s]  1%|          | 2215/400000 [00:01<49:04:44,  2.25it/s]  1%|          | 2962/400000 [00:01<34:17:43,  3.22it/s]  1%|          | 3692/400000 [00:01<23:58:01,  4.59it/s]  1%|          | 4425/400000 [00:01<16:45:01,  6.56it/s]  1%|▏         | 5183/400000 [00:01<11:42:25,  9.37it/s]  1%|▏         | 5935/400000 [00:02<8:11:01, 13.38it/s]   2%|▏         | 6644/400000 [00:02<5:43:22, 19.09it/s]  2%|▏         | 7386/400000 [00:02<4:00:10, 27.24it/s]  2%|▏         | 8102/400000 [00:02<2:48:05, 38.86it/s]  2%|▏         | 8849/400000 [00:02<1:57:42, 55.39it/s]  2%|▏         | 9608/400000 [00:02<1:22:29, 78.88it/s]  3%|▎         | 10371/400000 [00:02<57:53, 112.18it/s]  3%|▎         | 11113/400000 [00:02<40:43, 159.14it/s]  3%|▎         | 11837/400000 [00:02<28:43, 225.18it/s]  3%|▎         | 12563/400000 [00:03<20:20, 317.47it/s]  3%|▎         | 13318/400000 [00:03<14:27, 445.49it/s]  4%|▎         | 14056/400000 [00:03<10:22, 620.36it/s]  4%|▎         | 14793/400000 [00:03<07:30, 855.35it/s]  4%|▍         | 15551/400000 [00:03<05:29, 1165.49it/s]  4%|▍         | 16307/400000 [00:03<04:05, 1561.72it/s]  4%|▍         | 17053/400000 [00:03<03:08, 2035.63it/s]  4%|▍         | 17794/400000 [00:03<02:26, 2601.53it/s]  5%|▍         | 18554/400000 [00:03<01:57, 3240.47it/s]  5%|▍         | 19296/400000 [00:03<01:38, 3863.89it/s]  5%|▌         | 20038/400000 [00:04<01:24, 4512.34it/s]  5%|▌         | 20773/400000 [00:04<01:14, 5102.64it/s]  5%|▌         | 21514/400000 [00:04<01:07, 5627.54it/s]  6%|▌         | 22277/400000 [00:04<01:01, 6107.17it/s]  6%|▌         | 23021/400000 [00:04<00:58, 6426.27it/s]  6%|▌         | 23762/400000 [00:04<00:57, 6553.61it/s]  6%|▌         | 24515/400000 [00:04<00:55, 6817.70it/s]  6%|▋         | 25248/400000 [00:04<00:53, 6955.69it/s]  6%|▋         | 25980/400000 [00:04<00:53, 7045.09it/s]  7%|▋         | 26728/400000 [00:04<00:52, 7168.81it/s]  7%|▋         | 27481/400000 [00:05<00:51, 7271.79it/s]  7%|▋         | 28238/400000 [00:05<00:50, 7358.17it/s]  7%|▋         | 28984/400000 [00:05<00:50, 7351.92it/s]  7%|▋         | 29726/400000 [00:05<00:50, 7368.60it/s]  8%|▊         | 30481/400000 [00:05<00:49, 7421.57it/s]  8%|▊         | 31227/400000 [00:05<00:50, 7363.86it/s]  8%|▊         | 31966/400000 [00:05<00:51, 7136.90it/s]  8%|▊         | 32729/400000 [00:05<00:50, 7277.31it/s]  8%|▊         | 33494/400000 [00:05<00:49, 7382.96it/s]  9%|▊         | 34235/400000 [00:05<00:50, 7312.39it/s]  9%|▉         | 35000/400000 [00:06<00:49, 7410.45it/s]  9%|▉         | 35753/400000 [00:06<00:48, 7443.49it/s]  9%|▉         | 36516/400000 [00:06<00:48, 7496.11it/s]  9%|▉         | 37285/400000 [00:06<00:48, 7550.48it/s] 10%|▉         | 38045/400000 [00:06<00:47, 7562.82it/s] 10%|▉         | 38802/400000 [00:06<00:48, 7433.87it/s] 10%|▉         | 39547/400000 [00:06<00:48, 7403.11it/s] 10%|█         | 40302/400000 [00:06<00:48, 7444.18it/s] 10%|█         | 41047/400000 [00:06<00:49, 7294.45it/s] 10%|█         | 41798/400000 [00:06<00:48, 7355.00it/s] 11%|█         | 42565/400000 [00:07<00:47, 7446.66it/s] 11%|█         | 43311/400000 [00:07<00:47, 7432.20it/s] 11%|█         | 44055/400000 [00:07<00:47, 7424.60it/s] 11%|█         | 44811/400000 [00:07<00:47, 7462.45it/s] 11%|█▏        | 45559/400000 [00:07<00:47, 7466.09it/s] 12%|█▏        | 46317/400000 [00:07<00:47, 7497.30it/s] 12%|█▏        | 47067/400000 [00:07<00:47, 7453.44it/s] 12%|█▏        | 47825/400000 [00:07<00:47, 7490.30it/s] 12%|█▏        | 48585/400000 [00:07<00:46, 7521.31it/s] 12%|█▏        | 49338/400000 [00:07<00:46, 7487.13it/s] 13%|█▎        | 50099/400000 [00:08<00:46, 7522.48it/s] 13%|█▎        | 50857/400000 [00:08<00:46, 7537.72it/s] 13%|█▎        | 51611/400000 [00:08<00:46, 7508.85it/s] 13%|█▎        | 52378/400000 [00:08<00:46, 7554.45it/s] 13%|█▎        | 53134/400000 [00:08<00:45, 7541.42it/s] 13%|█▎        | 53896/400000 [00:08<00:45, 7562.88it/s] 14%|█▎        | 54660/400000 [00:08<00:45, 7585.44it/s] 14%|█▍        | 55421/400000 [00:08<00:45, 7590.80it/s] 14%|█▍        | 56181/400000 [00:08<00:46, 7346.73it/s] 14%|█▍        | 56918/400000 [00:08<00:47, 7194.73it/s] 14%|█▍        | 57640/400000 [00:09<00:47, 7184.64it/s] 15%|█▍        | 58375/400000 [00:09<00:47, 7231.57it/s] 15%|█▍        | 59134/400000 [00:09<00:46, 7333.15it/s] 15%|█▍        | 59887/400000 [00:09<00:46, 7388.95it/s] 15%|█▌        | 60627/400000 [00:09<00:46, 7358.39it/s] 15%|█▌        | 61365/400000 [00:09<00:45, 7364.11it/s] 16%|█▌        | 62120/400000 [00:09<00:45, 7418.66it/s] 16%|█▌        | 62872/400000 [00:09<00:45, 7447.76it/s] 16%|█▌        | 63626/400000 [00:09<00:45, 7474.33it/s] 16%|█▌        | 64374/400000 [00:09<00:45, 7422.82it/s] 16%|█▋        | 65117/400000 [00:10<00:45, 7411.32it/s] 16%|█▋        | 65875/400000 [00:10<00:44, 7458.51it/s] 17%|█▋        | 66626/400000 [00:10<00:44, 7473.65it/s] 17%|█▋        | 67374/400000 [00:10<00:45, 7370.89it/s] 17%|█▋        | 68112/400000 [00:10<00:45, 7306.81it/s] 17%|█▋        | 68844/400000 [00:10<00:45, 7202.65it/s] 17%|█▋        | 69586/400000 [00:10<00:45, 7264.88it/s] 18%|█▊        | 70344/400000 [00:10<00:44, 7355.76it/s] 18%|█▊        | 71081/400000 [00:10<00:44, 7319.36it/s] 18%|█▊        | 71819/400000 [00:11<00:44, 7336.63it/s] 18%|█▊        | 72579/400000 [00:11<00:44, 7412.54it/s] 18%|█▊        | 73321/400000 [00:11<00:44, 7381.15it/s] 19%|█▊        | 74063/400000 [00:11<00:44, 7392.67it/s] 19%|█▊        | 74814/400000 [00:11<00:43, 7425.58it/s] 19%|█▉        | 75564/400000 [00:11<00:43, 7445.82it/s] 19%|█▉        | 76309/400000 [00:11<00:43, 7437.86it/s] 19%|█▉        | 77053/400000 [00:11<00:43, 7363.73it/s] 19%|█▉        | 77790/400000 [00:11<00:44, 7310.67it/s] 20%|█▉        | 78538/400000 [00:11<00:43, 7360.22it/s] 20%|█▉        | 79286/400000 [00:12<00:43, 7393.00it/s] 20%|██        | 80027/400000 [00:12<00:43, 7396.59it/s] 20%|██        | 80775/400000 [00:12<00:43, 7421.31it/s] 20%|██        | 81524/400000 [00:12<00:42, 7439.77it/s] 21%|██        | 82269/400000 [00:12<00:42, 7430.04it/s] 21%|██        | 83013/400000 [00:12<00:42, 7387.56it/s] 21%|██        | 83768/400000 [00:12<00:42, 7434.93it/s] 21%|██        | 84523/400000 [00:12<00:42, 7467.54it/s] 21%|██▏       | 85270/400000 [00:12<00:43, 7279.68it/s] 22%|██▏       | 86027/400000 [00:12<00:42, 7362.27it/s] 22%|██▏       | 86765/400000 [00:13<00:43, 7282.24it/s] 22%|██▏       | 87495/400000 [00:13<00:43, 7123.06it/s] 22%|██▏       | 88237/400000 [00:13<00:43, 7170.13it/s] 22%|██▏       | 88992/400000 [00:13<00:42, 7278.01it/s] 22%|██▏       | 89752/400000 [00:13<00:42, 7364.26it/s] 23%|██▎       | 90490/400000 [00:13<00:42, 7313.47it/s] 23%|██▎       | 91232/400000 [00:13<00:42, 7344.85it/s] 23%|██▎       | 91969/400000 [00:13<00:41, 7351.84it/s] 23%|██▎       | 92725/400000 [00:13<00:41, 7412.46it/s] 23%|██▎       | 93467/400000 [00:13<00:42, 7180.91it/s] 24%|██▎       | 94194/400000 [00:14<00:42, 7204.72it/s] 24%|██▎       | 94916/400000 [00:14<00:42, 7100.49it/s] 24%|██▍       | 95665/400000 [00:14<00:42, 7212.45it/s] 24%|██▍       | 96415/400000 [00:14<00:41, 7294.52it/s] 24%|██▍       | 97167/400000 [00:14<00:41, 7360.67it/s] 24%|██▍       | 97905/400000 [00:14<00:41, 7355.66it/s] 25%|██▍       | 98646/400000 [00:14<00:40, 7369.74it/s] 25%|██▍       | 99384/400000 [00:14<00:41, 7318.28it/s] 25%|██▌       | 100133/400000 [00:14<00:40, 7368.05it/s] 25%|██▌       | 100887/400000 [00:14<00:40, 7418.55it/s] 25%|██▌       | 101630/400000 [00:15<00:40, 7413.13it/s] 26%|██▌       | 102372/400000 [00:15<00:40, 7407.64it/s] 26%|██▌       | 103124/400000 [00:15<00:39, 7439.17it/s] 26%|██▌       | 103869/400000 [00:15<00:40, 7315.02it/s] 26%|██▌       | 104632/400000 [00:15<00:39, 7406.48it/s] 26%|██▋       | 105374/400000 [00:15<00:39, 7366.89it/s] 27%|██▋       | 106112/400000 [00:15<00:40, 7287.30it/s] 27%|██▋       | 106878/400000 [00:15<00:39, 7394.55it/s] 27%|██▋       | 107630/400000 [00:15<00:39, 7431.31it/s] 27%|██▋       | 108375/400000 [00:15<00:39, 7434.49it/s] 27%|██▋       | 109135/400000 [00:16<00:38, 7480.22it/s] 27%|██▋       | 109884/400000 [00:16<00:38, 7468.99it/s] 28%|██▊       | 110632/400000 [00:16<00:39, 7312.49it/s] 28%|██▊       | 111395/400000 [00:16<00:38, 7402.64it/s] 28%|██▊       | 112148/400000 [00:16<00:38, 7438.01it/s] 28%|██▊       | 112893/400000 [00:16<00:38, 7412.67it/s] 28%|██▊       | 113654/400000 [00:16<00:38, 7468.49it/s] 29%|██▊       | 114419/400000 [00:16<00:37, 7519.93it/s] 29%|██▉       | 115183/400000 [00:16<00:37, 7554.14it/s] 29%|██▉       | 115946/400000 [00:16<00:37, 7576.21it/s] 29%|██▉       | 116704/400000 [00:17<00:38, 7326.63it/s] 29%|██▉       | 117447/400000 [00:17<00:38, 7314.41it/s] 30%|██▉       | 118180/400000 [00:17<00:39, 7175.38it/s] 30%|██▉       | 118916/400000 [00:17<00:38, 7227.59it/s] 30%|██▉       | 119666/400000 [00:17<00:38, 7305.41it/s] 30%|███       | 120409/400000 [00:17<00:38, 7340.95it/s] 30%|███       | 121153/400000 [00:17<00:37, 7369.74it/s] 30%|███       | 121898/400000 [00:17<00:37, 7390.88it/s] 31%|███       | 122667/400000 [00:17<00:37, 7476.05it/s] 31%|███       | 123434/400000 [00:18<00:36, 7530.43it/s] 31%|███       | 124188/400000 [00:18<00:36, 7516.99it/s] 31%|███       | 124941/400000 [00:18<00:36, 7448.99it/s] 31%|███▏      | 125704/400000 [00:18<00:36, 7501.49it/s] 32%|███▏      | 126464/400000 [00:18<00:36, 7529.99it/s] 32%|███▏      | 127233/400000 [00:18<00:36, 7575.85it/s] 32%|███▏      | 127991/400000 [00:18<00:35, 7563.24it/s] 32%|███▏      | 128748/400000 [00:18<00:36, 7463.03it/s] 32%|███▏      | 129495/400000 [00:18<00:36, 7407.48it/s] 33%|███▎      | 130237/400000 [00:18<00:36, 7298.73it/s] 33%|███▎      | 130968/400000 [00:19<00:37, 7228.42it/s] 33%|███▎      | 131716/400000 [00:19<00:36, 7301.08it/s] 33%|███▎      | 132474/400000 [00:19<00:36, 7380.12it/s] 33%|███▎      | 133243/400000 [00:19<00:35, 7467.92it/s] 33%|███▎      | 133991/400000 [00:19<00:35, 7432.23it/s] 34%|███▎      | 134754/400000 [00:19<00:35, 7487.80it/s] 34%|███▍      | 135504/400000 [00:19<00:35, 7435.66it/s] 34%|███▍      | 136267/400000 [00:19<00:35, 7491.60it/s] 34%|███▍      | 137031/400000 [00:19<00:34, 7533.24it/s] 34%|███▍      | 137785/400000 [00:19<00:35, 7445.59it/s] 35%|███▍      | 138531/400000 [00:20<00:35, 7440.10it/s] 35%|███▍      | 139276/400000 [00:20<00:35, 7394.78it/s] 35%|███▌      | 140044/400000 [00:20<00:34, 7476.84it/s] 35%|███▌      | 140809/400000 [00:20<00:34, 7527.48it/s] 35%|███▌      | 141579/400000 [00:20<00:34, 7576.01it/s] 36%|███▌      | 142337/400000 [00:20<00:34, 7460.82it/s] 36%|███▌      | 143084/400000 [00:20<00:34, 7428.82it/s] 36%|███▌      | 143828/400000 [00:20<00:34, 7417.74it/s] 36%|███▌      | 144592/400000 [00:20<00:34, 7482.95it/s] 36%|███▋      | 145357/400000 [00:20<00:33, 7531.87it/s] 37%|███▋      | 146122/400000 [00:21<00:33, 7565.20it/s] 37%|███▋      | 146879/400000 [00:21<00:33, 7460.46it/s] 37%|███▋      | 147626/400000 [00:21<00:34, 7295.26it/s] 37%|███▋      | 148392/400000 [00:21<00:33, 7400.94it/s] 37%|███▋      | 149160/400000 [00:21<00:33, 7481.53it/s] 37%|███▋      | 149927/400000 [00:21<00:33, 7535.44it/s] 38%|███▊      | 150682/400000 [00:21<00:33, 7461.71it/s] 38%|███▊      | 151429/400000 [00:21<00:33, 7410.27it/s] 38%|███▊      | 152181/400000 [00:21<00:33, 7441.03it/s] 38%|███▊      | 152950/400000 [00:21<00:32, 7513.79it/s] 38%|███▊      | 153721/400000 [00:22<00:32, 7571.25it/s] 39%|███▊      | 154479/400000 [00:22<00:32, 7551.32it/s] 39%|███▉      | 155235/400000 [00:22<00:32, 7433.62it/s] 39%|███▉      | 155980/400000 [00:22<00:34, 7167.86it/s] 39%|███▉      | 156744/400000 [00:22<00:33, 7302.51it/s] 39%|███▉      | 157502/400000 [00:22<00:32, 7383.25it/s] 40%|███▉      | 158255/400000 [00:22<00:32, 7423.78it/s] 40%|███▉      | 159021/400000 [00:22<00:32, 7483.00it/s] 40%|███▉      | 159771/400000 [00:22<00:33, 7274.21it/s] 40%|████      | 160540/400000 [00:23<00:32, 7391.79it/s] 40%|████      | 161309/400000 [00:23<00:31, 7476.40it/s] 41%|████      | 162059/400000 [00:23<00:31, 7461.60it/s] 41%|████      | 162825/400000 [00:23<00:31, 7518.26it/s] 41%|████      | 163580/400000 [00:23<00:31, 7525.56it/s] 41%|████      | 164334/400000 [00:23<00:31, 7505.04it/s] 41%|████▏     | 165095/400000 [00:23<00:31, 7534.96it/s] 41%|████▏     | 165849/400000 [00:23<00:31, 7494.84it/s] 42%|████▏     | 166607/400000 [00:23<00:31, 7517.92it/s] 42%|████▏     | 167369/400000 [00:23<00:30, 7546.53it/s] 42%|████▏     | 168124/400000 [00:24<00:31, 7261.70it/s] 42%|████▏     | 168873/400000 [00:24<00:31, 7325.87it/s] 42%|████▏     | 169613/400000 [00:24<00:31, 7347.39it/s] 43%|████▎     | 170376/400000 [00:24<00:30, 7429.58it/s] 43%|████▎     | 171121/400000 [00:24<00:30, 7383.95it/s] 43%|████▎     | 171885/400000 [00:24<00:30, 7457.88it/s] 43%|████▎     | 172632/400000 [00:24<00:30, 7352.74it/s] 43%|████▎     | 173370/400000 [00:24<00:30, 7358.05it/s] 44%|████▎     | 174107/400000 [00:24<00:30, 7331.06it/s] 44%|████▎     | 174864/400000 [00:24<00:30, 7400.77it/s] 44%|████▍     | 175626/400000 [00:25<00:30, 7463.76it/s] 44%|████▍     | 176378/400000 [00:25<00:29, 7478.94it/s] 44%|████▍     | 177127/400000 [00:25<00:30, 7392.45it/s] 44%|████▍     | 177875/400000 [00:25<00:29, 7417.88it/s] 45%|████▍     | 178621/400000 [00:25<00:29, 7429.06it/s] 45%|████▍     | 179386/400000 [00:25<00:29, 7491.44it/s] 45%|████▌     | 180138/400000 [00:25<00:29, 7497.67it/s] 45%|████▌     | 180888/400000 [00:25<00:29, 7381.50it/s] 45%|████▌     | 181649/400000 [00:25<00:29, 7448.11it/s] 46%|████▌     | 182414/400000 [00:25<00:28, 7505.12it/s] 46%|████▌     | 183179/400000 [00:26<00:28, 7546.52it/s] 46%|████▌     | 183945/400000 [00:26<00:28, 7578.75it/s] 46%|████▌     | 184704/400000 [00:26<00:29, 7387.76it/s] 46%|████▋     | 185452/400000 [00:26<00:28, 7412.37it/s] 47%|████▋     | 186218/400000 [00:26<00:28, 7482.16it/s] 47%|████▋     | 186981/400000 [00:26<00:28, 7524.10it/s] 47%|████▋     | 187746/400000 [00:26<00:28, 7558.44it/s] 47%|████▋     | 188503/400000 [00:26<00:28, 7412.40it/s] 47%|████▋     | 189246/400000 [00:26<00:28, 7400.56it/s] 48%|████▊     | 190011/400000 [00:26<00:28, 7473.06it/s] 48%|████▊     | 190777/400000 [00:27<00:27, 7526.69it/s] 48%|████▊     | 191544/400000 [00:27<00:27, 7567.72it/s] 48%|████▊     | 192302/400000 [00:27<00:27, 7511.25it/s] 48%|████▊     | 193063/400000 [00:27<00:27, 7538.20it/s] 48%|████▊     | 193818/400000 [00:27<00:27, 7489.98it/s] 49%|████▊     | 194579/400000 [00:27<00:27, 7523.40it/s] 49%|████▉     | 195332/400000 [00:27<00:27, 7514.28it/s] 49%|████▉     | 196084/400000 [00:27<00:27, 7487.25it/s] 49%|████▉     | 196841/400000 [00:27<00:27, 7511.06it/s] 49%|████▉     | 197603/400000 [00:27<00:26, 7540.34it/s] 50%|████▉     | 198358/400000 [00:28<00:26, 7492.66it/s] 50%|████▉     | 199108/400000 [00:28<00:26, 7459.15it/s] 50%|████▉     | 199861/400000 [00:28<00:26, 7479.63it/s] 50%|█████     | 200630/400000 [00:28<00:26, 7540.81it/s] 50%|█████     | 201394/400000 [00:28<00:26, 7568.69it/s] 51%|█████     | 202152/400000 [00:28<00:26, 7516.06it/s] 51%|█████     | 202914/400000 [00:28<00:26, 7546.10it/s] 51%|█████     | 203669/400000 [00:28<00:26, 7528.51it/s] 51%|█████     | 204426/400000 [00:28<00:25, 7540.09it/s] 51%|█████▏    | 205181/400000 [00:28<00:26, 7434.57it/s] 51%|█████▏    | 205943/400000 [00:29<00:25, 7487.94it/s] 52%|█████▏    | 206693/400000 [00:29<00:26, 7372.26it/s] 52%|█████▏    | 207433/400000 [00:29<00:26, 7378.28it/s] 52%|█████▏    | 208202/400000 [00:29<00:25, 7467.32it/s] 52%|█████▏    | 208967/400000 [00:29<00:25, 7521.18it/s] 52%|█████▏    | 209729/400000 [00:29<00:25, 7549.05it/s] 53%|█████▎    | 210485/400000 [00:29<00:25, 7538.10it/s] 53%|█████▎    | 211240/400000 [00:29<00:25, 7389.00it/s] 53%|█████▎    | 211998/400000 [00:29<00:25, 7444.68it/s] 53%|█████▎    | 212760/400000 [00:29<00:24, 7496.12it/s] 53%|█████▎    | 213522/400000 [00:30<00:24, 7529.70it/s] 54%|█████▎    | 214281/400000 [00:30<00:24, 7545.09it/s] 54%|█████▍    | 215036/400000 [00:30<00:24, 7490.07it/s] 54%|█████▍    | 215797/400000 [00:30<00:24, 7524.70it/s] 54%|█████▍    | 216553/400000 [00:30<00:24, 7534.63it/s] 54%|█████▍    | 217316/400000 [00:30<00:24, 7560.17it/s] 55%|█████▍    | 218073/400000 [00:30<00:24, 7423.29it/s] 55%|█████▍    | 218819/400000 [00:30<00:24, 7433.08it/s] 55%|█████▍    | 219563/400000 [00:30<00:24, 7394.32it/s] 55%|█████▌    | 220326/400000 [00:30<00:24, 7460.78it/s] 55%|█████▌    | 221089/400000 [00:31<00:23, 7510.54it/s] 55%|█████▌    | 221841/400000 [00:31<00:24, 7401.77it/s] 56%|█████▌    | 222585/400000 [00:31<00:23, 7410.79it/s] 56%|█████▌    | 223349/400000 [00:31<00:23, 7477.01it/s] 56%|█████▌    | 224098/400000 [00:31<00:23, 7439.41it/s] 56%|█████▌    | 224860/400000 [00:31<00:23, 7489.99it/s] 56%|█████▋    | 225610/400000 [00:31<00:23, 7314.45it/s] 57%|█████▋    | 226358/400000 [00:31<00:23, 7360.87it/s] 57%|█████▋    | 227122/400000 [00:31<00:23, 7441.70it/s] 57%|█████▋    | 227867/400000 [00:32<00:23, 7411.97it/s] 57%|█████▋    | 228614/400000 [00:32<00:23, 7427.75it/s] 57%|█████▋    | 229381/400000 [00:32<00:22, 7497.61it/s] 58%|█████▊    | 230132/400000 [00:32<00:22, 7463.95it/s] 58%|█████▊    | 230879/400000 [00:32<00:22, 7409.71it/s] 58%|█████▊    | 231643/400000 [00:32<00:22, 7476.70it/s] 58%|█████▊    | 232392/400000 [00:32<00:22, 7425.42it/s] 58%|█████▊    | 233135/400000 [00:32<00:22, 7376.27it/s] 58%|█████▊    | 233873/400000 [00:32<00:22, 7318.40it/s] 59%|█████▊    | 234627/400000 [00:32<00:22, 7382.90it/s] 59%|█████▉    | 235380/400000 [00:33<00:22, 7425.18it/s] 59%|█████▉    | 236138/400000 [00:33<00:21, 7468.84it/s] 59%|█████▉    | 236886/400000 [00:33<00:21, 7436.37it/s] 59%|█████▉    | 237630/400000 [00:33<00:21, 7391.70it/s] 60%|█████▉    | 238395/400000 [00:33<00:21, 7465.56it/s] 60%|█████▉    | 239163/400000 [00:33<00:21, 7527.23it/s] 60%|█████▉    | 239931/400000 [00:33<00:21, 7571.31it/s] 60%|██████    | 240689/400000 [00:33<00:21, 7378.13it/s] 60%|██████    | 241429/400000 [00:33<00:21, 7361.64it/s] 61%|██████    | 242190/400000 [00:33<00:21, 7433.34it/s] 61%|██████    | 242935/400000 [00:34<00:21, 7332.34it/s] 61%|██████    | 243684/400000 [00:34<00:21, 7377.60it/s] 61%|██████    | 244445/400000 [00:34<00:20, 7445.31it/s] 61%|██████▏   | 245191/400000 [00:34<00:20, 7413.03it/s] 61%|██████▏   | 245956/400000 [00:34<00:20, 7481.94it/s] 62%|██████▏   | 246711/400000 [00:34<00:20, 7500.25it/s] 62%|██████▏   | 247462/400000 [00:34<00:20, 7494.87it/s] 62%|██████▏   | 248212/400000 [00:34<00:20, 7448.74it/s] 62%|██████▏   | 248972/400000 [00:34<00:20, 7490.84it/s] 62%|██████▏   | 249722/400000 [00:34<00:20, 7485.19it/s] 63%|██████▎   | 250495/400000 [00:35<00:19, 7555.89it/s] 63%|██████▎   | 251269/400000 [00:35<00:19, 7608.52it/s] 63%|██████▎   | 252043/400000 [00:35<00:19, 7645.85it/s] 63%|██████▎   | 252808/400000 [00:35<00:19, 7597.05it/s] 63%|██████▎   | 253568/400000 [00:35<00:19, 7540.94it/s] 64%|██████▎   | 254333/400000 [00:35<00:19, 7572.11it/s] 64%|██████▍   | 255091/400000 [00:35<00:19, 7457.73it/s] 64%|██████▍   | 255838/400000 [00:35<00:19, 7298.98it/s] 64%|██████▍   | 256574/400000 [00:35<00:19, 7314.71it/s] 64%|██████▍   | 257322/400000 [00:35<00:19, 7360.82it/s] 65%|██████▍   | 258075/400000 [00:36<00:19, 7408.85it/s] 65%|██████▍   | 258843/400000 [00:36<00:18, 7485.76it/s] 65%|██████▍   | 259593/400000 [00:36<00:18, 7423.75it/s] 65%|██████▌   | 260342/400000 [00:36<00:18, 7441.36it/s] 65%|██████▌   | 261109/400000 [00:36<00:18, 7506.86it/s] 65%|██████▌   | 261870/400000 [00:36<00:18, 7535.25it/s] 66%|██████▌   | 262624/400000 [00:36<00:18, 7486.98it/s] 66%|██████▌   | 263373/400000 [00:36<00:18, 7435.62it/s] 66%|██████▌   | 264121/400000 [00:36<00:18, 7448.80it/s] 66%|██████▌   | 264875/400000 [00:36<00:18, 7474.19it/s] 66%|██████▋   | 265639/400000 [00:37<00:17, 7521.67it/s] 67%|██████▋   | 266392/400000 [00:37<00:17, 7474.09it/s] 67%|██████▋   | 267140/400000 [00:37<00:17, 7416.73it/s] 67%|██████▋   | 267882/400000 [00:37<00:17, 7394.94it/s] 67%|██████▋   | 268629/400000 [00:37<00:17, 7416.66it/s] 67%|██████▋   | 269386/400000 [00:37<00:17, 7460.69it/s] 68%|██████▊   | 270133/400000 [00:37<00:17, 7420.80it/s] 68%|██████▊   | 270876/400000 [00:37<00:17, 7412.16it/s] 68%|██████▊   | 271618/400000 [00:37<00:17, 7161.77it/s] 68%|██████▊   | 272384/400000 [00:37<00:17, 7302.22it/s] 68%|██████▊   | 273135/400000 [00:38<00:17, 7362.89it/s] 68%|██████▊   | 273898/400000 [00:38<00:16, 7439.28it/s] 69%|██████▊   | 274663/400000 [00:38<00:16, 7499.37it/s] 69%|██████▉   | 275414/400000 [00:38<00:16, 7350.51it/s] 69%|██████▉   | 276183/400000 [00:38<00:16, 7447.46it/s] 69%|██████▉   | 276939/400000 [00:38<00:16, 7479.46it/s] 69%|██████▉   | 277700/400000 [00:38<00:16, 7517.30it/s] 70%|██████▉   | 278453/400000 [00:38<00:16, 7358.13it/s] 70%|██████▉   | 279191/400000 [00:38<00:16, 7312.72it/s] 70%|██████▉   | 279924/400000 [00:39<00:16, 7241.19it/s] 70%|███████   | 280664/400000 [00:39<00:16, 7287.01it/s] 70%|███████   | 281428/400000 [00:39<00:16, 7386.85it/s] 71%|███████   | 282196/400000 [00:39<00:15, 7469.96it/s] 71%|███████   | 282953/400000 [00:39<00:15, 7498.44it/s] 71%|███████   | 283704/400000 [00:39<00:15, 7493.17it/s] 71%|███████   | 284470/400000 [00:39<00:15, 7540.68it/s] 71%|███████▏  | 285225/400000 [00:39<00:15, 7430.36it/s] 71%|███████▏  | 285991/400000 [00:39<00:15, 7495.88it/s] 72%|███████▏  | 286744/400000 [00:39<00:15, 7504.11it/s] 72%|███████▏  | 287504/400000 [00:40<00:14, 7532.45it/s] 72%|███████▏  | 288258/400000 [00:40<00:14, 7472.12it/s] 72%|███████▏  | 289026/400000 [00:40<00:14, 7530.58it/s] 72%|███████▏  | 289796/400000 [00:40<00:14, 7578.56it/s] 73%|███████▎  | 290555/400000 [00:40<00:14, 7559.41it/s] 73%|███████▎  | 291316/400000 [00:40<00:14, 7573.75it/s] 73%|███████▎  | 292074/400000 [00:40<00:14, 7505.98it/s] 73%|███████▎  | 292825/400000 [00:40<00:14, 7435.87it/s] 73%|███████▎  | 293569/400000 [00:40<00:14, 7370.09it/s] 74%|███████▎  | 294323/400000 [00:40<00:14, 7418.89it/s] 74%|███████▍  | 295086/400000 [00:41<00:14, 7478.84it/s] 74%|███████▍  | 295854/400000 [00:41<00:13, 7535.16it/s] 74%|███████▍  | 296608/400000 [00:41<00:13, 7394.46it/s] 74%|███████▍  | 297375/400000 [00:41<00:13, 7472.50it/s] 75%|███████▍  | 298124/400000 [00:41<00:13, 7362.24it/s] 75%|███████▍  | 298891/400000 [00:41<00:13, 7451.17it/s] 75%|███████▍  | 299658/400000 [00:41<00:13, 7512.64it/s] 75%|███████▌  | 300411/400000 [00:41<00:13, 7408.88it/s] 75%|███████▌  | 301170/400000 [00:41<00:13, 7461.70it/s] 75%|███████▌  | 301920/400000 [00:41<00:13, 7471.16it/s] 76%|███████▌  | 302679/400000 [00:42<00:12, 7505.18it/s] 76%|███████▌  | 303445/400000 [00:42<00:12, 7549.64it/s] 76%|███████▌  | 304213/400000 [00:42<00:12, 7587.02it/s] 76%|███████▌  | 304972/400000 [00:42<00:12, 7474.12it/s] 76%|███████▋  | 305720/400000 [00:42<00:12, 7453.64it/s] 77%|███████▋  | 306484/400000 [00:42<00:12, 7507.22it/s] 77%|███████▋  | 307243/400000 [00:42<00:12, 7529.24it/s] 77%|███████▋  | 308010/400000 [00:42<00:12, 7569.65it/s] 77%|███████▋  | 308772/400000 [00:42<00:12, 7581.88it/s] 77%|███████▋  | 309531/400000 [00:42<00:12, 7457.75it/s] 78%|███████▊  | 310296/400000 [00:43<00:11, 7513.71it/s] 78%|███████▊  | 311054/400000 [00:43<00:11, 7531.23it/s] 78%|███████▊  | 311820/400000 [00:43<00:11, 7568.46it/s] 78%|███████▊  | 312584/400000 [00:43<00:11, 7587.37it/s] 78%|███████▊  | 313343/400000 [00:43<00:11, 7464.29it/s] 79%|███████▊  | 314113/400000 [00:43<00:11, 7531.42it/s] 79%|███████▊  | 314877/400000 [00:43<00:11, 7562.15it/s] 79%|███████▉  | 315634/400000 [00:43<00:11, 7505.54it/s] 79%|███████▉  | 316387/400000 [00:43<00:11, 7511.70it/s] 79%|███████▉  | 317139/400000 [00:43<00:11, 7302.13it/s] 79%|███████▉  | 317884/400000 [00:44<00:11, 7345.83it/s] 80%|███████▉  | 318646/400000 [00:44<00:10, 7423.83it/s] 80%|███████▉  | 319390/400000 [00:44<00:10, 7419.03it/s] 80%|████████  | 320133/400000 [00:44<00:10, 7321.68it/s] 80%|████████  | 320866/400000 [00:44<00:10, 7305.47it/s] 80%|████████  | 321598/400000 [00:44<00:10, 7249.78it/s] 81%|████████  | 322347/400000 [00:44<00:10, 7319.88it/s] 81%|████████  | 323116/400000 [00:44<00:10, 7424.90it/s] 81%|████████  | 323874/400000 [00:44<00:10, 7470.73it/s] 81%|████████  | 324628/400000 [00:44<00:10, 7489.94it/s] 81%|████████▏ | 325390/400000 [00:45<00:09, 7527.51it/s] 82%|████████▏ | 326144/400000 [00:45<00:09, 7530.59it/s] 82%|████████▏ | 326913/400000 [00:45<00:09, 7575.12it/s] 82%|████████▏ | 327680/400000 [00:45<00:09, 7601.69it/s] 82%|████████▏ | 328441/400000 [00:45<00:09, 7579.57it/s] 82%|████████▏ | 329200/400000 [00:45<00:09, 7572.71it/s] 82%|████████▏ | 329958/400000 [00:45<00:09, 7476.70it/s] 83%|████████▎ | 330707/400000 [00:45<00:09, 7427.90it/s] 83%|████████▎ | 331451/400000 [00:45<00:09, 7408.28it/s] 83%|████████▎ | 332193/400000 [00:46<00:09, 7357.60it/s] 83%|████████▎ | 332956/400000 [00:46<00:09, 7435.93it/s] 83%|████████▎ | 333720/400000 [00:46<00:08, 7495.99it/s] 84%|████████▎ | 334470/400000 [00:46<00:08, 7461.60it/s] 84%|████████▍ | 335239/400000 [00:46<00:08, 7526.78it/s] 84%|████████▍ | 335994/400000 [00:46<00:08, 7532.02it/s] 84%|████████▍ | 336765/400000 [00:46<00:08, 7581.89it/s] 84%|████████▍ | 337529/400000 [00:46<00:08, 7598.38it/s] 85%|████████▍ | 338291/400000 [00:46<00:08, 7604.13it/s] 85%|████████▍ | 339052/400000 [00:46<00:08, 7557.80it/s] 85%|████████▍ | 339808/400000 [00:47<00:08, 7502.94it/s] 85%|████████▌ | 340559/400000 [00:47<00:07, 7495.99it/s] 85%|████████▌ | 341310/400000 [00:47<00:07, 7497.41it/s] 86%|████████▌ | 342063/400000 [00:47<00:07, 7507.12it/s] 86%|████████▌ | 342833/400000 [00:47<00:07, 7562.93it/s] 86%|████████▌ | 343590/400000 [00:47<00:07, 7438.07it/s] 86%|████████▌ | 344352/400000 [00:47<00:07, 7488.88it/s] 86%|████████▋ | 345102/400000 [00:47<00:07, 7328.75it/s] 86%|████████▋ | 345843/400000 [00:47<00:07, 7351.31it/s] 87%|████████▋ | 346608/400000 [00:47<00:07, 7435.72it/s] 87%|████████▋ | 347353/400000 [00:48<00:07, 7363.59it/s] 87%|████████▋ | 348092/400000 [00:48<00:07, 7369.68it/s] 87%|████████▋ | 348850/400000 [00:48<00:06, 7430.28it/s] 87%|████████▋ | 349594/400000 [00:48<00:06, 7375.00it/s] 88%|████████▊ | 350359/400000 [00:48<00:06, 7454.01it/s] 88%|████████▊ | 351114/400000 [00:48<00:06, 7480.29it/s] 88%|████████▊ | 351863/400000 [00:48<00:06, 7364.91it/s] 88%|████████▊ | 352607/400000 [00:48<00:06, 7386.00it/s] 88%|████████▊ | 353376/400000 [00:48<00:06, 7473.60it/s] 89%|████████▊ | 354141/400000 [00:48<00:06, 7524.29it/s] 89%|████████▊ | 354903/400000 [00:49<00:05, 7552.43it/s] 89%|████████▉ | 355659/400000 [00:49<00:05, 7482.27it/s] 89%|████████▉ | 356409/400000 [00:49<00:05, 7487.18it/s] 89%|████████▉ | 357159/400000 [00:49<00:05, 7428.12it/s] 89%|████████▉ | 357916/400000 [00:49<00:05, 7467.42it/s] 90%|████████▉ | 358676/400000 [00:49<00:05, 7506.21it/s] 90%|████████▉ | 359439/400000 [00:49<00:05, 7540.59it/s] 90%|█████████ | 360194/400000 [00:49<00:05, 7311.35it/s] 90%|█████████ | 360953/400000 [00:49<00:05, 7390.35it/s] 90%|█████████ | 361710/400000 [00:49<00:05, 7441.46it/s] 91%|█████████ | 362463/400000 [00:50<00:05, 7467.09it/s] 91%|█████████ | 363216/400000 [00:50<00:04, 7483.02it/s] 91%|█████████ | 363968/400000 [00:50<00:04, 7492.89it/s] 91%|█████████ | 364718/400000 [00:50<00:04, 7438.51it/s] 91%|█████████▏| 365483/400000 [00:50<00:04, 7498.87it/s] 92%|█████████▏| 366245/400000 [00:50<00:04, 7534.39it/s] 92%|█████████▏| 366999/400000 [00:50<00:04, 7485.66it/s] 92%|█████████▏| 367748/400000 [00:50<00:04, 7484.89it/s] 92%|█████████▏| 368497/400000 [00:50<00:04, 7457.29it/s] 92%|█████████▏| 369243/400000 [00:50<00:04, 7167.52it/s] 92%|█████████▏| 369994/400000 [00:51<00:04, 7266.42it/s] 93%|█████████▎| 370731/400000 [00:51<00:04, 7296.16it/s] 93%|█████████▎| 371481/400000 [00:51<00:03, 7353.59it/s] 93%|█████████▎| 372243/400000 [00:51<00:03, 7431.37it/s] 93%|█████████▎| 372994/400000 [00:51<00:03, 7452.37it/s] 93%|█████████▎| 373756/400000 [00:51<00:03, 7500.15it/s] 94%|█████████▎| 374507/400000 [00:51<00:03, 7499.51it/s] 94%|█████████▍| 375261/400000 [00:51<00:03, 7509.37it/s] 94%|█████████▍| 376016/400000 [00:51<00:03, 7520.19it/s] 94%|█████████▍| 376769/400000 [00:51<00:03, 7428.22it/s] 94%|█████████▍| 377514/400000 [00:52<00:03, 7433.73it/s] 95%|█████████▍| 378264/400000 [00:52<00:02, 7452.23it/s] 95%|█████████▍| 379034/400000 [00:52<00:02, 7524.70it/s] 95%|█████████▍| 379804/400000 [00:52<00:02, 7575.81it/s] 95%|█████████▌| 380567/400000 [00:52<00:02, 7590.55it/s] 95%|█████████▌| 381327/400000 [00:52<00:02, 7504.94it/s] 96%|█████████▌| 382078/400000 [00:52<00:02, 7318.54it/s] 96%|█████████▌| 382837/400000 [00:52<00:02, 7396.77it/s] 96%|█████████▌| 383578/400000 [00:52<00:02, 7317.01it/s] 96%|█████████▌| 384338/400000 [00:52<00:02, 7397.42it/s] 96%|█████████▋| 385095/400000 [00:53<00:02, 7446.88it/s] 96%|█████████▋| 385841/400000 [00:53<00:01, 7353.39it/s] 97%|█████████▋| 386592/400000 [00:53<00:01, 7399.05it/s] 97%|█████████▋| 387356/400000 [00:53<00:01, 7466.88it/s] 97%|█████████▋| 388113/400000 [00:53<00:01, 7495.29it/s] 97%|█████████▋| 388863/400000 [00:53<00:01, 7443.08it/s] 97%|█████████▋| 389613/400000 [00:53<00:01, 7458.63it/s] 98%|█████████▊| 390360/400000 [00:53<00:01, 7458.87it/s] 98%|█████████▊| 391107/400000 [00:53<00:01, 7457.00it/s] 98%|█████████▊| 391869/400000 [00:54<00:01, 7502.59it/s] 98%|█████████▊| 392620/400000 [00:54<00:00, 7503.93it/s] 98%|█████████▊| 393371/400000 [00:54<00:00, 7468.74it/s] 99%|█████████▊| 394118/400000 [00:54<00:00, 7352.63it/s] 99%|█████████▊| 394872/400000 [00:54<00:00, 7405.74it/s] 99%|█████████▉| 395615/400000 [00:54<00:00, 7410.98it/s] 99%|█████████▉| 396357/400000 [00:54<00:00, 7403.40it/s] 99%|█████████▉| 397109/400000 [00:54<00:00, 7429.81it/s] 99%|█████████▉| 397871/400000 [00:54<00:00, 7485.65it/s]100%|█████████▉| 398620/400000 [00:54<00:00, 7411.88it/s]100%|█████████▉| 399387/400000 [00:55<00:00, 7485.93it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7259.83it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f4384b020f0>, <torchtext.data.dataset.TabularDataset object at 0x7f4384b02240>, <torchtext.vocab.Vocab object at 0x7f4384b02160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.67 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.67 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.67 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.23 file/s]2020-06-13 00:19:02.585554: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 00:19:02.589965: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-13 00:19:02.590490: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56202890b670 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 00:19:02.590816: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3768320/9912422 [00:00<00:00, 35178479.23it/s]9920512it [00:00, 34309421.66it/s]                             
0it [00:00, ?it/s]32768it [00:00, 509399.17it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 460456.69it/s]1654784it [00:00, 10425398.20it/s]                         
0it [00:00, ?it/s]8192it [00:00, 131938.69it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
