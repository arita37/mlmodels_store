
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f619bc40b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f619bc40b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f62068b0510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f62068b0510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6225addea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6225addea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f61b3bdea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f61b3bdea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f61b3bdea60> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 21%|██        | 2097152/9912422 [00:00<00:00, 20663579.12it/s] 77%|███████▋  | 7618560/9912422 [00:00<00:00, 25403632.08it/s]9920512it [00:00, 26315846.73it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1305436.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 463286.74it/s]1654784it [00:00, 10861101.04it/s]                         
0it [00:00, ?it/s]8192it [00:00, 250072.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f619ae806d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f619ae8e978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f61b3bde6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f61b3bde6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f61b3bde6a8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:40,  1.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:40,  1.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:15,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:55,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:01<00:41,  3.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:41,  3.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:41,  3.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:31,  4.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:31,  4.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:31,  4.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:31,  4.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:23,  6.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:23,  6.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:23,  6.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:23,  6.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:17,  8.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:17,  8.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:17,  8.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 10.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:13, 10.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:13, 10.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:10, 13.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:10, 13.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:10, 13.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:10, 13.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 15.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:08, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:08, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:08, 15.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:07, 18.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:07, 18.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:07, 18.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:07, 18.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:07, 18.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:02<00:05, 22.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:02<00:04, 27.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:04, 27.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:02<00:03, 32.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:03, 32.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:02<00:02, 38.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 44.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 44.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:02<00:01, 48.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:01, 48.52 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 52.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 52.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 55.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 55.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 58.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 58.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 59.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 59.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:00, 61.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:00, 61.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 61.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 62.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 62.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 63.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 63.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 63.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 63.03 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 60.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 60.97 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 57.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 57.33 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 54.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 54.75 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 52.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 52.47 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 52.47 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 52.47 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 52.47 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.40 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ url]
0 examples [00:00, ? examples/s]2020-08-06 00:09:29.716679: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-06 00:09:29.731731: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-06 00:09:29.732010: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562079b233f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-06 00:09:29.732032: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.65 examples/s]92 examples [00:00,  5.20 examples/s]185 examples [00:00,  7.42 examples/s]279 examples [00:00, 10.56 examples/s]374 examples [00:00, 15.01 examples/s]468 examples [00:00, 21.30 examples/s]561 examples [00:00, 30.13 examples/s]656 examples [00:00, 42.46 examples/s]747 examples [00:01, 59.46 examples/s]840 examples [00:01, 82.67 examples/s]929 examples [00:01, 113.49 examples/s]1020 examples [00:01, 153.89 examples/s]1110 examples [00:01, 204.53 examples/s]1199 examples [00:01, 265.97 examples/s]1288 examples [00:01, 336.60 examples/s]1382 examples [00:01, 416.65 examples/s]1475 examples [00:01, 499.08 examples/s]1566 examples [00:01, 575.39 examples/s]1660 examples [00:02, 650.74 examples/s]1752 examples [00:02, 711.90 examples/s]1844 examples [00:02, 761.27 examples/s]1938 examples [00:02, 806.16 examples/s]2030 examples [00:02, 835.40 examples/s]2122 examples [00:02, 855.99 examples/s]2215 examples [00:02, 876.54 examples/s]2307 examples [00:02, 887.34 examples/s]2400 examples [00:02, 897.40 examples/s]2493 examples [00:02, 904.93 examples/s]2585 examples [00:03, 903.09 examples/s]2677 examples [00:03, 867.50 examples/s]2771 examples [00:03, 887.49 examples/s]2867 examples [00:03, 907.21 examples/s]2964 examples [00:03, 922.67 examples/s]3057 examples [00:03, 922.25 examples/s]3153 examples [00:03, 932.49 examples/s]3249 examples [00:03, 938.41 examples/s]3344 examples [00:03, 941.47 examples/s]3439 examples [00:04, 942.85 examples/s]3534 examples [00:04, 940.87 examples/s]3631 examples [00:04, 947.81 examples/s]3727 examples [00:04, 950.94 examples/s]3823 examples [00:04, 949.09 examples/s]3919 examples [00:04, 951.41 examples/s]4015 examples [00:04, 942.62 examples/s]4111 examples [00:04, 946.57 examples/s]4209 examples [00:04, 955.41 examples/s]4305 examples [00:04, 944.29 examples/s]4400 examples [00:05, 939.05 examples/s]4494 examples [00:05, 917.18 examples/s]4586 examples [00:05, 917.40 examples/s]4678 examples [00:05, 915.57 examples/s]4771 examples [00:05, 919.70 examples/s]4864 examples [00:05, 922.75 examples/s]4957 examples [00:05, 913.72 examples/s]5050 examples [00:05, 916.85 examples/s]5142 examples [00:05, 913.98 examples/s]5241 examples [00:05, 934.79 examples/s]5338 examples [00:06, 942.33 examples/s]5433 examples [00:06, 933.24 examples/s]5527 examples [00:06, 930.95 examples/s]5622 examples [00:06, 934.01 examples/s]5717 examples [00:06, 938.57 examples/s]5811 examples [00:06, 918.39 examples/s]5903 examples [00:06, 909.16 examples/s]5996 examples [00:06, 912.64 examples/s]6090 examples [00:06, 918.09 examples/s]6182 examples [00:06, 917.39 examples/s]6275 examples [00:07, 918.85 examples/s]6367 examples [00:07, 899.39 examples/s]6458 examples [00:07, 890.86 examples/s]6552 examples [00:07, 904.87 examples/s]6645 examples [00:07, 910.93 examples/s]6738 examples [00:07, 914.97 examples/s]6830 examples [00:07, 901.30 examples/s]6921 examples [00:07, 903.81 examples/s]7016 examples [00:07, 914.84 examples/s]7110 examples [00:07, 919.94 examples/s]7203 examples [00:08, 917.58 examples/s]7295 examples [00:08, 907.84 examples/s]7387 examples [00:08, 911.07 examples/s]7479 examples [00:08, 898.11 examples/s]7575 examples [00:08, 913.45 examples/s]7667 examples [00:08, 909.34 examples/s]7759 examples [00:08, 907.98 examples/s]7853 examples [00:08, 914.87 examples/s]7945 examples [00:08, 916.30 examples/s]8039 examples [00:09, 922.68 examples/s]8137 examples [00:09, 938.10 examples/s]8231 examples [00:09, 914.83 examples/s]8323 examples [00:09, 912.48 examples/s]8420 examples [00:09, 927.84 examples/s]8513 examples [00:09, 928.19 examples/s]8606 examples [00:09, 925.54 examples/s]8700 examples [00:09, 927.03 examples/s]8793 examples [00:09, 926.53 examples/s]8890 examples [00:09, 937.09 examples/s]8984 examples [00:10, 936.41 examples/s]9082 examples [00:10, 946.47 examples/s]9177 examples [00:10, 927.02 examples/s]9274 examples [00:10, 938.10 examples/s]9369 examples [00:10, 939.26 examples/s]9464 examples [00:10, 931.60 examples/s]9558 examples [00:10, 921.60 examples/s]9651 examples [00:10, 920.13 examples/s]9744 examples [00:10, 922.97 examples/s]9841 examples [00:10, 933.93 examples/s]9935 examples [00:11, 935.29 examples/s]10029 examples [00:11, 871.92 examples/s]10125 examples [00:11, 895.18 examples/s]10220 examples [00:11, 910.76 examples/s]10312 examples [00:11, 896.84 examples/s]10407 examples [00:11, 910.04 examples/s]10505 examples [00:11, 927.15 examples/s]10601 examples [00:11, 935.08 examples/s]10696 examples [00:11, 939.23 examples/s]10794 examples [00:11, 948.34 examples/s]10891 examples [00:12, 951.82 examples/s]10989 examples [00:12, 956.49 examples/s]11085 examples [00:12, 928.99 examples/s]11183 examples [00:12, 940.89 examples/s]11279 examples [00:12, 945.83 examples/s]11374 examples [00:12, 934.81 examples/s]11470 examples [00:12, 940.14 examples/s]11565 examples [00:12, 940.52 examples/s]11660 examples [00:12, 921.87 examples/s]11754 examples [00:13, 925.50 examples/s]11847 examples [00:13, 918.63 examples/s]11939 examples [00:13, 912.83 examples/s]12031 examples [00:13, 909.86 examples/s]12123 examples [00:13, 907.44 examples/s]12214 examples [00:13, 907.62 examples/s]12305 examples [00:13, 899.83 examples/s]12396 examples [00:13, 901.82 examples/s]12487 examples [00:13, 888.77 examples/s]12578 examples [00:13, 892.59 examples/s]12670 examples [00:14, 899.80 examples/s]12761 examples [00:14, 899.98 examples/s]12852 examples [00:14, 901.93 examples/s]12944 examples [00:14, 907.15 examples/s]13039 examples [00:14, 917.86 examples/s]13131 examples [00:14, 914.71 examples/s]13223 examples [00:14, 908.15 examples/s]13316 examples [00:14, 912.42 examples/s]13408 examples [00:14, 910.64 examples/s]13500 examples [00:14, 910.35 examples/s]13593 examples [00:15, 915.48 examples/s]13685 examples [00:15, 915.10 examples/s]13777 examples [00:15, 901.36 examples/s]13868 examples [00:15, 882.72 examples/s]13960 examples [00:15, 892.60 examples/s]14050 examples [00:15, 892.82 examples/s]14142 examples [00:15, 898.01 examples/s]14232 examples [00:15, 894.38 examples/s]14324 examples [00:15, 900.00 examples/s]14415 examples [00:15, 901.32 examples/s]14507 examples [00:16, 906.79 examples/s]14599 examples [00:16, 910.36 examples/s]14693 examples [00:16, 916.12 examples/s]14785 examples [00:16, 894.34 examples/s]14877 examples [00:16, 901.44 examples/s]14968 examples [00:16, 899.60 examples/s]15061 examples [00:16, 905.67 examples/s]15152 examples [00:16, 904.29 examples/s]15243 examples [00:16, 890.27 examples/s]15335 examples [00:16, 898.49 examples/s]15427 examples [00:17, 904.27 examples/s]15522 examples [00:17, 916.62 examples/s]15616 examples [00:17, 921.36 examples/s]15709 examples [00:17, 917.23 examples/s]15801 examples [00:17, 917.24 examples/s]15894 examples [00:17, 918.47 examples/s]15986 examples [00:17, 914.44 examples/s]16079 examples [00:17, 918.40 examples/s]16171 examples [00:17, 914.58 examples/s]16264 examples [00:17, 918.94 examples/s]16360 examples [00:18, 928.29 examples/s]16453 examples [00:18, 917.19 examples/s]16545 examples [00:18, 889.65 examples/s]16635 examples [00:18, 883.59 examples/s]16728 examples [00:18, 895.40 examples/s]16821 examples [00:18, 902.74 examples/s]16913 examples [00:18, 906.98 examples/s]17006 examples [00:18, 910.81 examples/s]17098 examples [00:18, 907.52 examples/s]17192 examples [00:19, 916.68 examples/s]17284 examples [00:19, 915.05 examples/s]17377 examples [00:19, 916.36 examples/s]17469 examples [00:19, 914.30 examples/s]17562 examples [00:19, 917.17 examples/s]17657 examples [00:19, 924.56 examples/s]17750 examples [00:19, 924.74 examples/s]17843 examples [00:19, 918.40 examples/s]17935 examples [00:19, 897.12 examples/s]18031 examples [00:19, 914.22 examples/s]18125 examples [00:20, 919.66 examples/s]18218 examples [00:20, 922.14 examples/s]18311 examples [00:20, 922.22 examples/s]18409 examples [00:20, 935.12 examples/s]18505 examples [00:20, 940.46 examples/s]18600 examples [00:20, 934.93 examples/s]18694 examples [00:20, 929.40 examples/s]18787 examples [00:20, 918.83 examples/s]18880 examples [00:20, 921.48 examples/s]18973 examples [00:20, 920.61 examples/s]19067 examples [00:21, 923.87 examples/s]19164 examples [00:21, 936.74 examples/s]19258 examples [00:21, 937.53 examples/s]19352 examples [00:21, 930.87 examples/s]19446 examples [00:21, 924.47 examples/s]19540 examples [00:21, 926.70 examples/s]19633 examples [00:21, 923.65 examples/s]19726 examples [00:21, 914.06 examples/s]19818 examples [00:21, 910.99 examples/s]19910 examples [00:21, 898.64 examples/s]20001 examples [00:22, 852.43 examples/s]20088 examples [00:22, 855.15 examples/s]20183 examples [00:22, 879.31 examples/s]20274 examples [00:22, 886.16 examples/s]20369 examples [00:22, 903.35 examples/s]20464 examples [00:22, 914.80 examples/s]20561 examples [00:22, 928.52 examples/s]20655 examples [00:22, 924.61 examples/s]20751 examples [00:22, 934.94 examples/s]20846 examples [00:22, 939.29 examples/s]20941 examples [00:23, 941.66 examples/s]21038 examples [00:23, 949.79 examples/s]21135 examples [00:23, 952.67 examples/s]21235 examples [00:23, 965.92 examples/s]21332 examples [00:23, 957.00 examples/s]21428 examples [00:23, 953.30 examples/s]21525 examples [00:23, 955.97 examples/s]21621 examples [00:23, 953.52 examples/s]21717 examples [00:23, 953.06 examples/s]21813 examples [00:24, 943.33 examples/s]21908 examples [00:24, 943.73 examples/s]22005 examples [00:24, 951.15 examples/s]22102 examples [00:24, 954.21 examples/s]22198 examples [00:24, 946.38 examples/s]22293 examples [00:24, 942.83 examples/s]22390 examples [00:24, 950.56 examples/s]22487 examples [00:24, 955.07 examples/s]22584 examples [00:24, 956.87 examples/s]22680 examples [00:24, 954.88 examples/s]22776 examples [00:25, 943.26 examples/s]22874 examples [00:25, 950.74 examples/s]22970 examples [00:25, 949.61 examples/s]23067 examples [00:25, 953.62 examples/s]23163 examples [00:25, 950.37 examples/s]23259 examples [00:25, 936.95 examples/s]23353 examples [00:25, 916.48 examples/s]23445 examples [00:25, 913.57 examples/s]23537 examples [00:25, 907.88 examples/s]23628 examples [00:25, 906.69 examples/s]23720 examples [00:26, 907.79 examples/s]23812 examples [00:26, 911.21 examples/s]23904 examples [00:26, 907.91 examples/s]23998 examples [00:26, 914.85 examples/s]24095 examples [00:26, 930.12 examples/s]24189 examples [00:26, 916.45 examples/s]24284 examples [00:26, 925.87 examples/s]24381 examples [00:26, 937.37 examples/s]24476 examples [00:26, 938.56 examples/s]24570 examples [00:26, 938.26 examples/s]24664 examples [00:27, 936.02 examples/s]24760 examples [00:27, 942.56 examples/s]24856 examples [00:27, 947.33 examples/s]24951 examples [00:27, 945.01 examples/s]25046 examples [00:27, 935.15 examples/s]25140 examples [00:27, 932.22 examples/s]25236 examples [00:27, 939.37 examples/s]25330 examples [00:27, 936.41 examples/s]25427 examples [00:27, 945.13 examples/s]25522 examples [00:27, 944.45 examples/s]25618 examples [00:28, 947.21 examples/s]25715 examples [00:28, 946.24 examples/s]25811 examples [00:28, 948.25 examples/s]25906 examples [00:28, 924.07 examples/s]25999 examples [00:28, 922.91 examples/s]26094 examples [00:28, 930.02 examples/s]26189 examples [00:28, 933.01 examples/s]26283 examples [00:28, 924.09 examples/s]26376 examples [00:28, 921.37 examples/s]26469 examples [00:28, 916.44 examples/s]26561 examples [00:29, 916.33 examples/s]26656 examples [00:29, 923.37 examples/s]26749 examples [00:29, 922.73 examples/s]26843 examples [00:29, 927.83 examples/s]26936 examples [00:29, 917.99 examples/s]27030 examples [00:29, 923.40 examples/s]27123 examples [00:29, 922.58 examples/s]27217 examples [00:29, 927.64 examples/s]27312 examples [00:29, 931.43 examples/s]27406 examples [00:30, 920.70 examples/s]27500 examples [00:30, 925.69 examples/s]27593 examples [00:30, 921.78 examples/s]27686 examples [00:30, 921.68 examples/s]27779 examples [00:30, 921.41 examples/s]27872 examples [00:30, 907.66 examples/s]27963 examples [00:30, 900.12 examples/s]28054 examples [00:30, 892.97 examples/s]28146 examples [00:30, 900.10 examples/s]28237 examples [00:30, 899.45 examples/s]28327 examples [00:31, 896.82 examples/s]28417 examples [00:31, 894.87 examples/s]28510 examples [00:31, 903.92 examples/s]28604 examples [00:31, 914.38 examples/s]28697 examples [00:31, 917.61 examples/s]28789 examples [00:31, 898.53 examples/s]28881 examples [00:31, 904.69 examples/s]28974 examples [00:31, 910.47 examples/s]29068 examples [00:31, 917.68 examples/s]29160 examples [00:31, 904.17 examples/s]29251 examples [00:32, 898.88 examples/s]29347 examples [00:32, 915.50 examples/s]29439 examples [00:32, 898.59 examples/s]29530 examples [00:32, 898.15 examples/s]29627 examples [00:32, 916.13 examples/s]29719 examples [00:32, 915.88 examples/s]29811 examples [00:32, 916.66 examples/s]29908 examples [00:32, 930.45 examples/s]30002 examples [00:32, 883.92 examples/s]30099 examples [00:32, 905.97 examples/s]30192 examples [00:33, 910.82 examples/s]30289 examples [00:33, 925.64 examples/s]30382 examples [00:33, 901.96 examples/s]30477 examples [00:33, 915.79 examples/s]30573 examples [00:33, 927.64 examples/s]30667 examples [00:33, 915.85 examples/s]30760 examples [00:33, 919.32 examples/s]30855 examples [00:33, 927.78 examples/s]30953 examples [00:33, 940.86 examples/s]31048 examples [00:33, 943.53 examples/s]31143 examples [00:34, 938.67 examples/s]31238 examples [00:34, 939.23 examples/s]31332 examples [00:34, 935.47 examples/s]31426 examples [00:34, 932.92 examples/s]31520 examples [00:34, 934.01 examples/s]31614 examples [00:34, 928.61 examples/s]31711 examples [00:34, 938.52 examples/s]31805 examples [00:34, 936.31 examples/s]31899 examples [00:34, 937.06 examples/s]31993 examples [00:35, 924.68 examples/s]32086 examples [00:35, 912.37 examples/s]32178 examples [00:35, 910.13 examples/s]32270 examples [00:35, 907.71 examples/s]32364 examples [00:35, 915.89 examples/s]32457 examples [00:35, 919.35 examples/s]32549 examples [00:35, 907.22 examples/s]32642 examples [00:35, 913.07 examples/s]32736 examples [00:35, 920.81 examples/s]32831 examples [00:35, 929.09 examples/s]32924 examples [00:36, 920.91 examples/s]33017 examples [00:36, 920.28 examples/s]33115 examples [00:36, 935.08 examples/s]33209 examples [00:36, 934.78 examples/s]33303 examples [00:36, 933.15 examples/s]33397 examples [00:36, 925.35 examples/s]33490 examples [00:36, 915.27 examples/s]33584 examples [00:36, 920.80 examples/s]33677 examples [00:36, 922.15 examples/s]33770 examples [00:36, 921.12 examples/s]33864 examples [00:37, 926.56 examples/s]33957 examples [00:37, 922.38 examples/s]34051 examples [00:37, 925.62 examples/s]34146 examples [00:37, 930.37 examples/s]34240 examples [00:37, 923.41 examples/s]34335 examples [00:37, 930.88 examples/s]34429 examples [00:37, 927.70 examples/s]34527 examples [00:37, 941.60 examples/s]34622 examples [00:37, 942.25 examples/s]34717 examples [00:37, 928.87 examples/s]34812 examples [00:38, 932.50 examples/s]34906 examples [00:38, 922.17 examples/s]34999 examples [00:38, 913.74 examples/s]35091 examples [00:38, 905.73 examples/s]35182 examples [00:38, 903.03 examples/s]35275 examples [00:38, 908.31 examples/s]35369 examples [00:38, 916.22 examples/s]35467 examples [00:38, 932.01 examples/s]35563 examples [00:38, 938.86 examples/s]35660 examples [00:38, 944.99 examples/s]35756 examples [00:39, 946.59 examples/s]35852 examples [00:39, 947.80 examples/s]35947 examples [00:39, 945.77 examples/s]36042 examples [00:39, 938.23 examples/s]36138 examples [00:39, 943.52 examples/s]36233 examples [00:39, 939.29 examples/s]36327 examples [00:39, 937.04 examples/s]36422 examples [00:39, 938.25 examples/s]36519 examples [00:39, 945.32 examples/s]36617 examples [00:39, 954.53 examples/s]36713 examples [00:40, 952.51 examples/s]36809 examples [00:40, 941.87 examples/s]36904 examples [00:40, 932.79 examples/s]36998 examples [00:40, 928.77 examples/s]37091 examples [00:40, 918.35 examples/s]37183 examples [00:40, 914.55 examples/s]37275 examples [00:40, 898.42 examples/s]37365 examples [00:40, 886.34 examples/s]37458 examples [00:40, 897.67 examples/s]37551 examples [00:41, 905.96 examples/s]37642 examples [00:41, 890.03 examples/s]37733 examples [00:41, 894.75 examples/s]37823 examples [00:41, 895.71 examples/s]37915 examples [00:41, 901.56 examples/s]38008 examples [00:41, 908.69 examples/s]38099 examples [00:41, 885.46 examples/s]38191 examples [00:41, 892.79 examples/s]38283 examples [00:41, 900.09 examples/s]38374 examples [00:41, 886.44 examples/s]38468 examples [00:42, 899.73 examples/s]38561 examples [00:42, 907.11 examples/s]38653 examples [00:42, 909.32 examples/s]38745 examples [00:42, 908.08 examples/s]38839 examples [00:42, 916.36 examples/s]38931 examples [00:42, 912.47 examples/s]39023 examples [00:42, 910.19 examples/s]39117 examples [00:42, 916.51 examples/s]39215 examples [00:42, 933.47 examples/s]39310 examples [00:42, 938.05 examples/s]39404 examples [00:43, 846.43 examples/s]39491 examples [00:43, 805.17 examples/s]39585 examples [00:43, 840.99 examples/s]39683 examples [00:43, 876.98 examples/s]39779 examples [00:43, 898.45 examples/s]39874 examples [00:43, 912.14 examples/s]39967 examples [00:43, 910.53 examples/s]40059 examples [00:43, 862.66 examples/s]40154 examples [00:43, 884.87 examples/s]40247 examples [00:44, 897.75 examples/s]40338 examples [00:44, 899.24 examples/s]40429 examples [00:44, 902.20 examples/s]40521 examples [00:44, 906.39 examples/s]40612 examples [00:44, 895.35 examples/s]40702 examples [00:44, 896.31 examples/s]40801 examples [00:44, 920.31 examples/s]40895 examples [00:44, 925.03 examples/s]40992 examples [00:44, 936.92 examples/s]41088 examples [00:44, 941.60 examples/s]41185 examples [00:45, 948.28 examples/s]41280 examples [00:45, 942.84 examples/s]41375 examples [00:45, 940.88 examples/s]41470 examples [00:45, 939.36 examples/s]41565 examples [00:45, 941.08 examples/s]41661 examples [00:45, 943.95 examples/s]41756 examples [00:45, 944.58 examples/s]41851 examples [00:45, 930.19 examples/s]41945 examples [00:45, 932.88 examples/s]42039 examples [00:45, 921.96 examples/s]42133 examples [00:46, 924.89 examples/s]42229 examples [00:46, 933.13 examples/s]42323 examples [00:46, 930.89 examples/s]42418 examples [00:46, 935.13 examples/s]42513 examples [00:46, 937.70 examples/s]42607 examples [00:46, 932.95 examples/s]42701 examples [00:46, 930.63 examples/s]42795 examples [00:46, 922.79 examples/s]42888 examples [00:46, 924.92 examples/s]42981 examples [00:46, 926.23 examples/s]43075 examples [00:47, 929.84 examples/s]43168 examples [00:47, 921.80 examples/s]43261 examples [00:47, 919.25 examples/s]43353 examples [00:47, 913.56 examples/s]43446 examples [00:47, 917.22 examples/s]43538 examples [00:47, 916.12 examples/s]43633 examples [00:47, 924.65 examples/s]43726 examples [00:47, 912.56 examples/s]43818 examples [00:47, 911.90 examples/s]43912 examples [00:47, 919.20 examples/s]44007 examples [00:48, 927.48 examples/s]44101 examples [00:48, 929.38 examples/s]44194 examples [00:48, 921.60 examples/s]44287 examples [00:48, 922.28 examples/s]44380 examples [00:48, 922.05 examples/s]44473 examples [00:48, 917.70 examples/s]44566 examples [00:48, 921.02 examples/s]44659 examples [00:48, 918.40 examples/s]44756 examples [00:48, 932.55 examples/s]44850 examples [00:48, 919.78 examples/s]44944 examples [00:49, 923.53 examples/s]45039 examples [00:49, 928.74 examples/s]45132 examples [00:49, 927.55 examples/s]45227 examples [00:49, 933.56 examples/s]45321 examples [00:49, 931.44 examples/s]45415 examples [00:49, 926.13 examples/s]45508 examples [00:49, 925.05 examples/s]45601 examples [00:49, 919.17 examples/s]45693 examples [00:49, 918.33 examples/s]45785 examples [00:50, 911.06 examples/s]45878 examples [00:50, 913.88 examples/s]45971 examples [00:50, 915.95 examples/s]46063 examples [00:50, 916.97 examples/s]46157 examples [00:50, 923.12 examples/s]46250 examples [00:50, 918.28 examples/s]46342 examples [00:50, 917.22 examples/s]46435 examples [00:50, 918.80 examples/s]46527 examples [00:50, 909.47 examples/s]46621 examples [00:50, 916.63 examples/s]46716 examples [00:51, 924.12 examples/s]46813 examples [00:51, 935.84 examples/s]46909 examples [00:51, 940.17 examples/s]47004 examples [00:51, 937.37 examples/s]47099 examples [00:51, 939.57 examples/s]47193 examples [00:51, 939.60 examples/s]47287 examples [00:51, 928.74 examples/s]47380 examples [00:51, 928.74 examples/s]47476 examples [00:51, 936.51 examples/s]47573 examples [00:51, 946.17 examples/s]47669 examples [00:52, 949.23 examples/s]47768 examples [00:52, 958.89 examples/s]47864 examples [00:52, 953.98 examples/s]47960 examples [00:52, 874.30 examples/s]48052 examples [00:52, 884.90 examples/s]48145 examples [00:52, 896.60 examples/s]48239 examples [00:52, 907.93 examples/s]48333 examples [00:52, 916.15 examples/s]48426 examples [00:52, 915.18 examples/s]48518 examples [00:52, 907.01 examples/s]48610 examples [00:53, 908.34 examples/s]48702 examples [00:53, 909.90 examples/s]48794 examples [00:53, 911.59 examples/s]48889 examples [00:53, 920.86 examples/s]48985 examples [00:53, 931.19 examples/s]49079 examples [00:53, 929.25 examples/s]49175 examples [00:53, 936.59 examples/s]49269 examples [00:53, 934.11 examples/s]49363 examples [00:53, 928.84 examples/s]49460 examples [00:53, 938.11 examples/s]49554 examples [00:54, 921.85 examples/s]49647 examples [00:54, 923.05 examples/s]49743 examples [00:54, 931.64 examples/s]49837 examples [00:54, 911.95 examples/s]49929 examples [00:54, 913.24 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6177/50000 [00:00<00:00, 61767.41 examples/s] 34%|███▍      | 17138/50000 [00:00<00:00, 71072.91 examples/s] 56%|█████▋    | 28151/50000 [00:00<00:00, 79534.31 examples/s] 79%|███████▉  | 39615/50000 [00:00<00:00, 87579.72 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 746.92 examples/s]176 examples [00:00, 808.67 examples/s]272 examples [00:00, 845.96 examples/s]369 examples [00:00, 877.51 examples/s]467 examples [00:00, 903.10 examples/s]563 examples [00:00, 918.98 examples/s]659 examples [00:00, 928.43 examples/s]753 examples [00:00, 929.52 examples/s]848 examples [00:00, 935.15 examples/s]939 examples [00:01, 919.76 examples/s]1030 examples [00:01, 916.14 examples/s]1126 examples [00:01, 925.92 examples/s]1220 examples [00:01, 929.75 examples/s]1314 examples [00:01, 931.54 examples/s]1411 examples [00:01, 942.13 examples/s]1507 examples [00:01, 946.75 examples/s]1602 examples [00:01, 946.10 examples/s]1698 examples [00:01, 950.23 examples/s]1793 examples [00:01, 940.52 examples/s]1888 examples [00:02, 939.15 examples/s]1984 examples [00:02, 942.54 examples/s]2079 examples [00:02, 943.51 examples/s]2174 examples [00:02, 941.17 examples/s]2269 examples [00:02, 904.70 examples/s]2366 examples [00:02, 920.95 examples/s]2463 examples [00:02, 933.22 examples/s]2560 examples [00:02, 941.41 examples/s]2655 examples [00:02, 936.54 examples/s]2753 examples [00:02, 947.87 examples/s]2851 examples [00:03, 956.16 examples/s]2947 examples [00:03, 957.11 examples/s]3043 examples [00:03, 955.10 examples/s]3139 examples [00:03, 923.54 examples/s]3235 examples [00:03, 931.62 examples/s]3331 examples [00:03, 938.67 examples/s]3430 examples [00:03, 951.11 examples/s]3526 examples [00:03, 941.64 examples/s]3623 examples [00:03, 949.84 examples/s]3719 examples [00:03, 948.61 examples/s]3816 examples [00:04, 953.05 examples/s]3913 examples [00:04, 956.63 examples/s]4014 examples [00:04, 970.56 examples/s]4112 examples [00:04, 967.64 examples/s]4209 examples [00:04, 959.36 examples/s]4306 examples [00:04, 960.02 examples/s]4403 examples [00:04, 961.13 examples/s]4500 examples [00:04, 936.02 examples/s]4594 examples [00:04, 927.73 examples/s]4689 examples [00:04, 933.84 examples/s]4787 examples [00:05, 946.48 examples/s]4882 examples [00:05, 936.77 examples/s]4980 examples [00:05, 949.31 examples/s]5076 examples [00:05, 930.67 examples/s]5170 examples [00:05, 931.19 examples/s]5265 examples [00:05, 934.75 examples/s]5360 examples [00:05, 938.02 examples/s]5456 examples [00:05, 943.83 examples/s]5551 examples [00:05, 940.90 examples/s]5646 examples [00:06, 930.01 examples/s]5740 examples [00:06, 911.79 examples/s]5834 examples [00:06, 919.31 examples/s]5928 examples [00:06, 924.84 examples/s]6021 examples [00:06, 922.15 examples/s]6118 examples [00:06, 933.40 examples/s]6215 examples [00:06, 942.76 examples/s]6311 examples [00:06, 947.28 examples/s]6406 examples [00:06, 947.10 examples/s]6501 examples [00:06, 937.55 examples/s]6595 examples [00:07, 921.32 examples/s]6692 examples [00:07, 933.37 examples/s]6788 examples [00:07, 939.13 examples/s]6882 examples [00:07, 917.41 examples/s]6974 examples [00:07, 906.40 examples/s]7068 examples [00:07, 915.42 examples/s]7165 examples [00:07, 930.93 examples/s]7261 examples [00:07, 939.22 examples/s]7361 examples [00:07, 956.57 examples/s]7457 examples [00:07, 953.59 examples/s]7553 examples [00:08, 944.58 examples/s]7648 examples [00:08, 936.77 examples/s]7743 examples [00:08, 938.12 examples/s]7844 examples [00:08, 956.56 examples/s]7940 examples [00:08, 943.40 examples/s]8035 examples [00:08, 933.05 examples/s]8130 examples [00:08, 936.61 examples/s]8226 examples [00:08, 940.81 examples/s]8321 examples [00:08, 935.14 examples/s]8415 examples [00:08, 924.43 examples/s]8508 examples [00:09, 921.25 examples/s]8603 examples [00:09, 927.59 examples/s]8696 examples [00:09, 917.57 examples/s]8788 examples [00:09, 915.19 examples/s]8880 examples [00:09, 912.78 examples/s]8973 examples [00:09, 916.36 examples/s]9067 examples [00:09, 920.84 examples/s]9162 examples [00:09, 927.21 examples/s]9255 examples [00:09, 924.70 examples/s]9348 examples [00:09, 925.22 examples/s]9444 examples [00:10, 933.61 examples/s]9538 examples [00:10, 930.24 examples/s]9632 examples [00:10, 922.81 examples/s]9725 examples [00:10, 921.44 examples/s]9818 examples [00:10, 886.34 examples/s]9913 examples [00:10, 902.56 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLL3QSA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLL3QSA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f61b3bdea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f61b3bdea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f61b3bdea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f61ff5c50b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f613df1efd0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f61b3bdea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f61b3bdea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f61b3bdea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f619ae8e0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f619ae8e240>), {}) 

  




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
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




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
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 15.19 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 25.08 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.39 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.27 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.67 file/s]2020-08-06 00:10:43.507412: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-06 00:10:43.512149: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-06 00:10:43.512319: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5600a2ab68a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-06 00:10:43.512336: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  1%|          | 90112/9912422 [00:00<00:10, 895013.69it/s] 70%|███████   | 6979584/9912422 [00:00<00:02, 1271503.91it/s]9920512it [00:00, 42176837.27it/s]                            
0it [00:00, ?it/s]32768it [00:00, 619672.19it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 488185.82it/s]1654784it [00:00, 12333065.88it/s]                         
0it [00:00, ?it/s]8192it [00:00, 266466.10it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
