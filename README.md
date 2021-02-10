#RCNN

<h3>Environment</h3>
<ul>
<li>System : Ubuntu 18.04.5 LTS</li>
<li>Language : Python 3.6.9</li>
<li> OpenCV : 4.5.1.48 </li>
<li> Tensorflow : 2.4.1  </li>
</ul>

<h3>Python File Description</h3>
<ul>
<li>CV_Detector.py: Use the OpenCV module to detect objects in the image.</li>
<li>TF_Detector.py : Use the Tensorflow module to detect objects in the image.</li>
<li>Convert2OpenCV.py : Convert Tensorflow model to OpenCV model.</li>
<li>ShowGPU.py : list available gpus. </li>
<li>Train.py : Run rcnn Train.  </li>
</ul>

<h3>Optional python file command arguments:</h3>
<ul>
<li>-h, --help : show this help message and exit</li>
<li>-c CONFIG, --config CONFIG :  Command '-c xxx' ,Choose config file, default file is
                        'default.cfg.ini.'</li>
<li>-n NEW, --new NEW : Create new config file.</li>
</ul>

<h3>Run Train.py usage custom.cfg.ini configure</h3>
<ul>
<li>Setp.1 : Create a configuration file, type commands in the terminal 'python Train.py -n custom'</li>
<li>Setp.2 : Set the parameters in custom.cfg.ini.</li>
<li>Setp.3 : Now run the training, type the command in the terminal 'python Train.py -c custom'</li>
</ul>