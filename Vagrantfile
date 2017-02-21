Vagrant.configure(2) do |config|

  config.vm.provider :virtualbox do |v|
    v.memory = 4096
  end

  config.vm.box = "ubuntu/trusty64"

  config.vm.network "forwarded_port", guest: 8888, host: 8888, host_ip: "127.0.0.1", auto_correct: true
  config.vm.network :forwarded_port, guest: 22, host: 22, id: "ssh", auto_correct: true
    
  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    apt-get dist-upgrade
    # Install PyMC and its requirements
    apt-get install -y build-essential gfortran libatlas-base-dev python-pip python-dev python-matplotlib ipython ipython-notebook python-pandas git-all
    pip install --upgrade pip
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
    pip install --upgrade $TF_BINARY_URL
    pip install edward
    # Edward installed these as requirements:
    # pip install numpy
    # pip install scipy
    mkdir -p /vagrant/notebooks
    git clone https://github.com/pymc-devs/pymc3
    cd pymc3
    pip install -r requirements.txt .
## Update:
    apt-get install libfreetype6-dev
    pip install -U git+https://github.com/pymc-devs/pymc3
    # Install additonal requirements for norm learning code
    apt-get -y install graphviz libgraphviz-dev gnuplot-nox pkg-config
    pip install psutil
    pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
  SHELL

  config.vm.provision "shell", run: "always", inline: <<-SHELL
    ipython notebook --notebook-dir=/vagrant/notebooks --no-browser --ip=0.0.0.0 &
  SHELL

end
