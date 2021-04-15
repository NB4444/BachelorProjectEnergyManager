FROM continuumio/anaconda3:latest

# Install dependencies
RUN \
	apt-get -y update \
	&& curl -sL https://deb.nodesource.com/setup_14.x | bash - \
	&& apt-get -y install \
		git \
		nodejs

# Add the environment file
ADD ./Visualizer/environment.yml /EnergyManager/Visualizer/environment.yml

# Set up the environment
RUN \
	cd /EnergyManager/Visualizer \
	&& conda env create --prefix /CondaEnvironment --file ./environment.yml \
    && /CondaEnvironment/bin/jupyter labextension install @jupyter-widgets/jupyterlab-manager @jupyterlab/toc jupyter-matplotlib

# Run the notebooks
CMD \
	cd /EnergyManager/Visualizer/Source \
	&& /CondaEnvironment/bin/jupyter lab --notebook-dir=./ --ip="*" --port=8888 --no-browser --NotebookApp.token="" --NotebookApp.password="" --allow-root
