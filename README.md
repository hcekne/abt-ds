# DS-ABM
This repository is built and maintained for providing tools to run Agent-Based models for data science tasks.

The tool is built on a docker image with a Jupyter notebook for easy testing, development and portability. Follow the guide to run and use.

The intended way to interact with and use the tool is through the docker container, however if you just want to grab the code and run it directly on your own machine that is totally possible. All the code for the agent-based models can be found in the /src folder.

----------------------------------------------------------------------------------

## Getting Started

Prerequisites: If you want to run the container make sure you have installed Docker and have permissions to run it.

To run:

1. Clone the repo and navigate to the root folder
2. Edit the runContainer.sh file with your values for NB_USER (your username, run "whoami" in a linux terminal), NB_UID and GID (run "id <your username>" on a linux system to see these) (The reason to do is because then when the container edits and makes files that are stored in your folders, they will be stored as if your user had made them and you don't get any issues with permissions). Also edit the line   -v "${PWD}":/home/hcekne/work/ \ and replace hcekne with your username.
3. Run the buildImage.sh file ( type "sudo bash ./buildImage.sh")
   You will get some warnings when you build the image but these can be ignored.
4. Then run the docker container with runContainer.sh  (type "sudo bash ./runContainer.sh")
   Access the notebook by CRTL click on the link that comes up in the terminal
5. Play around with the tools!

There are a few different ways to interact with the tools:

1. Either use the Jupyter notebook which we started with the above scripts
    This option is perhaps the easiest and only requires you to follow the browser link that was generated when you ran the runContainer.sh script
    Navigate to the notebooks folder and open the introduction notebook


2. Jump into the running container and run/write code directly in the terminal or
    To jump into the running container and work from inside the container
   a. run "sudo docker ps -a" to see the running containers
   b. identify the right container by finding the image with the image name "abt_ds" and copy the container ID
   c. run the following command in the terminal: "sudo docker exec -it <container ID> /bin/bash"
   d. you are now inside the container, navigate to the /work directory to access the project files
   e. to start running python and scripts directly simply type "python3.10" inside the container and get started!

4. Using a tool like VS code and remote development
  This link is a useful guide for how to do this with VS Code:
  https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host#_connect-using-the-remote-ssh-extension-recommended

  Follow the instructions for "Connect using the Docker CLI"

  

----------------------------------------------------------------------------------

## Developing source code

All the source code for the tool is located in the /src folder. This folder is mapped directly into the container when it is run.


