# abt-ds
This repository is built and maintained for providing tools to run Agent-Based models for data science tasks.

The tool is built on a docker image with a Jupyter notebook for easy testing, development and portability. Follow the guide to run and use.

The intended way to interact with and use the tool is through the docker container, however if you just want to grab the code and run it directly on your own machine that is totally possible. All the code for the agent-based models can be found in the /src folder.

----------------------------------------------------------------------------------

## Getting Started

To run:

1. Clone the repo and navigate to the root folder
2. Run the buildImage.sh file ( type "sudo bash ./buildImage.sh")
3. Then run the docker container with runContainer.sh  (type "sudo bash ./runContainer.sh")
4. Play around with the tools!

There are a few different ways to interact with the tools:

1. Either use the Jupyter notebook which we started with the above scripts
  This option is perhaps the easiest and only requires you to follow the browser link that was generated when you ran the runContainer.sh script  


2. Jump into the running container and run/write code directly in the terminal or
  To jump into the running container and

3. Using a tool like VS code and remote development
  This link is a useful guide for how to do this with VS Code:
  https://code.visualstudio.com/remote/advancedcontainers/develop-remote-host#_connect-using-the-remote-ssh-extension-recommended

  Follow the instructions for "Connect using the Docker CLI"

  

----------------------------------------------------------------------------------

## Developing source code

All the source code for the tool is located in the /src folder. This folder is mapped directly into the container when it is run.


