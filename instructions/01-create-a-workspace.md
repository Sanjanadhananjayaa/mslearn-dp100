# Create and Explore an Azure Machine Learning Workspace

## Overview

In this exercise, you will create and explore an Azure Machine Learning workspace.

## Create an Azure Machine Learning workspace

As its name suggests, a workspace is a centralized place to manage all of the Azure ML assets you need to work on a machine learning project.

1. Open the `Azure portal` shortcut present in the virtual machine desktop, and login to the azure portal using the credentials provided in the **Environment Details** tab.
2. On the top search bar, type and select **Azure Machine learning**. Click on **+ Create** and then **New workspace** to create a new **Machine Learning workspace**. Enter the following details under the **Basics** tab:

    ![](images/ml-workspace.png)

    - **Subscription**: *Your Azure subscription*
    - **Resource group**: Select **dp-100-<inject key="DeploymentID" enableCopy="false"/>**
    - **Workspace name**: **quick-starts-ws-<inject key="DeploymentID" enableCopy="false"/>**
    - **Region**: **Select the geographical region closest to you**
    - **Storage account**: *Note the default new storage account that will be created for your workspace*
    - **Key vault**: *Note the default new key vault that will be created for your workspace*
    - **Application insights**: *Note the default new application insights resource that will be created for your workspace*
    - **Container registry**: None (*one will be created automatically the first time you deploy a model to a container*)


    ![](images/mlcreate.png)

3. Click on **Review+create** then click on **Create**.

4. When the deployment is completed, verify the workspace and its associated resources in the **dp-100-<inject key="DeploymentID" enableCopy="false"/>** Resource Group.

## Explore Azure Machine Learning studio

You can manage some workspace assets in the Azure portal, but for data scientists, this tool contains lots of irrelevant information and links that relate to managing general Azure resources. *Azure Machine Learning studio* provides a dedicated web portal for working with your workspace.

1. On the `Azure portal` blade in the top search bar, search for **Machine learning** and select the Machine learning workspace you created in the previous task. 
2. On the **Overview** page click on the **launch studio** it will open your machine learning workspace in a new tab. Alternatively in a new browser tab, open [https://ml.azure.com](https://ml.azure.com). If prompted, log in using the credentials provided in the **Environment Details** tab and select your Azure subscription and workspace.

    ![](images/launchml.png)

    > **Tip** If you have multiple Azure subscriptions, you need to choose the Azure *directory* in which the subscription is defined; then choose the subscription, and finally choose the workspace.

3. View the Azure Machine Learning studio interface for your workspace - you can manage all of the assets in your workspace from here.
4. In Azure Machine Learning Studio, toggle the &#9776; icon at the top left to show and hide the various pages in the interface. You can use these pages to manage the resources in your workspace.

    ![](images/ml-dashboard.png)

## Create a compute instance

One of the benefits of `Azure Machine Learning` is the ability to create cloud-based compute on which you can run experiments and training scripts at scale.

1. In `Azure Machine Learning Studio`, view the **Compute** page on the left panel. This is where you'll manage to compute resources for your data science activities. There are four kinds of compute resources you can create:
    - **Compute instances**: Development of workstations that can be used by data scientists to work with data and models.
    - **Compute clusters**: Scalable clusters of virtual machines for on-demand processing of experiment code.
    - **Inference clusters**: Deployment targets for predictive services that use your trained models.
    - **Attached compute**: Links to other Azure compute resources, such as Virtual Machines or Azure Databricks clusters.

    For this exercise, you'll create a compute instance so you can run some code in your workspace.

    ![](images/compute-1.png)

2. On the **Compute instances** tab, Click on **NEW** to add a new compute instance with the following settings. You'll use this as a workstation to run code in notebooks.
    - **Compute name**: *enter a unique name eg:machinelearning<inject key="DeploymentID" enableCopy="false"/>*
    - **Location**: *The same region as your workspace*
    - **Virtual machine type**: CPU
    - **Total Available Quotas**:  This shows dedicated cores available.
        - **Virtual machine size**: Standard_DS11_v2 
    - Select **Next: Advanced Settings**
    - **Advanced settings**: Note the following settings, but do not select them:-
        - **Enable SSH access**: Unselected (you can use this to enable direct access to the virtual machine using an SSH client)
        - **Enable virtual network**: Unselected (you would typically use this in an enterprise environment to enhance network security)
        - **Assign to another user**: Unselected (you can use this to assign a compute instance to a data scientist)

    ![](images/12.png)
    
    ![](images/image13.png)
    
    ![](images/14.png)

3. Click on the **Create** button

4. Wait for the compute instance to start and its state to change to **Running**.

    ![](images/16.png)

## Clone and run a notebook

A lot of data science and machine learning experimentation is performed by running code in *notebooks*. Your compute instance includes fully featured Python notebook environments (*Jupyter* and *JuypyterLab*) that you can use for extensive work, but for basic notebook editing, you can use the built-in **Notebooks** page in Azure Machine learning studio.

1. In Azure Machine Learning Studio, view the **Notebooks** page on the left panel. If a message describing new features is displayed, close it.

2. Click on the **open a terminal** icon to use the terminal to clone a git repository, and ensure that its **Compute instance** is set to the instance that was created in the previous task. The current path must be **/Users/your-user-name** folder. Please refer to the below image for more clarity.

    ![](images/module1/1.png)

3. Enter the following commands to clone a Git repo containing notebooks, data, and other files to your workspace:

    ```bash
    git clone https://github.com/MicrosoftLearning/mslearn-dp100 mslearn-dp100
    ```

    ![](images/module1/2.png)

4. When the command has completed, in the **Files** pane, click **&#8635;** to refresh the view and verify that a new **Users/*odl_user_{DeploymentID}*/mslearn-dp100** folder has been created. This folder contains multiple **.ipynb** notebook files.

    ![](images/module1/3.png)

5. Close the terminal pane, terminating the session.
6. In the **Users/*odl_user_<inject key="DeploymentID" enableCopy="false"/>*/mslearn-dp100** folder, open the **Get Started with Notebooks** notebook. Then read the notes and follow the instructions it contains.

> **Tip**: To run a code cell, select the cell you want to run and then use the **&#9655;** button to run it.
