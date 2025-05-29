onda on# Assignment 1 EMLO end to end process

- Starting the coding part on local machine 
    . Starting with env setup using conda and poetry.
    . Create conda environment
        CMD: conda create -n emlo_ass2 python=3.10 
    . Get into the env.
        CMD: conda activate emlo_ass2 
    . Install poetry in the env. Dont forget to use -m while installing libraries to do env level installations as your pip might still be pointing to system level lib. you can verify it using CMDs: which python & which pip.
        CMD: python -m pip install poetry
    . To Enable tab completion for Bash, Fish, or Zsh for poetry: follow the below chain of commands
        CMD: python -m poetry completions zsh > ~/.zfunc/_poetry
        . You must then add the following lines in your ~/.zshrc, if they do not already exist:
            fpath+=~/.zfunc
            autoload -Uz compinit && compinit
        . Then run following commands
            CMD: export ZSH_CUSTOM="$HOME/.oh-my-zsh/custom"
            CMD: mkdir -p $ZSH_CUSTOM/plugins/poetry
            CMD: python -m poetry completions zsh > $ZSH_CUSTOM/plugins/poetry/_poetry
        . Add poetry in plugins array of .zshrc
            plugins=(poetry, ...existing plugins)
        . Then run:
            CMD: source ~/.zshrc
        . Try poetry auto completion now. it should work.
    . After setting up poetry, Let's create a new poetry project
        CMD: python -m poetry new train_mnist_on_cpu
        . Created project will have tree like:
            train_mnist_on_cpu
            ├── pyproject.toml
            ├── README.md
            ├── src
            │   └── train_mnist_on_cpu
            │       └── __init__.py
            └── tests
                └── __init__.py
        . The pyproject.toml file is what is the most important here. This will orchestrate your project and its dependencies
        . If you want to add dependencies to your project, you can specify them in the project section.
            eg: [project]
                # ...
                dependencies = [
                    "pendulum (>=2.1,<3.0)"
                ]
        . Also, instead of modifying the pyproject.toml file by hand, you can use the add command. eg:
            CMD: $ poetry add numpy
        . To run your script simply use poetry run python your_script.py. Likewise if you have command line tools such as pytest or black you can run them using poetry run pytest. However you not need to use poetry run since we are already using poetry inside conda env.
        . If managing your own virtual environment externally, you do not need to use poetry run since you will, presumably, already have activated that virtual environment and made available the correct python instance. For example, these commands should output the same python path: *verify before proceeding to avoid any env issues later
            CMDs: 
                conda activate your_env_name
                which python
                poetry run which python
                poetry env activate
                which python
        . To install the defined dependencies for your project, just run the install command.
            CMD: poetry run
    . Install the most important library : Pytorch
        . visit : https://pytorch.org
        . We have to install pytorch using pip, as poetry struggles to add all dependencies of pytorch: so run:
            CMD: python -m install torch torchvision torchaudio
        . Note: You can do dir(torch.Tensor) to see all the methods possible for Torch tensor. and you can do help(torch.Tensor.<method>) to see it's official docstring.
    . 
