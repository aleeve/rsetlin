# Tsetlin based language model

## Developer setup

* Make sure you installed rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

* python is easiest installed by going to https://www.python.org/downloads/
and getting a downloading an installer


* Now we need pipx, 
```
brew install pipx
pipx ensure path
```
see https://pypa.github.io/pipx/installation/ for alternative methods 

* Then install maturin
```
pipx install maturin
```

* Create a venv, for example at the root of the project
```
python3 -m venv .venv
```

## Develop

* Activate the venv
```
source .venv/bin/activate
```
* Build the rust libs and and install them in the active venv
```
maturin develop
```
repeat this step when for testing new changes to the rust code


