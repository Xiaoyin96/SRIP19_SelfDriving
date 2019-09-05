# Set up remote debugging in kubernetes with VS Code  

## Method 1 (prefered)
### Setp 1
Open a terminal at local. Start a pod in kubernetes:
```
kubectl create -f Your\path\to\base-pod.yaml
```
```
kubectl get pods
```
```
kubectl exec base-pod -it bash
```
In pod:
```
passwd
xx
xx
service ssh start
```
Open another terminal at local:
```
kubectl port-forward base-pod 2222:22
```
Don't close it.

### Setp 2
In VSCode, install remote develpoment extension.
Click configure (next to CONNECTIONS), add:
```
Host base-pod
    HostName localhost
    User root
    Port 2222
```
Save config.

Click Connect to host in new window.
Type passwd you just created.
Install Python extension in remote.
Start debugging!


## Method 2
### Setp 1

Start a pod in kubernetes:

```
kubectl create -f Your\path\to\mrcnn-pod.yaml
```
```
kubectl get pods
```
```
kubectl exec mrcnn-pod -it bash
```

Install ptvsd in pod if don't have:
```
pip install ptvsd
```

### Step 2

Prepare 2 same python script at both pod and local.  

Additionally, in VS Code, add debug configuration (open launch.json):
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach (Remote Debug)",
            "type": "python",
            "request": "attach",
            "localRoot": "${workspaceRoot}",
            "remoteRoot": "/home/selfdriving",
            "port": 5678,
            "host":"localhost"
        }
    ]
}
```
### Step 3

Run the python file in pod (--wait means wait until the debugger attaches before running your code):
```
python -m ptvsd --host 0.0.0.0 --port 5678 --wait temp.py
```

Open another terminal, run:
```
kubectl port-forward mrcnn-pod 5678:5678
```

### Step 4

Start dubugging!

## Method 2
### Setp 1

Same as above

### Step 2

Prepare 2 same python script at both pod and local.  

At beginning of each script, add:

```
import ptvsd
ptvsd.enable_attach()
ptvsd.wait_for_attach()
```

Additionally, in VS Code, add debug configuration (open launch.json):
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach (Remote Debug)",
            "type": "python",
            "request": "attach",
            "localRoot": "${workspaceRoot}",
            "remoteRoot": "/home/selfdriving",
            "port": 5678,
            "host":"localhost"
        }
    ]
}
```
### Step 3

Run the python file in pod:
```
python temp.py
```

Open another terminal, run:
```
kubectl port-forward mrcnn-pod 5678:5678
```

### Step 4

Start dubugging!
