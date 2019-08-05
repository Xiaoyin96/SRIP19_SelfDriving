# Set up remote debugging in kubernetes with VS Code  

## Setp 1

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

## Step 2

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
## Step 3

Run the python file in pod:
```
python temp.py
```

Open another terminal, run:
```
kubectl port-forward mrcnn-pod 5678:5678
```

## Step 4

Start dubugging!
