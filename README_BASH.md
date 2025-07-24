脚本
```bash
clash -d /etc/clash
redis-server
cd firecrawl/apps/api
pnpm run workers
pnpm run start
cd DEFAME
conda activate DEFAME
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
python -m scripts.run_config
```

error:
```
Error encountered in worker pool main thread:
Traceback (most recent call last):
  File "/root/autodl-tmp/DEFAME/defame/helpers/parallelization/pool.py", line 52, in run
    self.process_messages()
  File "/root/autodl-tmp/DEFAME/defame/helpers/parallelization/pool.py", line 110, in process_messages
    for msg in worker.get_messages():
  File "/root/autodl-tmp/DEFAME/defame/helpers/parallelization/worker.py", line 35, in get_messages
    msgs.append(self._connection.recv())
  File "/root/miniconda3/envs/DEFAME/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/root/miniconda3/envs/DEFAME/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "/root/miniconda3/envs/DEFAME/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError
```