# PyTorch training metrics viewer

A standalone browser dashboard for metric JSON files named like:

```text
exPreCastModel_2026_08_27_20_08_train_3.json
```

## Use

Put JSON files in `data`, then start the included local server:

```powershell
python server.py
```

Visit `http://127.0.0.1:8000`. The page automatically loads every `.json` file under `data`.

You can also open `index.html` directly, click **Choose folder**, or drag a folder onto the upload area. In that mode, all parsing happens locally in the browser.

The page groups files by model name and run type, then orders checkpoints by numeric run ID. If restarted training produced several files with the same run ID, the newest checkpoint is shown.

Use a different data folder or port when needed:

```powershell
python server.py --data D:\path\to\metrics --port 8080
```
