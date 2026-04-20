# HubDigitalCoWorker
# HubDigitalCoWorker
# IFRSHubDigitalCoWorker
# IFRSHubDigitalCoWorker

## API smoke test (poll-until-terminal)

Run:

```bash
python backend/api_smoke_test.py --base-url http://localhost:3000
```

This script starts `ask`, `followup`, and `translate`, and polls `/api/updatestatus` until each prompt reaches a terminal status (`Completed` or `Failed`).  
It intentionally has **no max polling limit and no timeout cap** on the polling loop.
