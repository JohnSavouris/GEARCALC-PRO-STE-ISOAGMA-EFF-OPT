# GEARCALC PRO READY PACKAGE (Efficiency + Optimization)

This folder is ready to upload to a **new GitHub repository** and deploy as a **new Render Web Service**.

## 1) Included files

- `index.html`
  - Home, Geometry, Strength (AGMA+ISO), STE (NN), Efficiency, Optimization Lab, Generation Lab, Theory, Exports
- `app/main.py`
  - API routes: `GET /`, `GET /healthz`, `POST /predict_ste`, `POST /predict_efficiency`, `POST /optimize_design`
- `app/ste_service.py`
- `app/efficiency_service.py`
  - 4 core curves: load distribution, sliding ratio, EHL friction, differential power loss
  - lubricant-aware model (family, ISO VG, oil temperature, additives, auto/manual viscosity)
- `app/optimization_service.py`
  - server-side intelligent redesign advisor with ranked scenarios
- `NN_model/`
  - STE model files
- `requirements.txt`
- `render.yaml`

## 2) New GitHub repo steps

1. Create a new empty GitHub repository.
2. Open this folder: `READY_TO_UPLOAD_WITH_EFF`.
3. Upload **all** files/folders from this folder to repository root.
4. Verify repository root contains:
   - `index.html`
   - `app/`
   - `NN_model/`
   - `requirements.txt`
   - `render.yaml`

Optional Git CLI flow:

```bash
git init
git add .
git commit -m "GEARCALC PRO: Efficiency + Optimization"
git branch -M main
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

## 3) Render deployment steps (detailed)

1. Go to Render dashboard.
2. Click `New` -> `Web Service`.
3. Connect the new GitHub repository.
4. Service settings:
   - Runtime: `Python`
   - Environment variable: `PYTHON_VERSION=3.11.9`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Create and deploy.
6. After deploy:
   - Open `https://<your-service>.onrender.com/healthz`
   - Expected output: `{"status":"ok"}`
7. Open the main URL and verify tabs:
   - `Efficiency`
   - `Optimization Lab`

## 4) What is upgraded in this version

1. Efficiency page has exactly the 4 requested charts:
   - Load distribution vs distance from pitch point
   - Sliding ratio vs distance from pitch point
   - EHL friction coefficient vs distance from pitch point
   - Differential power loss vs distance from pitch point
2. Added lubricant options:
   - lubricant family
   - ISO VG
   - oil temperature
   - additive package
   - auto/manual viscosity mode
3. Added server-side Optimization Lab:
   - diagnostics
   - ranked redesign scenarios
   - apply-ready parameter proposals
4. Upgraded visual quality:
   - refined theme
   - stronger chart readability
   - pitch-point marker and curve annotations in Efficiency charts
