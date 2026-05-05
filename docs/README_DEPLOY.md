# 🚀 Deploying AudLens for Free

I have prepared your project for free deployment on **Hugging Face Spaces**. This platform is the best choice because it provides 16GB of RAM for free, which is needed for your TensorFlow models.

### 1. Create a Hugging Face Account
Go to [huggingface.co](https://huggingface.co/) and create a free account.

### 2. Create a New Space
1. Click on **"New"** -> **"Space"**.
2. Give it a name (e.g., `audlens-ai`).
3. Select **Docker** as the SDK.
4. Choose the **"Blank"** template.
5. Select the **Public** or **Private** visibility.
6. Click **"Create Space"**.

### 3. Upload Your Files
Upload the following from your project folder to the new Space:
- `backend/` (all contents)
- `model/` (all contents, including your `.h5` model files)
- `src/`, `public/`, `index.html`, `vite.config.ts`, `tsconfig.json`, `package.json` (for the frontend build)
- `Dockerfile` (The one I created for you)
- `requirements.txt` (The one I created for you)

### 4. Wait for the Build
Hugging Face will automatically detect the `Dockerfile`, build your frontend, install the Python dependencies, and start the server. 

### Why this works:
- **Dockerfile**: Automates the build of both React and Python.
- **FastAPI Static Serving**: The backend is now configured to serve the frontend UI on the same URL.
- **Relative API Paths**: The frontend is now configured to talk to the backend without needing `localhost:8000`.

**Note**: Your large model files (`.h5`) must be uploaded. If they are over 100MB, Hugging Face might ask you to use **Git LFS** (Large File Storage), which is also free.
