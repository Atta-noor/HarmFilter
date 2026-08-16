# 🚀 How to Launch the App in Google Chrome

## Method 1: Automatic Launch (Recommended)

### Option A: Using the Batch File
1. Double-click `run_app_chrome.bat`
2. Chrome will open automatically with the app
3. The app will be available at `http://localhost:8501`

### Option B: Using the Simple Batch File
1. Double-click `run_app.bat`
2. The app will open in your default browser
3. If Chrome doesn't open automatically, copy the URL and paste it in Chrome

## Method 2: Manual Launch

### Step 1: Open Terminal/PowerShell
Navigate to your project directory:
```bash
cd "D:\one drive\OneDrive - Higher Education Commission\Desktop\archive (5)"
```

### Step 2: Run Streamlit
```bash
streamlit run app.py
```

### Step 3: Open in Chrome
1. The terminal will show a URL like: `http://localhost:8501`
2. Copy this URL
3. Open Google Chrome
4. Paste the URL in the address bar
5. Press Enter

## Method 3: Direct Chrome Launch

### Step 1: Run Streamlit in Background
```bash
streamlit run app.py
```

### Step 2: Open Chrome Manually
1. Open Google Chrome
2. Type in address bar: `localhost:8501`
3. Press Enter

## 🔧 Troubleshooting

### Port Already in Use
If you see "Port 8501 is already in use":
```bash
streamlit run app.py --server.port 8502
```
Then open `http://localhost:8502` in Chrome

### Chrome Doesn't Open Automatically
1. Run: `streamlit run app.py`
2. Look for the URL in the terminal (usually `http://localhost:8501`)
3. Manually open Chrome and navigate to that URL

### Models Not Found
Make sure you've trained the models first:
- For traditional ML: `python train_and_save_models.py`
- For LSTM: `python train_lstm_model.py`

## 📱 Using the App

1. **Select a Model**: Choose from the sidebar
   - Random Forest
   - Decision Tree
   - AdaBoost
   - BiLSTM (Advanced) - if you trained it
   - LSTM (Advanced) - if you trained it

2. **Enter Text**: Type or paste text in the text area

3. **Analyze**: Click "🔍 Analyze Text" button

4. **View Results**: See prediction, confidence, and probabilities

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where Streamlit is running.

## 💡 Tips

- The app runs on `localhost:8501` by default
- You can access it from any browser once it's running
- The app will automatically reload when you make code changes
- Keep the terminal window open while using the app

---

**Quick Start**: Just double-click `run_app_chrome.bat` and Chrome will open automatically! 🎉





