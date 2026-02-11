# NASDAQ Stock Streak Screener Dashboard

A professional Streamlit dashboard for screening NASDAQ stocks based on consecutive price streaks and analyzing subsequent returns versus the NASDAQ Composite Index.

**Features:**
- 📊 Interactive stock price explorer with comparison to NASDAQ Composite
- 🔍 Advanced streak analysis with customizable filters
- 📉 Macro-level probability tables and return visualizations
- 🔐 Password-protected access
- ☁️ Cloud-ready deployment (DigitalOcean, Heroku, etc.)

---

## Project Structure

```
.
├── nsdq_streamlit_core.py              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (PASSWORD)
├── .env.example                         # Example env file
├── .gitignore                          # Git ignore rules
├── .streamlit/
│   └── config.toml                     # Streamlit configuration
├── NK_market_cap_20260211.xlsx        # Market cap and company metadata (XLSX)
├── NK_stock_data_20260211.csv         # Daily OHLCV stock price data (CSV)
├── NK_nsdq_index_20260211.csv/xlsx    # NASDAQ Composite index data
└── README.md                            # This file
```

---

## Data Files

All three input data files must be placed in the **same directory** as the script:

### 1. `NK_market_cap_20260211.xlsx`
- **Format:** Excel spreadsheet
- **Sheet:** `NASDAQ_screener`
- **Required Columns:**
  - `Symbol` - Stock ticker symbol
  - `Name` (or) `Company` - Company name
  - `Market Cap` (or) `Market Cap $` - Market capitalization
  - `Sector` - Industry sector
  - `Industry` - Sub-industry classification

### 2. `NK_stock_data_20260211.csv`
- **Format:** CSV (comma-separated values)
- **Required Columns:**
  - `Date` - Trading date (YYYY-MM-DD format)
  - `Symbol` - Stock ticker symbol
  - `Close` - Closing price
  - `Open`, `High`, `Low`, `Volume` (optional but helpful)

### 3. `NK_nsdq_index_20260211.csv/xlsx`
- **Format:** CSV or XLSX
- **Required Columns:**
  - `Date` - Trading date (YYYY-MM-DD format)
  - `Symbol` - Should contain `^IXIC` (NASDAQ Composite ticker)
  - `Close` - Index closing value

---

## Local Development Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository** (from Gitlab)
   ```bash
   git clone https://gitlab.com/your-organization/nasdaq-streak-screener.git
   cd nasdaq-streak-screener
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and change DASHBOARD_PASSWORD to your desired password
   ```

5. **Run locally**
   ```bash
   streamlit run nsdq_streamlit_core.py
   ```

The dashboard will open at `http://localhost:8501`

---


## Deploy on DigitalOcean App Platform

1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com/) → **Apps** → **Create App**
2. **Source:** Connect your GitLab account → select the `nasdaq-streak-screener` repo, branch `main`
3. **Build:** DigitalOcean will auto-detect the `Dockerfile`
4. **Environment Variables:** Add:
   | Key            | Value                    | Type   |
   |----------------|--------------------------|--------|
   | `APP_PASSWORD` | `your_secure_password`   | Secret |
5. **Resources:** Select at least **Professional XS** (1 vCPU / 1GB RAM). For faster computation with many stocks, use **Professional S** (1 vCPU / 2GB).
6. **HTTP Port:** Confirm it's set to `8080`
7. Click **Create Resources**

The app will build and deploy. First deploy takes ~3-5 minutes.

---

## Step 3 — Access the Dashboard

- URL will be: `https://nasdaq-streak-screener-xxxxx.ondigitalocean.app`
- Enter the password you set in `APP_PASSWORD`
- All 3 tabs will be available

---

## Password Protection

- Controlled via the `APP_PASSWORD` environment variable
- If `APP_PASSWORD` is **not set**, the dashboard runs without a password (useful for local dev)
- Uses `hmac.compare_digest` for timing-safe comparison
- Password is never stored in session state after validation

**To change the password:** Update the `APP_PASSWORD` env var in DigitalOcean → App Settings → Environment Variables, then redeploy.


## Deployment to DigitalOcean

### Using DigitalOcean App Platform

1. **Push to GitLab**: Ensure all files are committed and pushed to your GitLab repository

2. **Create App in DigitalOcean**:
   - Go to https://cloud.digitalocean.com/
   - Navigate to App Platform
   - Click "Create App"
   - Connect your GitLab repository

3. **Configure Build Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `streamlit run nsdq_streamlit_core.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment**: Python 3.8+

4. **Set Environment Variables**:
   - In DigitalOcean App Platform settings, add environment variable:
     - `DASHBOARD_PASSWORD`: Your secure password
   - **Important**: Do not commit `.env` file - use DigitalOcean's environment variables instead

5. **Deploy**: Click "Deploy" and wait for the build to complete


## Cloud Deployment

### Option A: Deploy via Streamlit Community Cloud (Easiest)

1. **Push to Gitlab** (ensure `.env` is in `.gitignore` and NOT committed)
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Set up Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your Gitlab/GitHub account
   - Select your repository
   - Set deployment parameters

3. **Configure secrets in Streamlit Cloud**
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add: `DASHBOARD_PASSWORD=your_password_here`

### Option B: Docker Deployment on DigitalOcean App Platform

1. **Create `Dockerfile`** (add to project root)
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "nsdq_streamlit_core.py"]
   ```

2. **Create `docker-compose.yml`** (optional, for local testing)
   ```yaml
   version: '3'
   services:
     app:
       build: .
       ports:
         - "8501:8501"
       environment:
         - DASHBOARD_PASSWORD=your_password
       volumes:
         - ./:/app
   ```

3. **Push to Gitlab**
   - Ensure `.env` and secrets are NOT committed
   - Push Dockerfile and docker-compose.yml

4. **Deploy to DigitalOcean**
   - Connect your Gitlab account to DigitalOcean
   - Create new App → Select your repository
   - Configure environment:
     - Set `DASHBOARD_PASSWORD` secret in the environment settings
   - Deploy

### Option C: Manual Cloud Run on DigitalOcean Droplet

1. **Create a Droplet** (Ubuntu 22.04 recommended)
   - Size: At least 2GB RAM
   - Enable IPv4

2. **SSH into Droplet**
   ```bash
   ssh root@your_droplet_ip
   ```

3. **Install dependencies**
   ```bash
   apt-get update
   apt-get install -y python3-pip git
   ```

4. **Clone repository**
   ```bash
   git clone https://gitlab.com/your-organization/nasdaq-streak-screener.git
   cd nasdaq-streak-screener
   ```

5. **Set up Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Create `.env` file with password**
   ```bash
   echo "DASHBOARD_PASSWORD=your_secure_password" > .env
   ```

7. **Run with systemd service** (recommended)
   Create `/etc/systemd/system/streamlit-nasdaq.service`:
   ```ini
   [Unit]
   Description=NASDAQ Streak Screener Streamlit App
   After=network.target

   [Service]
   Type=simple
   User=app-user
   WorkingDirectory=/home/app-user/nasdaq-streak-screener
   Environment="PATH=/home/app-user/nasdaq-streak-screener/venv/bin"
   ExecStart=/home/app-user/nasdaq-streak-screener/venv/bin/streamlit run nsdq_streamlit_core.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Then:
   ```bash
   systemctl daemon-reload
   systemctl enable streamlit-nasdaq
   systemctl start streamlit-nasdaq
   ```

8. **Set up reverse proxy with Nginx**
   ```nginx
   server {
     listen 80;
     server_name your-domain.com;

     location / {
         proxy_pass http://localhost:8501;
         proxy_set_header Host $host;
         proxy_set_header X-Real-IP $remote_addr;
     }
   }
   ```

---

## Configuration

### Password Protection
The dashboard requires a password to access. Set this via the `DASHBOARD_PASSWORD` environment variable:

**Local Development:** Edit `.env`
```
DASHBOARD_PASSWORD=your_password
```

**Cloud Deployment:** Set environment variable in your cloud platform's settings

### Streamlit Configuration
Customize the app appearance and behavior by editing `.streamlit/config.toml`:
- Theme colors
- Page layout
- Server configuration
- Upload file size limits

---

## Updating Data Files

To update the dashboard with new data:

1. Replace the three CSV/XLSX files with newer versions
2. Use the **exact same filenames** (or update the script's data loading section)
3. Ensure columns match the expected schema (see Data Files section)
4. Commit and push to Gitlab
5. The cloud deployment will automatically reload with new data

---

## Troubleshooting

### "Failed to load data" Error
- Verify all three data files are in the correct directory
- Check that file names match exactly (case-sensitive on Linux/Mac)
- Ensure columns are named correctly (see Data Files section)
- Check for special characters in file paths

### Password Not Working
- Verify `.env` file exists and contains `DASHBOARD_PASSWORD` variable
- On cloud platform, ensure environment variable is properly set in app settings
- Restart the app after changing password

### Out of Memory
- Reduce the size of data files
- Increase droplet size if on DigitalOcean
- Consider caching data locally

### Slow Performance
- Data is cached for 1 hour by default (adjustable via `@st.cache_data(ttl=3600)`)
- Reduce the number of stocks analyzed
- Upgrade to larger Droplet/instance

---

## Security Best Practices

⚠️ **IMPORTANT:**
1. **Never commit `.env` file** to public repositories
2. **Change default password** before deploying
3. **Use strong passwords** (min 12 characters, mixed case, numbers, symbols)
4. **Limit access** to authorized users only
5. **Use HTTPS** in production (enable SSL/TLS)
6. **Rotate credentials** regularly
7. Keep dependencies updated

---

## Support & Maintenance

### Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt
```

### Monitoring
- Check Streamlit Cloud/DigitalOcean logs for errors
- Monitor memory and CPU usage
- Track dashboard access frequency

### Data Refresh Schedule
Recommend updating data files:
- Daily (for up-to-date stock analysis)
- Weekly (for broader trend analysis)
- Monthly (for historical comparisons)

---

## Built With

- **Streamlit** - Interactive web framework
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing
- **Openpyxl** - Excel file handling

---

## License

This project is confidential.

---

## Author


Cloud-ready edition: February 2026

For questions or support, contact your IT administrator.
